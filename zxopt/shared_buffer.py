"""Verified circuit restart exchange. PPO trajectories remain producer-local.

SQLite WAL transactions provide one durable, bounded pool per fixed-wire root.
Only unitary NAM Clifford+T circuits of <=7 wires are admitted in v1.
"""
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time

from qiskit import qasm2
from qiskit.quantum_info import Operator
import numpy as np


def canonical(qasm):
    circuit = qasm2.loads(qasm)
    if circuit.num_qubits > 7 or circuit.num_qubits < 1 or circuit.num_clbits:
        raise ValueError('shared exchange v1 supports 1..7 wires, no classical bits')
    if len(circuit) > 10000 or not set(circuit.count_ops()) <= {'h', 'x', 'rz', 'cx'}:
        raise ValueError('shared exchange requires bounded unconditional NAM')
    for item in circuit.data:
        if item.operation.name == 'rz':
            value = float(item.operation.params[0]) / (math.pi / 4)
            if not math.isfinite(value) or abs(value-round(value)) > 1e-9:
                raise ValueError('ZX Clifford+T boundary rejects non-pi/4 rotations')
    text = qasm2.dumps(circuit)
    return circuit, text, hashlib.sha256(text.encode()).hexdigest()


class SharedBuffer:
    def __init__(self, path, root_qasm, producer, capacity=512):
        if capacity < 2 or not producer:
            raise ValueError('capacity>=2 and producer required')
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.producer = producer
        self.root, self.root_qasm, self.root_key = canonical(root_qasm)
        self.reference = Operator(self.root).data
        self.db = sqlite3.connect(str(path), timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL'); self.db.execute('PRAGMA synchronous=FULL')
        with self.db:
            self.db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            self.db.execute('CREATE TABLE IF NOT EXISTS candidates (key TEXT PRIMARY KEY, qasm TEXT NOT NULL, gates INTEGER, cx INTEGER, depth INTEGER, producer TEXT, source TEXT, error REAL, created REAL)')
            self.db.execute('CREATE TABLE IF NOT EXISTS objects (key TEXT PRIMARY KEY, qasm TEXT NOT NULL)')
            self.db.execute('CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, time REAL, producer TEXT, kind TEXT, key TEXT, detail TEXT)')
            for key, value in [('schema', '1'), ('root', self.root_key), ('capacity', str(capacity))]:
                self.db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)', (key, value))
                if self.db.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()[0] != value:
                    raise ValueError('shared pool metadata mismatch: '+key)
        self.capacity = capacity
        if not self.db.execute('SELECT 1 FROM candidates WHERE key=?', (self.root_key,)).fetchone():
            self.publish(self.root_qasm, {'root': True})

    def event(self, kind, key=None, **detail):
        with self.db:
            self.db.execute('INSERT INTO events(time,producer,kind,key,detail) VALUES(?,?,?,?,?)',
                            (time.time(), self.producer, kind, key, json.dumps(detail, sort_keys=True)))

    def publish(self, qasm, source):
        circuit, text, key = canonical(qasm)
        if self.db.execute('SELECT 1 FROM candidates WHERE key=?', (key,)).fetchone():
            return key
        matrix = Operator(circuit).data
        if circuit.num_qubits != self.root.num_qubits:
            raise ValueError('wire count differs from root')
        overlap = np.vdot(matrix, self.reference)
        phase = overlap/abs(overlap) if abs(overlap) else 1
        error = float(np.max(np.abs(self.reference-phase*matrix)))
        if not math.isfinite(error) or error > 1e-9:
            self.event('rejected', key, error=error, source=source)
            raise ValueError('fixed-wire complete operator mismatch: '+str(error))
        row = (key, text, len(circuit), int(circuit.count_ops().get('cx', 0)), circuit.depth(),
               self.producer, json.dumps(source, sort_keys=True), error, time.time())
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO objects VALUES (?,?)', (key, text))
            self.db.execute('INSERT OR IGNORE INTO candidates VALUES(?,?,?,?,?,?,?,?,?)', row)
            # Root and best survive. Pool bounds graph memory; disk event log retains provenance.
            best = self.db.execute('SELECT key FROM candidates ORDER BY gates,cx,key LIMIT 1').fetchone()[0]
            excess = self.db.execute('SELECT count(*) FROM candidates').fetchone()[0]-self.capacity
            if excess > 0:
                self.db.execute('DELETE FROM candidates WHERE key IN (SELECT key FROM candidates WHERE key NOT IN (?,?) ORDER BY gates DESC,cx DESC,created LIMIT ?)', (self.root_key, best, excess))
            self.db.execute('INSERT INTO events(time,producer,kind,key,detail) VALUES(?,?,?,?,?)',
                            (time.time(), self.producer, 'publish', key, json.dumps(dict(source=source,error=error,gates=len(circuit)))))
        return key

    def choose(self, rng, foreign_only=True):
        sql = 'SELECT key,qasm,gates,cx,producer,source,error FROM candidates'
        args = ()
        if foreign_only:
            sql += ' WHERE producer != ?'; args = (self.producer,)
        rows = self.db.execute(sql+' ORDER BY gates,cx,key', args).fetchall()
        if not rows:
            return None
        # Separate exchange RNG, never consume policy/Quarl sampling RNG.
        row = rows[0] if rng.random() < .5 else rows[rng.randrange(len(rows))]
        key, text, gates, cx, producer, source, error = row
        _, canonical_text, actual = canonical(text)
        if actual != key or canonical_text != text:
            raise ValueError('shared candidate hash mismatch')
        # Recheck consumer-side; do not trust another process's proof metadata.
        matrix = Operator(qasm2.loads(text)).data
        overlap = np.vdot(matrix, self.reference)
        phase = overlap/abs(overlap) if abs(overlap) else 1
        error_now = float(np.max(np.abs(self.reference-phase*matrix)))
        if not math.isfinite(error_now) or error_now > 1e-9:
            raise ValueError('shared candidate consumer verification failed')
        record = dict(key=key,qasm=text,gates=gates,cx=cx,producer=producer,source=json.loads(source),error=error)
        self.event('sample', key, source_producer=producer)
        return record

    def snapshot(self, path):
        target = sqlite3.connect(str(path))
        try:
            self.db.backup(target)
        finally:
            target.close()

    def close(self):
        self.db.close()
