"""Cross-IR restart adapter. Imported states never become foreign PPO rollouts."""
import random
import time
import pyzx as zx
from .shared_buffer import SharedBuffer
from .complete.adapter import from_pyzx
from .general_archive import SearchState


class ZXSharedBridge:
    def __init__(self, path, root_qasm, seed, log):
        self.pool = SharedBuffer(path, root_qasm, 'zx')
        self.rng = random.Random(seed); self.next_publish = 0.
        self.log = log

    def publish(self, result, graph_key):
        if time.monotonic() < self.next_publish:
            return
        self.next_publish = time.monotonic()+1.
        key = self.pool.publish(result['qasm'], {'graph_key': graph_key, 'run': str(self.log)})
        self.pool.event('export', key, graph_key=graph_key, run=str(self.log))

    def choose(self, views, scorer, timeout):
        if self.rng.random() >= .5:
            return None
        record = self.pool.choose(self.rng)
        if record is None:
            return None
        diagram = from_pyzx(zx.Circuit.from_qasm(record['qasm']).to_graph())
        evaluation = scorer.evaluate(views.get(diagram), timeout)
        if evaluation.result is None:
            self.pool.event('conversion_unscored', record['key'], reason=evaluation.reason)
            return None
        cost = evaluation.result['cost']
        record['completion'] = evaluation.result
        self.pool.event('import', record['key'], graph_key=diagram.digest(), converted_cost=cost, run=str(self.log))
        return SearchState(diagram, cost, cost), record
