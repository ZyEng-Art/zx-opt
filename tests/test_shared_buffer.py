import concurrent.futures
import random
import sqlite3
import tempfile
import unittest
from pathlib import Path
from zxopt.shared_buffer import SharedBuffer

ROOT = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q[0]; cx q[0],q[1];'
ALT = ROOT+' x q[1]; x q[1];'
BAD = ROOT+' x q[0];'


def producer(path, index):
    pool = SharedBuffer(path, ROOT, 'worker'+str(index), capacity=3)
    pool.publish(ROOT+' h q[1]; h q[1];'*index, {'index': index})
    pool.close()


class SharedTest(unittest.TestCase):
    def test_foreign_roundtrip_isolation_rejection_and_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'pool.sqlite'
            quarl = SharedBuffer(path, ROOT, 'quarl')
            key = quarl.publish(ALT, {'trajectory': 'native/17'})
            zx = SharedBuffer(path, ROOT, 'zx')
            records = [zx.choose(random.Random(i)) for i in range(20)]
            self.assertIn(key, {r['key'] for r in records})
            self.assertTrue(all(r['producer']=='quarl' for r in records))
            with self.assertRaises(ValueError):zx.publish(BAD, {})
            with self.assertRaises(ValueError):zx.publish(ROOT+' rz(pi/8) q[0];', {})
            with self.assertRaises(ValueError):SharedBuffer(path, BAD, 'bad')
            snap = Path(tmp)/'snapshot.sqlite';zx.snapshot(snap)
            restored = SharedBuffer(snap, ROOT, 'zx')
            self.assertIsNotNone(restored.choose(random.Random(0)))
            restored.db.execute('UPDATE candidates SET qasm=? WHERE key=?', (BAD,key));restored.db.commit()
            class PickLast:
                def random(self):return .9
                def randrange(self,n):return n-1
            with self.assertRaises(ValueError):restored.choose(PickLast())
            restored.close();zx.close();quarl.close()

    def test_process_writers_keep_bounded_pool_and_all_source_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/'pool.sqlite')
            pool = SharedBuffer(path, ROOT, 'quarl', capacity=3)
            with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
                list(executor.map(producer, [path]*6, range(1,7)))
            self.assertEqual(pool.db.execute('SELECT count(*) FROM candidates').fetchone()[0], 3)
            self.assertEqual(pool.db.execute('SELECT count(*) FROM objects').fetchone()[0], 7)
            self.assertTrue(pool.db.execute('SELECT 1 FROM candidates WHERE key=?',(pool.root_key,)).fetchone())
            pool.close()

    def test_general_mode_cannot_silently_enter_ct_exchange(self):
        from zxopt.general_search import parser, run
        for flag in ('--general-zx','--allow-arbitrary-angles'):
            with tempfile.TemporaryDirectory() as tmp:
                output=Path(tmp)/'out'
                args=parser().parse_args([str(output),'--shared-buffer',str(Path(tmp)/'pool.sqlite'),flag])
                with self.assertRaisesRegex(ValueError,'Clifford\+T only'):
                    run(args)
                self.assertFalse(output.exists())

    def test_cost_not_producer_supplied_and_rng_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = SharedBuffer(Path(tmp)/'pool.sqlite',ROOT,'quarl')
            pool.publish(ALT, {'gates':0})
            before = random.getstate()
            pool.choose(random.Random(2),foreign_only=False)
            self.assertEqual(random.getstate(),before)
            self.assertEqual(pool.db.execute('SELECT max(gates) FROM candidates').fetchone()[0],4)
            pool.close()
