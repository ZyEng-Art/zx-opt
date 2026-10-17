"""A demonstration cannot lose its root or its per-anchor preparation visits."""
import json
from pathlib import Path
import tempfile
import unittest

from tools.prepare_imitation_path import recover


class ImitationAncestryTest(unittest.TestCase):
    def event(self, step, parent, child, gates, status='accepted'):
        return dict(transition=step, parent_graph_key=parent, graph_key=child,
                    anchor=dict(space='raw', kind='node', position=0), status=status, cost=dict(gates=gates))

    def test_restarts_reuse_the_first_creation_edge_and_count_failed_visits(self):
        events = [self.event(1,'R','R',40,'rejected'), self.event(2,'R','A',40),
                  self.event(3,'A','B',39), self.event(4,'R','A',40), self.event(5,'B','T',36)]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);(path/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
            result, target = recover(path, 'R', 36)
        self.assertEqual([event['transition'] for event,_ in result], [2,3,5])
        self.assertEqual([visits for _,visits in result], [1,0,0])
        self.assertEqual(target['graph_key'], 'T')

    def test_missing_predecessors_are_rejected_instead_of_truncating_the_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);(path/'events.jsonl').write_text(json.dumps(self.event(1,'missing','T',36))+'\n')
            with self.assertRaisesRegex(ValueError, 'Incomplete/cyclic ancestry'):
                recover(path, 'R', 36)
