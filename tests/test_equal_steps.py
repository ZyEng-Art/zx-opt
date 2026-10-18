"""Equal-step diagnostics count accepted verified improvements within the cut."""
import json
from pathlib import Path
import tempfile
import unittest

from tools.compare_equal_steps import best_prefix


class EqualStepTest(unittest.TestCase):
    def test_cut_and_status_do_not_import_later_or_unaccepted_improvements(self):
        events = [dict(transition=1, status="accepted", cost=dict(gates=39)),
                  dict(transition=2, status="accepted_unscored", cost=None),
                  dict(transition=3, status="excluded_after_deadline", cost=dict(gates=20)),
                  dict(transition=4, status="accepted", cost=dict(gates=39)),
                  dict(transition=5, status="accepted", cost=dict(gates=37))]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/"events.jsonl").write_text("".join(json.dumps(e)+"\n" for e in events))
            self.assertEqual(best_prefix(folder, 0, 40), dict(gates=40, first_best_transition=0))
            self.assertEqual(best_prefix(folder, 4, 40), dict(gates=39, first_best_transition=1))
            self.assertEqual(best_prefix(folder, 5, 40), dict(gates=37, first_best_transition=5))
