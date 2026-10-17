import unittest

from zxopt.archive import DeferredFrontier
from buffer import Entry


def entry(key, gates, payload=None):
    return Entry(key, dict(gates=gates, cx=14), payload)


class FrontierTest(unittest.TestCase):
    def test_resumes_remaining_parent_and_drops_exhausted_states(self):
        reserve = DeferredFrontier(3)
        reserve.retain(entry("a", 38, "parent-a"), 2)
        reserve.retain(entry("b", 39, "parent-b"), 1)
        reserve.retain(entry("c", 37, "exhausted"), 3)
        resumed = reserve.pop({"c"})
        self.assertEqual((resumed.key, resumed.payload), ("a", "parent-a"))
        reserve.retain(resumed, 1)
        reserve.retain(resumed, 0)
        self.assertEqual(reserve.pop(set()).key, "b")
        self.assertIsNone(reserve.pop(set()))
        self.assertEqual(reserve.summary()["resumptions"], 2)

    def test_capacity_is_bounded_and_disabled_mode_retains_nothing(self):
        reserve = DeferredFrontier(2)
        for key, gates in [("a", 38), ("b", 39), ("c", 40)]:
            reserve.retain(entry(key, gates), 4)
        self.assertEqual(set(reserve.entries), {"a", "b"})
        self.assertEqual(reserve.summary()["evictions"], 1)
        reserve.retain(entry("a", 38, "latest"), 2)
        self.assertEqual(reserve.pop(set()).payload, "latest")
        disabled = DeferredFrontier(0)
        disabled.retain(entry("a", 38), 5)
        self.assertIsNone(disabled.pop(set()))


if __name__ == "__main__":
    unittest.main()
