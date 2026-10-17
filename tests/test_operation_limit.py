"""Budget alarms survive vendor-style wrappers without hiding real failures."""
import signal
import unittest

from zxopt.search import operation_limit


class OperationLimitTest(unittest.TestCase):
    def fire(self):
        signal.getsignal(signal.SIGALRM)(signal.SIGALRM, None)

    def test_direct_alarm_restores_handler(self):
        before = signal.getsignal(signal.SIGALRM)
        with self.assertRaises(TimeoutError):
            with operation_limit(60):
                self.fire()
        self.assertIs(signal.getsignal(signal.SIGALRM), before)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0], 0)

    def test_vendor_wrapper_is_normalized_by_exception_identity(self):
        with self.assertRaises(TimeoutError) as raised:
            with operation_limit(60):
                try:
                    self.fire()
                except TimeoutError as exc:
                    raise RuntimeError("xvllm import hook failed") from exc
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)

    def test_swallowed_alarm_still_excludes_the_operation(self):
        with self.assertRaises(TimeoutError):
            with operation_limit(60):
                try:
                    self.fire()
                except TimeoutError:
                    pass

    def test_unrelated_import_error_with_same_words_is_preserved(self):
        original = RuntimeError("candidate operation exceeded deadline")
        with self.assertRaises(RuntimeError) as raised:
            with operation_limit(60):
                raise original
        self.assertIs(raised.exception, original)

    def test_real_error_after_handled_alarm_is_not_hidden(self):
        original = ValueError("independent verification failure")
        with self.assertRaises(ValueError) as raised:
            with operation_limit(60):
                try:
                    self.fire()
                except TimeoutError:
                    pass
                raise original
        self.assertIs(raised.exception, original)


if __name__ == "__main__":
    unittest.main()
