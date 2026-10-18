"""Run the real process termination policy against a compiled, driver-free HIP mock.

Compile tests/poly_gpu_failure_mock/main.cpp with its folder and include/ on the
include path, then pass the resulting executable with --exe.
"""
import argparse
import subprocess
import unittest

EXE = None

class GpuFailure(unittest.TestCase):
    def run_case(self, mode, error=0, sticky=0, fatal=False):
        p = subprocess.run([EXE, mode, str(error), str(sticky)], capture_output=True,
                           text=True, timeout=5)
        if fatal:
            self.assertEqual(p.returncode, 70, p.stdout+p.stderr)
            self.assertEqual(p.stdout.count('ERR GPU_FATAL '), 1)
            self.assertIn('backend=hip', p.stdout)
            self.assertIn('new process required', p.stdout)
            self.assertEqual(p.stderr.count('GPU_FATAL'), 1)
            for marker in ('CONTINUED', 'UNSAFE_RECOVERY', 'UNSAFE_GPU_CLEANUP'):
                self.assertNotIn(marker, p.stdout)
        else:
            self.assertEqual(p.returncode, 0, p.stdout+p.stderr)
            self.assertIn('CONTINUED', p.stdout)
            self.assertNotIn('GPU_FATAL', p.stdout+p.stderr)
        return p

    def test_fatal_device_errors_in_copy_query_sync_and_worker(self):
        for error in (700, 702, 709, 710, 719, 999):
            for mode in ('copy', 'query', 'sync', 'worker'):
                with self.subTest(mode=mode, error=error):
                    self.run_case(mode, error, fatal=True)

    def test_pending_launch_failure_is_caught_before_sync(self):
        self.run_case('sync', sticky=700, fatal=True)

    def test_query_does_not_erase_unrelated_error(self):
        for error in (0, 600):
            with self.subTest(error=error):
                self.run_case('query', error, sticky=709, fatal=True)

    def test_benign_queries_and_successful_sync(self):
        for error in (0, 600):
            for sticky in (0, 600):
                self.run_case('query', error, sticky)
        self.assertIn('sync=1', self.run_case('sync').stdout)
        self.run_case('copy')

    def test_unexpected_argument_or_allocation_error_also_fails_closed(self):
        # A checked call during inference has no safe rollback, even for errors
        # that need not mean hardware damage. Optional startup OOM is handled by
        # the caller before it reaches this checked path.
        for error in (1, 2):
            self.run_case('copy', error, fatal=True)

    def test_unknown_inference_state_is_not_reused(self):
        self.run_case('state', fatal=True)

    def test_optional_oom_cannot_hide_a_fatal_sticky_error(self):
        for sticky in (0, 2, 600):
            self.run_case('expected', 2, sticky)
        self.run_case('expected', 2, 700, fatal=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--exe', required=True)
    EXE = ap.parse_args().exe
    unittest.main(argv=['test_gpu_failure'])
