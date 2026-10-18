"""Regression checks for the changed statistical gate and unchanged quality gates."""
import math
import unittest

from kv_q8_t4_policy import evaluate


def case(**overrides):
    values = dict(dnll_mean=0.0, dnll_se=0.001, kl_mean=0.01, top1=500, steps=512, length_tag="1k")
    values.update(overrides)
    return evaluate(**values)


class T4PolicyTests(unittest.TestCase):
    def test_previous_chat130k_failure_now_passes(self):
        r = case(dnll_mean=.004617638671875, dnll_se=.002106391645246467,
                 kl_mean=.002364638421327684, top1=508, length_tag="130k")
        self.assertEqual(r["result"], "PASS")
        self.assertAlmostEqual(r["nll_upper_2se"], .008830421962367934)

    def test_small_point_estimate_with_high_uncertainty_fails(self):
        r = case(dnll_mean=.014076277343750014, dnll_se=.007089641516312145, top1=492)
        self.assertEqual(r["result"], "FAIL")
        self.assertEqual(r["failed_checks"], ["nll_upper_le_margin"])

    def test_closed_boundary_and_next_float(self):
        self.assertEqual(case(dnll_mean=.02, dnll_se=0)["result"], "PASS")
        self.assertEqual(case(dnll_mean=math.nextafter(.02, math.inf), dnll_se=0)["result"], "FAIL")

    def test_large_negative_delta_not_failed_for_distance_from_zero(self):
        self.assertEqual(case(dnll_mean=-.05, dnll_se=.001)["result"], "PASS")

    def test_missing_and_nonfinite_metrics_are_errors(self):
        for key in ("dnll_mean", "dnll_se", "kl_mean"):
            for value in (None, float("nan"), float("inf"), True):
                with self.subTest(key=key, value=value):
                    self.assertEqual(case(**{key: value})["result"], "ERROR")

    def test_invalid_sample_and_negative_uncertainty_are_errors(self):
        for overrides in ({"dnll_se": -.001}, {"kl_mean": -.1}, {"steps": 511},
                          {"top1": 513}, {"top1": 500.0}, {"length_tag": "unknown"}):
            with self.subTest(overrides=overrides):
                self.assertEqual(case(**overrides)["result"], "ERROR")

    def test_kl_thresholds_are_unchanged(self):
        self.assertEqual(case(kl_mean=.03)["result"], "PASS")
        self.assertEqual(case(kl_mean=.031)["result"], "FAIL")
        self.assertEqual(case(kl_mean=.15, length_tag="8k")["result"], "PASS")
        self.assertEqual(case(kl_mean=.151, length_tag="8k")["result"], "FAIL")

    def test_top1_thresholds_are_unchanged(self):
        self.assertEqual(case(top1=487)["result"], "PASS")
        self.assertEqual(case(top1=486)["result"], "FAIL")
        self.assertEqual(case(top1=461, length_tag="8k")["result"], "PASS")
        self.assertEqual(case(top1=460, length_tag="8k")["result"], "FAIL")


if __name__ == "__main__":
    unittest.main()
