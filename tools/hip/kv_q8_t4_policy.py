"""Versioned T4 acceptance criteria, adopted by the user on 2026-10-17.

V2 replaces distance from zero by an upper bound on tolerated delta NLL.
The 2-SE quantity is descriptive, not a calibrated confidence interval for
correlated tokens. Historical V1 results must never be overwritten.
"""
from __future__ import annotations

import math
from numbers import Real

POLICY_ID = "T4-v2-2026-10-17"
POLICY = {
    "id": POLICY_ID,
    "adopted": "2026-10-17",
    "authorization": "User requested applying delta_NLL + 2*SE <= 0.02 and rerunning the evaluation",
    "nll_upper_formula": "dnll_mean + 2 * dnll_se <= 0.02",
    "nll_margin_nats_per_token": 0.02,
    "se_multiplier": 2,
    "forced_steps": 512,
    "short_context_kl_max": 0.03,
    "long_context_kl_max": 0.15,
    "short_context_top1_min": 0.95,
    "long_context_top1_min": 0.90,
    "historical_v1": "abs(dnll_mean) <= 2*dnll_se AND dnll_mean <= 0.02, plus unchanged KL/top1",
}


def evaluate(*, dnll_mean, dnll_se, kl_mean, top1, steps, length_tag):
    """Fail closed on absent/nonfinite/invalid metrics. No inference or file IO."""
    errors = []
    for key, value in (("dnll_mean", dnll_mean), ("dnll_se", dnll_se), ("kl_mean", kl_mean)):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            errors.append(f"{key} must be a finite number")
    if not errors:
        if dnll_se < 0:
            errors.append("dnll_se must be nonnegative")
        if kl_mean < 0:
            errors.append("kl_mean must be nonnegative")
    if type(steps) is not int or steps != POLICY["forced_steps"]:
        errors.append("steps must be 512")
    if type(top1) is not int or not 0 <= top1 <= POLICY["forced_steps"]:
        errors.append("top1 must be an integer between zero and 512")
    if length_tag not in ("1k", "8k", "32k", "130k", "261k"):
        errors.append("unsupported context tag")
    if errors:
        return {"policy": POLICY_ID, "result": "ERROR", "errors": errors, "checks": {}}
    short = length_tag == "1k"
    upper = dnll_mean + POLICY["se_multiplier"] * dnll_se
    checks = {
        "nll_upper_le_margin": upper <= POLICY["nll_margin_nats_per_token"],
        "kl_mean": kl_mean <= POLICY["short_context_kl_max" if short else "long_context_kl_max"],
        "top1": top1 / steps >= POLICY["short_context_top1_min" if short else "long_context_top1_min"],
    }
    return {"policy": POLICY_ID, "result": "PASS" if all(checks.values()) else "FAIL",
            "nll_upper_2se": upper, "margin": POLICY["nll_margin_nats_per_token"],
            "checks": checks, "failed_checks": [key for key, passed in checks.items() if not passed]}
