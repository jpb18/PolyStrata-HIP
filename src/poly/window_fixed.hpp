// src/poly/window_fixed.hpp - --window-fixed: a verified window of the same size in every round, and
// what the block that guesses ahead says about a forced sequence.
//
// The adaptive MTP decides its window from measured round times (Gate, Guesses, DraftPolicy), so the
// rate right/guessed of a run is a function of the machine's scheduling too, not just of the text
// (bench/results/2026-10-16-hip-mtp-t7-investigation/, CORRECTION-PROPOSAL.md §3).  --window-fixed N
// asks that window's tokens every round instead, which the engine verifies and keeps as the adaptive
// windows did; what it drops is only the decision.  Nothing here knows the engine or the model.
#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::poly {

/// --window-fixed N (N >= 1): the window every round verifies, at most `most`, the engine's own.
inline int window_fixed_effective(int n, int most) { return std::clamp(n, 1, most); }

/// The drafts a window of `t` tokens carries: `t` at most, the first of them is not a draft.
inline int drafts_in(int t) { return std::clamp(t, 1, 8) - 1; }

/// How many of the window's leading drafts `ahead` accepts, the way a decoded round counts them
/// (right, main.cpp): `d[j]` guessed the token at `ahead[j]`, and the first wrong one ends the round.
/// Returns 0..n_d.  When --window-fixed measures at a forced step, `ahead` is the given tokens'
/// positions after the forced one's, so the two formats see the block on the same tokens.
inline int drafts_right(const int32_t* d, int n_d, const int32_t* ahead) {
    int right = 0;
    for (int j = 0; j < n_d; ++j)
        if (d[j] == ahead[j]) ++right;
        else break;
    return right;
}

}  // namespace strata::poly
