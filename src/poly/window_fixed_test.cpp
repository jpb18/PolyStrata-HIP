// src/poly/window_fixed_test.cpp - the pure pieces of --window-fixed, without an engine or a GPU:
//
//   g++ -std=c++20 -Iinclude src/poly/window_fixed_test.cpp -o window_fixed_test && ./window_fixed_test
//   (build-hip-poly-*/poly_window_fixed_test: the CMake target, ctest -R poly_window_fixed_test)
//
// The tip: --window-fixed is an experimental instrument; its decode and forced paths are checked by
// the logs they print (STRATA_POLY_TRACE, --force-tokens), not here.  What is checkable without the
// engine is the arithmetic the two paths share.
#include "window_fixed.hpp"

#include <initializer_list>

#include <cstdio>

using strata::poly::drafts_in;
using strata::poly::drafts_right;
using strata::poly::window_fixed_effective;

namespace {

int failed = 0;

void expect(bool ok, const char* what) {
    if (!ok) std::printf("FAIL %s\n", what);
    else std::printf("ok %s\n", what);
    failed += !ok;
}

}  // namespace

int main() {
    // --window-fixed N: the effective size is N in the engine's own bounds and at least 1.
    expect(window_fixed_effective(4, 4) == 4, "window_fixed_effective(4, 4) == 4");
    expect(window_fixed_effective(8, 4) == 4, "a window the engine does not hold is clamped");
    expect(window_fixed_effective(8, 8) == 8, "8 with 8 stays 8");
    expect(window_fixed_effective(1, 4) == 1, "1 is no drafts");
    expect(window_fixed_effective(3, 1) == 1, "an engine without window(s) verifies one token");

    // drafts a window carries: n-1 (the first token of the window is not a draft).
    for (int t : {1, 2, 3, 4, 8}) expect(drafts_in(t) == t - 1, "drafts_in(n) == n - 1");
    expect(drafts_in(0) == 0, "drafts_in(0) == 0");

    // what the block's drafts say about a forced sequence: the first wrong one ends, as a decoded
    // round's right does.
    {
        const int32_t d[3] = {7, 8, 9}, ahead[3] = {7, 8, 9};
        expect(drafts_right(d, 3, ahead) == 3, "all three right");
    }
    {
        const int32_t d[3] = {7, 8, 9}, ahead[3] = {7, 5, 9};
        expect(drafts_right(d, 3, ahead) == 1, "the wrong one ends the round");
    }
    {
        const int32_t d[3] = {7, 8, 9}, ahead[3] = {6, 8, 9};
        expect(drafts_right(d, 3, ahead) == 0, "nothing right from a wrong first");
    }
    {
        const int32_t d[3] = {9, 9, 9}, ahead[3] = {7, 4, 9};
        expect(drafts_right(d, 0, ahead) == 0, "no drafts: nothing right");
        expect(drafts_right(nullptr, 0, nullptr) == 0, "nullptr drafts: nothing right");
    }

    std::printf(failed == 0 ? "window_fixed_test: all passed\n" : "window_fixed_test: %d failed\n",
                failed);
    return failed == 0 ? 0 : 1;
}
