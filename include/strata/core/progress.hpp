// include/strata/core/progress.hpp - issue #29: whether a request is still moving, and where it is.
//
// The engine's host loop and the GPU wait on each other through flags; a protocol bug there does not crash, it
// spins forever (the GPU "100%", one CPU core busy, no tokens).  `--serve` runs a watchdog thread over this: a
// request whose heartbeat stops for `STRATA_WATCHDOG_S` seconds (default 120; 0 = off) ends the engine with the
// stage it was stuck in, and the server starts it again instead of hanging.  Tokens, prompt chunks and verify
// windows beat; the stage is two relaxed stores per layer, which the token path does not notice.
#pragma once

#include <atomic>
#include <cstdint>

namespace strata::core {

struct Progress {
    std::atomic<uint64_t> beats{0};
    std::atomic<bool> busy{false};
    std::atomic<const char*> where{"idle"};
    std::atomic<int64_t> detail{-1};
};

inline Progress& progress() {
    static Progress p;
    return p;
}

inline void progress_at(const char* where, int64_t detail = -1) {
    Progress& p = progress();
    p.where.store(where, std::memory_order_relaxed);
    p.detail.store(detail, std::memory_order_relaxed);
}

inline void progress_beat() { progress().beats.fetch_add(1, std::memory_order_relaxed); }

}  // namespace strata::core
