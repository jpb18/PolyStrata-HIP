// src/glm/main.cpp - strata-glm: GLM 5.3 Flash from its GGUF.
//
//   strata-glm --model M.gguf --tokens 1,2,3 [--max-new N]   one prompt; prints the generated ids
//   strata-glm --model M.gguf --serve                        the line protocol serve/server.py speaks
//
// The protocol is the one `strata --serve` speaks (src/program/generate.cpp): GEN, STOP and QUIT in, and RESUME,
// PP, T, DONE and ERR out.
#include "conversations.hpp"
#include "engine.hpp"
#include "strata/core/conversation_memory.hpp"
#include "strata/spec/draft_policy.hpp"
#include "strata/spec/suffix_drafter.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifndef STRATA_VERSION
#define STRATA_VERSION "dev"
#endif

namespace {

using strata::glm::CacheOptions;
using strata::glm::Conversations;
using strata::glm::Engine;
using strata::glm::Options;
using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

struct Sampling {
    float temperature = 0.f;   // 0: the most likely token
    float top_p = 1.f, min_p = 0.f;
    int top_k = 20;            // 0: no limit
    int penalty_last_n = 0;
    float penalty_repeat = 1.f, penalty_freq = 0.f, penalty_present = 0.f;
    unsigned long long seed = 0;
};

// llama.cpp's sampler chain in its default order: penalties, top-k, top-p, min-p, temperature, then the draw.
class Sampler {
public:
    int pick(const float* logits, int n_vocab, const Sampling& s, const std::vector<int32_t>& history,
             std::mt19937_64& rng) {
        cand_.resize((size_t) n_vocab);
        for (int i = 0; i < n_vocab; ++i) cand_[(size_t) i] = {logits[i], i};
        penalize(s, history);
        auto by_logit = [](const Cand& a, const Cand& b) { return a.logit != b.logit ? a.logit > b.logit : a.id < b.id; };
        if (s.temperature <= 0.f) return std::min_element(cand_.begin(), cand_.end(), by_logit)->id;

        const size_t k = s.top_k > 0 ? std::min((size_t) s.top_k, cand_.size()) : cand_.size();
        std::partial_sort(cand_.begin(), cand_.begin() + (std::ptrdiff_t) k, cand_.end(), by_logit);
        cand_.resize(k);

        if (s.top_p < 1.f) {
            softmax();
            double cum = 0.0;
            size_t keep = cand_.size();
            for (size_t i = 0; i < cand_.size(); ++i) {
                cum += p_[i];
                if (cum >= s.top_p) {
                    keep = i + 1;
                    break;
                }
            }
            cand_.resize(keep);
        }
        if (s.min_p > 0.f) {
            const float floor = cand_[0].logit + std::log(s.min_p);
            size_t keep = 1;
            while (keep < cand_.size() && cand_[keep].logit >= floor) ++keep;
            cand_.resize(keep);
        }
        for (Cand& c : cand_) c.logit /= s.temperature;
        softmax();
        return cand_[std::discrete_distribution<size_t>(p_.begin(), p_.end())(rng)].id;
    }

private:
    struct Cand {
        float logit;
        int id;
    };

    void penalize(const Sampling& s, const std::vector<int32_t>& history) {
        if (s.penalty_last_n == 0 ||
            (s.penalty_repeat == 1.f && s.penalty_freq == 0.f && s.penalty_present == 0.f))
            return;
        const size_t n = s.penalty_last_n < 0 ? history.size() : std::min(history.size(), (size_t) s.penalty_last_n);
        seen_.clear();
        for (size_t i = history.size() - n; i < history.size(); ++i) ++seen_[history[i]];
        for (const auto& [id, count] : seen_) {
            float& l = cand_[(size_t) id].logit;
            l = l <= 0.f ? l * s.penalty_repeat : l / s.penalty_repeat;
            l -= (float) count * s.penalty_freq + s.penalty_present;
        }
    }

    void softmax() {
        p_.resize(cand_.size());
        float mx = cand_[0].logit;
        for (const Cand& c : cand_) mx = std::max(mx, c.logit);
        double sum = 0.0;
        for (size_t i = 0; i < cand_.size(); ++i) sum += p_[i] = std::exp((double) (cand_[i].logit - mx));
        for (double& p : p_) p /= sum;
    }

    std::vector<Cand> cand_;
    std::vector<double> p_;
    std::unordered_map<int32_t, int> seen_;
};

bool parse_ids(const char* s, std::vector<int32_t>& out) {
    out.clear();
    while (*s == ' ') ++s;
    while (*s != '\0') {
        char* end = nullptr;
        const long long v = std::strtoll(s, &end, 10);
        if (end == s || v < 0 || v > INT32_MAX) return false;
        out.push_back((int32_t) v);
        s = end;
        while (*s == ' ') ++s;
        if (*s == ',') ++s;
        else if (*s != '\0') return false;
    }
    return !out.empty();
}

// stdin on its own thread, so a STOP is seen while a request runs
class Lines {
public:
    Lines() : th_([this] { read(); }) { th_.detach(); }

    std::string pop() {
        std::unique_lock<std::mutex> l(m_);
        cv_.wait(l, [this] { return !q_.empty(); });
        std::string s = std::move(q_.front());
        q_.pop_front();
        return s;
    }

    std::atomic<bool> stop{false};

private:
    void read() {
        std::string l;
        while (std::getline(std::cin, l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            if (l == "STOP") stop.store(true);
            else push(l);
        }
        push("QUIT");
    }
    void push(std::string s) {
        { std::lock_guard<std::mutex> l(m_); q_.push_back(std::move(s)); }
        cv_.notify_one();
    }

    std::mutex m_;
    std::condition_variable cv_;
    std::deque<std::string> q_;
    std::thread th_;
};

// What the conversation cache is told on the command line.
struct CacheFlags {
    CacheOptions o;
    long long min_free_mib = 2560;   // parking leaves this much physical RAM free
};

struct Session {
    Engine& engine;
    std::vector<int32_t> eos;
    CacheFlags flags;
    Conversations<Engine> kept;   // the conversation in the engine, and the parked ones
    Sampler sampler;
    std::mt19937_64 rng{std::random_device{}()};
    long long asked = 0, reused = 0;   // prompt tokens of every request, and those of them not read again
    // Guesses at the tokens to come, checked a window at a time (Engine::window).  They come from the block some
    // files carry for it (Engine::draft), and from the conversation: when the last tokens repeat an earlier stretch
    // of it, what followed there is the guess.  The policy learns from the windows so far, their hits and their
    // time, which of the two pays.  It and the repeats are the other engine's (strata/spec).
    int most;                             // the tokens of a window at most; 1: no guesses
    bool look;                            // guess from repeats
    strata::spec::SuffixDrafter repeats;
    strata::spec::DraftPolicy policy;

    Session(Engine& e, std::vector<int32_t> eos_ids, const CacheFlags& f, int lookup)
        : engine(e), eos(std::move(eos_ids)), flags(f), kept(e, with_floor(f)),
          most(lookup > 0 || e.drafts() ? std::min<int>(e.max_window(), strata::spec::DraftPolicy::kMaxT) : 1),
          look(lookup > 0 && most > 1),
          repeats(std::max(lookup, 3), 64, (size_t) e.max_context() + 4096), policy(most) {}

    static CacheOptions with_floor(const CacheFlags& f) {
        CacheOptions o = f.o;
        o.log = stderr;
        o.admit = [floor = (uint64_t) f.min_free_mib << 20](size_t bytes) {
            return strata::core::conversation_memory_admit(strata::core::conversation_available_memory(), bytes, floor);
        };
        return o;
    }

    bool is_eos(int32_t t) const { return std::find(eos.begin(), eos.end(), t) != eos.end(); }

    struct Result {
        int64_t produced = 0, resume = 0, read = 0;
        int64_t guessed = 0, guessed_right = 0;
        double prompt_ms = 0, decode_ms = 0;
        const char* finish = "length";
    };

    // Reads the prompt and generates.  on_token gets every token, the end-of-sequence one included.
    template<class OnToken, class OnProgress>
    Result generate(const std::vector<int32_t>& ids, int64_t max_new, const Sampling& s, const std::atomic<bool>& stop,
                    OnToken on_token, OnProgress on_progress) {
        Result r;
        const int64_t n = (int64_t) ids.size();
        Clock::time_point t0 = Clock::now(), shown = t0;   // bringing a parked conversation back is prompt time
        r.resume = kept.resume(ids);
        asked += n;
        reused += r.resume;
        on_progress(r.resume, n, 0.0, true);
        if (s.seed != 0) rng.seed(s.seed);

        // The last token goes in alone, for the logits that come with it.  The tokens before it go in as many at
        // a time as the engine takes, up to the next checkpoint.
        for (int64_t i = r.resume; i < n;) {
            if (stop.load()) {
                r.finish = "cancel";
                break;
            }
            const bool last = i == n - 1;
            const int64_t m = last ? 1 : kept.until_checkpoint(i, std::min<int64_t>(n - 1 - i, engine.max_rows()));
            kept.before(i, last);
            engine.eval(ids.data() + i, (int) m, last);
            kept.read(ids.data() + i, m);
            i += m;
            r.read += m;
            if (ms_since(shown) >= 500.0 || last) {
                on_progress(i, n, ms_since(t0), false);
                shown = Clock::now();
            }
        }
        kept.prompt_was(n);
        r.prompt_ms = ms_since(t0);
        if (std::strcmp(r.finish, "cancel") == 0) return r;

        t0 = Clock::now();
        std::vector<int32_t> history = kept.seq();
        if (look) {
            repeats.reset();
            repeats.append(history.data(), history.size());
        }
        const auto pick = [&](const float* logits) {
            return (int32_t) sampler.pick(logits, engine.n_vocab(), s, history, rng);
        };
        // Shows a token.  False when the answer ends with it.
        const auto show = [&](int32_t t) {
            on_token(t);
            ++r.produced;
            history.push_back(t);
            if (look) repeats.append(t);
            if (is_eos(t)) r.finish = "stop";
            else if (stop.load()) r.finish = "cancel";
            else return r.produced < max_new;
            return false;
        };
        if (max_new > 0) {
            int32_t w[strata::spec::DraftPolicy::kMaxT], d[strata::spec::DraftPolicy::kMaxT];
            int32_t t = pick(engine.logits());   // picked, and not shown yet
            while (show(t)) {
                // t goes into the model now, and with it guesses at what follows when there are any that pay
                const Clock::time_point round = Clock::now();
                const bool room = engine.pos() + most <= engine.max_context();
                const int k = look && room ? repeats.propose(most - 1, w + 1) : 0;
                const int match = repeats.last_match();
                // A repeat that fills the window, from a match of a length that has been right nine times in ten,
                // is taken as it is: the block would mostly say the same, and asking it takes time.  Otherwise the
                // block guesses, each guess from the one before.  A guess joins the window while the window is then
                // expected to write more tokens a millisecond.  The chance that the guesses up to it are all right
                // is the product of the block's own.  What one more token costs is what the windows so far have
                // measured, held to 0.2 to 0.8 of a token alone: one slow window must not end the guessing.
                const bool ask = engine.drafts() && room && !(k == most - 1 && policy.lookup_rate(match) >= 0.9);
                strata::spec::DraftPolicy::Pick use = policy.choose(1, k, match);
                if (ask) {
                    const double one = policy.cost_ms(1);
                    double sure = 1.0, tokens = 1.0;
                    int n_d = 0;
                    for (int32_t from = t; n_d < most - 1; from = d[n_d++]) {
                        float p = 0.f;
                        d[n_d] = engine.draft(from, n_d, &p);
                        sure *= p;
                        const double more =
                            std::clamp((policy.cost_ms(n_d + 2) - policy.cost_ms(n_d + 1)) / one, 0.2, 0.8);
                        if (sure * policy.cost_ms(n_d + 1) <= tokens * more * one) break;
                        tokens += sure;
                    }
                    use = policy.choose(1 + n_d, k, match);
                }
                if (use.t == 1) {
                    engine.eval(&t, 1, true);
                    kept.read(&t, 1);
                    t = pick(engine.logits());
                    policy.observe(false, 1, 0, 0, ms_since(round));
                    continue;
                }
                w[0] = t;
                if (!use.lookup) std::copy(d, d + use.t - 1, w + 1);
                engine.window(w, use.t);
                // logits_of(i) is what follows w[i].  A guess that is what was picked has been through the model
                // already; the token an answer ends with is shown and never fed, as without guesses.
                int right = 0;
                bool more = true;
                t = pick(engine.logits_of(0));
                while (right < use.t - 1 && t == w[right + 1] && (more = show(t))) t = pick(engine.logits_of(++right));
                engine.keep(right + 1);
                kept.read(w, right + 1);
                r.guessed += use.t - 1;
                r.guessed_right += right + (more ? 0 : 1);
                policy.observe(use.lookup, use.t, right, match, ms_since(round));
                if (!more) break;
            }
        }
        r.decode_ms = ms_since(t0);
        return r;
    }
};

bool parse_gen(const std::string& line, long long& max_new, Sampling& s, std::vector<int32_t>& ids) {
    char* p = nullptr;
    max_new = std::strtoll(line.c_str() + 4, &p, 10);
    if (max_new < 1) return false;
    for (;;) {   // key=value tokens, then the ids
        while (*p == ' ') ++p;
        const char* start = p;
        while (*p != '\0' && *p != ' ') ++p;
        const std::string tok(start, (size_t) (p - start));
        const size_t eq = tok.find('=');
        if (eq == std::string::npos) {
            p = const_cast<char*>(start);
            break;
        }
        const std::string key = tok.substr(0, eq);
        const char* v = tok.c_str() + eq + 1;
        if (key == "temperature") s.temperature = std::strtof(v, nullptr);
        else if (key == "top_p") s.top_p = std::strtof(v, nullptr);
        else if (key == "top_k") s.top_k = std::atoi(v);
        else if (key == "min_p") s.min_p = std::strtof(v, nullptr);
        else if (key == "penalty_last_n") s.penalty_last_n = std::atoi(v);
        else if (key == "penalty_repeat") s.penalty_repeat = std::strtof(v, nullptr);
        else if (key == "penalty_freq") s.penalty_freq = std::strtof(v, nullptr);
        else if (key == "penalty_present") s.penalty_present = std::strtof(v, nullptr);
        else if (key == "seed") s.seed = std::strtoull(v, nullptr, 10);
        // other keys belong to the other engine and are skipped
    }
    return parse_ids(p, ids);
}

int serve(Session& ss) {
    Engine& engine = ss.engine;
    std::printf("INFO context=%d expert_slots=%d expert_cache_mib=%lld pool_workers=%d conversation_cache_mib=%lld "
                "conversation_cache_slots=%d conversation_cache_min_free_mib=%lld engine=" STRATA_VERSION "\n",
                engine.max_context(), engine.cached_experts(), (long long) (engine.cached_gib() * 1024.0),
                engine.threads(), (long long) ss.flags.o.park_mib, ss.flags.o.park_slots, ss.flags.min_free_mib);
    std::printf("READY %d stop\n", engine.max_context());
    std::fflush(stdout);

    static Lines lines;   // its thread may still be in a read when this function returns
    for (;;) {
        const std::string line = lines.pop();
        if (line == "QUIT") break;
        if (line.empty()) continue;
        lines.stop.store(false);   // a STOP that came between requests is stale
        long long max_new = 0;
        Sampling s;
        std::vector<int32_t> ids;
        if (line.rfind("GEN ", 0) != 0) {
            std::printf("ERR expected: GEN <max_new> <id,id,...>\n");
            std::fflush(stdout);
            continue;
        }
        if (!parse_gen(line, max_new, s, ids)) {
            std::printf("ERR bad request\n");
            std::fflush(stdout);
            continue;
        }
        const long long n = (long long) ids.size();
        if (n + max_new > engine.max_context()) {
            std::printf("ERR prompt (%lld tokens) + max_new (%lld) exceeds the context (%d)\n", n, max_new,
                        engine.max_context());
            std::fflush(stdout);
            continue;
        }
        if (std::any_of(ids.begin(), ids.end(), [&](int32_t t) { return t >= engine.n_vocab(); })) {
            std::printf("ERR a token id is outside the vocabulary\n");
            std::fflush(stdout);
            continue;
        }

        const strata::glm::Stats before = engine.stats();
        int64_t resume = 0;
        Session::Result r;
        try {
            r = ss.generate(
                ids, max_new, s, lines.stop,
                [](int32_t t) {
                    std::printf("T %d\n", (int) t);
                    std::fflush(stdout);
                },
                [&](int64_t at, int64_t total, double ms, bool first) {
                    if (first) {
                        resume = at;
                        std::printf("RESUME %lld\n", (long long) at);
                    } else {
                        std::printf("PP %lld %lld %.0f %.1f\n", (long long) at, (long long) total, ms,
                                    ms > 0.0 ? 1000.0 * (double) (at - resume) / ms : 0.0);
                    }
                    std::fflush(stdout);
                });
        } catch (const std::exception& e) {
            // the engine's state is unknown now: the conversation it was reading is lost
            ss.kept.drop();
            try { engine.reset(); } catch (const std::exception&) {}
            std::printf("ERR %s\n", e.what());
            std::fflush(stdout);
            continue;
        }
        // DONE <generated> <prompt> <prompt ms> <decode ms> <finish> <drafts accepted> <drafts offered> <reused>
        //      <experts from VRAM> <experts> <RAM blobs> <file blobs> <file MB> <prompt tokens read> <offloaded>
        const strata::glm::Stats& now = engine.stats();
        // the cache's share of the experts looked up while answering: a prompt batch's go through VRAM either way
        const long long hits = now.gpu_experts - before.gpu_experts;
        const long long misses = now.cpu_experts - before.cpu_experts;
        std::printf("DONE %lld %lld %.1f %.1f %s %lld %lld %lld %lld %lld 0 0 0.0 %lld 0\n", (long long) r.produced, n,
                    r.prompt_ms, r.decode_ms, r.finish, (long long) r.guessed_right, (long long) r.guessed,
                    (long long) r.resume, hits, hits + misses, (long long) r.read);
        std::fflush(stdout);
    }
    std::fprintf(stderr, "strata-glm: %lld of %lld prompt tokens were not read again (%.1f%%); %zu conversations "
                         "parked in %zu MiB, %zu pushed out\n",
                 ss.reused, ss.asked, ss.asked > 0 ? 100.0 * (double) ss.reused / (double) ss.asked : 0.0,
                 ss.kept.parked(), ss.kept.parked_bytes() >> 20, ss.kept.evictions());
    return 0;
}

// The n most likely next tokens with their probabilities, to hold against another engine's on the same prompt.
void print_top(const float* logits, int n_vocab, int n) {
    std::vector<int> order((size_t) n_vocab);
    for (int i = 0; i < n_vocab; ++i) order[(size_t) i] = i;
    n = std::min(n, n_vocab);
    std::partial_sort(order.begin(), order.begin() + n, order.end(),
                      [logits](int a, int b) { return logits[a] > logits[b]; });
    const float top = logits[order[0]];
    double sum = 0.0;
    for (int i = 0; i < n_vocab; ++i) sum += std::exp((double) (logits[i] - top));
    std::fprintf(stderr, "top");
    for (int i = 0; i < n; ++i)
        std::fprintf(stderr, " %d:%.4f", order[(size_t) i], std::exp((double) (logits[order[(size_t) i]] - top)) / sum);
    std::fprintf(stderr, "\n");
}

int once(Session& ss, const std::vector<int32_t>& ids, long long max_new, const Sampling& s, int top_probs) {
    std::atomic<bool> stop{false};
    std::string out;
    const Session::Result r = ss.generate(
        ids, max_new, s, stop,
        [&](int32_t t) {
            // the logits the token was picked from are still the engine's
            if (top_probs > 0) print_top(ss.engine.logits(), ss.engine.n_vocab(), top_probs);
            out += (out.empty() ? "" : ",") + std::to_string(t);
        },
        [](int64_t, int64_t, double, bool) {});
    std::printf("%s\n", out.c_str());
    const strata::glm::Stats& st = ss.engine.stats();
    const double experts = (double) std::max<int64_t>(st.gpu_experts + st.cpu_experts, 1);
    std::fprintf(stderr,
                 "prompt %lld tokens %.1f tok/s, generated %lld tokens %.2f tok/s (%s); experts from VRAM %.1f%%, "
                 "CPU experts %.1f ms/token, router wait %.1f ms/token, %lld experts swapped into VRAM, "
                 "%lld of %lld guessed tokens right\n",
                 (long long) r.read, r.prompt_ms > 0 ? 1000.0 * (double) r.read / r.prompt_ms : 0.0,
                 (long long) r.produced, r.decode_ms > 0 ? 1000.0 * (double) (r.produced - 1) / r.decode_ms : 0.0,
                 r.finish, 100.0 * (double) st.gpu_experts / experts, st.cpu_ms / (double) std::max<int64_t>(st.tokens, 1),
                 st.route_wait_ms / (double) std::max<int64_t>(st.tokens, 1), (long long) st.swapped_experts,
                 (long long) r.guessed_right, (long long) r.guessed);
    return 0;
}

void usage() {
    std::fprintf(stderr,
                 "usage: strata-glm --model FILE.gguf (--serve | --tokens ID,ID,... [--max-new N])\n"
                 "  --max-context N        tokens the attention caches hold (default 8192)\n"
                 "  --threads N            CPU threads for the experts that are not in VRAM (default: all usable)\n"
                 "  --vram-reserve-mib N   VRAM left free beside the expert cache (default 1024)\n"
                 "  --expert-profile FILE  expert use counts: read to choose the cached experts, updated on exit\n"
                 "  --no-gpu-experts       keep every routed expert on the CPU\n"
                 "  --any-node             on a PC with several NUMA nodes, let the CPU threads run on any of them\n"
                 "                         (default: on the node whose memory holds most of the experts)\n"
                 "  --window N             the most tokens that go through the model at once while an answer is\n"
                 "                         written: the one that is known and guesses at the ones after it, which\n"
                 "                         count when they turn out right (default 4, 1 = no guesses)\n"
                 "  --lookup N             guess what followed an earlier repeat of the last N or more tokens\n"
                 "                         (default 3, 0 = never)\n"
                 "  --no-draft             do not run the block that guesses ahead\n"
                 "  --draft-model FILE     a file of the same model to take that block from, when --model has none\n"
                 "  --no-graphs            launch a token's kernels one by one (default: a layer's are recorded once\n"
                 "                         as a CUDA graph and replayed)\n"
                 "  --expert-swaps N       while an answer is written, up to N experts it keeps using are on their way\n"
                 "                         into VRAM, in place of cached ones it uses less (default 16, 0 = the cache\n"
                 "                         stays as it was loaded)\n"
                 "  --prompt-batch N       prompt tokens read at a time (default 1024); about 1.4 MiB of VRAM each\n"
                 "  --stage-mib N          VRAM the experts outside the cache pass through in a prompt batch (default\n"
                 "                         2048); with less, more of them are copied after the layer has routed\n"
                 "  --stage-ahead N        from N tokens, a batch copies those experts before it routes (default 256,\n"
                 "                         0 = never)\n"
                 "  --prompt-cache N       checkpoints kept of a conversation, about 150 MB of RAM each (default 6;\n"
                 "                         0 = a prompt is read from its start unless it goes on from the last token)\n"
                 "  --prompt-cache-every N a checkpoint every N prompt tokens, besides the one at a prompt's end\n"
                 "                         (default 2048, 0 = none)\n"
                 "  --conversation-cache-mib N   RAM for parked conversations, so that several can take turns\n"
                 "                         without being read again (default 0 = off)\n"
                 "  --conversation-cache-slots N        parked conversations at most (default 4)\n"
                 "  --conversation-cache-min-tokens N   a conversation shorter than this is not parked (default 1024)\n"
                 "  --conversation-cache-min-free-mib N park only while this much physical RAM stays free (default\n"
                 "                         2560)\n"
                 "  --eos-ids ID,ID,...    tokens that end a reply (default: the GGUF's eos, eot and eom)\n"
                 "  --temperature F --top-k N --top-p F --min-p F --seed N   sampling for --tokens (default greedy)\n"
                 "  --dump-layers          print a checksum of every layer's stages to stderr\n"
                 "  --top-probs N          with --tokens: print the N most likely ids of every step to stderr\n");
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    CacheFlags cache;
    Sampling s;
    bool want_serve = false;
    long long max_new = 32;
    int top_probs = 0, lookup = 3;
    std::vector<int32_t> ids, eos;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "strata-glm: %s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--model") o.model = val();
        else if (a == "--serve") want_serve = true;
        else if (a == "--max-context") o.max_context = std::atoi(val());
        else if (a == "--threads") o.threads = std::atoi(val());
        else if (a == "--vram-reserve-mib") o.vram_reserve_mib = std::atoi(val());
        else if (a == "--expert-profile") o.profile = val();
        else if (a == "--no-gpu-experts") o.gpu_experts = false;
        else if (a == "--any-node") o.stay_near = false;
        else if (a == "--no-graphs") o.graphs = false;
        else if (a == "--window") o.window = std::atoi(val());
        else if (a == "--lookup") lookup = std::atoi(val());
        else if (a == "--no-draft") o.draft = false;
        else if (a == "--draft-model") o.draft_model = val();
        else if (a == "--expert-swaps") o.expert_swaps = std::atoi(val());
        else if (a == "--prompt-batch") o.prompt_batch = std::atoi(val());
        else if (a == "--stage-mib") o.stage_mib = std::atoi(val());
        else if (a == "--stage-ahead") o.stage_ahead = std::atoi(val());
        else if (a == "--prompt-cache") cache.o.checkpoints = std::atoi(val());
        else if (a == "--prompt-cache-every") cache.o.every = std::atoll(val());
        else if (a == "--conversation-cache-mib") cache.o.park_mib = std::atoll(val());
        else if (a == "--conversation-cache-slots") cache.o.park_slots = std::atoi(val());
        else if (a == "--conversation-cache-min-tokens") cache.o.park_min = std::atoll(val());
        else if (a == "--conversation-cache-min-free-mib") cache.min_free_mib = std::atoll(val());
        else if (a == "--dump-layers") o.dump = true;
        else if (a == "--top-probs") top_probs = std::atoi(val());
        else if (a == "--max-new") max_new = std::atoll(val());
        else if (a == "--temperature") s.temperature = std::strtof(val(), nullptr);
        else if (a == "--top-k") s.top_k = std::atoi(val());
        else if (a == "--top-p") s.top_p = std::strtof(val(), nullptr);
        else if (a == "--min-p") s.min_p = std::strtof(val(), nullptr);
        else if (a == "--seed") s.seed = std::strtoull(val(), nullptr, 10);
        else if (a == "--tokens" || a == "--eos-ids") {
            if (!parse_ids(val(), a == "--tokens" ? ids : eos)) {
                std::fprintf(stderr, "strata-glm: %s takes token ids separated by commas\n", a.c_str());
                return 2;
            }
        } else {
            if (a != "--help" && a != "-h") std::fprintf(stderr, "strata-glm: unknown option %s\n", a.c_str());
            usage();
            return 2;
        }
    }
    if (o.model.empty() || (!want_serve && ids.empty()) || max_new < 1) {
        usage();
        return 2;
    }
    const long long most = 1LL << 24;   // MiB: 16 TiB
    if (cache.o.checkpoints < 0 || cache.o.every < 0 || cache.o.park_mib < 0 || cache.o.park_mib > most ||
        cache.o.park_slots < 0 || cache.o.park_slots > 1024 || cache.o.park_min < 0 || cache.min_free_mib < 0 ||
        cache.min_free_mib > most) {
        std::fprintf(stderr, "strata-glm: a cache option is negative or too large\n");
        return 2;
    }

    Engine engine;
    try {
        const Clock::time_point t0 = Clock::now();
        engine.load(o);
        std::fprintf(stderr, "strata-glm: loaded in %.1f s; %d of %d routed experts in VRAM (%.1f GiB), %d CPU threads\n",
                     ms_since(t0) / 1000.0, engine.cached_experts(), engine.total_experts(), engine.cached_gib(),
                     engine.threads());
        if (engine.numa_node() >= 0)
            std::fprintf(stderr, "strata-glm: the CPU threads stay on NUMA node %d, which has %.0f%% of the experts' "
                                 "pages\n", engine.numa_node(), 100.0 * engine.numa_share());
        if (eos.empty())
            for (const char* key : {"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id",
                                    "tokenizer.ggml.eom_token_id"}) {
                const int64_t id = engine.meta_u(key, -1);
                if (id >= 0 && std::find(eos.begin(), eos.end(), (int32_t) id) == eos.end()) eos.push_back((int32_t) id);
            }
        Session ss(engine, eos, cache, lookup);
        int rc = 0;
        if (want_serve) {
            rc = serve(ss);
        } else if ((long long) ids.size() + max_new > engine.max_context()) {
            throw std::runtime_error("the prompt and --max-new together exceed --max-context");
        } else {
            rc = once(ss, ids, max_new, s, top_probs);
        }
        engine.save_profile();
        return rc;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-glm: %s\n", e.what());
        if (want_serve) {
            std::printf("ERR %s\n", e.what());
            std::fflush(stdout);
        }
        return 1;
    }
}
