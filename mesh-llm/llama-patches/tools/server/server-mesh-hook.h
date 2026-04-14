#pragma once

// Mesh hook system — callbacks from llama-server into mesh-llm during inference.
// See mesh-llm/docs/VIRTUAL_LLM.md for the full design.
//
// Three hooks, all synchronous:
//   Hook 1 (pre_inference)  — before generation starts, can inject into prompt
//   Hook 2 (post_prefill)   — after prompt eval, when first token is uncertain
//     2b   (mid_generation) — during generation, when sustained entropy spike detected
//   Hook 3 (pre_response)   — before response is sent, can replace output
//
// mesh-llm may start background work on Hook 1 and collect results on Hook 3.
// No polling — the C++ side just calls hooks and uses whatever comes back.

#include <nlohmann/json.hpp>
#include <cpp-httplib/httplib.h>

#include <string>
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>

using json = nlohmann::ordered_json;

// Rolling window of per-token signal stats.
// Updated every token during generation — just cheap arithmetic.
//
// Tracks three independent trigger signals:
//   1. Entropy spike  — sustained high entropy (original Hook 2b)
//   2. Repetition     — 3-gram loops in generated token IDs
//   3. Surprise break — sudden NLL spike after a calm generation run
//
struct mesh_signal_window {
    static constexpr int SIZE = 16;
    float entropy[SIZE] = {};
    float margin[SIZE]  = {};
    int   pos   = 0;
    int   count = 0;

    float entropy_mean  = 0;
    float entropy_max   = 0;
    float margin_min    = 1;
    int   uncertain_count = 0;  // total tokens with entropy > 4.0

    // --- Repetition detection ---
    // Track last 32 token IDs and count repeated 3-grams.
    static constexpr int REP_WINDOW = 32;
    int   token_ids[REP_WINDOW] = {};
    int   token_pos = 0;
    int   token_count = 0;

    // --- Surprise break detection ---
    // EWMA of -log(p_chosen) with z-score spike detection.
    float surprise_ewma   = 0;     // exponential weighted moving average
    float surprise_ewma_sq = 0;    // EWMA of squared surprise (for variance)
    static constexpr float SURPRISE_ALPHA = 0.15f;  // smoothing factor
    static constexpr int   SURPRISE_WARMUP = 8;     // tokens before z-scores are meaningful
    int   recent_spike_count = 0;  // spikes in last 8 tokens
    int   recent_calm_count  = 0;  // calm tokens before last 8

    void push(float e, float m) {
        entropy[pos] = e;
        margin[pos]  = m;
        pos = (pos + 1) % SIZE;
        count++;
        entropy_max = std::max(entropy_max, e);
        margin_min  = std::min(margin_min, m);
        entropy_mean = ((entropy_mean * (count - 1)) + e) / count;
        if (e > 4.0f) {
            uncertain_count++;
        }
    }

    // Call after sampling to record the chosen token ID and its probability.
    void push_token(int token_id, float p_chosen) {
        // --- Repetition ---
        token_ids[token_pos] = token_id;
        token_pos = (token_pos + 1) % REP_WINDOW;
        token_count++;

        // --- Surprise break ---
        float surprise = (p_chosen > 1e-10f) ? -std::log2(p_chosen) : 20.0f;

        if (count <= 1) {
            // first token: init EWMA
            surprise_ewma    = surprise;
            surprise_ewma_sq = surprise * surprise;
        } else {
            surprise_ewma    = SURPRISE_ALPHA * surprise + (1.0f - SURPRISE_ALPHA) * surprise_ewma;
            surprise_ewma_sq = SURPRISE_ALPHA * (surprise * surprise) + (1.0f - SURPRISE_ALPHA) * surprise_ewma_sq;
        }

        // Track recent spikes for surprise_break()
        float var = surprise_ewma_sq - surprise_ewma * surprise_ewma;
        float std_dev = (var > 1e-6f) ? std::sqrt(var) : 1.0f;
        float z = (count > SURPRISE_WARMUP) ? (surprise - surprise_ewma) / std_dev : 0.0f;

        // Shift spike/calm tracking (simple 8-token lookback)
        if (z > 2.5f) {
            recent_spike_count++;
        }
        // "calm" = low z-score tokens before current 8-token window
        if (count > 8 && z < 0.5f) {
            recent_calm_count++;
        }
    }

    float uncertain_ratio() const {
        return count > 0 ? (float)uncertain_count / count : 0;
    }

    float tail_entropy_mean() const {
        int n = std::min(count, SIZE);
        if (n == 0) return 0;
        float sum = 0;
        for (int i = 0; i < n; i++) {
            sum += entropy[i];
        }
        return sum / n;
    }

    bool tail_entropy_spike() const {
        return count > SIZE && tail_entropy_mean() > entropy_mean * 2.0f;
    }

    // Check if the model has been consistently uncertain over the recent window.
    // Used for mid-generation Hook 2b: sustained spike = model is lost.
    bool sustained_spike(float spike_ratio) const {
        if (count < SIZE) return false;
        int high_count = 0;
        for (int i = 0; i < SIZE; i++) {
            if (entropy[i] > 4.0f) high_count++;
        }
        return (float)high_count / SIZE >= spike_ratio;
    }

    // Repetition detection: fraction of repeated 3-grams in the token window.
    // Returns ratio in [0,1]. >0.18 suggests looping/degenerate output.
    float repetition_ratio() const {
        int n = std::min(token_count, REP_WINDOW);
        if (n < 6) return 0;  // need at least 2 trigrams

        int n_trigrams = n - 2;
        int repeats = 0;

        // Brute force O(n^2) over tiny window — 32 tokens max = ~900 comparisons
        for (int i = 0; i < n_trigrams; i++) {
            for (int j = i + 1; j < n_trigrams; j++) {
                int ai = (token_pos - n + i + REP_WINDOW) % REP_WINDOW;
                int bi = (token_pos - n + j + REP_WINDOW) % REP_WINDOW;
                if (token_ids[ai] == token_ids[bi] &&
                    token_ids[(ai + 1) % REP_WINDOW] == token_ids[(bi + 1) % REP_WINDOW] &&
                    token_ids[(ai + 2) % REP_WINDOW] == token_ids[(bi + 2) % REP_WINDOW]) {
                    repeats++;
                    break;  // count each source trigram at most once
                }
            }
        }
        return (float)repeats / n_trigrams;
    }

    // Surprise break: was the model calm, then suddenly spiked?
    // True when 2+ recent tokens had z-score >2.5 AND the preceding run was calm.
    bool surprise_break() const {
        if (count < SURPRISE_WARMUP + 8) return false;
        // Need at least 2 spikes in recent generation after a calm run
        return recent_spike_count >= 2 && recent_calm_count >= 4;
    }

    void reset() {
        pos = 0;
        count = 0;
        entropy_mean = 0;
        entropy_max = 0;
        margin_min = 1;
        uncertain_count = 0;
        for (int i = 0; i < SIZE; i++) {
            entropy[i] = 0;
            margin[i] = 0;
        }
        // repetition
        token_pos = 0;
        token_count = 0;
        for (int i = 0; i < REP_WINDOW; i++) {
            token_ids[i] = 0;
        }
        // surprise
        surprise_ewma = 0;
        surprise_ewma_sq = 0;
        recent_spike_count = 0;
        recent_calm_count = 0;
    }

    json to_json() const {
        return {
            {"mean_entropy",          entropy_mean},
            {"max_entropy",           entropy_max},
            {"min_margin",            margin_min},
            {"uncertain_token_count", uncertain_count},
            {"tail_entropy_mean",     tail_entropy_mean()},
            {"total_tokens",          count},
            {"repetition_ratio",      repetition_ratio()},
            {"surprise_ewma",         surprise_ewma},
            {"surprise_break",        surprise_break()},
        };
    }
};

// Per-slot mesh hook context.
struct mesh_hook_ctx {
    bool enabled = false;
    bool debug   = false;  // --mesh-hook-debug: lower all thresholds
    int  port    = 0;
    std::string request_id;

    // configured by Hook 1 response (or defaults)
    float entropy_threshold = 3.0f;   // default: fire Hook 2 when entropy > 3.0 (~8 equally-likely tokens)

    // mid-generation hook state
    int   last_midgen_token = -32;    // token index of last mid-gen hook fire (-32 = never)
    static constexpr int MIDGEN_COOLDOWN = 32;  // minimum tokens between mid-gen hooks
    static constexpr float MIDGEN_SPIKE_RATIO = 0.75f;  // 75% of window must be high entropy

    // pre-inference trigger state (computed from request)
    bool has_images_no_multimodal = false;
    bool has_audio_no_support     = false;

    // signal window — updated every token, read at Hook 2b and Hook 3
    mesh_signal_window signals;

    // reusable HTTP client (one per slot, kept alive across requests)
    std::unique_ptr<httplib::Client> client;

    void init(int mesh_port, bool debug_mode) {
        port = mesh_port;
        debug = debug_mode;
        enabled = true;
        client = std::make_unique<httplib::Client>("localhost", mesh_port);
        client->set_connection_timeout(0, 200000);  // 200ms connect
        client->set_read_timeout(60);               // 60s read (hooks may block for consultation)

        if (debug) {
            // Debug mode: fire hooks on almost anything
            entropy_threshold = 0.5f;
        }
    }

    void reset() {
        request_id.clear();
        entropy_threshold = debug ? 0.5f : 3.0f;
        has_images_no_multimodal = false;
        has_audio_no_support = false;
        last_midgen_token = -MIDGEN_COOLDOWN;
        signals.reset();
    }

    bool any_pre_inference_trigger() const {
        return has_images_no_multimodal
            || has_audio_no_support;
    }

    std::string first_trigger_name() const {
        if (has_images_no_multimodal) return "images_no_multimodal";
        if (has_audio_no_support)     return "audio_no_support";
        return "unknown";
    }

    // Check if mid-generation hook should fire.
    // Any of three independent signals can trigger:
    //   1. Sustained entropy spike (original) — model is lost/incoherent
    //   2. Repetition loop — model is degenerating into 3-gram repeats
    //   3. Surprise break — model was calm then suddenly spiked (hallucination onset)
    // All require cooldown to have elapsed and warmup period to have passed.
    bool should_fire_midgen() const {
        float ratio = debug ? 0.25f : MIDGEN_SPIKE_RATIO;
        int cooldown = debug ? 8 : MIDGEN_COOLDOWN;
        float rep_threshold = debug ? 0.10f : 0.18f;

        // Cooldown check — shared across all triggers
        if ((signals.count - last_midgen_token) < cooldown) return false;

        // Must be past warmup (skip <think> region on thinking models)
        if (signals.count < 12) return false;

        // Trigger 1: sustained entropy spike
        if (signals.sustained_spike(ratio)) return true;

        // Trigger 2: repetition loop
        if (signals.repetition_ratio() >= rep_threshold) return true;

        // Trigger 3: surprise break (calm then suddenly spiked)
        if (signals.surprise_break()) return true;

        return false;
    }

    // Return the name of the trigger that caused should_fire_midgen() to be true.
    // Call only when should_fire_midgen() returned true.
    std::string midgen_trigger_name() const {
        float ratio = debug ? 0.25f : MIDGEN_SPIKE_RATIO;
        float rep_threshold = debug ? 0.10f : 0.18f;
        if (signals.sustained_spike(ratio)) return "sustained_entropy_spike";
        if (signals.repetition_ratio() >= rep_threshold) return "repetition_loop";
        if (signals.surprise_break()) return "surprise_break";
        return "unknown";
    }

    // --- Hook helpers ---

    // POST to mesh-llm and return parsed response, or empty json on failure.
    json call_hook(const json & payload) {
        if (!client) return {};
        try {
            auto res = client->Post("/mesh/hook", payload.dump(), "application/json");
            if (res && res->status == 200) {
                return json::parse(res->body);
            }
        } catch (...) {
            // mesh-llm may not be listening — hooks are best-effort
        }
        return {};
    }

    // Process a hook response: apply entropy_threshold.
    // Returns inject text (empty = no injection).
    std::string process_response(const json & resp) {
        if (resp.empty()) return "";

        auto action = resp.value("action", "none");

        if (resp.contains("entropy_threshold")) {
            entropy_threshold = resp["entropy_threshold"].get<float>();
        }

        if (action == "inject") {
            return resp.value("text", "");
        }

        // "none" or "stop"
        return "";
    }
};

// Compute entropy from a sorted probability distribution.
inline float mesh_compute_entropy(const std::vector<llama_token_data> & probs) {
    float h = 0.0f;
    for (const auto & p : probs) {
        if (p.p > 0.0f) {
            h -= p.p * std::log2(p.p);
        }
    }
    return h;
}
