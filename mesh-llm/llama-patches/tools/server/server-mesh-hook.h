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

// Rolling window of per-token signal stats (entropy, margin).
// Updated every token during generation — just cheap arithmetic.
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
        // Check how many of the last SIZE tokens had entropy > 4.0
        int high_count = 0;
        for (int i = 0; i < SIZE; i++) {
            if (entropy[i] > 4.0f) high_count++;
        }
        return (float)high_count / SIZE >= spike_ratio;
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
    }

    json to_json() const {
        return {
            {"mean_entropy",          entropy_mean},
            {"max_entropy",           entropy_max},
            {"min_margin",            margin_min},
            {"uncertain_token_count", uncertain_count},
            {"tail_entropy_mean",     tail_entropy_mean()},
            {"total_tokens",          count},
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
    bool  verify            = false;  // false = Hook 3 only on triggers

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
        verify = false;
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
    // Requires: sustained spike in rolling window + cooldown elapsed.
    bool should_fire_midgen() const {
        float ratio = debug ? 0.25f : MIDGEN_SPIKE_RATIO;
        int cooldown = debug ? 8 : MIDGEN_COOLDOWN;
        return signals.sustained_spike(ratio)
            && (signals.count - last_midgen_token) >= cooldown;
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

    // Process a hook response: apply entropy_threshold, verify.
    // Returns inject text (empty = no injection).
    std::string process_response(const json & resp) {
        if (resp.empty()) return "";

        auto action = resp.value("action", "none");

        // configure downstream hooks
        if (resp.contains("entropy_threshold")) {
            entropy_threshold = resp["entropy_threshold"].get<float>();
        }
        if (resp.contains("verify")) {
            verify = resp["verify"].get<bool>();
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
