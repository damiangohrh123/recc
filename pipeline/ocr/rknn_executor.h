#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Thin C++ wrapper around librknnrt's C API (third_party/rknn_api.h).
// Port of python/rknn_executor.py.

// One model output: dequantized float data plus its tensor shape (the C API
// doesn't report shape the way a numpy array would).
struct RknnOutput {
    std::vector<float> data;
    std::vector<int> shape;  // e.g. {1, 1, 480, 480} for det's output map
};

class RknnExecutor {
public:
    RknnExecutor() = default;  // Starts unloaded; call load() before use.
    ~RknnExecutor();           // Releases the model if still loaded.

    RknnExecutor(const RknnExecutor&) = delete;
    RknnExecutor& operator=(const RknnExecutor&) = delete;
    RknnExecutor(RknnExecutor&& other) noexcept;             // Transfers ownership of the loaded model.
    RknnExecutor& operator=(RknnExecutor&& other) noexcept;  // Releases any current model, then transfers ownership.

    // Loads the model and initializes the runtime. Returns false on failure
    // instead of exiting the process. target/device_id are unused, kept only
    // for parity with the Python constructor.
    //
    // num_cores duplicates the loaded context that many times (capped at 3,
    // the number of NPU cores on RK3588) and pins each duplicate to its own
    // core, so concurrent run() calls on different core_slot values execute
    // on different physical cores. num_cores=1 (the default) keeps the old
    // single-context, single-core behavior.
    bool load(const std::string& model_path,
              const std::string& target = "",
              const std::string& device_id = "",
              int num_cores = 1);

    // Runs inference on one input tensor, using the context pinned to
    // core_slot (0-based; see load()'s num_cores). Returns one RknnOutput
    // per model output, always dequantized to float32.
    std::vector<RknnOutput> run(const std::vector<float>& input_nhwc, int core_slot = 0);

    void release();  // Frees all contexts; safe to call even if not loaded.

    bool is_loaded() const { return loaded_; }  // True once load() has succeeded.

    // Number of core-pinned contexts available (see load()'s num_cores).
    int num_cores() const { return static_cast<int>(ctxs_.size()); }

private:
    std::vector<uint64_t> ctxs_;  // One RKNN context handle per core slot; empty when unloaded.
    bool loaded_ = false;         // Whether ctxs_ currently owns loaded model(s).

    // Per-output shape, cached once in load() since it's fixed for the life
    // of the model -- avoids re-querying it on every run() call.
    std::vector<std::vector<int>> output_shapes_;
};

// Constructs and loads an RknnExecutor in one step. Returns nullptr on failure.
std::unique_ptr<RknnExecutor> load_model(const std::string& model_path,
                                          const std::string& target = "",
                                          const std::string& device_id = "",
                                          int num_cores = 1);
