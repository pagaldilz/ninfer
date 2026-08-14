#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/sparse_moe.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace ninfer::ops::detail {

// One reusable full-address device bank. Only the expert row spans selected by the router are
// populated before the existing sparse CUDA leaf runs, so host residency costs PCIe bandwidth for
// active experts rather than CPU compute for whole MoE layers.
[[nodiscard]] std::size_t streaming_sparse_moe_device_bytes(bool compact = false) noexcept;

class StreamingSparseMoeExecutor {
public:
    StreamingSparseMoeExecutor(DeviceArena& weights_arena, bool compact = false);
    ~StreamingSparseMoeExecutor();

    StreamingSparseMoeExecutor(const StreamingSparseMoeExecutor&)            = delete;
    StreamingSparseMoeExecutor& operator=(const StreamingSparseMoeExecutor&) = delete;

    void run(const Tensor& device_input, const SparseMoeWeights& mixed_weights,
             Tensor& device_residual, WorkspaceArena& workspace, cudaStream_t stream);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::ops::detail
