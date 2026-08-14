#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/sparse_moe.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace ninfer::ops {

// Exact host execution leaf for the registered Qwen3.6-35B-A3B sparse MoE.
// Large encoded expert banks remain in ordinary cacheable host memory. Only
// BF16 activations and results use the reusable pinned transfer buffers.
class CpuSparseMoeExecutor {
public:
    explicit CpuSparseMoeExecutor(std::uint32_t worker_threads);
    ~CpuSparseMoeExecutor();

    CpuSparseMoeExecutor(const CpuSparseMoeExecutor&)            = delete;
    CpuSparseMoeExecutor& operator=(const CpuSparseMoeExecutor&) = delete;

    [[nodiscard]] std::uint32_t worker_threads() const noexcept;

    void run(const Tensor& device_input, const SparseMoeWeights& host_weights,
             Tensor& device_residual, WorkspaceArena& workspace, cudaStream_t stream);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::ops
