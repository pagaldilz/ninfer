#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"

#include <cstdint>

namespace ninfer::targets::qwen3_6::detail {

// Executes only the large vocabulary embedding and projection weights on a second CUDA device.
// The supported RTX 50-series pair has no CUDA peer access, so represented inputs and outputs
// cross the device boundary through stable pinned-host buffers. Transformer state remains owned
// by the primary device and no runtime tensor is shared between CUDA contexts.
class EndpointExecution {
public:
    EndpointExecution(DeviceContext& primary, DeviceContext& secondary, std::int32_t hidden,
                      std::int32_t vocabulary, std::int32_t maximum_embedding_columns,
                      std::int32_t maximum_score_columns);
    ~EndpointExecution() noexcept;

    EndpointExecution(const EndpointExecution&)            = delete;
    EndpointExecution& operator=(const EndpointExecution&) = delete;

    void embedding(const Tensor& primary_ids, const Weight& secondary_weight,
                   Tensor& primary_output);
    void linear(const Tensor& primary_input, const Weight& secondary_weight,
                Tensor& primary_output);

private:
    DeviceContext& primary_;
    DeviceContext& secondary_;
    std::int32_t hidden_;
    std::int32_t vocabulary_;
    std::int32_t maximum_embedding_columns_;
    std::int32_t maximum_score_columns_;

    DeviceBuffer secondary_ids_;
    DeviceBuffer secondary_hidden_;
    DeviceBuffer secondary_logits_;
    PinnedHostBuffer host_ids_;
    PinnedHostBuffer host_hidden_;
    PinnedHostBuffer host_logits_;
};

} // namespace ninfer::targets::qwen3_6::detail
