#include "targets/qwen3_6/impl/runtime/endpoint_execution.h"

#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail {
namespace {

std::size_t matrix_bytes(std::int32_t rows, std::int32_t columns, std::size_t element_bytes,
                         const char* label) {
    if (rows <= 0 || columns <= 0 ||
        static_cast<std::size_t>(rows) >
            std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(columns) /
                element_bytes) {
        throw std::invalid_argument(label);
    }
    return static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns) * element_bytes;
}

void require_contiguous(const Tensor& tensor, DType dtype, const char* label) {
    if (tensor.data == nullptr || tensor.dtype != dtype || !tensor.is_contiguous()) {
        throw std::invalid_argument(label);
    }
}

} // namespace

EndpointExecution::EndpointExecution(DeviceContext& primary, DeviceContext& secondary,
                                     std::int32_t hidden, std::int32_t vocabulary,
                                     std::int32_t maximum_embedding_columns,
                                     std::int32_t maximum_score_columns)
    : primary_(primary), secondary_(secondary), hidden_(hidden), vocabulary_(vocabulary),
      maximum_embedding_columns_(maximum_embedding_columns),
      maximum_score_columns_(maximum_score_columns),
      secondary_ids_((secondary.activate(),
                      matrix_bytes(1, maximum_embedding_columns, sizeof(std::int32_t),
                                   "endpoint embedding capacity is invalid"))),
      secondary_hidden_(matrix_bytes(
          hidden, std::max(maximum_embedding_columns, maximum_score_columns), sizeof(std::uint16_t),
          "endpoint hidden capacity is invalid")),
      secondary_logits_(matrix_bytes(vocabulary, maximum_score_columns, sizeof(std::uint16_t),
                                     "endpoint logits capacity is invalid")),
      host_ids_(matrix_bytes(1, maximum_embedding_columns, sizeof(std::int32_t),
                             "endpoint host id capacity is invalid")),
      host_hidden_(matrix_bytes(
          hidden, std::max(maximum_embedding_columns, maximum_score_columns), sizeof(std::uint16_t),
          "endpoint host hidden capacity is invalid")),
      host_logits_(matrix_bytes(vocabulary, maximum_score_columns, sizeof(std::uint16_t),
                                "endpoint host logits capacity is invalid")) {
    if (primary.device == secondary.device) {
        throw std::invalid_argument("endpoint execution requires two distinct CUDA devices");
    }
    primary_.activate();
}

EndpointExecution::~EndpointExecution() noexcept {
    if (secondary_.stream != nullptr) {
        (void)cudaSetDevice(secondary_.device);
        (void)cudaStreamSynchronize(secondary_.stream);
    }
    (void)cudaSetDevice(primary_.device);
}

void EndpointExecution::embedding(const Tensor& primary_ids, const Weight& secondary_weight,
                                  Tensor& primary_output) {
    require_contiguous(primary_ids, DType::I32, "endpoint ids must be contiguous I32");
    require_contiguous(primary_output, DType::BF16,
                       "endpoint embedding output must be contiguous BF16");
    const std::int32_t columns = static_cast<std::int32_t>(primary_ids.numel());
    if (columns <= 0 || columns > maximum_embedding_columns_ ||
        primary_output.numel() != static_cast<std::int64_t>(hidden_) * columns ||
        secondary_weight.n != vocabulary_ || secondary_weight.k != hidden_) {
        throw std::invalid_argument("endpoint embedding shape exceeds the frozen bridge");
    }
    const std::size_t id_bytes =
        matrix_bytes(1, columns, sizeof(std::int32_t), "endpoint id copy size is invalid");
    const std::size_t hidden_bytes =
        matrix_bytes(hidden_, columns, sizeof(std::uint16_t),
                     "endpoint embedding copy size is invalid");

    primary_.activate();
    CUDA_CHECK(cudaMemcpyAsync(host_ids_.data(), primary_ids.data, id_bytes,
                               cudaMemcpyDeviceToHost, primary_.stream));
    primary_.synchronize();

    secondary_.activate();
    CUDA_CHECK(cudaMemcpyAsync(secondary_ids_.p, host_ids_.data(), id_bytes,
                               cudaMemcpyHostToDevice, secondary_.stream));
    Tensor ids(secondary_ids_.p, DType::I32, {columns});
    Tensor output(secondary_hidden_.p, DType::BF16, {hidden_, columns});
    ops::embedding(ids, secondary_weight, output, secondary_.stream);
    CUDA_CHECK(cudaMemcpyAsync(host_hidden_.data(), output.data, hidden_bytes,
                               cudaMemcpyDeviceToHost, secondary_.stream));
    secondary_.synchronize();

    primary_.activate();
    CUDA_CHECK(cudaMemcpyAsync(primary_output.data, host_hidden_.data(), hidden_bytes,
                               cudaMemcpyHostToDevice, primary_.stream));
}

void EndpointExecution::linear(const Tensor& primary_input, const Weight& secondary_weight,
                               Tensor& primary_output) {
    require_contiguous(primary_input, DType::BF16,
                       "endpoint linear input must be contiguous BF16");
    require_contiguous(primary_output, DType::BF16,
                       "endpoint linear output must be contiguous BF16");
    if (primary_input.ne[0] != hidden_ || secondary_weight.k != hidden_) {
        throw std::invalid_argument("endpoint linear hidden dimension is invalid");
    }
    const std::int32_t columns =
        static_cast<std::int32_t>(primary_input.numel() / hidden_);
    const std::int32_t rows = secondary_weight.n;
    if (columns <= 0 || columns > maximum_score_columns_ || rows <= 0 || rows > vocabulary_ ||
        primary_output.numel() != static_cast<std::int64_t>(rows) * columns) {
        throw std::invalid_argument("endpoint linear shape exceeds the frozen bridge");
    }
    const std::size_t hidden_bytes =
        matrix_bytes(hidden_, columns, sizeof(std::uint16_t),
                     "endpoint hidden copy size is invalid");
    const std::size_t logits_bytes =
        matrix_bytes(rows, columns, sizeof(std::uint16_t),
                     "endpoint logits copy size is invalid");

    primary_.activate();
    CUDA_CHECK(cudaMemcpyAsync(host_hidden_.data(), primary_input.data, hidden_bytes,
                               cudaMemcpyDeviceToHost, primary_.stream));
    primary_.synchronize();

    secondary_.activate();
    CUDA_CHECK(cudaMemcpyAsync(secondary_hidden_.p, host_hidden_.data(), hidden_bytes,
                               cudaMemcpyHostToDevice, secondary_.stream));
    Tensor input(secondary_hidden_.p, DType::BF16, {hidden_, columns});
    Tensor output(secondary_logits_.p, DType::BF16, {rows, columns});
    ops::linear(input, secondary_weight, output, secondary_.stream);
    CUDA_CHECK(cudaMemcpyAsync(host_logits_.data(), output.data, logits_bytes,
                               cudaMemcpyDeviceToHost, secondary_.stream));
    secondary_.synchronize();

    primary_.activate();
    CUDA_CHECK(cudaMemcpyAsync(primary_output.data, host_logits_.data(), logits_bytes,
                               cudaMemcpyHostToDevice, primary_.stream));
}

} // namespace ninfer::targets::qwen3_6::detail
