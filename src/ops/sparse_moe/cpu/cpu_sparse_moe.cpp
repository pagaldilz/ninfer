#include "ninfer/ops/cpu_sparse_moe.h"

#include "core/device.h"
#include "ninfer/ops/residual_add.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHidden       = 2048;
constexpr std::int32_t kExperts      = 256;
constexpr std::int32_t kTopK         = 8;
constexpr std::int32_t kIntermediate = 512;

float bf16_to_float(std::uint16_t bits) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
}

std::uint16_t float_to_bf16(float value) noexcept {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t magnitude = bits & 0x7fffffffU;
    if (magnitude > 0x7f800000U) {
        return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
    }
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

float fp16_to_float(std::uint16_t h) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000U) << 16U;
    std::uint32_t exponent   = (h >> 10U) & 0x1fU;
    std::uint32_t mantissa   = h & 0x03ffU;
    std::uint32_t bits       = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x0400U) == 0) {
                mantissa <<= 1U;
                ++shift;
            }
            mantissa &= 0x03ffU;
            bits = sign | (static_cast<std::uint32_t>(127 - 15 - shift) << 23U) |
                   (mantissa << 13U);
        }
    } else if (exponent == 0x1fU) {
        bits = sign | 0x7f800000U | (mantissa << 13U);
    } else {
        bits = sign | ((exponent + (127U - 15U)) << 23U) | (mantissa << 13U);
    }
    return std::bit_cast<float>(bits);
}

struct RowSplitView {
    const std::uint8_t* low   = nullptr;
    const std::uint8_t* high  = nullptr;
    const std::uint8_t* scale = nullptr;
    QType qtype               = QType::Q4G64_F16S;
    std::int32_t rows         = 0;
    std::int32_t columns      = 0;
    std::int32_t groups       = 0;
};

RowSplitView row_split(const Weight& weight) {
    if (weight.layout != QuantLayout::RowSplit || weight.qdata == nullptr ||
        weight.scales == nullptr || weight.group <= 0 || weight.k <= 0) {
        throw std::invalid_argument("cpu_sparse_moe: invalid row-split weight");
    }
    return {static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.qhigh),
            static_cast<const std::uint8_t*>(weight.scales), weight.qtype, weight.n, weight.k,
            weight.k / weight.group};
}

float quantized_dot(const RowSplitView& weight, std::int32_t row, const float* input) {
    float sum = 0.0F;
    const std::int64_t row_group = static_cast<std::int64_t>(row) * weight.groups;
    const int group_size = weight.qtype == QType::W8G32_F16S ? 32 : 64;
    for (int group = 0; group < weight.groups; ++group) {
        const std::int64_t index = row_group + group;
        const auto scale_bits = *reinterpret_cast<const std::uint16_t*>(weight.scale + index * 2);
        const float scale      = fp16_to_float(scale_bits);
        const float* x         = input + group * group_size;
        if (weight.qtype == QType::W8G32_F16S) {
            const auto* codes = reinterpret_cast<const std::int8_t*>(weight.low + index * 32);
            for (int k = 0; k < 32; ++k) { sum += static_cast<float>(codes[k]) * scale * x[k]; }
            continue;
        }
        const std::uint8_t* codes = weight.low + index * 32;
        const std::uint8_t* high  = weight.high;
        for (int lane = 0; lane < 32; ++lane) {
            const std::uint8_t packed = codes[lane];
            int q0 = packed & 0x0fU;
            int q1 = packed >> 4U;
            if (weight.qtype == QType::Q4G64_F16S) {
                q0 = (q0 ^ 0x08) - 0x08;
                q1 = (q1 ^ 0x08) - 0x08;
            } else if (weight.qtype == QType::Q5G64_F16S) {
                const std::uint8_t high_byte = high[index * 8 + (lane >> 2)];
                const int shift = (lane & 3) * 2;
                q0 = ((q0 | (((high_byte >> shift) & 1) << 4)) ^ 0x10) - 0x10;
                q1 = ((q1 | (((high_byte >> (shift + 1)) & 1) << 4)) ^ 0x10) - 0x10;
            } else if (weight.qtype == QType::Q6G64_F16S) {
                const std::uint8_t high_byte = high[index * 16 + (lane >> 1)];
                const int shift = (lane & 1) * 4;
                q0 = ((q0 | (((high_byte >> shift) & 3) << 4)) ^ 0x20) - 0x20;
                q1 = ((q1 | (((high_byte >> (shift + 2)) & 3) << 4)) ^ 0x20) - 0x20;
            } else {
                throw std::invalid_argument("cpu_sparse_moe: unsupported quantized weight");
            }
            sum += static_cast<float>(q0) * scale * x[lane * 2];
            sum += static_cast<float>(q1) * scale * x[lane * 2 + 1];
        }
    }
    return sum;
}

float silu(float x) noexcept { return x / (1.0F + std::exp(-x)); }

} // namespace

class CpuSparseMoeExecutor::Impl {
public:
    explicit Impl(std::uint32_t worker_threads)
        : worker_threads_(std::clamp(worker_threads, 1U, 32U)) {
        workers_.reserve(worker_threads_ > 1 ? worker_threads_ - 1 : 0);
        for (std::uint32_t i = 1; i < worker_threads_; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    ~Impl() {
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
            ++generation_;
        }
        ready_.notify_all();
        for (auto& worker : workers_) { worker.join(); }
    }

    std::uint32_t worker_threads() const noexcept { return worker_threads_; }

    void run(const Tensor& device_input, const SparseMoeWeights& weights, Tensor& device_residual,
             WorkspaceArena& workspace, cudaStream_t stream) {
        if (device_input.dtype != DType::BF16 || device_residual.dtype != DType::BF16 ||
            device_input.ne[0] != kHidden || device_residual.ne[0] != kHidden ||
            device_input.ne[1] != device_residual.ne[1] || !device_input.is_contiguous() ||
            !device_residual.is_contiguous()) {
            throw std::invalid_argument("cpu_sparse_moe: invalid input or residual tensor");
        }
        const std::int32_t tokens = device_input.ne[1];
        ensure_staging(tokens);
        const std::size_t transfer_bytes = static_cast<std::size_t>(tokens) * kHidden * 2;
        CUDA_CHECK(cudaMemcpyAsync(input_->data(), device_input.data, transfer_bytes,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        weights_ = &weights;
        if (tokens == 1) {
            execute_decode(static_cast<const std::uint16_t*>(input_->data()),
                           static_cast<std::uint16_t*>(output_->data()));
        } else {
            token_input_  = static_cast<const std::uint16_t*>(input_->data());
            token_output_ = static_cast<std::uint16_t*>(output_->data());
            parallel_for(static_cast<std::size_t>(tokens), Phase::Tokens);
        }

        auto scope        = workspace.scope();
        Tensor device_out = workspace.alloc(DType::BF16, {kHidden, tokens});
        CUDA_CHECK(cudaMemcpyAsync(device_out.data, output_->data(), transfer_bytes,
                                   cudaMemcpyHostToDevice, stream));
        residual_add(device_out, device_residual, stream);
    }

private:
    enum class Phase : std::uint8_t { Idle, Router, GateUp, Down, Tokens };

    void ensure_staging(std::int32_t tokens) {
        const std::size_t bytes = static_cast<std::size_t>(tokens) * kHidden * 2;
        if (input_ == nullptr || input_->size() < bytes) {
            input_  = std::make_unique<PinnedHostBuffer>(bytes);
            output_ = std::make_unique<PinnedHostBuffer>(bytes);
        }
    }

    void parallel_for(std::size_t count, Phase phase) {
        count_ = count;
        next_.store(0, std::memory_order_relaxed);
        completed_.store(0, std::memory_order_relaxed);
        {
            std::lock_guard lock(mutex_);
            phase_ = phase;
            ++generation_;
        }
        ready_.notify_all();
        process_phase(phase);
        if (worker_threads_ > 1) {
            std::unique_lock lock(mutex_);
            done_.wait(lock, [&] {
                return completed_.load(std::memory_order_acquire) == worker_threads_ - 1;
            });
        }
        phase_ = Phase::Idle;
    }

    void worker_loop() {
        std::uint64_t observed = 0;
        for (;;) {
            Phase phase;
            {
                std::unique_lock lock(mutex_);
                ready_.wait(lock, [&] { return stop_ || generation_ != observed; });
                if (stop_) { return; }
                observed = generation_;
                phase    = phase_;
            }
            process_phase(phase);
            if (completed_.fetch_add(1, std::memory_order_acq_rel) + 1 == worker_threads_ - 1) {
                done_.notify_one();
            }
        }
    }

    void process_phase(Phase phase) {
        constexpr std::size_t chunk = 4;
        for (;;) {
            const std::size_t begin = next_.fetch_add(chunk, std::memory_order_relaxed);
            if (begin >= count_) { return; }
            const std::size_t end = std::min(count_, begin + chunk);
            for (std::size_t index = begin; index < end; ++index) { run_index(phase, index); }
        }
    }

    void run_index(Phase phase, std::size_t index) {
        switch (phase) {
        case Phase::Router:
            router_row(index);
            break;
        case Phase::GateUp:
            gate_up_row(index);
            break;
        case Phase::Down:
            down_row(index);
            break;
        case Phase::Tokens:
            execute_token_sequential(token_input_ + index * kHidden,
                                     token_output_ + index * kHidden);
            break;
        case Phase::Idle:
            break;
        }
    }

    void prepare_input(const std::uint16_t* input) {
        for (int k = 0; k < kHidden; ++k) { input_fp32_[k] = bf16_to_float(input[k]); }
    }

    void router_row(std::size_t row) {
        const auto* router = static_cast<const std::uint16_t*>(weights_->router_shared_gate.qdata);
        const auto* values = router + row * kHidden;
        float sum = 0.0F;
        for (int k = 0; k < kHidden; ++k) { sum += bf16_to_float(values[k]) * input_fp32_[k]; }
        scores_[row] = sum;
    }

    void select_routes() {
        std::array<float, kTopK> selected;
        selected.fill(-std::numeric_limits<float>::infinity());
        ids_.fill(std::numeric_limits<std::int32_t>::max());
        for (int expert = 0; expert < kExperts; ++expert) {
            const float value = scores_[expert];
            int position = kTopK - 1;
            if (value < selected[position] ||
                (value == selected[position] && expert > ids_[position])) {
                continue;
            }
            while (position > 0 &&
                   (value > selected[position - 1] ||
                    (value == selected[position - 1] && expert < ids_[position - 1]))) {
                selected[position] = selected[position - 1];
                ids_[position]     = ids_[position - 1];
                --position;
            }
            selected[position] = value;
            ids_[position]     = expert;
        }
        float denominator = 0.0F;
        for (int rank = 0; rank < kTopK; ++rank) {
            alpha_[rank] = std::exp(selected[rank] - selected[0]);
            denominator += alpha_[rank];
        }
        for (float& value : alpha_) { value /= denominator; }
        shared_scale_ = 1.0F / (1.0F + std::exp(-scores_[kExperts]));
    }

    void gate_up_row(std::size_t j) {
        const RowSplitView routed = row_split(weights_->routed_gate_up);
        const RowSplitView shared = row_split(weights_->shared_gate_up);
        for (int rank = 0; rank < kTopK; ++rank) {
            const int base = ids_[rank] * 2 * kIntermediate;
            const float gate = quantized_dot(routed, base + static_cast<int>(j), input_fp32_.data());
            const float up = quantized_dot(routed, base + kIntermediate + static_cast<int>(j),
                                           input_fp32_.data());
            activation_[rank * kIntermediate + j] = silu(gate) * up;
        }
        const float shared_gate = quantized_dot(shared, static_cast<int>(j), input_fp32_.data());
        const float shared_up =
            quantized_dot(shared, kIntermediate + static_cast<int>(j), input_fp32_.data());
        activation_[kTopK * kIntermediate + j] = silu(shared_gate) * shared_up;
    }

    void down_row(std::size_t row) {
        const RowSplitView routed = row_split(weights_->routed_down);
        const RowSplitView shared = row_split(weights_->shared_down);
        float sum = shared_scale_ * quantized_dot(
                                        shared, static_cast<int>(row),
                                        activation_.data() + kTopK * kIntermediate);
        for (int rank = 0; rank < kTopK; ++rank) {
            sum += alpha_[rank] * quantized_dot(
                                      routed, ids_[rank] * kHidden + static_cast<int>(row),
                                      activation_.data() + rank * kIntermediate);
        }
        result_[row] = sum;
    }

    void execute_decode(const std::uint16_t* input, std::uint16_t* output) {
        prepare_input(input);
        parallel_for(kExperts + 1, Phase::Router);
        select_routes();
        parallel_for(kIntermediate, Phase::GateUp);
        parallel_for(kHidden, Phase::Down);
        for (int row = 0; row < kHidden; ++row) { output[row] = float_to_bf16(result_[row]); }
    }

    void execute_token_sequential(const std::uint16_t* input, std::uint16_t* output) {
        // Prefill parallelizes across independent tokens. Each worker uses local
        // scratch to avoid serial phase barriers and false sharing.
        std::array<float, kHidden> x{};
        std::array<float, kExperts + 1> scores{};
        std::array<std::int32_t, kTopK> ids{};
        std::array<float, kTopK> alpha{};
        std::array<float, (kTopK + 1) * kIntermediate> act{};
        for (int k = 0; k < kHidden; ++k) { x[k] = bf16_to_float(input[k]); }
        const auto* router = static_cast<const std::uint16_t*>(weights_->router_shared_gate.qdata);
        for (int row = 0; row <= kExperts; ++row) {
            float sum = 0.0F;
            for (int k = 0; k < kHidden; ++k) {
                sum += bf16_to_float(router[row * kHidden + k]) * x[k];
            }
            scores[row] = sum;
        }
        std::array<float, kTopK> selected;
        selected.fill(-std::numeric_limits<float>::infinity());
        ids.fill(std::numeric_limits<std::int32_t>::max());
        for (int expert = 0; expert < kExperts; ++expert) {
            int position = kTopK - 1;
            const float value = scores[expert];
            if (value < selected[position] ||
                (value == selected[position] && expert > ids[position])) { continue; }
            while (position > 0 &&
                   (value > selected[position - 1] ||
                    (value == selected[position - 1] && expert < ids[position - 1]))) {
                selected[position] = selected[position - 1];
                ids[position]      = ids[position - 1];
                --position;
            }
            selected[position] = value;
            ids[position]      = expert;
        }
        float denom = 0.0F;
        for (int rank = 0; rank < kTopK; ++rank) {
            alpha[rank] = std::exp(selected[rank] - selected[0]);
            denom += alpha[rank];
        }
        for (float& value : alpha) { value /= denom; }
        const float shared_scale = 1.0F / (1.0F + std::exp(-scores[kExperts]));
        const RowSplitView gate_up = row_split(weights_->routed_gate_up);
        const RowSplitView shared_gate_up = row_split(weights_->shared_gate_up);
        for (int j = 0; j < kIntermediate; ++j) {
            for (int rank = 0; rank < kTopK; ++rank) {
                const int base = ids[rank] * 2 * kIntermediate;
                const float gate = quantized_dot(gate_up, base + j, x.data());
                const float up = quantized_dot(gate_up, base + kIntermediate + j, x.data());
                act[rank * kIntermediate + j] = silu(gate) * up;
            }
            const float gate = quantized_dot(shared_gate_up, j, x.data());
            const float up = quantized_dot(shared_gate_up, kIntermediate + j, x.data());
            act[kTopK * kIntermediate + j] = silu(gate) * up;
        }
        const RowSplitView down = row_split(weights_->routed_down);
        const RowSplitView shared_down = row_split(weights_->shared_down);
        for (int row = 0; row < kHidden; ++row) {
            float sum = shared_scale *
                        quantized_dot(shared_down, row, act.data() + kTopK * kIntermediate);
            for (int rank = 0; rank < kTopK; ++rank) {
                sum += alpha[rank] * quantized_dot(down, ids[rank] * kHidden + row,
                                                  act.data() + rank * kIntermediate);
            }
            output[row] = float_to_bf16(sum);
        }
    }

    std::uint32_t worker_threads_ = 1;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::condition_variable done_;
    bool stop_ = false;
    std::uint64_t generation_ = 0;
    Phase phase_ = Phase::Idle;
    std::size_t count_ = 0;
    std::atomic<std::size_t> next_{0};
    std::atomic<std::uint32_t> completed_{0};

    std::unique_ptr<PinnedHostBuffer> input_;
    std::unique_ptr<PinnedHostBuffer> output_;
    const SparseMoeWeights* weights_ = nullptr;
    const std::uint16_t* token_input_ = nullptr;
    std::uint16_t* token_output_ = nullptr;

    std::array<float, kHidden> input_fp32_{};
    std::array<float, kExperts + 1> scores_{};
    std::array<std::int32_t, kTopK> ids_{};
    std::array<float, kTopK> alpha_{};
    float shared_scale_ = 0.0F;
    std::array<float, (kTopK + 1) * kIntermediate> activation_{};
    std::array<float, kHidden> result_{};
};

CpuSparseMoeExecutor::CpuSparseMoeExecutor(std::uint32_t worker_threads)
    : impl_(std::make_unique<Impl>(worker_threads)) {}

CpuSparseMoeExecutor::~CpuSparseMoeExecutor() = default;

std::uint32_t CpuSparseMoeExecutor::worker_threads() const noexcept {
    return impl_->worker_threads();
}

void CpuSparseMoeExecutor::run(const Tensor& device_input, const SparseMoeWeights& host_weights,
                               Tensor& device_residual, WorkspaceArena& workspace,
                               cudaStream_t stream) {
    impl_->run(device_input, host_weights, device_residual, workspace, stream);
}

} // namespace ninfer::ops
