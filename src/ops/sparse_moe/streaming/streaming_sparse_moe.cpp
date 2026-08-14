#include "ops/sparse_moe/streaming/streaming_sparse_moe.h"

#include "ops/sparse_moe/decode/sparse_moe_decode.h"
#include "ops/sparse_moe/prefill/sparse_moe_prefill.h"
#include "ops/sparse_moe/small_t/sparse_moe_small_t.h"
#include "core/device.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <execution>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::ops::detail {
namespace {

constexpr int kExperts                 = 256;
constexpr int kTopK                    = 8;
constexpr int kCacheBanks              = 3;
constexpr int kMaximumRouteTokens      = kSparseMoePrefillSliceMax;
constexpr int kGateRowsPerExpert       = 1024;
constexpr int kDownRowsPerExpert       = 2048;
constexpr int kGateGroupsPerRow        = 32;
constexpr int kDownGroupsPerRow        = 8;
constexpr std::size_t kGateCodeBytes   = 256ULL * 1024ULL * 1024ULL;
constexpr std::size_t kGateScaleBytes  = 16ULL * 1024ULL * 1024ULL;
constexpr std::size_t kDownCodeBytes   = 128ULL * 1024ULL * 1024ULL;
constexpr std::size_t kDownHighBytes   = 64ULL * 1024ULL * 1024ULL;
constexpr std::size_t kDownScaleBytes  = 8ULL * 1024ULL * 1024ULL;
constexpr std::size_t kBankBytes =
    kGateCodeBytes + kGateScaleBytes + kDownCodeBytes + kDownHighBytes + kDownScaleBytes;
constexpr std::size_t kDeviceBytes = kCacheBanks * kBankBytes;
constexpr std::size_t kQ3GateBytes = 218'103'808ULL;
constexpr std::size_t kCompactDownBytes = kDownCodeBytes + kDownScaleBytes;
constexpr std::size_t kCompactBankBytes = kQ3GateBytes + kCompactDownBytes;
constexpr std::size_t kCompactDeviceBytes = kCacheBanks * kCompactBankBytes;

struct HostCopy {
    const std::byte* source = nullptr;
    std::byte* destination  = nullptr;
    std::size_t bytes       = 0;
};

struct DeviceCopy {
    const std::byte* source = nullptr;
    std::byte* destination  = nullptr;
    std::size_t bytes       = 0;
};

struct CacheSlot {
    const void* layer = nullptr;
    int expert        = -1;
    std::uint64_t use = 0;
};

struct CacheMiss {
    int expert = -1;
    int slot   = -1;
};

Weight make_row_split_weight(QType qtype, void* codes, void* high, void* scales,
                             std::int32_t rows, std::int32_t columns, std::int32_t group,
                             std::size_t bytes, std::size_t high_bytes) {
    Weight out{};
    out.payload          = codes;
    out.payload_bytes    = bytes;
    out.high_plane_bytes = high_bytes;
    out.qtype            = qtype;
    out.group_size       = static_cast<std::uint32_t>(group);
    out.qdata            = codes;
    out.qhigh            = high;
    out.scales           = scales;
    out.n                = rows;
    out.k                = columns;
    out.group            = group;
    out.layout           = QuantLayout::RowSplit;
    out.scale_dtype      = DType::FP16;
    out.ndim             = 2;
    out.shape[0]         = rows;
    out.shape[1]         = columns;
    out.padded_shape[0]  = rows;
    out.padded_shape[1]  = columns;
    return out;
}

Weight make_q3_weight(void* payload) {
    Weight out{};
    out.payload          = payload;
    out.payload_bytes    = kQ3GateBytes;
    out.qtype            = QType::Q3G64_F16S;
    out.group_size       = 64;
    out.qdata            = payload;
    out.n                = 262144;
    out.k                = 2048;
    out.group            = 64;
    out.layout           = QuantLayout::GroupInterleaved;
    out.scale_dtype      = DType::FP16;
    out.ndim             = 2;
    out.shape[0]         = 262144;
    out.shape[1]         = 2048;
    out.padded_shape[0]  = 262144;
    out.padded_shape[1]  = 2048;
    return out;
}

} // namespace

std::size_t streaming_sparse_moe_device_bytes(bool compact) noexcept {
    return compact ? kCompactDeviceBytes : kDeviceBytes;
}

class StreamingSparseMoeExecutor::Impl {
public:
    Impl(DeviceArena& arena, bool compact)
        : compact_(compact),
          route_ids_(kMaximumRouteTokens * kTopK * sizeof(std::int32_t)),
          transfer_(compact ? kCompactBankBytes : kBankBytes) {
        gate_codes_ = arena.alloc_bytes(kCacheBanks * (compact ? kQ3GateBytes : kGateCodeBytes),
                                        256).data;
        if (!compact) {
            gate_scales_ = arena.alloc_bytes(kCacheBanks * kGateScaleBytes, 256).data;
        }
        down_codes_ = arena.alloc_bytes(kCacheBanks * kDownCodeBytes, 256).data;
        if (!compact) {
            down_high_ = arena.alloc_bytes(kCacheBanks * kDownHighBytes, 256).data;
        }
        down_scales_ = arena.alloc_bytes(kCacheBanks * kDownScaleBytes, 256).data;
    }

    void run(const Tensor& input, const SparseMoeWeights& mixed, Tensor& residual,
             WorkspaceArena& workspace, cudaStream_t stream) {
        if (input.ne[1] <= 0 || input.ne[1] != residual.ne[1]) {
            throw std::invalid_argument("streaming_sparse_moe: invalid token count");
        }
        auto scope = workspace.scope();
        const int bank = bank_for_layer(mixed.routed_gate_up.qdata);
        SparseMoeWeights device = device_weights(mixed, bank);
 
        if (input.ne[1] == 1) {
            const auto plan = resolve_sparse_moe_decode_plan(device.routed_gate_up.qtype,
                                                             device.routed_down.qtype);
            const auto route = allocate_sparse_moe_decode_workspace(workspace);
            sparse_moe_decode_launch_d1(input, device.router_shared_gate, route, plan.d1, stream);
            sparse_moe_decode_launch_d2(route, plan.d2, stream);
            CUDA_CHECK(cudaMemcpyAsync(route_ids_.data(), route.ids.data, kTopK * sizeof(int),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            stage_selected(mixed, device, kTopK, route.ids.data, true, bank, stream);
            sparse_moe_decode_launch_d3(input, device, route, plan.d3, stream);
            sparse_moe_decode_launch_d4(device, residual, route, plan.d4, stream);
            return;
        }

        if (input.ne[1] <= kSparseMoeSmallTMax) {
            const auto plan = resolve_sparse_moe_small_t_plan(
                input.ne[1], device.routed_gate_up.qtype, device.routed_down.qtype);
            const auto route = allocate_sparse_moe_small_t_workspace(workspace, input.ne[1]);
            sparse_moe_small_t_launch_s1(input, device.router_shared_gate, route, stream);
            sparse_moe_small_t_launch_s2(plan, route, stream);
            const std::size_t id_count = static_cast<std::size_t>(input.ne[1]) * kTopK;
            CUDA_CHECK(cudaMemcpyAsync(route_ids_.data(), route.token_ids.data,
                                       id_count * sizeof(int), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            stage_selected(mixed, device, id_count, route.token_ids.data, true, bank, stream);
            sparse_moe_small_t_launch_s3(input, device, plan, route, stream);
            sparse_moe_small_t_launch_s4(device, residual, plan, route, stream);
            return;
        }

        const auto plan = resolve_sparse_moe_prefill_plan(
            input.ne[1], device.routed_gate_up.qtype, device.routed_down.qtype);
        const auto route = allocate_sparse_moe_prefill_workspace(workspace, plan.slice_tokens);
        for (std::int32_t token0 = 0; token0 < plan.tokens; token0 += plan.slice_tokens) {
            const std::int32_t tokens =
                std::min(plan.slice_tokens, static_cast<std::int32_t>(plan.tokens - token0));
            const Tensor input_slice = input.slice(1, token0, tokens);
            Tensor residual_slice    = residual.slice(1, token0, tokens);
            sparse_moe_prefill_launch_route(input_slice, device.router_shared_gate, route, stream);
            const std::size_t id_count = static_cast<std::size_t>(tokens) * kTopK;
            CUDA_CHECK(cudaMemcpyAsync(route_ids_.data(), route.token_ids.data,
                                       id_count * sizeof(int), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            stage_selected(mixed, device, id_count, nullptr, false, bank, stream);
            sparse_moe_prefill_launch_selected(input_slice, device, residual_slice, route, stream);
        }
    }

private:
    SparseMoeWeights device_weights(const SparseMoeWeights& mixed, int bank) const {
        auto* gate_codes = static_cast<std::byte*>(gate_codes_) +
                           bank * (compact_ ? kQ3GateBytes : kGateCodeBytes);
        auto* down_codes = static_cast<std::byte*>(down_codes_) + bank * kDownCodeBytes;
        auto* down_scales = static_cast<std::byte*>(down_scales_) + bank * kDownScaleBytes;
        SparseMoeWeights device = mixed;
        if (compact_) {
            if (mixed.routed_gate_up.qtype != QType::Q3G64_F16S ||
                mixed.routed_down.qtype != QType::Q4G64_F16S) {
                throw std::logic_error("compact sparse-MoE streamer received a noncompact layer");
            }
            device.routed_gate_up = make_q3_weight(gate_codes);
            device.routed_down = make_row_split_weight(
                QType::Q4G64_F16S, down_codes, nullptr, down_scales, 524288, 512, 64,
                kCompactDownBytes, 0);
            return device;
        }
        auto* gate_scales = static_cast<std::byte*>(gate_scales_) + bank * kGateScaleBytes;
        auto* down_high = static_cast<std::byte*>(down_high_) + bank * kDownHighBytes;
        device.routed_gate_up = make_row_split_weight(
            QType::Q4G64_F16S, gate_codes, nullptr, gate_scales, 262144, 2048, 64,
            kGateCodeBytes + kGateScaleBytes, 0);
        const bool q6 = mixed.routed_down.qtype == QType::Q6G64_F16S;
        const std::size_t down_high_bytes = q6 ? kDownHighBytes : kDownHighBytes / 2;
        device.routed_down = make_row_split_weight(
            mixed.routed_down.qtype, down_codes, down_high, down_scales, 524288, 512, 64,
            kDownCodeBytes + down_high_bytes + kDownScaleBytes, down_high_bytes);
        return device;
    }

    int bank_for_layer(const void* layer) {
        const auto found = std::find(layer_keys_.begin(), layer_keys_.end(), layer);
        if (found != layer_keys_.end()) {
            return static_cast<int>(found - layer_keys_.begin()) % kCacheBanks;
        }
        layer_keys_.push_back(layer);
        return static_cast<int>(layer_keys_.size() - 1) % kCacheBanks;
    }

    void parallel_copy() {
        std::for_each(std::execution::par, host_copies_.begin(), host_copies_.end(),
                      [](const HostCopy& copy) {
                          std::memcpy(copy.destination, copy.source, copy.bytes);
                      });
    }

    void stage_selected(const SparseMoeWeights& host, const SparseMoeWeights& device,
                        std::size_t id_count, void* device_ids, bool cached,
                        int bank, cudaStream_t stream) {
        auto* ids = static_cast<int*>(route_ids_.data());
        std::array<bool, kExperts> seen{};
        std::vector<int> unique;
        unique.reserve(id_count);
        for (std::size_t i = 0; i < id_count; ++i) {
            if (ids[i] < 0 || ids[i] >= kExperts) {
                throw std::runtime_error("streaming_sparse_moe: router produced invalid expert");
            }
            if (!seen[ids[i]]) {
                seen[ids[i]] = true;
                unique.push_back(ids[i]);
            }
        }

        const void* layer_key = host.routed_gate_up.qdata;
        std::array<int, kExperts> route_slot{};
        route_slot.fill(-1);
        std::vector<CacheMiss> misses;
        misses.reserve(unique.size());
        if (!cached) {
            cache_[bank].fill(CacheSlot{});
            for (const int expert : unique) { misses.push_back({expert, expert}); }
        } else for (const int expert : unique) {
            int selected_slot = -1;
            for (int candidate = 0; candidate < kExperts; ++candidate) {
                if (cache_[bank][candidate].layer == layer_key &&
                    cache_[bank][candidate].expert == expert) {
                    selected_slot = candidate;
                    break;
                }
            }
            if (selected_slot < 0) {
                selected_slot = 0;
                for (int candidate = 1; candidate < kExperts; ++candidate) {
                    if (cache_[bank][candidate].expert < 0 ||
                        cache_[bank][candidate].use < cache_[bank][selected_slot].use) {
                        selected_slot = candidate;
                        if (cache_[bank][candidate].expert < 0) { break; }
                    }
                }
                cache_[bank][selected_slot].layer  = layer_key;
                cache_[bank][selected_slot].expert = expert;
                misses.push_back({expert, selected_slot});
            }
            cache_[bank][selected_slot].use = ++cache_clock_[bank];
            route_slot[expert]        = selected_slot;
        }
        std::sort(misses.begin(), misses.end(),
                  [](const CacheMiss& a, const CacheMiss& b) { return a.slot < b.slot; });

        const std::size_t gate_code_per = compact_ ? 1024ULL * 32ULL * 26ULL
                                                   : kGateRowsPerExpert * kGateGroupsPerRow * 32ULL;
        const std::size_t gate_scale_per = kGateRowsPerExpert * kGateGroupsPerRow * 2ULL;
        const std::size_t down_code_per = kDownRowsPerExpert * kDownGroupsPerRow * 32ULL;
        const std::size_t down_high_per = compact_ ? 0 :
            kDownRowsPerExpert * kDownGroupsPerRow *
            (host.routed_down.qtype == QType::Q6G64_F16S ? 16ULL : 8ULL);
        const std::size_t down_scale_per = kDownRowsPerExpert * kDownGroupsPerRow * 2ULL;

        std::byte* packed = static_cast<std::byte*>(transfer_.data());
        std::size_t cursor = 0;
        host_copies_.clear();
        device_copies_.clear();
        host_copies_.reserve(misses.size() * 5);
        device_copies_.reserve(misses.size() * 5);
        const auto pack_plane = [&](const void* source_plane, const void* destination_plane,
                                    std::size_t bytes) {
            const std::size_t plane_start = cursor;
            for (const CacheMiss& miss : misses) {
                const auto* source = static_cast<const std::byte*>(source_plane) +
                                     static_cast<std::size_t>(miss.expert) * bytes;
                std::byte* staging = packed + cursor;
                host_copies_.push_back({source, staging, bytes});
                cursor += bytes;
            }
            for (std::size_t begin = 0; begin < misses.size();) {
                std::size_t end = begin + 1;
                while (end < misses.size() && misses[end].slot == misses[end - 1].slot + 1) { ++end; }
                auto* destination = const_cast<std::byte*>(
                    static_cast<const std::byte*>(destination_plane) +
                    static_cast<std::size_t>(misses[begin].slot) * bytes);
                device_copies_.push_back(
                    {packed + plane_start + begin * bytes, destination, (end - begin) * bytes});
                begin = end;
            }
        };
        pack_plane(host.routed_gate_up.qdata, device.routed_gate_up.qdata, gate_code_per);
        if (!compact_) {
            pack_plane(host.routed_gate_up.scales, device.routed_gate_up.scales, gate_scale_per);
        }
        pack_plane(host.routed_down.qdata, device.routed_down.qdata, down_code_per);
        if (!compact_) {
            pack_plane(host.routed_down.qhigh, device.routed_down.qhigh, down_high_per);
        }
        pack_plane(host.routed_down.scales, device.routed_down.scales, down_scale_per);
        if (cursor > transfer_.size()) {
            throw std::logic_error("streaming_sparse_moe: transfer staging capacity exceeded");
        }
        parallel_copy();
        for (const DeviceCopy& copy : device_copies_) {
            CUDA_CHECK(cudaMemcpyAsync(copy.destination, copy.source, copy.bytes,
                                       cudaMemcpyHostToDevice, stream));
        }
        if (cached) {
            if (device_ids == nullptr) {
                throw std::logic_error("streaming_sparse_moe: cached route has no device ids");
            }
            for (std::size_t i = 0; i < id_count; ++i) { ids[i] = route_slot[ids[i]]; }
            CUDA_CHECK(cudaMemcpyAsync(device_ids, ids, id_count * sizeof(int),
                                       cudaMemcpyHostToDevice, stream));
        }
    }

    void* gate_codes_  = nullptr;
    void* gate_scales_ = nullptr;
    void* down_codes_  = nullptr;
    void* down_high_   = nullptr;
    void* down_scales_ = nullptr;
    bool compact_ = false;
    PinnedHostBuffer route_ids_;
    PinnedHostBuffer transfer_;
    std::vector<HostCopy> host_copies_;
    std::vector<DeviceCopy> device_copies_;
    std::array<std::array<CacheSlot, kExperts>, kCacheBanks> cache_{};
    std::array<std::uint64_t, kCacheBanks> cache_clock_{};
    std::vector<const void*> layer_keys_;
};

StreamingSparseMoeExecutor::StreamingSparseMoeExecutor(DeviceArena& weights_arena, bool compact)
    : impl_(std::make_unique<Impl>(weights_arena, compact)) {}

StreamingSparseMoeExecutor::~StreamingSparseMoeExecutor() = default;

void StreamingSparseMoeExecutor::run(const Tensor& device_input,
                                     const SparseMoeWeights& mixed_weights,
                                     Tensor& device_residual, WorkspaceArena& workspace,
                                     cudaStream_t stream) {
    impl_->run(device_input, mixed_weights, device_residual, workspace, stream);
}

} // namespace ninfer::ops::detail
