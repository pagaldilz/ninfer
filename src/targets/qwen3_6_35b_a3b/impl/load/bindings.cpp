#include "targets/qwen3_6_35b_a3b/impl/load/bindings.h"

#include "artifact/typed_binding.h"

#include <cstddef>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::targets::qwen3_6_35b_a3b::detail {
namespace {

using artifact::NumericFormat;

bool is_full_layer(std::size_t layer) { return layer >= 3 && (layer - 3) % 4 == 0; }

NumericFormat routed_down_format(std::size_t layer) {
    return layer == 34 || layer == 38 || layer == 39 ? NumericFormat::Q6G64_F16S
                                                     : NumericFormat::Q5G64_F16S;
}

MoePlan bind_moe(artifact::Binder& binder, const std::string& prefix, NumericFormat routed_gate_up,
                 NumericFormat routed_down, bool host = false) {
    return MoePlan{
        .router_shared_gate = artifact::bind_device_tensor(
            binder, prefix + "router_shared_gate", NumericFormat::BF16, {257, 2048}),
        .routed_gate_up = host ? artifact::bind_host_tensor(
                                    binder, prefix + "routed_gate_up", routed_gate_up,
                                    {262144, 2048})
                              : artifact::bind_device_tensor(
                                    binder, prefix + "routed_gate_up", routed_gate_up,
                                    {262144, 2048}),
        .routed_down = host ? artifact::bind_host_tensor(
                                 binder, prefix + "routed_down", routed_down, {524288, 512})
                           : artifact::bind_device_tensor(
                                 binder, prefix + "routed_down", routed_down, {524288, 512}),
        .shared_gate_up = artifact::bind_device_tensor(
            binder, prefix + "shared_gate_up", NumericFormat::W8G32_F16S, {1024, 2048}),
        .shared_down = artifact::bind_device_tensor(
            binder, prefix + "shared_down", NumericFormat::W8G32_F16S, {2048, 512}),
        .routed_gate_up_format = routed_gate_up,
        .routed_down_format = routed_down,
        .host = host,
    };
}

SparseMoePayload load_moe(const MoePlan& plan, const artifact::MaterializedArtifact& materialized,
                          const std::shared_ptr<ops::detail::StreamingSparseMoeExecutor>&
                              streamer = {}) {
    const auto routed = [&](artifact::ObjectHandle handle, NumericFormat format,
                            std::int32_t rows, std::int32_t columns) {
        return plan.host
                   ? artifact::materialized_host_weight(materialized, handle, format, rows, columns)
                   : artifact::materialized_weight(materialized, handle, format, rows, columns);
    };
    return SparseMoePayload{
        .op = {
            .router_shared_gate = artifact::materialized_weight(
                materialized, plan.router_shared_gate, NumericFormat::BF16, 257, 2048),
            .routed_gate_up = routed(plan.routed_gate_up, plan.routed_gate_up_format, 262144, 2048),
            .routed_down    = routed(plan.routed_down, plan.routed_down_format, 524288, 512),
            .shared_gate_up = artifact::materialized_weight(
                materialized, plan.shared_gate_up, NumericFormat::W8G32_F16S, 1024, 2048),
            .shared_down = artifact::materialized_weight(
                materialized, plan.shared_down, NumericFormat::W8G32_F16S, 2048, 512),
        },
        .streamer = plan.host ? streamer : nullptr};
}

std::array<bool, kTextLayers> host_moe_placement(std::uint64_t device_weight_budget,
                                                 bool compact) {
    constexpr std::uint64_t kFullDeviceArena = 22'360'207'360ULL;
    constexpr std::uint64_t kQ5LayerBytes    = 461'373'440ULL;
    constexpr std::uint64_t kQ6LayerBytes    = 494'927'872ULL;
    constexpr std::uint64_t kCompactDeviceArena = 18'501'447'680ULL;
    constexpr std::uint64_t kCompactLayerBytes  = 360'710'144ULL;
    const std::uint64_t full_device_arena = compact ? kCompactDeviceArena : kFullDeviceArena;
    std::array<bool, kTextLayers> host{};
    if (device_weight_budget >= full_device_arena) { return host; }
    const std::uint64_t streaming_bytes =
        ops::detail::streaming_sparse_moe_device_bytes(compact);
    if (device_weight_budget <= streaming_bytes) {
        throw artifact::ArtifactError("5070 Ti device budget cannot reserve sparse-MoE staging");
    }
    const std::uint64_t resident_budget = device_weight_budget - streaming_bytes;
    std::uint64_t savings = 0;
    const auto select = [&](std::size_t layer, std::uint64_t bytes) {
        if (full_device_arena - savings <= resident_budget) { return; }
        host[layer] = true;
        savings += bytes;
    };
    if (compact) {
        // The compact staging bank admits only the uniform Q3+Q4 codec. Keep the two promoted
        // Q6-down layers and the final Q4+Q6 layer resident, then stream uniform layers in order.
        for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
            if (layer != 34 && layer != 38 && layer != 39) {
                select(layer, kCompactLayerBytes);
            }
        }
    } else {
        // Minimize the number of CPU layers first; the three larger Q6-down banks therefore move
        // before the uniform Q5-down banks.
        select(34, kQ6LayerBytes);
        select(38, kQ6LayerBytes);
        select(39, kQ6LayerBytes);
        for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
            if (!host[layer]) { select(layer, kQ5LayerBytes); }
        }
    }
    if (full_device_arena - savings > resident_budget) {
        throw artifact::ArtifactError("5070 Ti device budget cannot retain the non-MoE weights");
    }
    return host;
}

void validate_draft_ids(const artifact::Binder& binder, artifact::ObjectHandle handle) {
    constexpr std::size_t kDraftVocab     = 131072;
    constexpr std::size_t kTokenizerVocab = 248077;
    const auto bytes                      = binder.payload(handle).data;
    std::vector<bool> seen(kTokenizerVocab, false);
    for (std::size_t i = 0; i < kDraftVocab; ++i) {
        const std::byte* value = bytes.data() + i * sizeof(std::uint32_t);
        const std::uint32_t id = std::to_integer<std::uint32_t>(value[0]) |
                                 (std::to_integer<std::uint32_t>(value[1]) << 8U) |
                                 (std::to_integer<std::uint32_t>(value[2]) << 16U) |
                                 (std::to_integer<std::uint32_t>(value[3]) << 24U);
        if (id >= kTokenizerVocab || seen[id]) {
            throw artifact::ArtifactError("invalid optimized draft-head token ids");
        }
        seen[id] = true;
    }
}

} // namespace

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, std::uint64_t device_weight_budget) {
    ArtifactLoadPlan load_plan;
    BindingPlan& out    = load_plan.bindings;
    out.compact_5070ti_experts =
        binder.tensor_format("text/layers/0/moe/routed_gate_up") == NumericFormat::Q3G64_F16S;
    const auto host_moe = host_moe_placement(device_weight_budget, out.compact_5070ti_experts);
    out.frontend        = qwen3_6::bind_frontend_resources(binder);
    out.token_embedding = artifact::bind_device_tensor(binder, "text/token_embedding",
                                                       NumericFormat::W8G32_F16S, {248320, 2048});

    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = artifact::bind_device_tensor(binder, prefix + "input_norm",
                                                                NumericFormat::BF16, {2048});
        target.is_full_attention = is_full_layer(layer);
        if (target.is_full_attention) {
            target.attention.query_key_gate_value =
                artifact::bind_device_tensor(binder, prefix + "attention/query_key_gate_value",
                                             NumericFormat::W8G32_F16S, {9216, 2048});
            target.attention.query_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/query_norm", NumericFormat::BF16, {256});
            target.attention.key_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/key_norm", NumericFormat::BF16, {256});
            target.attention.output = artifact::bind_device_tensor(
                binder, prefix + "attention/output", NumericFormat::W8G32_F16S, {2048, 4096});
        } else {
            target.gdn.a_log       = artifact::bind_device_tensor(binder, prefix + "gdn/a_log",
                                                                  NumericFormat::FP32, {32});
            target.gdn.dt_bias     = artifact::bind_device_tensor(binder, prefix + "gdn/dt_bias",
                                                                  NumericFormat::FP32, {32});
            target.gdn.convolution = artifact::bind_device_tensor(
                binder, prefix + "gdn/convolution", NumericFormat::BF16, {4, 8192});
            target.gdn.a_b_projection = artifact::bind_device_tensor(
                binder, prefix + "gdn/a_b_projection", NumericFormat::BF16, {64, 2048});
            target.gdn.query_key_value_z = artifact::bind_device_tensor(
                binder, prefix + "gdn/query_key_value_z", NumericFormat::W8G32_F16S, {12288, 2048});
            target.gdn.norm   = artifact::bind_device_tensor(binder, prefix + "gdn/norm",
                                                             NumericFormat::BF16, {128});
            target.gdn.output = artifact::bind_device_tensor(
                binder, prefix + "gdn/output", NumericFormat::W8G32_F16S, {2048, 4096});
        }
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {2048});
        const NumericFormat gate_format = out.compact_5070ti_experts && layer < 39
                                              ? NumericFormat::Q3G64_F16S
                                              : NumericFormat::Q4G64_F16S;
        const NumericFormat down_format =
            out.compact_5070ti_experts && layer < 39 && layer != 34 && layer != 38
                ? NumericFormat::Q4G64_F16S
                : routed_down_format(layer);
        target.moe = bind_moe(binder, prefix + "moe/", gate_format, down_format,
                              host_moe[layer]);
        out.host_moe_layers += host_moe[layer] ? 1U : 0U;
    }

    out.final_norm =
        artifact::bind_device_tensor(binder, "text/final_norm", NumericFormat::BF16, {2048});
    out.output_head          = artifact::bind_device_tensor(binder, "text/output_head",
                                                            NumericFormat::Q6G64_F16S, {248320, 2048});
    out.draft_head           = artifact::bind_device_tensor(binder, "text/draft_head",
                                                            NumericFormat::Q4G64_F16S, {131072, 2048});
    out.draft_head_token_ids = artifact::bind_device_tensor(binder, "text/draft_head_token_ids",
                                                            NumericFormat::I32, {131072});
    validate_draft_ids(binder, out.draft_head_token_ids);

    out.mtp.input_projection = artifact::bind_device_tensor(
        binder, "mtp/input_projection", NumericFormat::W8G32_F16S, {2048, 4096});
    out.mtp.embedding_norm =
        artifact::bind_device_tensor(binder, "mtp/embedding_norm", NumericFormat::BF16, {2048});
    out.mtp.hidden_norm =
        artifact::bind_device_tensor(binder, "mtp/hidden_norm", NumericFormat::BF16, {2048});
    out.mtp.input_norm =
        artifact::bind_device_tensor(binder, "mtp/layer/input_norm", NumericFormat::BF16, {2048});
    out.mtp.attention.query_key_gate_value =
        artifact::bind_device_tensor(binder, "mtp/layer/attention/query_key_gate_value",
                                     NumericFormat::W8G32_F16S, {9216, 2048});
    out.mtp.attention.query_norm = artifact::bind_device_tensor(
        binder, "mtp/layer/attention/query_norm", NumericFormat::BF16, {256});
    out.mtp.attention.key_norm = artifact::bind_device_tensor(
        binder, "mtp/layer/attention/key_norm", NumericFormat::BF16, {256});
    out.mtp.attention.output = artifact::bind_device_tensor(
        binder, "mtp/layer/attention/output", NumericFormat::W8G32_F16S, {2048, 4096});
    out.mtp.post_attention_norm = artifact::bind_device_tensor(
        binder, "mtp/layer/post_attention_norm", NumericFormat::BF16, {2048});
    out.mtp.moe =
        bind_moe(binder, "mtp/layer/moe/", NumericFormat::W8G32_F16S, NumericFormat::W8G32_F16S);
    out.mtp.final_norm =
        artifact::bind_device_tensor(binder, "mtp/final_norm", NumericFormat::BF16, {2048});

    out.vision_backbone     = qwen3_6::bind_vision_backbone(binder);
    out.vision_merger_input = qwen3_6::bind_vision_merger_input(binder);
    out.vision_merger_fc2   = artifact::bind_device_tensor(binder, "vision/merger/fc2",
                                                           NumericFormat::W8G32_F16S, {2048, 4608});
    out.vision_merger_fc2_bias =
        artifact::bind_device_tensor(binder, "vision/merger/fc2_bias", NumericFormat::BF16, {2048});
    out.vision_merger_norm = qwen3_6::bind_vision_merger_norm(binder);

    load_plan.materialization = binder.finish();
    if (out.host_moe_layers != 0) {
        load_plan.materialization.device_capacity_bytes +=
            ops::detail::streaming_sparse_moe_device_bytes(out.compact_5070ti_experts);
    }
    return load_plan;
}

LoadedModelData::LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized)
    : backing(std::move(materialized)) {
    frontend = qwen3_6::take_frontend_resources(backing, plan.frontend);

    runtime.weights_arena      = &backing.device_arena();
    auto& token_embedding      = runtime.token_embedding;
    auto& full_layers          = runtime.full_layers;
    auto& gdn_layers           = runtime.gdn_layers;
    auto& final_norm           = runtime.final_norm;
    auto& output_head          = runtime.output_head;
    auto& draft_head           = runtime.draft_head;
    auto& draft_head_token_ids = runtime.draft_head_token_ids;
    auto& mtp                  = runtime.mtp;
    auto& vision               = runtime.vision;
    std::shared_ptr<ops::detail::StreamingSparseMoeExecutor> streamer;
    if (plan.host_moe_layers != 0) {
        streamer =
            std::make_shared<ops::detail::StreamingSparseMoeExecutor>(
                backing.device_arena(), plan.compact_5070ti_experts);
    }

    token_embedding = artifact::materialized_weight(backing, plan.token_embedding,
                                                    NumericFormat::W8G32_F16S, 248320, 2048);

    std::size_t full_index = 0;
    std::size_t gdn_index  = 0;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        const TextLayerPlan& source = plan.text_layers[layer];
        if (source.is_full_attention) {
            FullAttentionWeights& target = full_layers.at(full_index++);
            target.input_norm            = artifact::materialized_tensor(backing, source.input_norm,
                                                                         NumericFormat::BF16, {2048});
            target.projection.query_key_gate_value =
                artifact::materialized_weight(backing, source.attention.query_key_gate_value,
                                              NumericFormat::W8G32_F16S, 9216, 2048);
            target.query_norm = artifact::materialized_tensor(backing, source.attention.query_norm,
                                                              NumericFormat::BF16, {256});
            target.key_norm   = artifact::materialized_tensor(backing, source.attention.key_norm,
                                                              NumericFormat::BF16, {256});
            target.output     = artifact::materialized_weight(backing, source.attention.output,
                                                              NumericFormat::W8G32_F16S, 2048, 4096);
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {2048});
            target.post_mixer = load_moe(source.moe, backing, streamer);
        } else {
            GdnWeights& target = gdn_layers.at(gdn_index++);
            target.input_norm  = artifact::materialized_tensor(backing, source.input_norm,
                                                               NumericFormat::BF16, {2048});
            target.projection.a_log =
                artifact::materialized_tensor(backing, source.gdn.a_log, NumericFormat::FP32, {32});
            target.projection.dt_bias = artifact::materialized_tensor(backing, source.gdn.dt_bias,
                                                                      NumericFormat::FP32, {32});
            target.convolution = artifact::materialized_tensor(backing, source.gdn.convolution,
                                                               NumericFormat::BF16, {8192, 4});
            target.projection.a_b_projection = artifact::materialized_weight(
                backing, source.gdn.a_b_projection, NumericFormat::BF16, 64, 2048);
            target.projection.query_key_value_z = artifact::materialized_weight(
                backing, source.gdn.query_key_value_z, NumericFormat::W8G32_F16S, 12288, 2048);
            target.norm =
                artifact::materialized_tensor(backing, source.gdn.norm, NumericFormat::BF16, {128});
            target.output              = artifact::materialized_weight(backing, source.gdn.output,
                                                                       NumericFormat::W8G32_F16S, 2048, 4096);
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {2048});
            target.post_mixer = load_moe(source.moe, backing, streamer);
        }
    }
    if (full_index != full_layers.size() || gdn_index != gdn_layers.size()) {
        throw std::logic_error("35B Text topology binding is incomplete");
    }

    final_norm =
        artifact::materialized_tensor(backing, plan.final_norm, NumericFormat::BF16, {2048});
    output_head = artifact::materialized_weight(backing, plan.output_head,
                                                NumericFormat::Q6G64_F16S, 248320, 2048);
    draft_head  = artifact::materialized_weight(backing, plan.draft_head, NumericFormat::Q4G64_F16S,
                                                131072, 2048);
    draft_head_token_ids = artifact::materialized_tensor(backing, plan.draft_head_token_ids,
                                                         NumericFormat::I32, {131072});

    mtp.input_projection = artifact::materialized_weight(backing, plan.mtp.input_projection,
                                                         NumericFormat::W8G32_F16S, 2048, 4096);
    mtp.embedding_norm   = artifact::materialized_tensor(backing, plan.mtp.embedding_norm,
                                                         NumericFormat::BF16, {2048});
    mtp.hidden_norm =
        artifact::materialized_tensor(backing, plan.mtp.hidden_norm, NumericFormat::BF16, {2048});
    mtp.input_norm =
        artifact::materialized_tensor(backing, plan.mtp.input_norm, NumericFormat::BF16, {2048});
    mtp.attention.query_key_gate_value = artifact::materialized_weight(
        backing, plan.mtp.attention.query_key_gate_value, NumericFormat::W8G32_F16S, 9216, 2048);
    mtp.query_norm          = artifact::materialized_tensor(backing, plan.mtp.attention.query_norm,
                                                            NumericFormat::BF16, {256});
    mtp.key_norm            = artifact::materialized_tensor(backing, plan.mtp.attention.key_norm,
                                                            NumericFormat::BF16, {256});
    mtp.output              = artifact::materialized_weight(backing, plan.mtp.attention.output,
                                                            NumericFormat::W8G32_F16S, 2048, 4096);
    mtp.post_attention_norm = artifact::materialized_tensor(backing, plan.mtp.post_attention_norm,
                                                            NumericFormat::BF16, {2048});
    mtp.post_mixer =
        load_moe(plan.mtp.moe, backing);
    mtp.final_norm =
        artifact::materialized_tensor(backing, plan.mtp.final_norm, NumericFormat::BF16, {2048});

    vision.common = qwen3_6::materialize_vision_common(
        backing, plan.vision_backbone, plan.vision_merger_input, plan.vision_merger_norm);
    vision.merger_fc2      = artifact::materialized_weight(backing, plan.vision_merger_fc2,
                                                           NumericFormat::W8G32_F16S, 2048, 4608);
    vision.merger_fc2_bias = artifact::materialized_tensor(backing, plan.vision_merger_fc2_bias,
                                                           NumericFormat::BF16, {2048});
}

} // namespace ninfer::targets::qwen3_6_35b_a3b::detail
