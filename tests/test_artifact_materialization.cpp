#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "artifact_fixture.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

constexpr std::array<std::byte, 3> kResource = {
    std::byte{1},
    std::byte{1},
    std::byte{1},
};
constexpr std::array<std::byte, 4> kTensor = {
    std::byte{2},
    std::byte{2},
    std::byte{2},
    std::byte{2},
};
constexpr std::array<std::byte, 8> kSecondTensor = {
    std::byte{3}, std::byte{3}, std::byte{3}, std::byte{3},
    std::byte{3}, std::byte{3}, std::byte{3}, std::byte{3},
};

ninfer::test::artifact_fixture::TemporaryArtifact write_fixture() {
    using Json = ninfer::test::artifact_fixture::Json;
    return ninfer::test::artifact_fixture::write_fixture(
        {
            {"identity", {{"model_id", "fixture-model"}, {"weights_id", "fixture-weights"}}},
            {"objects", Json::array({
                            {{"name", "frontend/test.json"},
                             {"kind", "resource"},
                             {"encoding", "raw-bytes-v1"},
                             {"offset", 0},
                             {"bytes", 3}},
                            {{"name", "weights/test"},
                             {"kind", "tensor"},
                             {"shape", {2}},
                             {"format", "BF16"},
                             {"layout", "contiguous-le-v1"},
                             {"offset", 256},
                             {"bytes", 4}},
                            {{"name", "weights/second"},
                             {"kind", "tensor"},
                             {"shape", {4}},
                             {"format", "BF16"},
                             {"layout", "contiguous-le-v1"},
                             {"offset", 8192},
                             {"bytes", 8}},
                        })},
        },
        "materialization");
}

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

} // namespace

int main() {
    try {
        auto fixture = write_fixture();
        ninfer::artifact::Reader reader(fixture.path);
        ninfer::artifact::Binder validation_binder(reader);
        const auto validated_resource = validation_binder.require_resource(
            "frontend/test.json", ninfer::artifact::ResourceEncoding::RawBytesV1);
        validation_binder.retain_on_host(validated_resource);
        constexpr std::array<std::uint64_t, 1> validated_shape = {2};
        const auto validated_only                              = validation_binder.require_tensor(
            "weights/test", ninfer::artifact::NumericFormat::BF16,
            ninfer::artifact::StorageLayout::ContiguousLeV1, validated_shape);
        validation_binder.validate_only(validated_only);
        constexpr std::array<std::uint64_t, 1> retained_shape = {4};
        const auto retained_tensor                            = validation_binder.require_tensor(
            "weights/second", ninfer::artifact::NumericFormat::BF16,
            ninfer::artifact::StorageLayout::ContiguousLeV1, retained_shape);
        validation_binder.materialize_on_device(retained_tensor);
        const auto validation_plan = validation_binder.finish();
        require(validation_plan.object_count == 3 && validation_plan.host_objects.size() == 1 &&
                    validation_plan.device_objects.size() == 1 &&
                    validation_plan.device_capacity_bytes == kSecondTensor.size(),
                "validate-only tensor was included in the materialization plan");

        int device_count              = 0;
        const cudaError_t count_error = cudaGetDeviceCount(&device_count);
        if (cuda_unavailable(count_error)) {
            std::cout << "SKIP: no usable CUDA device\n";
            return 77;
        }
        CUDA_CHECK(count_error);
        if (device_count == 0) {
            std::cout << "SKIP: no CUDA devices\n";
            return 77;
        }

        ninfer::artifact::Binder binder(reader);

        const auto resource = binder.require_resource(
            "frontend/test.json", ninfer::artifact::ResourceEncoding::RawBytesV1);
        binder.retain_on_host(resource);
        constexpr std::array<std::uint64_t, 1> second_shape = {4};
        const auto second =
            binder.require_tensor("weights/second", ninfer::artifact::NumericFormat::BF16,
                                  ninfer::artifact::StorageLayout::ContiguousLeV1, second_shape);
        binder.materialize_on_device(second);

        // Bind in the opposite order from the artifact. Device placement order and file read order
        // are intentionally independent, exercising the direct-I/O scatter path.
        constexpr std::array<std::uint64_t, 1> tensor_shape = {2};
        const auto tensor =
            binder.require_tensor("weights/test", ninfer::artifact::NumericFormat::BF16,
                                  ninfer::artifact::StorageLayout::ContiguousLeV1, tensor_shape);
        binder.materialize_on_device(tensor);

        const ninfer::artifact::MaterializationPlan plan = binder.finish();
        require(plan.object_count == 3 && plan.host_objects.size() == 1 &&
                    plan.device_objects.size() == 2 && plan.device_capacity_bytes == 260,
                "binder produced the wrong materialization plan");

        ninfer::DeviceContext device(0);
        auto materialized = ninfer::artifact::materialize(reader, plan, device);

        std::array<std::byte, kTensor.size()> copied{};
        CUDA_CHECK(cudaMemcpy(copied.data(), materialized.device_data(tensor), copied.size(),
                              cudaMemcpyDeviceToHost));
        require(copied == kTensor, "device tensor payload differs from the artifact");
        std::array<std::byte, kSecondTensor.size()> second_copied{};
        CUDA_CHECK(cudaMemcpy(second_copied.data(), materialized.device_data(second),
                              second_copied.size(), cudaMemcpyDeviceToHost));
        require(second_copied == kSecondTensor,
                "second device tensor payload differs from the artifact");

        const auto retained = materialized.resource_bytes(resource);
        require(std::equal(retained.begin(), retained.end(), kResource.begin(), kResource.end()),
                "retained resource payload differs from the artifact");

        const auto& stats = materialized.stats();
        require(stats.tensor_count == 2 && stats.resource_count == 1 &&
                    stats.h2d_bytes == kTensor.size() + kSecondTensor.size() &&
                    stats.retained_resource_bytes == kResource.size() &&
                    stats.file_bytes == kResource.size() +
                                            ninfer::artifact::Reader::direct_io_alignment +
                                            kSecondTensor.size(),
                "materialization statistics are incomplete");
        require(materialized.device_arena().capacity() == plan.device_capacity_bytes &&
                    materialized.device_arena().used() == plan.device_capacity_bytes,
                "materialized tensor does not own the planned device backing");

        if (device_count >= 2) {
            ninfer::artifact::Binder split_binder(reader);
            const auto split_resource = split_binder.require_resource(
                "frontend/test.json", ninfer::artifact::ResourceEncoding::RawBytesV1);
            split_binder.retain_on_host(split_resource);
            const auto primary_tensor = split_binder.require_tensor(
                "weights/test", ninfer::artifact::NumericFormat::BF16,
                ninfer::artifact::StorageLayout::ContiguousLeV1, tensor_shape);
            split_binder.materialize_on_device(primary_tensor);
            const auto secondary_tensor = split_binder.require_tensor(
                "weights/second", ninfer::artifact::NumericFormat::BF16,
                ninfer::artifact::StorageLayout::ContiguousLeV1, second_shape);
            split_binder.materialize_on_device(
                secondary_tensor, ninfer::artifact::DevicePartition::Secondary);
            const auto split_plan = split_binder.finish();
            require(split_plan.device_capacity_bytes == kTensor.size() &&
                        split_plan.secondary_device_capacity_bytes == kSecondTensor.size(),
                    "binder did not preserve independent device partitions");

            ninfer::DeviceContext secondary_device(1);
            auto split = ninfer::artifact::materialize(reader, split_plan, device,
                                                        &secondary_device, nullptr);
            device.activate();
            copied.fill(std::byte{});
            CUDA_CHECK(cudaMemcpy(copied.data(), split.device_data(primary_tensor), copied.size(),
                                  cudaMemcpyDeviceToHost));
            secondary_device.activate();
            second_copied.fill(std::byte{});
            CUDA_CHECK(cudaMemcpy(second_copied.data(), split.device_data(secondary_tensor),
                                  second_copied.size(), cudaMemcpyDeviceToHost));
            require(copied == kTensor && second_copied == kSecondTensor,
                    "split-device tensor payload differs from the artifact");
            require(split.stats().secondary_h2d_bytes == kSecondTensor.size() &&
                        split.device_arena(ninfer::artifact::DevicePartition::Secondary)
                                .capacity() == kSecondTensor.size(),
                    "secondary materialization statistics or ownership are incomplete");
            device.activate();
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
