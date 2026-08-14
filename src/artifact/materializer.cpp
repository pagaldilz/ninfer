#include "artifact/materializer.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::artifact {
namespace {

constexpr std::size_t kSlotBytes        = 64ULL * 1024ULL * 1024ULL;
constexpr std::size_t kMaximumSlotCount = 4;

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, const char* label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) { throw ArtifactError(label); }
    return a + b;
}

std::uint64_t align_down(std::uint64_t value, std::uint64_t alignment) {
    return value / alignment * alignment;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, const char* label) {
    return checked_add(value, alignment - 1, label) / alignment * alignment;
}

class Slot {
public:
    explicit Slot(std::size_t bytes) : buffer(bytes) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }

    ~Slot() {
        if (pending) { (void)cudaEventSynchronize(event); }
        if (event != nullptr) { (void)cudaEventDestroy(event); }
    }

    void wait() {
        if (pending) {
            CUDA_CHECK(cudaEventSynchronize(event));
            pending = false;
        }
    }

    PinnedHostBuffer buffer;
    cudaEvent_t event = nullptr;
    bool pending      = false;
};

struct CopyRange {
    std::uint64_t source_begin = 0;
    std::uint64_t source_end   = 0;
    std::byte* destination     = nullptr;
};

struct ReadSpan {
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
};

} // namespace

void* MaterializedArtifact::device_data(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || objects_[handle.index].device == nullptr) {
        throw ArtifactError("object handle does not name a materialized tensor");
    }
    return objects_[handle.index].device;
}

std::span<const std::byte> MaterializedArtifact::resource_bytes(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || objects_[handle.index].resource.empty()) {
        throw ArtifactError("object handle does not name a materialized resource");
    }
    return objects_[handle.index].resource;
}

std::vector<std::byte> MaterializedArtifact::take_resource_bytes(ObjectHandle handle) {
    if (handle.index >= objects_.size() || objects_[handle.index].resource.empty()) {
        throw ArtifactError("object handle does not name a materialized resource");
    }
    auto& resource = objects_[handle.index].resource;
    stats_.retained_resource_bytes -= resource.size();
    return std::move(resource);
}

DeviceArena& MaterializedArtifact::device_arena() {
    if (!device_arena_) { throw ArtifactError("artifact has no device tensor backing"); }
    return *device_arena_;
}

DeviceArena& MaterializedArtifact::device_arena(DevicePartition partition) {
    if (partition == DevicePartition::Primary) { return device_arena(); }
    if (!secondary_device_arena_) {
        throw ArtifactError("artifact has no secondary-device tensor backing");
    }
    return *secondary_device_arena_;
}

MaterializedArtifact materialize(const Reader& reader, const MaterializationPlan& plan,
                                 DeviceContext& device, DeviceContext* secondary_device,
                                 LoadProgress* progress) {
    MaterializedArtifact out;
    out.objects_.resize(plan.object_count);
    const std::uint64_t primary_capacity   = plan.device_capacity_bytes;
    const std::uint64_t secondary_capacity = plan.secondary_device_capacity_bytes;
    if (primary_capacity == 0 || primary_capacity > static_cast<std::uint64_t>(SIZE_MAX) ||
        secondary_capacity > static_cast<std::uint64_t>(SIZE_MAX)) {
        throw ArtifactError("artifact tensor backing size is invalid");
    }
    if ((secondary_capacity != 0) != (secondary_device != nullptr)) {
        throw ArtifactError(secondary_capacity != 0
                                ? "materialization plan requires a secondary CUDA device"
                                : "secondary CUDA device supplied for a single-device plan");
    }
    device.activate();
    out.device_arena_ =
        std::make_unique<DeviceArena>(static_cast<std::size_t>(primary_capacity));
    if (secondary_device != nullptr) {
        secondary_device->activate();
        out.secondary_device_arena_ =
            std::make_unique<DeviceArena>(static_cast<std::size_t>(secondary_capacity));
    }
    out.stats_.device_capacity_bytes           = primary_capacity;
    out.stats_.secondary_device_capacity_bytes = secondary_capacity;
    out.stats_.tensor_count          = plan.device_objects.size();
    out.stats_.resource_count        = plan.host_objects.size();

    for (const HostMaterialization& placement : plan.host_objects) {
        auto& resource            = out.objects_.at(placement.object.index).resource;
        const PayloadSpan payload = reader.payload(reader.objects().at(placement.object.index));
        resource.assign(payload.data.begin(), payload.data.end());
        out.stats_.retained_resource_bytes += resource.size();
        out.stats_.file_bytes =
            checked_add(out.stats_.file_bytes, resource.size(), "artifact read bytes overflow u64");
    }

    std::uint64_t total = 0;
    for (const DeviceMaterialization& placement : plan.device_objects) {
        total = checked_add(total, placement.bytes, "artifact tensor byte count overflows u64");
    }
    if (total == 0) { throw ArtifactError("materialization plan has no device tensors"); }

    std::uint64_t copied_total   = 0;
    std::uint64_t last_published = 0;
    const auto start       = std::chrono::steady_clock::now();
    if (progress != nullptr && progress->callback) { progress->callback("weights", 0, total); }

    const auto load_partition = [&](DevicePartition partition, DeviceArena& arena,
                                    DeviceContext& context) {
        context.activate();
        std::vector<CopyRange> ranges;
        for (const DeviceMaterialization& placement : plan.device_objects) {
            if (placement.partition != partition) { continue; }
            const PayloadSpan payload =
                reader.payload(reader.objects().at(placement.object.index));
            DeviceSpan storage = arena.alloc_bytes(static_cast<std::size_t>(placement.bytes),
                                                   static_cast<std::size_t>(placement.alignment));
            const auto actual_offset = static_cast<std::uint64_t>(
                static_cast<std::byte*>(storage.data) - static_cast<std::byte*>(arena.base()));
            if (actual_offset != placement.offset || payload.data.size() != placement.bytes) {
                throw ArtifactError("materialization plan does not match artifact payload");
            }
            out.objects_.at(placement.object.index).device = storage.data;
            ranges.push_back(CopyRange{
                .source_begin = payload.absolute_offset,
                .source_end   = checked_add(payload.absolute_offset, placement.bytes,
                                            "artifact tensor source range overflows u64"),
                .destination  = static_cast<std::byte*>(storage.data),
            });
        }
        if (ranges.empty()) { return; }
        std::sort(ranges.begin(), ranges.end(), [](const CopyRange& a, const CopyRange& b) {
            return a.source_begin < b.source_begin;
        });
        for (std::size_t i = 1; i < ranges.size(); ++i) {
            if (ranges[i].source_begin < ranges[i - 1].source_end) {
                throw ArtifactError("materialization source ranges overlap");
            }
        }

        constexpr std::uint64_t alignment = Reader::direct_io_alignment;
        std::vector<ReadSpan> read_spans;
        std::uint64_t aligned_read_bytes = 0;
        for (const CopyRange& range : ranges) {
            const std::uint64_t begin = align_down(range.source_begin, alignment);
            if (read_spans.empty() ||
                begin > align_up(read_spans.back().end, alignment,
                                 "artifact direct I/O span overflows u64")) {
                read_spans.push_back(ReadSpan{begin, range.source_end});
            } else {
                read_spans.back().end = std::max(read_spans.back().end, range.source_end);
            }
        }
        for (const ReadSpan& span : read_spans) {
            aligned_read_bytes = checked_add(
                aligned_read_bytes,
                align_up(span.end - span.begin, alignment,
                         "artifact direct I/O span overflows u64"),
                "artifact direct I/O byte count overflows u64");
        }
        const std::size_t slot_bytes =
            static_cast<std::size_t>(std::min<std::uint64_t>(kSlotBytes, aligned_read_bytes));
        const std::size_t slot_count = static_cast<std::size_t>(std::min<std::uint64_t>(
            kMaximumSlotCount, 1 + (aligned_read_bytes - 1) / slot_bytes));
        std::vector<std::unique_ptr<Slot>> slots;
        for (std::size_t i = 0; i < slot_count; ++i) {
            slots.push_back(std::make_unique<Slot>(slot_bytes));
        }
        out.stats_.peak_staging_bytes =
            std::max(out.stats_.peak_staging_bytes,
                     static_cast<std::uint64_t>(slot_bytes) * slot_count);

        std::size_t next_slot  = 0;
        std::size_t next_range = 0;
        std::uint64_t partition_copied = 0;
        for (const ReadSpan& span : read_spans) {
            for (std::uint64_t source = span.begin; source < span.end; source += slot_bytes) {
                Slot& slot = *slots[next_slot++ % slot_count];
                slot.wait();
                const std::uint64_t remaining = span.end - source;
                const std::size_t request = static_cast<std::size_t>(std::min<std::uint64_t>(
                    slot_bytes,
                    align_up(remaining, alignment,
                             "artifact direct I/O request overflows u64")));
                auto destination =
                    std::span<std::byte>(static_cast<std::byte*>(slot.buffer.data()), request);
                const std::size_t bytes_read = reader.read_direct(source, destination);
                const std::uint64_t required = std::min<std::uint64_t>(request, remaining);
                if (bytes_read < required) {
                    throw ArtifactError(
                        "direct artifact read ended before the planned tensor range");
                }
                out.stats_.file_bytes = checked_add(out.stats_.file_bytes, bytes_read,
                                                    "artifact read bytes overflow u64");
                const std::uint64_t chunk_end =
                    checked_add(source, bytes_read, "artifact direct I/O result overflows u64");
                while (next_range < ranges.size() && ranges[next_range].source_end <= source) {
                    ++next_range;
                }
                std::size_t range_index = next_range;
                while (range_index < ranges.size() &&
                       ranges[range_index].source_begin < chunk_end) {
                    const CopyRange& range = ranges[range_index];
                    const std::uint64_t copy_begin = std::max(source, range.source_begin);
                    const std::uint64_t copy_end   = std::min(chunk_end, range.source_end);
                    if (copy_begin < copy_end) {
                        const auto amount = static_cast<std::size_t>(copy_end - copy_begin);
                        CUDA_CHECK(cudaMemcpyAsync(
                            range.destination +
                                static_cast<std::size_t>(copy_begin - range.source_begin),
                            static_cast<std::byte*>(slot.buffer.data()) +
                                static_cast<std::size_t>(copy_begin - source),
                            amount, cudaMemcpyHostToDevice, context.load_stream));
                        partition_copied = checked_add(
                            partition_copied, amount,
                            "artifact copied byte count overflows u64");
                    }
                    if (range.source_end <= chunk_end) {
                        ++range_index;
                    } else {
                        break;
                    }
                }
                next_range = range_index;
                CUDA_CHECK(cudaEventRecord(slot.event, context.load_stream));
                slot.pending = true;
                const std::uint64_t published = copied_total + partition_copied;
                if (progress != nullptr && progress->callback && published != last_published &&
                    published < total) {
                    last_published = published;
                    progress->callback("weights", published, total);
                }
            }
        }
        for (const auto& slot : slots) { slot->wait(); }
        CUDA_CHECK(cudaStreamSynchronize(context.load_stream));
        std::uint64_t expected = 0;
        for (const CopyRange& range : ranges) {
            expected = checked_add(expected, range.source_end - range.source_begin,
                                   "artifact tensor byte count overflows u64");
        }
        if (partition_copied != expected || next_range != ranges.size()) {
            throw ArtifactError("direct materialization did not cover every tensor byte");
        }
        copied_total = checked_add(copied_total, partition_copied,
                                   "artifact copied byte count overflows u64");
        if (partition == DevicePartition::Secondary) {
            out.stats_.secondary_h2d_bytes = partition_copied;
        }
    };

    load_partition(DevicePartition::Primary, *out.device_arena_, device);
    if (secondary_device != nullptr) {
        load_partition(DevicePartition::Secondary, *out.secondary_device_arena_,
                       *secondary_device);
    }
    if (copied_total != total) {
        throw ArtifactError("direct materialization did not cover every planned tensor byte");
    }
    out.stats_.h2d_bytes = copied_total;
    out.stats_.upload_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (progress != nullptr && progress->callback) {
        progress->callback("weights", copied_total, total);
    }
    device.activate();
    return out;
}

} // namespace ninfer::artifact
