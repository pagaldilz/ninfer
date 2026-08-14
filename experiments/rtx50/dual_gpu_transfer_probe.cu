// Measures the explicit pinned-host boundary required when two CUDA devices lack P2P access.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define CUDA_CHECK(expr)                                                                       \
    do {                                                                                       \
        const cudaError_t error_ = (expr);                                                     \
        if (error_ != cudaSuccess) {                                                           \
            throw std::runtime_error(std::string(#expr) + ": " + cudaGetErrorString(error_)); \
        }                                                                                      \
    } while (false)

constexpr std::size_t kHidden = 5120;

struct DeviceAllocation {
    int device = 0;
    void* pointer = nullptr;

    DeviceAllocation(int device_index, std::size_t bytes) : device(device_index) {
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaMalloc(&pointer, bytes));
        CUDA_CHECK(cudaMemset(pointer, 0x5a, bytes));
    }

    ~DeviceAllocation() {
        if (pointer != nullptr) {
            cudaSetDevice(device);
            cudaFree(pointer);
        }
    }

    DeviceAllocation(const DeviceAllocation&)            = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;
};

struct DeviceStream {
    int device = 0;
    cudaStream_t stream = nullptr;

    explicit DeviceStream(int device_index) : device(device_index) {
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }

    ~DeviceStream() {
        if (stream != nullptr) {
            cudaSetDevice(device);
            cudaStreamDestroy(stream);
        }
    }

    DeviceStream(const DeviceStream&)            = delete;
    DeviceStream& operator=(const DeviceStream&) = delete;
};

void staged_copy(int source_device, const void* source, cudaStream_t source_stream,
                 int destination_device, void* destination, cudaStream_t destination_stream,
                 void* host, std::size_t bytes) {
    CUDA_CHECK(cudaSetDevice(source_device));
    CUDA_CHECK(cudaMemcpyAsync(host, source, bytes, cudaMemcpyDeviceToHost, source_stream));
    CUDA_CHECK(cudaStreamSynchronize(source_stream));
    CUDA_CHECK(cudaSetDevice(destination_device));
    CUDA_CHECK(
        cudaMemcpyAsync(destination, host, bytes, cudaMemcpyHostToDevice, destination_stream));
    CUDA_CHECK(cudaStreamSynchronize(destination_stream));
}

double median_us(std::vector<double>& samples) {
    std::sort(samples.begin(), samples.end());
    const std::size_t middle = samples.size() / 2;
    if (samples.size() % 2 != 0) { return samples[middle]; }
    return 0.5 * (samples[middle - 1] + samples[middle]);
}

void measure_direction(int source_device, const DeviceAllocation& source,
                       const DeviceStream& source_stream, int destination_device,
                       const DeviceAllocation& destination,
                       const DeviceStream& destination_stream, void* host, std::size_t bytes,
                       int maximum_iterations) {
    const std::size_t target_bytes = 1ULL << 30;
    const int iterations = std::max(
        20, std::min(maximum_iterations,
                     static_cast<int>(target_bytes / std::max<std::size_t>(bytes, 1))));
    const int warmup = std::min(20, iterations);

    for (int iteration = 0; iteration < warmup; ++iteration) {
        staged_copy(source_device, source.pointer, source_stream.stream, destination_device,
                    destination.pointer, destination_stream.stream, host, bytes);
    }

    std::vector<double> samples;
    samples.reserve(iterations);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const auto begin = std::chrono::steady_clock::now();
        staged_copy(source_device, source.pointer, source_stream.stream, destination_device,
                    destination.pointer, destination_stream.stream, host, bytes);
        const auto end = std::chrono::steady_clock::now();
        samples.push_back(
            std::chrono::duration<double, std::micro>(end - begin).count());
    }

    const double latency = median_us(samples);
    const double gib_s = (2.0 * static_cast<double>(bytes)) / (latency * 1.0e-6) /
                         static_cast<double>(1ULL << 30);
    std::printf("staged,%d,%d,%zu,%d,%.3f,%.3f\n", source_device, destination_device, bytes,
                iterations, latency, gib_s);
}

} // namespace

int main(int argc, char** argv) {
    try {
        int maximum_iterations = 2000;
        if (argc == 3 && std::string(argv[1]) == "--max-iterations") {
            maximum_iterations = std::stoi(argv[2]);
        } else if (argc != 1) {
            std::fprintf(stderr, "usage: %s [--max-iterations N]\n", argv[0]);
            return 2;
        }
        if (maximum_iterations < 20) {
            throw std::invalid_argument("--max-iterations must be at least 20");
        }

        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count != 2) {
            throw std::runtime_error("probe requires exactly two visible CUDA devices");
        }

        for (int device = 0; device < device_count; ++device) {
            cudaDeviceProp properties{};
            CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
            std::printf("device,%d,%s,%d.%d,%zu\n", device, properties.name, properties.major,
                        properties.minor, properties.totalGlobalMem);
        }
        for (int source = 0; source < device_count; ++source) {
            const int destination = 1 - source;
            int can_access = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, source, destination));
            std::printf("peer_access,%d,%d,%d\n", source, destination, can_access);
        }

        const std::vector<std::size_t> sizes{
            kHidden * sizeof(std::uint16_t),       // one BF16 decode row
            8 * kHidden * sizeof(std::uint16_t),   // maximum compact decode batch
            48 * kHidden * sizeof(std::uint16_t),  // representative speculative/prefill tile
            1024 * kHidden * sizeof(std::uint16_t) // one prefill chunk
        };
        const std::size_t maximum_bytes = *std::max_element(sizes.begin(), sizes.end());

        DeviceAllocation device0(0, maximum_bytes);
        DeviceAllocation device1(1, maximum_bytes);
        DeviceStream stream0(0);
        DeviceStream stream1(1);
        void* host = nullptr;
        CUDA_CHECK(cudaMallocHost(&host, maximum_bytes));

        std::printf("route,source,destination,bytes,iterations,median_us,effective_gib_s\n");
        for (const std::size_t bytes : sizes) {
            measure_direction(0, device0, stream0, 1, device1, stream1, host, bytes,
                              maximum_iterations);
            measure_direction(1, device1, stream1, 0, device0, stream0, host, bytes,
                              maximum_iterations);
        }

        CUDA_CHECK(cudaFreeHost(host));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "dual_gpu_transfer_probe: %s\n", error.what());
        return 1;
    }
}
