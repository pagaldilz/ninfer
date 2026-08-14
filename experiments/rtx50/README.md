# RTX 5070 Ti + RTX 5060 Ti feasibility experiment

Status: **GO for a native NInfer dual-device implementation, using endpoint offload.**

This experiment answers two separate questions:

1. Can NInfer's CUDA 13.1 `sm_120a` kernels execute correctly and efficiently on both cards?
   **Yes.**
2. Can the existing Qwen3.8-27B artifact run on either 16 GB card as a normal one-device Engine?
   **No.** The text weights alone leave only 991,232 bytes on the 5070 Ti before CUDA state, KV,
   workspace, and graph allocations.

The recommended design keeps the latency-critical transformer and MTP weights on the 5070 Ti and
places the two large vocabulary matrices, draft head, and Vision weights on the 5060 Ti. It is a
better single-request design than layer splitting because all 64 text layers stay on the card with
twice the measured memory bandwidth. It is also a better fit than tensor parallelism because this
machine has no CUDA peer access between the cards.

This branch contains the reproducible feasibility tools and measurements. It does **not** yet add
multi-device ownership to the public Engine, so the projected NInfer end-to-end rate below is a
target for the implementation, not a completed benchmark claim.

## Tested machine

Measurements were taken on 2026-08-14 under Docker Desktop/WSL2 with CUDA Toolkit 13.1.80 and
NVIDIA driver 610.88.

| CUDA device | Physical VRAM | Compute capability | Sustained cold read |
|---|---:|---:|---:|
| GeForce RTX 5070 Ti | 17,094,475,776 bytes | 12.0 | 852.7 GB/s |
| GeForce RTX 5060 Ti | 17,102,864,384 bytes | 12.0 | 427.2 GB/s |

Both binaries were compiled as native `sm_120a`, rather than generic `sm_120`. NVIDIA documents
`a`-suffix targets as architecture-specific targets; execution on both local devices is the final
compatibility check for this exact setup. See the
[CUDA 13.1 compute-capability documentation](https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/05-appendices/compute-capabilities.html).

`nvidia-smi topo -p2p r`, `w`, and `a` all report `NS` in both directions, and
`cudaDeviceCanAccessPeer` returns zero in both directions. A hidden state must therefore cross via
pinned host memory.

## Native-kernel results

The existing NInfer production Q4 SwiGLU and Q5 residual-projection binaries compiled unchanged.
Their independent FP32-oracle tests passed on each GPU:

```text
ninfer_linear_add_q5_a16_test       passed on 5070 Ti and 5060 Ti
ninfer_linear_swiglu_q4_a16_test    passed on 5070 Ti and 5060 Ti
```

Cold-cache production shapes (`warmup=3`, `repeat=10`):

| Operator | Rows | 5070 Ti | 5060 Ti |
|---|---:|---:|---:|
| Q4 LinearSwiGLU | T=1 | 132.320 us, 716.0 GB/s | 251.424 us, 376.8 GB/s |
| Q4 LinearSwiGLU | T=8 | 169.120 us, 562.1 GB/s | 331.904 us, 286.4 GB/s |
| Q5 LinearAdd, K=17408 | T=1 | 107.456 us, 544.8 GB/s | 174.528 us, 335.5 GB/s |
| Q5 LinearAdd, K=17408 | T=8 | 177.280 us, 332.4 GB/s | 335.488 us, 175.7 GB/s |

The exact Qwen3.8 W8 output head (`N=248320`, `K=5120`) takes 3,283.232 us at T=1 and
3,254.720 us at T=8 on the 5060 Ti. This is the large operation intentionally assigned to the
second card.

The explicit pinned-host boundary probe measured:

| BF16 hidden payload | 5070 Ti to 5060 Ti | 5060 Ti to 5070 Ti |
|---:|---:|---:|
| 10,240 bytes (C=1) | 67.406 us | 74.965 us |
| 81,920 bytes (C=8) | 96.928 us | 100.895 us |
| 491,520 bytes | 231.442 us | 231.507 us |
| 10,485,760 bytes | 3,054.906 us | 3,084.067 us |

The two C=1 crossings add 142.371 us per base round. That is small beside the 3.283 ms output head
and the transformer-layer traversal.

## Exact residency plan

The downloaded
`model-cards/Qwen3.8-27B-NInfer/qwen3_8_27b.ninfer` is 18,210,531,328 bytes. Its tensor
directory produces this placement without estimating from the file size:

| Device | Contents | Weights | Physical headroom |
|---|---|---:|---:|
| 5070 Ti | all 64 text layers, final norm, MTP | 13.824 GiB | 2.097 GiB |
| 5060 Ti | token embedding, output head, draft head, Vision | 3.124 GiB | 12.804 GiB |

The 5070 Ti headroom can hold the execution workspace, fixed state, and a useful INT8 KV cache.
At 33,792 bytes per token for the 27B main INT8 KV layout, the physical upper bound from headroom
alone is about 66,600 tokens; allocator, graph, state, workspace, and the configured 1 GiB automatic
reserve reduce the actual supported capacity. A 16K or 32K first implementation target is
therefore realistic and must be checked by the Engine capacity solver rather than hard-coded.

The measured-component lower bound is:

```text
14,391,762,944 fast-device bytes / 852.7 GB/s
+ 3.283232 ms output head
+ 0.142371 ms two-device boundary
= 20.303 ms, or a 49.25 base-round/s ceiling
```

This omits attention, recurrent state, launch, synchronization, sampling, and other work. A
reasonable implementation target is **40-49 non-speculative tok/s**, versus the measured LM Studio
GGUF median of **28.69 wall tok/s**: approximately **1.4-1.7x** if integration overhead remains
controlled. MTP can commit more than one token per target round, but no MTP throughput number is
claimed before the complete Engine route is running and acceptance is measured.

## GGUF result

The local LM Studio file
`Qwen3.8-27B-Q4_K_M.gguf` is GGUF v3, 17,106,773,984 bytes, and contains 866 tensors. Its stored
payload is approximately:

| GGML tensor type | Stored bytes |
|---|---:|
| Q4_K | 11,370,332,160 |
| Q5_K | 1,038,090,240 |
| Q6_K | 4,526,592,000 |
| Q8_0 | 55,705,600 |
| F32 | 105,058,304 |

These are not NInfer's `Q4G64_F16S`, `Q5G64_F16S`, `Q6G64_F16S`, and `W8G32_F16S`
row-split-k128 layouts. GGUF provides metadata and tensor encodings for GGML-family executors; the
[GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) does not make those
encodings interchangeable with another engine's kernel ABI.

Therefore a generic GGUF hot-path loader is the wrong performance boundary. It would require a
second set of kernels for the GGML layouts or startup repacking with duplicate peak memory. If a
GGUF-facing product is required, the strong design is an **offline, exact-identity GGUF importer**
that writes a native `.ninfer` cache once. For the current local experiment, the downloaded native
artifact is already the appropriate kernel input and avoids a lossy Q4_K_M-to-NInfer
requantization.

## LM Studio control measurement

The GGUF was loaded across both cards at 4,096 context, one parallel request, no speculative draft.
A deterministic 256-token completion was measured after warmup.

| LM Studio GPU strategy | Trial rates (wall tok/s) | Median |
|---|---|---:|
| Priority order | 26.70, 28.69, 29.18 | 28.69 |
| Tensor parallelism | 25.87, 28.63, 29.09 | 28.63 |

Tensor parallelism changed the median by -0.2%, consistent with the absent P2P path. The LM Studio
global strategy was restored to Priority order and the benchmark model was unloaded after the
measurement.

## Implementation cut

The next code cut should be one exact Qwen3.8 endpoint-offload execution profile, not a generic
multi-GPU framework:

1. Add an optional endpoint device to `EngineOptions` and the CLI/server startup options. Omission
   preserves the existing one-device profiles; Qwen3.8 endpoint offload requires two distinct
   devices.
2. Let the target binder assign each tensor to one of two materialization partitions. The generic
   artifact layer owns the two arenas and uploads, while the Qwen3.8 target owns the placement.
3. Keep all text state, KV, layer workspace, and MTP tensors on the primary device. Put the token
   embedding, full output head, draft head, and Vision weights on the endpoint device.
4. Make embedding and scoring compile-time execution leaves of the family schedule. The ordinary,
   prefill, MTP, and Vision schedules call the same leaf contract; the dual Qwen3.8 leaf performs
   explicit pinned-host staging and device/stream selection.
5. Capture device-local CUDA graphs separately. Host orchestration performs the cross-device
   boundary; graph capture must not assume one stream owns pointers from both devices.
6. First qualify C=1 at 16K/32K INT8 KV with greedy MTP0, then MTP3, then C=2..8. Report server
   decode timing and acceptance, not the 49.25 component ceiling.

Layer splitting remains a fallback only if actual runtime allocations exceed the 2.097 GiB primary
headroom. Tensor parallelism is rejected for this topology.

## Reproduction

Build the CUDA 13.1 environment and focused targets:

```powershell
docker build -f experiments/rtx50/Dockerfile -t ninfer-rtx50-dev:cuda13.1 .

docker run --rm --gpus all -v "${PWD}:/workspace" -w /workspace `
  ninfer-rtx50-dev:cuda13.1 cmake -S . -B build-rtx50-linux -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DNINFER_BUILD_APPS=OFF `
  -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=ON

docker run --rm --gpus all -v "${PWD}:/workspace" -w /workspace `
  ninfer-rtx50-dev:cuda13.1 cmake --build build-rtx50-linux --parallel 12 --target `
  ninfer_q4_linear_swiglu_bench ninfer_q5_linear_add_bench ninfer_linear_bench `
  ninfer_linear_swiglu_q4_a16_test ninfer_linear_add_q5_a16_test
```

Compile and run the boundary probe:

```powershell
docker run --rm --gpus all -v "${PWD}:/workspace" -w /workspace `
  ninfer-rtx50-dev:cuda13.1 nvcc -std=c++20 -O3 -arch=sm_120a `
  experiments/rtx50/dual_gpu_transfer_probe.cu `
  -o build-rtx50-linux/dual_gpu_transfer_probe

docker run --rm --gpus all -v "${PWD}:/workspace" -w /workspace `
  ninfer-rtx50-dev:cuda13.1 ./build-rtx50-linux/dual_gpu_transfer_probe
```

Inspect the two local formats and compute placement:

```powershell
python experiments/rtx50/plan_residency.py `
  model-cards/Qwen3.8-27B-NInfer/qwen3_8_27b.ninfer --json

python experiments/rtx50/inspect_gguf.py `
  "$env:USERPROFILE/.lmstudio/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf" `
  --json
```

`benchmark_openai.py` repeats the wall-throughput control against any loaded OpenAI-compatible
local server:

```powershell
python experiments/rtx50/benchmark_openai.py --model qwen38-bench
```
