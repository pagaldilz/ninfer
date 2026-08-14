# RTX 5070 Ti + RTX 5060 Ti feasibility experiment

Status: **IMPLEMENTED AND QUALIFIED through the public NInfer Engine and HTTP server.**

This experiment answers two separate questions:

1. Can NInfer's CUDA 13.1 `sm_120a` kernels execute correctly and efficiently on both cards?
   **Yes.**
2. Can the existing Qwen3.8-27B artifact run on either 16 GB card as a normal one-device Engine?
   **No.** The text weights alone leave only 991,232 bytes on the 5070 Ti before CUDA state, KV,
   workspace, and graph allocations.

The implemented design keeps the latency-critical transformer, MTP, Vision, state, KV, and
workspace on the 5070 Ti and places the two large vocabulary matrices plus the optional optimized
draft head on the 5060 Ti. It is a
better single-request design than layer splitting because all 64 text layers stay on the card with
twice the measured memory bandwidth. It is also a better fit than tensor parallelism because this
machine has no CUDA peer access between the cards.

The public `EngineOptions`, CLI, and server expose the profile as `endpoint_device` /
`--endpoint-device`. The complete `.ninfer` artifact is split into two device-owned arenas at load
time, while all runtime crossings use fixed pinned-host buffers. The ordinary one-device profiles
remain unchanged when the option is omitted.

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

One base round transfers two 10,240-byte hidden vectors in opposite directions and returns roughly
497 KB of logits from the endpoint. Using the nearest measured payload gives about 0.38 ms of
host-staged transfer per round. That is still small beside the 3.283 ms output head and the
transformer-layer traversal.

## Measured residency

The downloaded
`model-cards/Qwen3.8-27B-NInfer/qwen3_8_27b.ninfer` is 18,210,531,328 bytes. Its tensor
directory produces this placement without estimating from the file size:

| Profile and device | Contents | Resident weight arena |
|---|---|---:|
| MTP0, 5070 Ti primary | 64 text layers and final norm | 14,391,769,088 bytes (13.404 GiB) |
| MTP0, 5060 Ti endpoint | token embedding and output head | 2,701,721,600 bytes (2.516 GiB) |
| MTP3 optimized, 5070 Ti primary | text, final norm, and MTP | 14,843,560,960 bytes (13.824 GiB) |
| MTP3 optimized, 5060 Ti endpoint | embedding, output head, and draft head | 3,058,761,728 bytes (2.848 GiB) |

With MTP3, one active request, a 2,048-token INT8 KV capacity, and no Vision, the primary retained
293 MiB free after startup. The MTP0 4,096-token profile retained 671 MiB. These are measured Engine
memory ledgers rather than file-size estimates. A larger context or Vision profile must be sized by
the Engine on this tight 16 GB primary; 16K/32K is not supported by the measured MTP3 profile.

The pre-implementation component model was:

```text
14,391,762,944 fast-device bytes / 852.7 GB/s
+ 3.283232 ms output head
+ approximately 0.38 ms two-device boundary
= approximately 20.54 ms, or a 48.7 base-round/s ceiling
```

It correctly identified the endpoint projection as the useful split but, as expected, omitted
attention, recurrent state, launch, synchronization, and sampling costs. The complete route
measured **34.9 non-speculative decode tok/s** on a deterministic 256-token counting request.

## End-to-end generation

All rows below ran through `ninfer-serve` and `POST /v1/chat/completions` after server warm-up with
one request, greedy decoding, no thinking, INT8 KV, and 256 generated tokens. Decode throughput is
the server's `(completion_tokens - 1) / decode_seconds` metric.

| Workload | Profile | Decode | TTFT | Wall | Acceptance | Tokens/round |
|---|---|---:|---:|---:|---:|---:|
| Natural-language photosynthesis prose | MTP0 | 34.9 tok/s | 651 ms | 7.96 s | n/a | 1.00 |
| Natural-language photosynthesis prose | MTP3 optimized | 73.9 tok/s | 545 ms | 4.00 s | 64.9% | 2.93 |
| Counting sequence | MTP3 optimized | 101.3 tok/s | 1,061 ms | 3.58 s | 100.0% | 3.98 |

The prose pair is the useful headline for this setup: MTP3 improves decode by **2.12×** on the same
prompt and produces the same greedy response. The counting row is a valid predictable-output best
case. The non-speculative rate also exceeds the earlier 28.69 wall tok/s LM Studio GGUF control,
but that comparison is directional because the two servers expose different prompt rendering
paths.

## Storage result

The original `H:` model location is a WDC WD140EDFZ 14 TB SATA HDD. Loading the 15.92 GiB MTP0
resident payload from that path took **258.257 s**. The same profile from an SHA-256-verified copy
on the `D:` Micron NVMe loaded in **105.437 s**, a **2.45× startup improvement**. A separate MTP3
load from D: completed in 71.156 s, but its selected tensor ranges and cache state differ, so it is
not used for the disk multiplier. Storage location does not affect steady token generation after
the weights are resident in VRAM.

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

## Implemented execution profile

The change is one exact Qwen3.8 endpoint-offload profile, not a generic multi-GPU framework:

1. Add an optional endpoint device to `EngineOptions` and the CLI/server startup options. Omission
   preserves the existing one-device profiles; Qwen3.8 endpoint offload requires two distinct
   devices.
2. Let the target binder assign each tensor to one of two materialization partitions. The generic
   artifact layer owns the two arenas and uploads, while the Qwen3.8 target owns the placement.
3. Keep all text state, KV, layer workspace, MTP, and Vision on the primary device. Put the token
   embedding, full output head, and optional optimized draft head on the endpoint device.
4. Make embedding and scoring compile-time execution leaves of the family schedule. The ordinary,
   prefill, MTP, and Vision schedules call the same leaf contract; the dual Qwen3.8 leaf performs
   explicit pinned-host staging and device/stream selection.
5. Disable CUDA Graph capture for this explicit host-staged route. Existing one-device profiles
   retain their graph behavior.
6. Qualify C=1 at 4,096-token MTP0 and 2,048-token MTP3 capacities, reporting server decode timing
   and acceptance rather than the component ceiling.

Layer splitting and tensor parallelism remain rejected for this topology.

## Reproduction

Build the CUDA 13.1 environment and focused targets:

```powershell
docker build -f experiments/rtx50/Dockerfile -t ninfer-rtx50-dev:cuda13.1 .

docker run --rm --gpus all -v "${PWD}:/workspace" -w /workspace `
  ninfer-rtx50-dev:cuda13.1 cmake -S . -B build-rtx50-linux -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DNINFER_BUILD_APPS=OFF `
  -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=ON

docker run --rm --gpus all -v "${PWD}:/workspace" -w /workspace `
  ninfer-rtx50-dev:cuda13.1 cmake --build build-rtx50-linux --parallel 4 --target `
  ninfer-serve ninfer_artifact_materialization_test ninfer_serve_options_test
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

Run the measured MTP3 endpoint from the NVMe copy:

```powershell
docker run --rm --gpus all -p 127.0.0.1:8085:8080 `
  -v "${PWD}:/workspace" -v "D:\AiModels\NInfer:/models:ro" `
  -w /workspace ninfer-rtx50-dev:cuda13.1 `
  ./build-rtx50-linux/apps/ninfer-serve /models/qwen3_8_27b.ninfer `
  --host 0.0.0.0 --port 8080 --api-key local-secret --model-id qwen3.8-27b `
  --device 0 --endpoint-device 1 --max-context 2048 --kv-capacity 2048 `
  --kv-dtype int8 --spec mtp --draft-tokens 3 --lm-head-draft `
  --no-thinking --greedy
```
