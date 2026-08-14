# Single-GPU serving performance

## Windows RTX 5070 Ti hardware profile

The Windows build has a dedicated single-request profile for an NVIDIA GeForce RTX 5070 Ti
(16 GiB), Ryzen 9 9950X, and 64 GiB system. It uses CUDA 12.8, MSVC 14.42, INT8 group-64 KV,
eager decode. Two artifacts were measured: the published `qwen3_6_35b_a3b.ninfer`
(22,373,184,256 bytes) and the GGUF-sourced compact `.ninfer` profile
(18,514,424,576 bytes, SHA-256
`c989d0e1a6e3c31b2e175361075fa6eca48bfa2a055509d1490066b1896461ec`).

This profile is deliberately different from the 32-GiB RTX 5090 route:

- `--text-only` omits the 1.90-GiB worst-case Vision workspace reservation; Vision requests are
  rejected by that Engine instance;
- three device expert-cache banks retain selected `(layer, expert)` payloads for hosted MoE
  layers and transfer only cache misses during decode;
- cooperative GDN launches use runtime occupancy and fall back to an unsplit grid when the exact
  launch cannot reside on the 5070 Ti;
- Text GDN convolution uses the qualified one-thread-per-channel sequence kernel through 4,096
  tokens; and
- CUDA Graphs remain disabled because the artifact, sequence state, and graph allowance do not fit
  the 16-GiB envelope together.

The local llama.cpp results use the same machine, checkpoint family, one-slot workload, and exact
context allocations. The compact artifact uses native Q3/Q4 expert layouts derived offline from
the exercised 16.96-GiB IQ4_XS GGUF. It is still a `.ninfer` product artifact; GGUF parsing and
I-quant decode are not part of startup or inference.

Cross-engine prompts are token-count matched or near-matched, not byte-identical. The saved
llama.cpp requests used chat-template text and varied synthetic records; `ninfer_bench` used the
committed meaningful-token corpus, deterministically tiled to 240,000 tokens for the two longest
tests. Since prompt content changes MTP acceptance, the table is an operational profile comparison,
not a kernel-identical A/B experiment.

| Reserved context and short request | Compact NInfer | llama.cpp saved result | NInfer change |
|---|---:|---:|---:|
| 32K, 54 prompt + 256 output | 100.09 +/- 1.11 tok/s | 141.96 tok/s | -29.5% |
| 128K, 54 prompt + 256 output | 90.01 +/- 0.77 tok/s | 126.17 tok/s | -28.7% |
| 256K, 54 prompt + 128 output | 79.36 +/- 0.54 tok/s | 78.54 tok/s Max Performance | +1.0% |

The NInfer short-request sequence produced only 47.7% MTP acceptance at 32K and 128K and 58.6% at
256K. These results supersede the earlier one-token-seed comparison for cross-engine claims. With
that older one-token seed, compact NInfer measured 129.77, 118.72, and 101.53 tok/s respectively;
the difference demonstrates that MTP acceptance, not context reservation alone, controls the short
result. On the controlled 54-token sequence, NInfer loses 20.7% from 32K to 256K.

| Filled workload | Compact NInfer prefill | llama.cpp prefill | Compact NInfer generation | llama.cpp generation |
|---|---:|---:|---:|---:|
| 32K: 30,615 / 30,634 prompt tokens | 4,047.63 +/- 0.65 | 1,772.73 | 111.05 +/- 0.37 | 28.73 |
| 128K: 102,642 prompt tokens | 3,036.64 | 1,350.59 | 86.33 | 16.45 |
| 256K: 234,984 prompt tokens | 2,116.30 | 871.89 | 65.94 | 54.11 |

Across the three filled-context points, compact NInfer is 2.28x, 2.25x, and 2.43x faster in
prefill, and 3.87x, 5.25x, and 1.22x faster in generation than the corresponding saved llama.cpp
profiles. Their geometric-mean speedups are 2.32x for prefill and 2.91x for generation. The 128K
NInfer result also exceeds the saved no-MTP Long profile's 62.72 generated tok/s by 37.6%.

The full 262,144-token NInfer allocation fits with INT8-G64 KV: the near-full run reserves 10.51
GiB of device weight capacity and 3.09 GiB of sequence state. llama.cpp uses Q4 KV for its 256K
profiles. The final 32K compact prefill is also 19.1% faster than the published full NInfer profile.

The retained compact prefill kernels decode Q3/Q4/Q6 weights and stage weight and activation MMA
operands as FP16, use Tensor Core contractions with FP32 accumulation, keep the routed activation
and grouped output in FP32, and perform the sole residual write in BF16. The independent Q3+Q4 and
Q3+Q6 operator checks retain a maximum absolute prefill error of `1.953e-3`. Before this route was
selected, a representative 4,096-token Nsight trace attributed 50.8% of GPU time to scalar Q3
gate/up and 31.2% to scalar Q4/Q6 down. The final long-prompt report used one warm-up and two
measured repetitions: `4,047.63 +/- 0.65` prefill tok/s and `111.05 +/- 0.37` generated tok/s.

Build and reproduce both retained Windows profiles from a Developer Command Prompt:

```bat
set NINFER_BUILD_DIR=H:\AiModelLearning\Ninfer\build-win-5070ti-bench
set NINFER_BUILD_BENCHMARKS=ON
tools\windows\configure-5070ti.cmd
tools\windows\benchmark-5070ti.cmd out\qwen3_6_35b_a3b_5070ti_gguf.ninfer
```

These measurements characterize the two registered NInfer targets independently on one NVIDIA
GeForce RTX 5090. They cover long-context prefill and baseline decode with MTP disabled, plus
long-reasoning and cross-scenario decode with MTP enabled.

All requests were submitted serially to a persistent `ninfer-serve` process over the loopback
OpenAI-compatible HTTP endpoint. Each reported fixture used five fixed seeds. Values are arithmetic
mean ± sample standard deviation; warm-up requests are excluded.

## Test method

| Setting | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32 GiB |
| CUDA compile/runtime/driver API | 13.1 / 13.1 / 13.1 |
| Request mode | One active request, `stream=false` |
| Maximum context | 262,144 tokens |
| Prefill chunk | 1,024 tokens |
| KV cache | INT8 group-64 |
| CUDA Graph | Enabled |
| Prefix reuse | Disabled |
| Sampling | Temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.0 |
| MTP0 | `--mtp-draft-tokens 0` |
| MTP3 | `--mtp-draft-tokens 3 --lm-head-draft` |

The MTP0 profile uses four Long NIAH prompts with approximately 8K, 64K, 128K, and 256K tokens.
Thinking is disabled and the output budget is 128 tokens. These runs measure prefill throughput,
server-internal time to first token, and baseline decode throughput at each context length. Content
scenarios are not repeated with MTP disabled because they do not change the baseline decode path.

The MTP3 corpus contains three long-reasoning fixtures with thinking enabled and a 65,536-token
output limit, followed by twelve fixtures covering code, story, translation, and structured output.
The cross-scenario fixtures disable thinking and use a 4,096-token output limit. The tables report
actual completion lengths rather than assuming that every request reaches its limit.

Metrics are computed from the server's unrounded phase timings and MTP counters:

```text
prefill_tok_s = prompt_tokens / prefill_seconds
server_ttft_ms = 1000 * (prepare_seconds + vision_seconds + prefill_seconds)
decode_tok_s = (completion_tokens - 1) / decode_seconds
mtp_acceptance = accepted_tokens / drafted_tokens
mtp_tokens_per_round = 1 + accepted_tokens / mtp_rounds
```

## Reproduction

Build `ninfer-serve`, prepare both registered `.ninfer` artifacts, and run:

```bash
python3 tools/bench/run_serve_corpus.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_35b_a3b=out/qwen3_6_35b_a3b.ninfer \
  --artifact qwen3_6_27b=out/qwen3_6_27b.ninfer \
  --output profiles/bench/serve_corpus_20260720
```

## `qwen3_6_35b_a3b`

### MTP0 context-length profile

| Prompt tokens | Samples | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 15,544.3 ± 242.4 | 500.2 ± 7.8 | 271.1 ± 3.6 |
| 64,512 | 5 | 10,809.0 ± 95.3 | 6,009.9 ± 52.6 | 242.9 ± 1.3 |
| 130,048 | 5 | 7,828.4 ± 34.1 | 16,693.3 ± 71.2 | 219.4 ± 1.6 |
| 260,096 | 5 | 5,157.1 ± 52.4 | 50,598.8 ± 519.7 | 188.2 ± 2.1 |

### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 8,675.4 ± 1,565.6 | 634.3 ± 14.2 | 82.7% ± 2.6% | 3.48 ± 0.08 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 542.8 ± 12.5 | 73.0% ± 2.5% | 3.19 ± 0.07 |
| `long_decode_aime26_30` | 5 | 55,171.0 ± 5,407.1 | 572.9 ± 9.1 | 77.7% ± 1.4% | 3.33 ± 0.04 |

### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 576.5 ± 21.7 | 71.0% ± 4.0% | 3.13 ± 0.12 |
| Story | 15 | 395.9 ± 30.9 | 37.7% ± 5.8% | 2.13 ± 0.17 |
| Translation | 15 | 559.3 ± 28.1 | 66.6% ± 5.1% | 3.00 ± 0.15 |
| Structured | 15 | 661.2 ± 29.5 | 87.2% ± 6.0% | 3.62 ± 0.18 |

## `qwen3_6_27b`

### MTP0 context-length profile

| Prompt tokens | Samples | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 3,218.1 ± 4.3 | 2,392.4 ± 3.0 | 77.6 ± 0.1 |
| 64,512 | 5 | 2,655.9 ± 2.9 | 24,335.7 ± 25.2 | 70.7 ± 0.1 |
| 130,048 | 5 | 2,185.3 ± 0.3 | 59,590.3 ± 8.9 | 64.5 ± 0.1 |
| 260,096 | 5 | 1,614.8 ± 0.6 | 161,221.8 ± 62.5 | 54.8 ± 0.1 |

### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 11,009.4 ± 419.1 | 174.2 ± 3.3 | 79.9% ± 2.0% | 3.40 ± 0.06 |
| `long_decode_aime26_15` | 5 | 62,652.6 ± 3,000.4 | 158.7 ± 5.2 | 73.3% ± 3.4% | 3.20 ± 0.10 |
| `long_decode_aime26_30` | 5 | 47,837.8 ± 5,882.7 | 169.0 ± 2.7 | 79.3% ± 2.0% | 3.38 ± 0.06 |

### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 163.9 ± 6.2 | 72.5% ± 3.9% | 3.18 ± 0.12 |
| Story | 15 | 110.4 ± 9.2 | 37.9% ± 6.0% | 2.14 ± 0.18 |
| Translation | 15 | 153.6 ± 11.7 | 65.7% ± 7.5% | 2.97 ± 0.23 |
| Structured | 15 | 189.1 ± 15.7 | 88.9% ± 10.2% | 3.67 ± 0.31 |

The MTP0 and MTP3 suites intentionally measure different supported workloads. No per-scenario
MTP0/MTP3 speedup is reported.
