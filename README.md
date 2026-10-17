# NInfer RTX 5080 Laptop

[简体中文](README.zh-CN.md) · [Quickstart](docs/quickstart.md) · [Documentation](docs/README.md)

CUDA inference optimizations for **Qwen3.8-27B on RTX 5080 Laptop 16 GB**. This fork extends [toddballinger/ninfer-5080](https://github.com/toddballinger/ninfer-5080) v1.5, built on [Neroued/ninfer](https://github.com/Neroued/ninfer).

## Fixed configuration

The laptop comparisons retain the **same RTX 5080 Laptop 16 GB GPU (60 SMs) with a 175 W power limit**, the **original 3.953 BPW model weights**, **131,072-token context and KV capacity**, and **Q4 group64 KV**. CUDA Graph, host-mapped embeddings and single-request execution remain enabled in both configurations. Each workload uses the same corpus and environment: 32 GB DDR5, Windows 11, WSL2 Ubuntu 26.04, CUDA 13.4.92 and NVIDIA driver 592.27.

The optimizations retain the artifact's Q3/Q4/Q5 integer codes and FP16 scales, activation and KV formats, and persistent-state formats. Operators are validated at the original numerical tolerances.

## Inference optimizations

### v1.5 MTP checkpoint-restoration bounds fix

This repair establishes the common functional baseline for subsequent long-context, MTP and cache-reuse experiments. In [toddballinger/ninfer-5080 v1.5](https://github.com/toddballinger/ninfer-5080/releases/tag/qwen3.8-27b-rtx5080-128k-vision-v1.5), a full prefix hit followed by response-checkpoint restoration and reconstruction of multiple MTP drafts reaches `mtp_bridge_and_propose`. That function uses the persistent one-column `prefill_hidden` tensor as a multi-column temporary. Starting with the second draft, the column slice throws `slice range out of bounds` and the inference request fails.

The fix allocates a separate BF16 `[hidden,1]` tensor from the GPU workspace (temporary device memory) for each additional draft step, then copies the result into the stable autoregressive hidden state while retaining the persistent prefill tail. Repeated long prompts can restore a full prefix hit and continue MTP inference. Regression comparisons of cold requests, exact checkpoint replay and forced recomputation preserve token sequences and termination state. Both MTP-3 and MTP-4 use the repair, and it is included in the comparison baseline and optimized build. See the [MTP bridge fix](src/targets/qwen3_6/impl/runtime/mtp_impl.h).

### MTP weight sharing and draft window

The artifact provides a 248,320-row full output head and a 131,072-entry proposal shortlist. At load time, the loader compares every proposal row's packed Q4 codes and FP16 scales with the corresponding full-head row. At runtime, indexed reads reuse the full head while preserving the official shortlist and its order. Only the **512 KiB token-ID map** requires additional GPU storage, saving **340 MiB of VRAM** compared with a separately copied compact proposal head.

An indexed Tensor Core projection retains BF16 logits, shortlist argmax and remapping to original token IDs. The recommended configuration increases the draft window from three to four: **MTP-4** proposes four tokens per round and normally verifies them with the target model at width T5; MTP-3 normally verifies at T4. Both modes share model storage and runtime, with kernels selected for their verification widths. See [weight binding](src/targets/qwen3_6_27b/impl/load/bindings.cpp) and [indexed projection](src/ops/linear/q4/q4_gather_rows.cu).

### Q3 SwiGLU: double buffering and one-pass verification

The Q3 gate/up projection alternates between two shared-memory weight buffers. `cp.async` overlaps the next K tile's transfer with the current tile's computation. T5–T8 verification batches now use one eight-column MMA tile, sharing a single weight traversal across all active token columns and producing SwiGLU output directly in the epilogue.

Packed shared-memory rows gain 16 bytes of padding to improve bank distribution. The specialized T4 route uses exact PRMT lookups to construct BF16 code pairs. For dense prefill, integer bit operations pack INT3 codes into INT8 lanes while preserving the existing activation quantization and MMA route. See the [Q3 small-T kernel](src/ops/linear_swiglu/q3/q3_small_t_mma.cuh) and [integer packing](src/ops/common/int8_mma.cuh).

### Q4 Linear: native packed matrix loads

MLP down, GDN output projection and the full vocabulary head gain one-pass T5–T8 MMA dispatch, reusing weights across token columns. Their packed shared-memory rows also receive 16-byte padding to reduce bank conflicts between rows.

For these three ordinary Linear shapes at T5, producer threads permute their completed packed transfers before Blackwell's native `ldmatrix … s8.s4` loads signed Q4 codes. This reduces fragment rearrangement and warp shuffle work while retaining the integer values, FP32 group accumulation and existing scales. See [Q4 dispatch](src/ops/linear/q4/q4_dispatch.cpp) and the [small-T MMA kernel](src/ops/linear/q4/q4_small_t_mma.cuh).

### Q5 and GDN: Tensor Core verification paths

Small attention and GDN Q5 projections gain T4–T6 Tensor Core paths. The kernel asynchronously stages the low-nibble and high-bit planes, represents signed Q5 integers exactly in BF16, accumulates each group dot product in FP32 and then applies the original FP16 scale. This reduces scalar decoding and accumulation work while retaining the established output layouts and convolution-state boundaries.

GDN layers with Q4 value/z weights also move that projection from SIMT to T4/T5 MMA, writing directly to the existing strided output views. QK projection, the materialized BF16 convolution boundary, and state snapshot/restore ordering remain intact. T4 and T5 cover MTP-3 and MTP-4, respectively. See the [Q5 kernel](src/ops/linear/q5/q5_small_t_mma.cuh) and [GDN integration](src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_independent.cu).

### Long-context decode: compute from packed KV

Q4 attention extends asynchronous packed K/V staging to T5/T6. Native signed-Q4 matrix loads construct K's INT8 MMA fragments directly, reducing intermediate expansion in shared memory. Consumers decode V into the established BF16 PV tile, retaining the QK, softmax and PV precision policies.

For single-request T4/T5 long-context routes, one 16-byte packed transfer replaces two 8-byte copies while preserving address swizzling, page mapping and causal boundaries. These changes serve both MTP-3 and MTP-4 with the full Q4 group64 KV capacity. See [Q4 decode attention](src/ops/kernel/gqa_attention_decode_q4.cuh).

### Prefill: QK/PV pipelining and vectorized decoding

Q4 prefill attention uses a **384-thread** schedule: four warps produce QK and move data, while eight warps consume probabilities and values. Producers hold the next tile's probabilities and rescaling factors alpha in registers until current PV consumption finishes, then publish them to shared memory. This overlaps QK production with PV consumption while retaining the established Q8 QK, FP16 probability/value and FP32 accumulation path.

Dense MLP down and GDN output projections load and decode eight Q4 codes at a time, then use a 128-bit shared store in place of pairwise stores. Q3 prefill also uses exact integer packing. These shared paths accelerate long-prompt and tool-result prefill for both MTP configurations. See the [384-thread pipeline](src/ops/kernel/gqa_attention_prefill_q4_pipeline384.cuh) and [dense Q4 projection](src/ops/linear/q4/q4_rowsplit_gemm_mma.cuh).

### Exact tokenization caching and runtime accounting

The tokenizer adds an **8 MiB bounded LRU cache** keyed by the complete raw bytes of ordinary text runs. It reuses their exact token sequences while preserving special-token scanning, BPE boundaries and token order. Repeated long segments in Agent conversations avoid repeated CPU tokenization; this host cache complements GPU KV prefix reuse. See the [tokenization cache](src/targets/qwen3_6/impl/frontend/tokenizer.cpp).

The shared runtime also aligns greedy token-count updates and charges forced-token bridges and state commitment to the complete MTP round. The [optimization guide](docs/optimization.md) details shapes, instructions, MTP coverage and operator results.

## Performance

![Qwen3.8-27B 118K decode throughput](assets/throughput-comparison.png)

| Laptop workload | v1.5 baseline | This fork · MTP-4 | Increase |
|---|---:|---:|---:|
| 118K sustained decode | 67.845 tok/s | **94.295 tok/s** | **+39.0%** |
| Near-128K Chinese reasoning, code and Agent mix | 60.2901 tok/s | **85.7366 tok/s** | **+42.2%** |

The same implementation achieves **81.4659 tok/s with MTP-3** on the mixed serving workload. The default MTP-4 configuration has an observed cached-Agent TTFT P95 of **1.729 seconds** with a 100K prefix across 20 stages.

The **RTX 5080 Desktop** reference in the figure is **112.215 tok/s**, published by [toddballinger/ninfer-5080 v1.5](https://github.com/toddballinger/ninfer-5080/releases/tag/qwen3.8-27b-rtx5080-128k-vision-v1.5). See the [benchmark methodology](docs/benchmark-methodology.md) for measurement definitions and numerical validation, and the [sustained decode](bench/results/laptop-comparison.json) and [serving](bench/results/real-workload-summary.json) records for measured results.

## Quickstart

Follow the [quickstart](docs/quickstart.md) to install dependencies, build and download the model, then start the recommended profile:

```bash
python3 scripts/launch_ninfer.py --profile configs/laptop-128k-mtp4.json
```

The OpenAI-compatible API listens at `http://127.0.0.1:8080`. Select [`configs/laptop-128k-mtp3.json`](configs/laptop-128k-mtp3.json) for MTP-3. See the [optimization guide](docs/optimization.md) for implementation details and the [serving guide](docs/serving.md) for API options.

## License

Source code is released under **[Apache License 2.0](LICENSE)**, with existing copyright notices and dependency licenses preserved. The [Qwen3.8-27B-RTX5080 model](https://huggingface.co/ninfer-5080/Qwen3.8-27B-RTX5080) is distributed under its model repository's license.
