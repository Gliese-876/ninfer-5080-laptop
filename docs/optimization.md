# RTX 5080 Laptop optimization guide

This fork tunes Qwen3.8-27B inference for a 60-SM RTX 5080 Laptop GPU with 16 GB VRAM. It starts from [toddballinger/ninfer-5080 v1.5](https://github.com/toddballinger/ninfer-5080/releases/tag/qwen3.8-27b-rtx5080-128k-vision-v1.5), commit `e4edd6d5c5f9f7996de0f3d9f6c311e883452580`, and retains the official model artifact, its mixed Q3/Q4/Q5 text-core precision, and full 131,072-token Q4 KV capacity.

The recommended profile uses MTP-4. MTP-3 uses the same implementation, with its draft window changed at launch. The target verification width is normally the draft window plus one, so shared optimizations cover both T4 and T5 where those geometries occur.

## Exact-hit MTP restoration: common functional baseline

The v1.5 `AfterExactHit` path restores a response checkpoint and calls `mtp_bridge_and_propose` to reconstruct multiple drafts. Its original `prefill_hidden.slice(1, i, 1)` accesses column `i≥1` of a persistent one-column BF16 tensor, throwing `slice range out of bounds` and failing the request. The bridge now allocates a separate BF16 `[hidden,1]` temporary buffer for each additional draft hidden state and copies the result back to stable autoregressive storage. The persistent prefill tail retains its actual one-column shape.

This single-line repair establishes the common functional baseline for subsequent long-context, MTP and cache experiments. Both the v1.5 comparison baseline and the optimized build include the same repair. Cold requests, exact checkpoint replay and forced recomputation passed token-sequence and termination-state regression comparisons.

Source: [MTP bridge repair](../src/targets/qwen3_6/impl/runtime/mtp_impl.h).

## Main changes

| Area | Implementation | MTP-3 / MTP-4 coverage |
|---|---|---|
| Exact-hit MTP restoration | Allocate correctly shaped BF16 workspace for additional draft hidden states, establishing the shared functional baseline for long-context, MTP and cache execution. | Both; baseline and optimized build |
| Proposal-head residency | Index the artifact's existing shortlist rows in the full output head, upload the 0.5 MiB ID map and avoid a separate 340 MiB compact-head allocation. Cold binding verifies every selected code and scale row. | Both |
| Indexed T1 projection | Use a qualified Tensor Core path for the canonical indexed single-token proposal head, retaining explicit BF16 logits, local argmax ties and token remapping. | Both |
| Q3 SwiGLU | Stage packed weights asynchronously, add native small-T MMA admission, pad shared packed-code rows and use exact INT3 packing. A composed PRMT decoder is admitted for the measured T4 geometry. | Shared path; T4 specialization benefits MTP-3 |
| Q4 small-T Linear | Admit one-pass MMA for relevant T5–T8 shapes, pad packed-code rows and use native packed signed-Q4 matrix loads for the qualified T5 producer. | Shared path; T5 specialization benefits MTP-4 |
| Q5 projection | Compute exact integer-code BF16 MMA before the original FP16 group scale, with qualified T4/T5/T6 paths for attention and GDN projection. | Both |
| GDN Q4 value/z | Route canonical T4/T5 value/z projection through small-T MMA while preserving the materialized convolution boundary, state and producer/consumer ordering. | Both |
| Long-context Q4 attention | Stage packed K/V asynchronously, load signed-Q4 K fragments with native `ldmatrix`, and use 16-byte packed transfers for qualified T4/T5 cases. | Both |
| Q4 prefill attention | Pipeline QK production and PV consumption with a qualified 384-thread schedule and packed staging. Preserve the established Q8 QK, FP16 probability/value and FP32 accumulation sequence. | Both |
| Dense prefill | Vectorize Q4 shared-memory decoding/stores and parallelize exact INT3 packing for the canonical large-prefill shapes. | Both |
| Tokenization | Cache exact ordinary-run tokenization by raw bytes in a bounded, thread-safe cache, retaining the added-token scanner and token-ID sequence. | Both |
| Runtime consistency and accounting | Align greedy token-count updates with ordinary sampling and include forced-token bridges, synchronization and row updates in complete MTP timing. | Both |

Admission remains specific to the qualified model geometry, token width and numeric policy. Existing paths serve the other geometries.

## Indexed proposal storage and computation

The artifact contains the full Q4 output head `[248320,5120]`, a compact proposal head `[131072,5120]` and its token-ID map. `validate_indexed_proposal_head` compares every compact row's packed code bytes and FP16 scale bytes with the selected full-head row. The compact head is bound for validation, while runtime proposal weights alias the complete resident head with its actual physical shape.

Only the I32 ID vector is materialized separately: **512 KiB**, replacing the need for a **340 MiB duplicate device weight allocation**. The model file, official shortlist, row order, integer codes and scales remain intact. Target verification continues to use the full vocabulary head.

`TextContext::proposal_argmax` calls `linear_gather_rows` to produce BF16 shortlist logits, selects the local argmax and remaps the result to the original token ID. `IndexedRows` maps logical rows to physical weight rows. The final T1 route uses the padded eight-column Tensor Core kernel with one active column; T2–T8 use the same kernel family, with larger extents partitioned into bounded groups.

Sources: [binding and row validation](../src/targets/qwen3_6_27b/impl/load/bindings.cpp), [proposal runtime](../src/targets/qwen3_6/impl/runtime/text_context_impl.h), [indexed projection](../src/ops/linear/q4/q4_gather_rows.cu).

## Q3/Q4 small-T kernels

The canonical Q3 gate/up matrix `[34816,5120]` produces `[17408,T]` SwiGLU output. Gate/up rows share one CTA and the epilogue writes the fused activation. Two shared weight buffers alternate across K tiles, allowing `cp.async` to stage the next tile while Tensor Cores process the current tile. Direct T5–T8 admission uses one eight-column MMA tile and one weight traversal across the active token columns.

Q4 one-pass T5–T8 dispatch covers MLP down `[5120,17408]`, GDN output `[5120,6144]` and the full head `[248320,5120]`. The Q3/Q4 kernels represent signed integer codes exactly in BF16, accumulate G64 dot products in FP32 and then apply the original FP16 scale in FP32.

Both kernel families add 16 bytes of padding to shared packed-code rows while retaining asynchronous-copy alignment:

| Shared layout | Original row stride | Padded row stride |
|---|---:|---:|
| Q3 small-T | 192 bytes | 208 bytes |
| Q4 small-T | 256 bytes | 272 bytes |

The canonical Q3 T4 route also uses composed `prmt.b32` lookups to construct exact BF16 code pairs. T5/T6 retain their qualified decoder. For the three ordinary Q4 Linear shapes at T5, copying threads permute their completed 16-byte chunks before `ldmatrix.sync.aligned.m8n16.x2.shared::cta.s8.s4` loads the signed fragments. This removes consumer fragment-rearrangement work without adding a CTA barrier or shared allocation. Admission selects Tile8/T5/Pad16, identity row mapping and the ordinary store epilogue; indexed rows and strided GDN value/z stores use separately qualified producers.

Sources: [Q3 dispatch](../src/ops/linear_swiglu/q3/q3_linear_swiglu.cu), [Q3 small-T kernel](../src/ops/linear_swiglu/q3/q3_small_t_mma.cuh), [Q4 dispatch](../src/ops/linear/q4/q4_dispatch.cpp), [Q4 small-T kernel](../src/ops/linear/q4/q4_small_t_mma.cuh).

## Q5 and GDN state integration

The Q5 kernel asynchronously stages low-nibble and high-bit planes with padded shared rows. Signed codes `[-16,15]` are exactly representable as BF16 Tensor Core operands. Each raw-code G64 dot product accumulates in FP32 before applying the original FP16 scale. Qualified T4/T5/T6 routes cover the `[7168,5120]` attention projection and `[12288,5120]` GDN value/z parent geometry while preserving their split output strides.

GDN layers using Q4 value/z weights gain T4/T5 admission to the padded small-T MMA kernel. The `[12288,5120]` projection writes directly to the existing value/z views with a 6144-element split. QK retains its existing route. These Q4 paths preserve the materialized BF16 convolution boundary, snapshot/record state publication and producer/consumer ordering. The Q5 T4 route retains its materialized BF16 boundary and Programmatic Dependent Launch schedule; Q5 T5/T6 retain the established fused FP32 projection/convolution epilogues and state publication.

These concrete T4/T5 leaves carry the GDN improvement into MTP-3 and MTP-4. The original output tolerances qualify their floating-point reduction organization.

Sources: [Q5 exact-code MMA](../src/ops/linear/q5/q5_small_t_mma.cuh), [attention projection](../src/ops/attn_input_proj/q4_q5/q4_q5_attn_input_small_t.cu), [GDN projection](../src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_independent.cu), [convolution/snapshot integration](../src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_conv_snapshot.cu).

## Long-context Q4 decode attention

The v1.5 kernel already stages packed K/V asynchronously at T4/group6/Bc32. The new routes extend staging to T5/T6 and add native signed-Q4 K matrix loads. K stays packed in shared memory until `ldmatrix … s8.s4` constructs its exact INT8 MMA fragments. Removing intermediate K expansion reduces the canonical Bc32/D256 shared arena from **32 KiB to 24 KiB per CTA**.

The final 16-byte-copy policy covers Q24/KV4, T4/T5, eight warps, MinBlocks2, Bc32, static arena and single-request execution. One 16-byte transfer replaces two 8-byte copies while preserving the swizzle's low four address bits, page lookup and tail handling. The corresponding long-context T4/T5 schedules serve MTP-3/MTP-4.

The arithmetic retains group64 Q8 queries, exact signed K codes, INT32 QK MMA and FP32 scaling/online softmax. V is decoded into the existing BF16 tile; decode probabilities and PV operands remain BF16 with FP32 accumulation. Split-KV reduction, causal boundaries, stored Q4 bytes and FP16 KV scales remain in place.

Sources: [Q4 decode kernel](../src/ops/kernel/gqa_attention_decode_q4.cuh), [decode schedule selection](../src/ops/launcher/gqa_attention_decode.cu).

## Q4 prompt pipeline and dense prefill

The canonical Q24/KV4 prompt route uses 64-query × 64-key tiles. Packed K/V staging and native K matrix loads reduce shared storage from **92,672 to 76,288 bytes per CTA**. V is decoded directly into the existing FP16 PV tile.

The final production schedule selects **384 threads** and register cap 168: four warps copy the next packed KV tile and produce its QK/softmax, while eight warps consume current P/V. Next-tile probabilities and rescaling factors alpha stay in producer registers until the current PV completes. Named barriers then publish the new values and permit reuse of the shared regions. Folding data loading into the QK producers removes the dedicated-loader role of the earlier 512-thread pipeline.

Prompt arithmetic retains BF16 Q input, the existing group64 Q8 quantizer, INT32 QK MMA, FP32 score/softmax, FP16 probability rounding and V decode, and FP16 PV MMA with FP32 accumulators. Output remains BF16. Both MTP windows share this prompt/suffix-prefill route.

For Q4 MLP down `[5120,17408]` and GDN output `[5120,6144]` at T≥128, each shared 32-bit code load decodes eight signed weights. Four rows per warp distribute accesses across banks, and a 128-bit shared store writes eight BF16 values. The route keeps the original four-warp schedule, asynchronous global staging, FP32 scale multiplication, scaled-weight BF16 rounding and K64 MMA sequence.

Q3 dense prefill uses `q3_codes_to_s8x4` to spread a 12-bit payload into four byte lanes and sign-extend them with integer operations. It preserves the existing INT8 activation quantizer, group64 INT32 MMA, FP32 rescaling and SwiGLU boundary. This exact decoder optimization is shared by both MTP configurations.

Sources: [prompt dispatch](../src/ops/launcher/gqa_attention_prefill.cu), [packed layouts](../src/ops/kernel/gqa_attention_prefill_q4.cuh), [pipeline helpers](../src/ops/kernel/gqa_attention_prefill_q4_pipeline.cuh), [384-thread schedule](../src/ops/kernel/gqa_attention_prefill_q4_pipeline384.cuh), [dense Q4 decode/store](../src/ops/linear/q4/q4_rowsplit_gemm_mma.cuh), [INT3 packing](../src/ops/common/int8_mma.cuh).

## Exact host caching and runtime consistency

The tokenizer caches complete ordinary text runs between added tokens. Lookup checks the hash, byte length and complete raw-byte equality; a hit appends the run's exact cached token sequence. The full added-token scan and rewrite-checkpoint prefix verification retain their original behavior.

The default LRU bounds managed storage at **8 MiB**, with at most 512 entries and 2 MiB per entry; runs shorter than 256 bytes bypass the cache. A mutex protects lookup, promotion, eviction and copying, while normalization and BPE execute outside the lock. Rendered-chat preparation and token counting use the cached encoder. Repeated long segments therefore reuse CPU work while GPU KV reuse handles retained model state.

The common speculative runtime also aligns greedy token-count updates with ordinary sampling semantics. MTP timing is charged after forced-token bridges, their synchronization and row updates, so runtime accounting includes the complete transaction.

Sources: [cache policy](../src/targets/qwen3_6/impl/frontend/tokenizer.h), [cache implementation](../src/targets/qwen3_6/impl/frontend/tokenizer.cpp), [chat preparation](../src/targets/qwen3_6/impl/frontend/processor.cpp), [acceptance](../src/ops/kernel/speculative_round.cuh), [round accounting](../src/targets/qwen3_6/impl/runtime/program_impl.h).

## Individual operator results

These measurements describe complete operators at their stated shapes and widths. Each row names its comparison scope; complete-model throughput is reported in the [README](../README.md#performance).

| Operator and comparison | Case | Before → after | Latency reduction |
|---|---|---:|---:|
| Q3 SwiGLU, native-width two-stage MMA versus original split SIMT/GEMV | T5, `[34816,5120]` | — | 59.9% |
| Same Q3 comparison | T6, `[34816,5120]` | — | 61.5% |
| Q4/Q4 GDN independent projection, QK + value/z | T4, canonical geometry | 128.688 → 89.856 µs | 30.2% |
| Same GDN comparison | T5, canonical geometry | 206.592 → 118.480 µs | 42.6% |
| Dense Q4 GDN output, vector shared decode/store | T936, `[5120,6144]` | 1.024836 → 0.951339 ms | 7.17% |
| Dense Q4 MLP down, vector shared decode/store | T936, `[5120,17408]` | 3.042610 → 2.816381 ms | 7.44% |

## Validation

Affected operators are checked against independent FP64 references at the original tolerances, with relevant token widths, full production shapes, CUDA Graph replay, guards and state checks. Exact codec and row-map changes also use exhaustive integer proofs or complete BF16 comparisons. Shared T4/T5 changes receive coverage for both MTP windows.

The final indexed-row and GDN projection public tests passed for the selected implementation. Long decode attention checks cover complete output comparisons at 102,400 and 131,072 keys. Long prompt qualification combines sampled-query FP64 evaluation over all causal keys with complete BF16 output comparisons. The [validation methodology](benchmark-methodology.md#numerical-validation) describes the numerical criteria.

Generated-content checks assess the answer and tool workflow separately from numerical equivalence. Complete-model performance is measured after operator qualification using the [benchmark contracts](benchmark-methodology.md).

## Source map

- [`src/ops/linear/q4/`](../src/ops/linear/q4/): Q4 small-T and indexed projection.
- [`src/ops/linear/q5/q5_small_t_mma.cuh`](../src/ops/linear/q5/q5_small_t_mma.cuh): exact-code Q5 MMA.
- [`src/ops/linear_swiglu/q3/q3_small_t_mma.cuh`](../src/ops/linear_swiglu/q3/q3_small_t_mma.cuh): Q3 SwiGLU staging and decoding.
- [`src/ops/gdn_input_proj/q4_q5/`](../src/ops/gdn_input_proj/q4_q5/): GDN projection and convolution integration.
- [`src/ops/kernel/gqa_attention_decode_q4.cuh`](../src/ops/kernel/gqa_attention_decode_q4.cuh): Q4 decode attention.
- [`src/ops/kernel/gqa_attention_prefill_q4_pipeline384.cuh`](../src/ops/kernel/gqa_attention_prefill_q4_pipeline384.cuh): pipelined Q4 prefill.
- [`src/targets/qwen3_6/impl/frontend/tokenizer.cpp`](../src/targets/qwen3_6/impl/frontend/tokenizer.cpp): exact tokenization cache.
- [`tests/README.md`](../tests/README.md): public numerical and runtime tests.

The preserved [maintainer references](README.md#maintainer-references) describe the underlying contracts, storage formats and operator-development rules.
