# Performance measurement

The measurement records accompany the CUDA changes described in the [optimization guide](optimization.md). The reference implementation is [toddballinger/ninfer-5080 v1.5](https://github.com/toddballinger/ninfer-5080/releases/tag/qwen3.8-27b-rtx5080-128k-vision-v1.5), based on [Neroued/ninfer](https://github.com/Neroued/ninfer).

## Fixed configuration

The laptop configurations use the same RTX 5080 Laptop GPU with 16 GB VRAM and 60 SMs, a 175 W power limit and 32 GB DDR5 memory. Software versions are Windows 11, WSL2 Ubuntu 26.04, NVIDIA 592.27, CUDA 13.4.92 and GCC 15.2. The desktop reference uses RTX 5080 16 GB, CUDA 13.4.92 and NVIDIA 615.71.09.

The official `qwen3_8_27b.ninfer` artifact is 16,461,267,456 bytes, SHA-256 `c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21`. Its effective main-model weight precision is 3.952986229 BPW. Both laptop configurations allocate 131,072-token context and KV capacity with Q4 group64 KV.

## 118K sustained decode

The `ninfer_bench` workload uses `bench/fixtures/workflow-118k-v1/ninfer_bench_118001.ids`, SHA-256 `5b08da2c7b7ea5cafad2fab5699dccbcbce86040d8a37219b8c21f094d1d1eb7`.

| Parameter | Value |
|---|---|
| Input / timed decode | 118,001 / 2,048 tokens |
| Warmup / measured repetitions | 1 / 2 |
| Rate aggregation | Arithmetic mean of `2048 / decode_seconds` |
| Prefill chunk | 1792 |
| CUDA Graph / embedding | Enabled / host mapped |
| Baseline | v1.5 with the exact-hit MTP bridge repair; MTP-3; full proposal head |
| Optimized | MTP-4; indexed proposal head |

The first generated token belongs to prefill; the following 2,048 tokens form the decode interval. Model-default stopping is disabled to complete the fixed output length. Prefix reuse is disabled. Decode time includes speculative proposals, target verification, KV operations and state commitment.

Both laptop runs remained on external power with a sampled enforced limit of 175 W. Thermal limiting occurred during sustained execution. Per-run durations, power and temperature summaries are available in [the measurement record](../bench/results/laptop-comparison.json).

```bash
./build/bench/ninfer_bench \
  --weights models/qwen3_8_27b.ninfer \
  --corpus bench/fixtures/workflow-118k-v1/ninfer_bench_118001.ids \
  -pg '118001,2048' --max-ctx 131072 \
  --prefill-chunk 1792 --kv-dtype q4 \
  --mtp-draft-tokens 4 --lm-head-draft --embedding-host \
  --warmup 1 -r 2 -o json --output-file benchmark.json
```

The baseline command selects `--mtp-draft-tokens 3` and the full proposal head by omitting `--lm-head-draft`.

## Near-128K serving workloads

The serving evaluation combines Chinese reasoning, code generation and a four-stage Agent tool workflow. Actual input lengths are 124,920–126,559 tokens, with output space retained inside the full context capacity. The primary metric is:

```text
mixed decode TPS = sum(max(completion_tokens - 1, 0))
                   / sum(server_total_seconds - server_ttft_seconds)
```

The six-request aggregate counts committed output tokens and includes all work after the first token. Initial speculative preparation belongs to TTFT. Each optimized MTP window has one complete serving observation. The baseline combines a preselected reasoning observation with subsequent code and Agent measurements. [Serving results](../bench/results/real-workload-summary.json) contain the measured token counts, durations and configuration.

## Cached Agent latency

A prefix of at least 102,400 tokens is cached before five four-stage Agent trials. All 20 stages verify the actual cache-hit count. Client TTFT ends at the first visible content or tool output and includes tool-stream buffering and cache restoration. P95 is the nearest-rank quantile across these 20 stages.

## Numerical validation

Affected operators retain the original numerical tolerances and are checked against independently decoded FP64 references. Exact codec and transfer changes use integer proofs or complete BF16 comparisons. CUDA Graph replay, guards and state checks cover the relevant paths. Generated-content checks and measurement completeness are recorded separately.

## References

- [toddballinger/ninfer-5080](https://github.com/toddballinger/ninfer-5080): source runtime and desktop release records.
- [Neroued/ninfer](https://github.com/Neroued/ninfer): inference framework.
- [Official model artifact](https://huggingface.co/ninfer-5080/Qwen3.8-27B-RTX5080): model and license information.
