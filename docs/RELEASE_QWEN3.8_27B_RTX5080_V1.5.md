> Record from [toddballinger/ninfer-5080](https://github.com/toddballinger/ninfer-5080). The [project README](../README.md) describes the laptop implementation.

# Qwen3.8-27B RTX 5080 128K + Vision v1.5

v1.5 is the current production-qualified RTX 5080 runtime for Qwen3.8-27B. It moves the project to **CUDA 13.4.92 / NVIDIA 615.71.09**, retunes the Q3/A8 large-prefill SwiGLU path for Blackwell/CC 12.0, and standardizes release performance reporting on `ninfer_bench`.

## Headline result

Canonical `pp118001+tg2048` benchmark on a single RTX 5080 16 GB:

| Metric | v1.5 |
|---|---:|
| Prefill | **1,374.383 tok/s** |
| Sustained decode | **112.215 tok/s** |
| Improvement | **+4.46% prefill / +2.32% decode** |
| Context / KV capacity | **131,072 / 131,072** |
| KV dtype | **Q4 group64** |
| Prefill chunk | **1792** |
| Speculation | **MTP-3** |
| CUDA Graph | **enabled** |
| Host-mapped embeddings | **enabled** |
| Vision profile | **2048 tokens** |

The model artifact is unchanged:

```text
MODEL_SHA256=c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
```

Full benchmark methodology, run-level results, memory measurements and compiler evidence are in [BENCHMARKS.md](BENCHMARKS.md).

## Canonical benchmark contract

`ninfer_bench` is the source of truth for whole-model release performance.

```text
fixture=bench/fixtures/workflow-118k-v1/ninfer_bench_118001.ids
tokens=118001
corpus_sha256=5b08da2c7b7ea5cafad2fab5699dccbcbce86040d8a37219b8c21f094d1d1eb7
test=pp118001+tg2048
warmup=1
measured_repetitions=2
max_context=131072
kv_capacity=131072
kv_dtype=q4-group64
mtp_draft_tokens=3
cuda_graph=on
embedding_host=on
```

Other context sweeps, microbenchmarks, Nsight captures and serving probes remain diagnostic tools rather than competing release benchmarks.

## Blackwell Q3/A8 retune

The large-prefill schedule is retuned from:

```cpp
Q3Int8SwiGluSchedule<64, 256, 16, 128, 3, 1>
```

to:

```cpp
Q3Int8SwiGluSchedule<64, 128, 16, 64, 3, 1>
```

The change removes the CUDA 13.4 spill behavior observed in the previous large-prefill route. Decode and small-T kernels are unchanged.

## Recommended production profile

```bash
./build/apps/ninfer-serve /path/to/qwen3_8_27b.ninfer \
  --host 0.0.0.0 \
  --port 8080 \
  --model-id local-model \
  --max-context 131072 \
  --kv-capacity 131072 \
  --prefill-chunk 1792 \
  --kv-dtype q4 \
  --spec mtp \
  --draft-tokens 3 \
  --max-concurrency 1 \
  --max-pending-requests 16 \
  --pending-timeout-ms 180000 \
  --embedding-host \
  --vision \
  --vision-max-tokens 2048 \
  --default-thinking-budget 2048 \
  --prefix-checkpoint-policy rolling-tool
```

CUDA Graph remains enabled by default.

## Production validation

Validated on:

```text
GPU=NVIDIA GeForce RTX 5080 16 GB
CUDA=13.4.92
DRIVER=615.71.09
SOURCE=e4353f061bf0e378c83472cb2bcaf99e65681f4e
NINFER_SERVE_SHA256=928e5615ef453786f47f79b6af2152d2f8f8d61307656f23c47fa45b5ed41167
```

Final Brain deployment passed:

- service enabled and active;
- `/v1/models` HTTP 200;
- text generation smoke returned `OK`;
- full 131,072 Q4 KV capacity;
- MTP-3;
- CUDA Graph enabled;
- host-mapped embeddings;
- Vision 2048;
- **885.94 MiB free after startup**;
- **806.92 MiB planned slack**.

**V1.5 PRODUCTION RELEASE VALIDATED.**
