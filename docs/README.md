# Documentation

Start with the [English README](../README.md) or [简体中文 README](../README.zh-CN.md) for the RTX 5080 Laptop build, official model and recommended MTP-4 text-serving profile.

## Current laptop fork

| Guide | Purpose |
|---|---|
| [Benchmark methodology](benchmark-methodology.md) | 118K measurement matching [toddballinger/ninfer-5080](https://github.com/toddballinger/ninfer-5080), near-128K real serving workloads and cached Agent latency |
| [Optimization guide](optimization.md) | CUDA, proposal-head, KV and cache changes; shared MTP-3/MTP-4 coverage |
| [HTTP serving](serving.md) | APIs, streaming, tools, state, authentication and request options |
| [CLI](cli.md) | Text, chat history, sampling and MTP options |
| [Constrained decisions](CONSTRAINED_DECISIONS_USAGE.md) | C++ finite-decision API and supported behavior |
| [Benchmarks](../bench/README.md) | Public Engine and operator benchmark tools |
| [Tests](../tests/README.md) | Numerical and runtime test tools |

The executable `--help` output provides the exact CLI option spelling. The root README defines this fork's recommended profile.

## Release records from [toddballinger/ninfer-5080](https://github.com/toddballinger/ninfer-5080)

These records describe the desktop RTX 5080 branch and earlier releases. Their measured values and historical runtime settings remain attributable to those releases.

| Record | Scope |
|---|---|
| [v1.5 release](RELEASE_QWEN3.8_27B_RTX5080_V1.5.md) | Desktop RTX 5080 release used as this fork's starting point |
| [v1.4](RELEASE_QWEN3.8_27B_RTX5080_V1.4.md), [v1.3](RELEASE_QWEN3.8_27B_RTX5080_V1.3.md), [v1.2](RELEASE_QWEN3.8_27B_RTX5080_V1.2.md) | Earlier desktop releases |
| [Benchmark history](BENCHMARKS.md) | Published desktop results and historical regression data |
| [Validated manifest](VALIDATED_MANIFEST.md) | Source, binary and artifact identities |
| [Vision 128K](VISION_128K.md) | Multimodal profiles and validation |
| [Memory profile](MEMORY_PROFILE.md) | Weight, KV and Vision memory records |
| [Reproducibility history](REPRODUCIBILITY.md) | Historical desktop build and launch settings |
| [Integration history](UPSTREAM_SYNC_STATUS.md) | Desktop branch's selective-port ledger |
| [Technical deep dive](TECHNICAL_DEEP_DIVE.md) | Architecture and optimization explanation |

The official artifact is [ninfer-5080/Qwen3.8-27B-RTX5080](https://huggingface.co/ninfer-5080/Qwen3.8-27B-RTX5080). Its SHA-256 remains `c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21`.

## Maintainer references

The files in [`maintainer/`](maintainer/) describe architecture and implementation contracts:

- [Operator development](maintainer/op-development.md), [Linear benchmarks](maintainer/linear-benchmark.md) and [concurrent inference](maintainer/concurrent-inference-architecture.md).
- [Paged KV](maintainer/paged-kv-cache.md), [embedding residency](maintainer/embedding-host-offload.md) and [ReplaySSM GDN](maintainer/replayssm-gdn.md).
- [Artifact container](maintainer/artifact-container.md), [numeric formats](maintainer/tensor-formats.md) and [storage layouts](maintainer/storage-layouts.md).
- [Qwen3.8-27B artifact contract](maintainer/qwen3.8-27b-artifact.md), with its inherited [Qwen model semantics](maintainer/qwen3.6-27b-model.md).
- [Qwen3.6-27B](maintainer/qwen3.6-27b-model.md), [Qwen3.6-35B-A3B](maintainer/qwen3.6-35b-a3b-model.md) and the other retained model references.

[Softmax Attention](maintainer/softmax-attention.md) records a planned migration. Current dispatch and runtime behavior are defined by the source and public API.

## Additional material from [toddballinger/ninfer-5080](https://github.com/toddballinger/ninfer-5080)

The repository retains [CLI examples](../examples/cli/), [maintainer tools](../tools/README.md), [capability evaluation](../eval/README.md) and [model cards](../model-cards/). Their licenses and source attribution remain with the corresponding material.
