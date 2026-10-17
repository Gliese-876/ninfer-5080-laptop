# Quickstart

Build and run the text-serving configuration for Qwen3.8-27B on an RTX 5080 Laptop GPU with 16 GB VRAM. Execute the commands in an Ubuntu terminal under WSL2, from the repository root.

## Requirements

The validated environment uses Windows 11, WSL2 Ubuntu 26.04, NVIDIA driver 592.27, CUDA 13.4.92 and GCC 15.2. The build requires CUDA 13.4 or newer, CMake 3.28 or newer, a C++20 compiler, Python 3.11 or newer, FFmpeg development libraries and libcurl. Its CUDA architecture is `sm_120a`.

Install the NVIDIA driver in Windows and the CUDA toolkit-only package in Ubuntu following the [NVIDIA CUDA on WSL installation guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html). The commands below select the toolkit at `/usr/local/cuda-13.4`; adjust that directory to match the installed toolkit. Store the checkout and model in the Linux filesystem for WSL2 inference.

## Install build dependencies

```bash
sudo apt-get update
sudo apt-get install build-essential cmake ninja-build pkg-config python3 \
  libavformat-dev libavcodec-dev libavutil-dev libswscale-dev libcurl4-openssl-dev
export CUDACXX=/usr/local/cuda-13.4/bin/nvcc
"$CUDACXX" --version
```

## Build the server and benchmark

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120a \
  -DBUILD_TESTING=OFF \
  -DNINFER_BUILD_BENCHMARKS=ON
cmake --build build --target ninfer-serve ninfer_bench -j
```

For the numerical and runtime test targets, configure with `-DBUILD_TESTING=ON` and follow the [test guide](../tests/README.md).

## Download and verify the model

The [model manifest](../configs/model_manifest.json) pins the official [Qwen3.8-27B-RTX5080 artifact](https://huggingface.co/ninfer-5080/Qwen3.8-27B-RTX5080):

| Field | Value |
|---|---|
| Filename | `qwen3_8_27b.ninfer` |
| File size | 16,461,267,456 bytes |
| SHA-256 | `c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21` |
| Model revision | `e23a3301eed611ac3196e3389c3d572bc5ddeb9a` |
| Main-model precision | 3.952986229 BPW, rounded to 3.953 |

Download with 16 parallel byte ranges:

```bash
python3 scripts/download_model.py --workers 16
```

The downloader writes `models/qwen3_8_27b.ninfer`, resumes completed parts and verifies the complete file against the manifest. Resume an interrupted transfer with the same command.

To perform another full SHA-256 verification:

```bash
python3 scripts/verify_model.py
```

## Start MTP-4

```bash
python3 scripts/launch_ninfer.py \
  --profile configs/laptop-128k-mtp4.json \
  --cuda-dir /usr/local/cuda-13.4
```

The launcher selects `build/apps/ninfer-serve`, validates the model and CUDA runtime, and saves the launch command, binary identity and request logs under `logs/`.

| Profile setting | Value |
|---|---|
| Model ID | `qwen3.8-27b` |
| Context / KV capacity | 131,072 / 131,072 |
| KV format | Q4 group64 |
| Prefill chunk | 1792 |
| MTP draft window | 4 |
| Proposal head | Indexed |
| CUDA Graph / host-mapped embeddings | Enabled / enabled |
| Request concurrency | 1 |
| Prefix checkpoint policy | `rolling-tool` |

Requests use the model's native thinking behavior unless the client selects a thinking policy. The configuration serves text workloads with Vision allocation disabled.

In another Ubuntu terminal, check the API:

```bash
curl --fail --silent http://127.0.0.1:8080/v1/models
```

The [serving guide](serving.md) documents OpenAI-compatible requests, streaming, tools and authentication.

## Select MTP-3 or an alternative model location

MTP-3 uses the same optimized binary:

```bash
python3 scripts/launch_ninfer.py \
  --profile configs/laptop-128k-mtp3.json \
  --cuda-dir /usr/local/cuda-13.4
```

For an artifact stored elsewhere, pass its location with `--model`. The launcher accepts `--port` for an alternative listening port and `--dry-run` to print the resolved command.

Performance reproduction is documented in the [benchmark methodology](benchmark-methodology.md); implementation details are in the [optimization guide](optimization.md).
