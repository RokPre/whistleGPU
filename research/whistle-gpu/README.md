# Experimental Cactus Whistle CUDA reference

This directory is a reproducible research experiment for running the released Cactus Whistle ASR weights on an NVIDIA GPU. It does **not** add CUDA inference to Cactus's native Needle engine.

## What runs on the GPU

Cactus documents Whistle's `.cact` model through Needle's native CPU inference engine. The supported-device list does not currently advertise NVIDIA inference, and the package's `gpu` extra installs CUDA-enabled JAX for the training/reference stack; installing that extra alone does not make `needle.Whistle` run on CUDA.

This experiment therefore keeps the two paths distinct:

- **CPU baseline:** `needle.Whistle(weights=".../whistle.cact")` through the released native Needle shared library.
- **CUDA reference:** the *released* `whistle.safetensors` checkpoint, loaded by the repository's JAX `Whistle` model, quantized with Cactus's own `cq_ste_params(..., WEIGHT_BITS)` helper, then greedily decoded through `Whistle.encode` and `Whistle.logits`. The audio is processed in 30-second windows. This is an experimental reference loop, not the native engine, a supported production interface, or a compiled CUDA port.

The GPU test checks for non-empty Whistle output, JAX's GPU backend, process-specific GPU SM and VRAM samples from `nvidia-smi pmon`, and a JAX profiler trace. In the recorded run, the trace included cuDNN Maxwell convolution-kernel events during the Whistle encode/decode profile; this is direct device-work evidence in addition to GPU selection and model allocation.

## Reproduce

From the repository root on Linux x86_64, with Python 3.12, `ffmpeg`, `nvidia-smi`, a supported NVIDIA driver, network access, and enough free GPU memory:

```bash
./research/whistle-gpu/run.sh all
```

The first run creates an isolated `.venv` and caches under `research/whistle-gpu/`, installs this checkout's `[train,gpu]` extra with the compatibility pins in `requirements-gpu.txt` from PyPI, downloads the pinned released weights and public test audio, fetches the local Needle CPU engine, then runs CPU and CUDA cases.

For staged runs:

```bash
./research/whistle-gpu/run.sh prepare
./research/whistle-gpu/run.sh cpu
# Check current free VRAM with nvidia-smi; do not stop other GPU services.
./research/whistle-gpu/run.sh gpu
```

`prepare`, `cpu`, and `gpu` each install/update the local editable package and prepare assets; cached downloads make subsequent preparations quick. CUDA profiling can be skipped only by not running the `gpu` action; that action fails rather than claiming success if its inference or GPU-evidence checks fail.

The script writes the model checkpoint, native CPU engine, WAV/FLAC samples, virtual environment, caches, temporary files, and profiler output beneath this directory. `results/prepare.json`, `results/cpu.json`, and `results/gpu.json` contain compact metadata and measurements. The larger profiler traces are intentionally ignored by Git and are recreated by `gpu`.

The experiment downloads no private speech. The short smoke input is OpenAI's public JFK sample; the duration series is assembled from pinned public LibriSpeech dummy utterances with 250 ms silences. The prepared audio is mono 16 kHz and has durations of 11, 10, 30, 60, and 133.22 seconds. The longer files are prefixes of the same joined utterance stream, so these points show measured runtime changes for these inputs, not a general scaling law.

## Recorded run

Measured on **GeForce GTX 1070 (8 GiB, compute capability 6.1), NVIDIA driver 580.178.04**, Linux x86_64, Python 3.12, this fork's `cactus-needle` source version 3.2.1, JAX 0.12.0 with CUDA 12.9 libraries and cuDNN 9.10.2.21. No system `nvcc` was installed; the CUDA runtime and cuDNN came from PyPI wheels. cuDNN 9.10.2 is pinned because newer cuDNN releases that exclude Pascal do not support this GPU.

Warm times below are the median of two calls after the first call for that case; RTF is warm time divided by audio duration. CPU and CUDA ran the exact same prepared WAVs with the same released Whistle revision.

| Audio | Duration | CPU warm | CPU RTF | CUDA warm | CUDA RTF | CPU/CUDA speedup |
|---|---:|---:|---:|---:|---:|---:|
| JFK smoke | 11.00 s | 1.750 s | 0.159 | 0.302 s | 0.027 | 5.8× |
| LibriSpeech | 10.00 s | 3.163 s | 0.316 | 0.467 s | 0.047 | 6.8× |
| LibriSpeech | 30.00 s | 5.832 s | 0.194 | 1.422 s | 0.047 | 4.1× |
| LibriSpeech | 60.00 s | 17.397 s | 0.290 | 3.160 s | 0.053 | 5.5× |
| LibriSpeech | 133.22 s | 38.824 s | 0.291 | 5.908 s | 0.044 | 6.6× |

Both paths returned non-empty smoke and long transcriptions. On the 11-second JFK sample, WER was 0.000 for CPU and 0.045 for CUDA; on the 133.22-second sample it was 0.084 for CPU and 0.091 for CUDA. The long CUDA case was about **6.6× faster** in warm transcription time in this run; this is a measurement on this hardware and test set, not a speedup guarantee.

Timing and resource measurements are separated:

- First full `prepare` command: 103.8 s wall time including the initial Python/CUDA dependency install. The asset-preparation phase itself took 15.6 s; `prepare.json` records the individual download times and SHA-256 hashes.
- Model startup, excluding downloads: native CPU 0.061 s; JAX/CUDA 15.869 s. JAX startup includes importing the GPU stack, loading/quantizing the checkpoint, and placing parameters on the GPU.
- First 11-second inference call: CPU 3.175 s; CUDA 8.263 s. This includes first-call compilation/warm-up effects. The warm medians are shown in the table.
- Peak process RSS: CPU 125.2 MiB; CUDA 3,052.6 MiB.
- CUDA peak **sampled process VRAM**: 1,154 MiB, with 76% process SM utilization at peak across 75 one-second `nvidia-smi pmon` samples. Sampling can miss shorter spikes. The same PID produced non-empty transcriptions and the profiler trace; the trace included cuDNN Maxwell convolution kernels. `results/gpu.json` records the evidence and sets `cuda_inference_verified` only when the output, GPU backend, process GPU activity/memory, and trace checks all pass.

An earlier, separate same-host Whisper small.en comparison on the same long audio measured 28.378 s, about 1,690 MiB sampled VRAM, and 1,658 MiB peak RSS. Against that historical measurement, this Whistle CUDA run used about 536 MiB (32%) less sampled VRAM and was about 4.8× faster, but used about 1.4 GiB more process RSS. That Whisper run is not part of this repository's reproducible script, so treat it as a host observation rather than a controlled model comparison.

## Notes and limitations

- The native `.cact`/Needle path remains CPU-only here. The CUDA result comes from the separate JAX reference loop and the real `.safetensors` Whistle checkpoint.
- A CUDA-enabled Python install, `jax.devices("gpu")`, model allocation, or another process's utilization is not sufficient proof. The `gpu` action requires inference output plus this process's SM/VRAM samples and a profile trace.
- The local loop uses Cactus's current JAX model and quantization helpers; changes to those APIs may require updating this experiment. It is not a maintained upstream inference API.
- The 30-second windows are intentional. Runtime data for 10/30/60/133.22 seconds is reported as measured; no linear, quadratic, or exponential scaling claim is made.
- CPU and CUDA outputs differ slightly because these are separate inference implementations; the recorded WERs make that visible.

## Sources

- [Cactus Whistle overview](https://cactuscompute.com/blog/whistle)
- [Needle supported devices](https://cactuscompute.com/blog/needle-supported-devices)
- [Needle Python documentation](https://cactuscompute.com/blog/needle-python-docs)
- [Needle package extras](https://github.com/cactus-compute/needle/blob/main/pyproject.toml)
- [Whistle JAX model](https://github.com/RokPre/whistleGPU/blob/main/needle/model/whistle.py)
- [Whistle fine-tuning/reference scorer and quantization use](https://github.com/RokPre/whistleGPU/blob/main/needle/model/whistle_finetune.py)
- [Pinned released Whistle files](https://huggingface.co/Cactus-Compute/whistle/tree/741118687e1f008f06d1351a402a731274789790)
- [JAX NVIDIA GPU installation guide](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu)
- [NVIDIA cuDNN 9.10.2 support matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.10.2/reference/support-matrix.html)
- [Pinned public LibriSpeech dummy dataset revision](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/tree/5be91486e11a2d616f4ec5db8d3fd248585ac07a)
- [OpenAI public JFK smoke audio](https://github.com/openai/whisper/blob/main/tests/jfk.flac)
