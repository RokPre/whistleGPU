from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import resource
import shutil
import statistics
import subprocess
import threading
import time
import traceback
import wave
from pathlib import Path
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).absolute().parent
MODEL_REVISION = "741118687e1f008f06d1351a402a731274789790"
DATASET_REVISION = "5be91486e11a2d616f4ec5db8d3fd248585ac07a"
MODEL_CACT_SHA256 = "b6e02f048568ac5d01a2042556c658061e699acbc0aa2a1439f52f3d461dffeb"
CHECKPOINT_SHA256 = "5fd58c246f522ecee2b568b306598a3befdc274c225e3a7376abce277ab3d107"
JFK_SHA256 = "63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715"
SAMPLE_RATE = 16000
WINDOW_SECONDS = 30
DATASET_ROWS = "https://datasets-server.huggingface.co/rows?" + urlencode({
    "dataset": "hf-internal-testing/librispeech_asr_dummy",
    "config": "clean",
    "split": "validation",
    "offset": 0,
    "length": 73,
    "revision": DATASET_REVISION,
})


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    root = ROOT.resolve()
    resolved = path.resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"refusing path outside experiment directory: {path}")
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f"refusing symlink inside experiment directory: {current}")
    return path


def ensure_dir(path: str | Path) -> Path:
    path = repo_path(path)
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise RuntimeError(f"not a directory: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: str | Path) -> dict:
    destination = repo_path(destination)
    ensure_dir(destination.parent)
    if destination.exists():
        if not destination.is_file() or not destination.stat().st_size:
            raise RuntimeError(f"invalid existing download target: {destination}")
        return {"path": str(destination.relative_to(ROOT)), "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination), "already_present": True, "seconds": 0.0}

    partial = repo_path(destination.with_name(destination.name + ".part"))
    partial.unlink(missing_ok=True)
    request = Request(url, headers={"User-Agent": "cactus-whistle-gpu-reference/1.0"})
    digest = hashlib.sha256()
    size = 0
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=120) as response, partial.open("xb") as handle:
            while block := response.read(1024 * 1024):
                handle.write(block)
                digest.update(block)
                size += len(block)
        if not size:
            raise RuntimeError(f"download was empty: {url}")
        os.replace(partial, destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return {"path": str(destination.relative_to(ROOT)), "bytes": size,
            "sha256": digest.hexdigest(), "already_present": False,
            "seconds": time.perf_counter() - started}


def fetch_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": "cactus-whistle-gpu-reference/1.0"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=60) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code not in (500, 502, 503, 504) or attempt == 2:
                raise
            error.close()
            time.sleep(2 ** attempt)


def pcm_from_flac(path: Path) -> bytes:
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le", "-acodec", "pcm_s16le",
         "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if not result.stdout:
        raise RuntimeError(f"ffmpeg decoded no samples from {path}")
    return result.stdout


def write_pcm_wav(path: str | Path, pcm: bytes) -> None:
    path = repo_path(path)
    ensure_dir(path.parent)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm)


def wav_duration(path: str | Path) -> float:
    with wave.open(str(repo_path(path)), "rb") as source:
        if source.getframerate() != SAMPLE_RATE or source.getnchannels() != 1:
            raise RuntimeError(f"expected 16 kHz mono WAV: {path}")
        return source.getnframes() / source.getframerate()


def write_json(path: str | Path, value: object) -> None:
    path = repo_path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def prepare() -> None:
    started = time.perf_counter()
    ensure_dir("models/cactus")
    ensure_dir("checkpoints")
    ensure_dir("data/librispeech")
    ensure_dir("results")
    manifest: dict = {"model_revision": MODEL_REVISION, "dataset_revision": DATASET_REVISION, "files": {}}

    manifest["files"]["whistle_cact"] = download(
        f"https://huggingface.co/Cactus-Compute/whistle/resolve/{MODEL_REVISION}/whistle.cact?download=true",
        "models/cactus/whistle.cact",
    )
    manifest["files"]["whistle_checkpoint"] = download(
        f"https://huggingface.co/Cactus-Compute/whistle/resolve/{MODEL_REVISION}/checkpoints/whistle.safetensors?download=true",
        "checkpoints/whistle.safetensors",
    )
    manifest["files"]["jfk_flac"] = download(
        "https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac", "data/jfk.flac",
    )
    expected = {"whistle_cact": MODEL_CACT_SHA256,
                "whistle_checkpoint": CHECKPOINT_SHA256, "jfk_flac": JFK_SHA256}
    for key, digest in expected.items():
        if manifest["files"][key]["sha256"] != digest:
            raise RuntimeError(f"{key} checksum does not match the pinned public source")
    write_pcm_wav("data/jfk-16k-mono.wav", pcm_from_flac(repo_path("data/jfk.flac")))

    library = repo_path("models/cactus/libneedle.so")
    library_was_present = library.is_file()
    engine_started = time.perf_counter()
    if not library_was_present:
        subprocess.run([str(repo_path(".venv/bin/needle")), "fetch", "--out",
                        str(repo_path("models/cactus")), "--generation", "3"], check=True)
    if not library.is_file():
        raise RuntimeError(f"Needle fetch did not create the local CPU engine: {library}")
    manifest["files"]["needle_cpu_engine"] = {
        "path": str(library.relative_to(ROOT)), "bytes": library.stat().st_size,
        "sha256": sha256_file(library), "already_present": library_was_present,
        "seconds": 0.0 if library_was_present else time.perf_counter() - engine_started,
    }

    rows = fetch_json(DATASET_ROWS)["rows"]
    selected, pcm_parts, audio_downloads = [], [], []
    silence = bytes(SAMPLE_RATE // 4 * 2)
    frames = 0
    for item in rows:
        row = item["row"]
        if len(row.get("audio", [])) != 1:
            continue
        clip_id = row["id"]
        flac = repo_path(Path("data/librispeech") / f"{clip_id}.flac")
        asset = download(row["audio"][0]["src"], flac)
        audio_downloads.append(asset)
        pcm = pcm_from_flac(flac)
        clip_frames = len(pcm) // 2
        selected.append({"id": clip_id, "text": row["text"], "frames": clip_frames,
                         "duration_seconds": clip_frames / SAMPLE_RATE,
                         "flac": str(flac.relative_to(ROOT))})
        pcm_parts.append(pcm)
        frames += clip_frames
        if frames >= 120 * SAMPLE_RATE:
            break
        pcm_parts.append(silence)
        frames += len(silence) // 2
    if not selected or frames < 60 * SAMPLE_RATE:
        raise RuntimeError(f"public sample selection is too short: {frames / SAMPLE_RATE:.1f} seconds")

    long_pcm = b"".join(pcm_parts)
    write_pcm_wav("data/speech-long.wav", long_pcm)
    for seconds in (10, 30, 60):
        write_pcm_wav(f"data/speech-{seconds}s.wav", long_pcm[:seconds * SAMPLE_RATE * 2])
    transcripts = {
        "dataset": "hf-internal-testing/librispeech_asr_dummy",
        "revision": DATASET_REVISION,
        "audio_format": "16 kHz mono PCM WAV; 250 ms silence between source utterances",
        "smoke": {"audio": "data/jfk-16k-mono.wav",
                  "text": "And so, my fellow Americans: ask not what your country can do for you—ask what you can do for your country.",
                  "duration_seconds": wav_duration("data/jfk-16k-mono.wav")},
        "long": {"ids": [row["id"] for row in selected],
                 "text": " ".join(row["text"] for row in selected),
                 "duration_seconds": wav_duration("data/speech-long.wav"), "clips": selected},
    }
    write_json("data/transcripts.json", transcripts)
    manifest["audio_downloads"] = audio_downloads
    manifest["audio_seconds"] = {
        name: wav_duration(path) for name, path in {
            "smoke": "data/jfk-16k-mono.wav", "10s": "data/speech-10s.wav",
            "30s": "data/speech-30s.wav", "60s": "data/speech-60s.wav",
            "long": "data/speech-long.wav",
        }.items()
    }
    manifest["total_seconds"] = time.perf_counter() - started
    manifest["created_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json("results/prepare.json", manifest)
    print(json.dumps(manifest, indent=2))


class NvidiaMonitor:
    def __init__(self) -> None:
        self.pid = os.getpid()
        self.samples: list[dict[str, float]] = []
        self.process: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None

    def __enter__(self):
        executable = shutil.which("nvidia-smi")
        if executable:
            self.process = subprocess.Popen(
                [executable, "pmon", "-s", "um", "-d", "1"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
            )
            self.thread = threading.Thread(target=self._read, daemon=True)
            self.thread.start()
        return self

    def _read(self) -> None:
        assert self.process and self.process.stdout
        for line in self.process.stdout:
            fields = line.split()
            if len(fields) < 10 or not fields[0].isdigit() or not fields[1].isdigit():
                continue
            if int(fields[1]) != self.pid:
                continue
            sample = {}
            for key, value in (("sm_pct", fields[3]), ("mem_pct", fields[4]), ("vram_mib", fields[9])):
                if value.isdigit():
                    sample[key] = float(value)
            if sample:
                self.samples.append(sample)

    def __exit__(self, *_exc) -> None:
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.thread:
            self.thread.join(timeout=2)

    def summary(self) -> dict:
        peak = lambda key: max((row[key] for row in self.samples if key in row), default=None)
        return {"pid": self.pid, "process_vram_peak_mib": peak("vram_mib"),
                "process_sm_peak_pct": peak("sm_pct"), "process_mem_peak_pct": peak("mem_pct"),
                "process_pmon_samples": len(self.samples)}


def peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def normalized_wer(reference: str | None, hypothesis: str) -> float | None:
    if reference is None:
        return None
    try:
        from whisper_normalizer.english import EnglishTextNormalizer
    except ImportError:
        return None
    normalize = EnglishTextNormalizer()
    ref, hyp = normalize(reference).split(), normalize(hypothesis).split()
    if not ref:
        return None
    previous = list(range(len(hyp) + 1))
    for i, word in enumerate(ref, 1):
        current = [i]
        for j, candidate in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (word != candidate)))
        previous = current
    return previous[-1] / len(ref)


def benchmark_one(name: str, duration: float, infer, reference: str | None = None) -> dict:
    started = time.perf_counter()
    result = infer()
    first = time.perf_counter() - started
    warm = []
    for _ in range(2):
        started = time.perf_counter()
        result = infer()
        warm.append(time.perf_counter() - started)
    text = result["text"]
    median = statistics.median(warm)
    return {"name": name, "audio_seconds": duration, "first_call_seconds": first,
            "warm_call_seconds": warm, "warm_median_seconds": median, "warm_rtf": median / duration,
            "text": text, "language": result.get("language"),
            "nonempty_transcription": bool(text.strip()),
            "reference_wer": normalized_wer(reference, text)}


def _cases(transcripts: dict) -> list[tuple[str, str, str | None]]:
    return [
        ("smoke", "data/jfk-16k-mono.wav", transcripts["smoke"]["text"]),
        ("10s", "data/speech-10s.wav", None),
        ("30s", "data/speech-30s.wav", None),
        ("60s", "data/speech-60s.wav", None),
        ("long", "data/speech-long.wav", transcripts["long"]["text"]),
    ]


def cpu_experiment() -> None:
    started = time.perf_counter()
    import needle

    model = needle.Whistle(weights=str(repo_path("models/cactus/whistle.cact")))
    startup = time.perf_counter() - started
    with NvidiaMonitor() as monitor:
        results = [benchmark_one(name, wav_duration(path),
                                 lambda path=path: model.transcribe(str(repo_path(path))), reference)
                   for name, path, reference in _cases(json.loads(repo_path("data/transcripts.json").read_text()))]
    if not results[0]["nonempty_transcription"] or not results[-1]["nonempty_transcription"]:
        raise RuntimeError("native CPU Whistle returned empty smoke or long output")
    report = {"backend": "Cactus Needle native CPU engine with released whistle.cact",
              "package_version": needle.__version__, "startup_seconds_no_download": startup,
              "short_smoke_nonempty": results[0]["nonempty_transcription"],
              "long_transcription_nonempty": results[-1]["nonempty_transcription"],
              "cases": results, "process_peak_rss_mib": peak_rss_mib(),
              "gpu_sampling": monitor.summary(), "created_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    write_json("results/cpu.json", report)
    for case in results:
        print(f"CPU {case['name']:>5}: {case['audio_seconds']:6.2f}s audio, "
              f"{case['warm_median_seconds']:7.3f}s warm, RTF {case['warm_rtf']:.3f}, "
              f"WER {case['reference_wer']}")
    print(f"CPU startup {startup:.3f}s; peak RSS {report['process_peak_rss_mib']:.1f} MiB")


def gpu_experiment() -> None:
    monitor = NvidiaMonitor()
    try:
        with monitor:
            report = _run_gpu()
        sampling = monitor.summary()
        report["gpu_sampling"] = sampling
        report["cuda_inference_verified"] = bool(
            report["jax_default_backend"] == "gpu" and report["gpu_devices"]
            and report["short_smoke_nonempty"] and report["long_transcription_nonempty"]
            and sampling["process_sm_peak_pct"] and sampling["process_sm_peak_pct"] > 0
            and sampling["process_vram_peak_mib"] and sampling["process_vram_peak_mib"] > 0
            and report["profile_trace_files"] and not report["profiler_start_error"]
        )
        if not report["cuda_inference_verified"]:
            raise RuntimeError("CUDA evidence incomplete: require non-empty output, GPU JAX, "
                               "process SM/VRAM samples, and a profiler trace")
        report["cuda_evidence"] = {
            "per_process_gpu_monitor": "nvidia-smi pmon -s um -d 1",
            "process_sm_peak_pct": sampling["process_sm_peak_pct"],
            "process_peak_vram_mib": sampling["process_vram_peak_mib"],
            "profile_trace_directory": report["profile_trace_directory"],
            "profile_trace_files": report.pop("profile_trace_files"),
            "profiler_start_error": report.pop("profiler_start_error"),
        }
        write_json("results/gpu.json", report)
        for case in report["cases"]:
            print(f"GPU {case['name']:>5}: {case['audio_seconds']:6.2f}s audio, "
                  f"{case['warm_median_seconds']:7.3f}s warm, RTF {case['warm_rtf']:.3f}, "
                  f"WER {case['reference_wer']}")
        print(f"CUDA verified: {report['cuda_inference_verified']}; "
              f"peak process VRAM {sampling['process_vram_peak_mib']} MiB; "
              f"peak RSS {report['process_peak_rss_mib']:.1f} MiB")
        print(f"CUDA trace: {report['profile_trace_directory']}")
    except Exception as error:
        write_json("results/gpu_failure.json", {
            "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(), "gpu_sampling": monitor.summary(),
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        })
        raise


def _run_gpu() -> dict:
    started = time.perf_counter()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from dataclasses import replace
    from needle.model.quantize import WEIGHT_BITS, cq_ste_params
    from needle.model.whistle import Whistle, load_whistle
    from needle.model.whistle_finetune import WhistleTokenizer, read_audio

    devices = jax.devices("gpu")
    if not devices:
        raise RuntimeError("JAX found no CUDA GPU device")
    device = devices[0]
    params, cfg = load_whistle(str(repo_path("checkpoints/whistle.safetensors")))
    params = jax.tree.map(lambda value: np.asarray(value).astype(np.float32), params)
    cfg = replace(cfg, dtype="float32", flash=False, remat=False)
    model = Whistle(cfg)
    params = jax.device_put(cq_ste_params(params, WEIGHT_BITS), device)
    jax.tree.leaves(params)[0].block_until_ready()
    startup = time.perf_counter() - started
    tokenizer = WhistleTokenizer().sp

    @jax.jit
    def encode(weights, wav, lengths):
        return model.apply({"params": weights}, wav, lengths, True, method=Whistle.encode)

    @jax.jit
    def step(weights, tokens, enc, frames):
        return model.apply({"params": weights}, tokens, enc, frames, True, method=Whistle.logits)

    tag_mask = np.arange(cfg.vocab) >= cfg.text

    def infer_window(samples: np.ndarray) -> dict:
        wav = np.zeros((1, WINDOW_SECONDS * SAMPLE_RATE), np.float32)
        count = min(len(samples), wav.shape[1])
        wav[0, :count] = samples[:count]
        lengths = np.asarray([count], np.int32)
        enc, frames = encode(params, wav, lengths)
        enc.block_until_ready()
        tokens = jnp.full((1, cfg.max_tokens), cfg.pad, dtype=jnp.int32).at[:, 0].set(cfg.bos)
        text_ids, language = [], ""
        for pos in range(cfg.max_tokens - 1):
            size = min(cfg.max_tokens, (pos // 64 + 1) * 64)
            logits = np.array(step(params, tokens[:, :size], enc, frames)[:, pos])
            logits[:, tag_mask != (pos == 0)] = -np.inf
            token = int(logits.argmax(-1)[0])
            if pos == 0:
                language_index = token - cfg.text
                if 0 <= language_index < len(cfg.languages):
                    language = cfg.languages[language_index]
            elif token == cfg.eos:
                break
            else:
                text_ids.append(token)
            tokens = tokens.at[:, pos + 1].set(token)
        return {"text": tokenizer.decode(text_ids), "language": language}

    def infer_path(path: str) -> dict:
        samples = np.asarray(read_audio(str(repo_path(path))), np.float32)
        pieces = [infer_window(samples[offset:offset + WINDOW_SECONDS * SAMPLE_RATE])
                  for offset in range(0, len(samples), WINDOW_SECONDS * SAMPLE_RATE)]
        pieces = [piece for piece in pieces if piece["text"].strip()]
        return {"text": " ".join(piece["text"].strip() for piece in pieces),
                "language": pieces[0]["language"] if pieces else ""}

    transcripts = json.loads(repo_path("data/transcripts.json").read_text())
    cases = _cases(transcripts)
    profile_dir = repo_path(Path("results/profiles") /
                            ("whistle-jax-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")))
    ensure_dir(profile_dir)
    profile_error, profile_started = None, False
    results = []
    try:
        jax.profiler.start_trace(str(profile_dir))
        profile_started = True
    except Exception as error:
        profile_error = f"{type(error).__name__}: {error}"
    try:
        name, path, reference = cases[0]
        results.append(benchmark_one(name, wav_duration(path),
                                     lambda: infer_path(path), reference))
    finally:
        if profile_started:
            jax.profiler.stop_trace()
    for name, path, reference in cases[1:]:
        results.append(benchmark_one(name, wav_duration(path),
                                     lambda path=path: infer_path(path), reference))
    jax.effects_barrier()
    profile_files = [str(path.relative_to(ROOT)) for path in profile_dir.rglob("*") if path.is_file()]
    return {
        "backend": "Cactus Whistle JAX reference model with Cactus quantization helpers on CUDA",
        "package_version": __import__("needle").__version__, "jax_version": jax.__version__,
        "jax_default_backend": jax.default_backend(), "gpu_devices": [str(item) for item in devices],
        "compute_dtype": "float32", "flash_attention": False, "weight_quantization_bits": WEIGHT_BITS,
        "startup_seconds_no_download": startup,
        "short_smoke_nonempty": bool(results[0]["text"].strip()),
        "long_transcription_nonempty": bool(results[-1]["text"].strip()),
        "profile_trace_directory": str(profile_dir.relative_to(ROOT)), "cases": results,
        "process_peak_rss_mib": peak_rss_mib(), "profile_trace_files": profile_files,
        "profiler_start_error": profile_error, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Whistle's native CPU and JAX/CUDA reference paths")
    parser.add_argument("action", choices=("prepare", "cpu", "gpu"))
    action = parser.parse_args().action
    if action == "prepare":
        prepare()
    elif action == "cpu":
        cpu_experiment()
    else:
        gpu_experiment()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
