"""
test_align.py
─────────────
End-to-end test of the forced alignment system.

Runs directly (no server needed). Accepts WAV files as CLI arguments
or downloads a public sample if none are provided.

Usage:
    # With your own WAV files (16kHz mono):
    python test_align.py path/to/audio1.wav path/to/audio2.wav

    # With surah:ayah references:
    python test_align.py --surah 1 --ayah 1 path/to/basmala.wav

    # Quick self-test using a generated sine-wave (no real audio needed):
    python test_align.py --selftest

    # Performance benchmark only:
    python test_align.py --benchmark path/to/audio.wav
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

MODEL_NAME = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
SAMPLE_RATE = 16000


def load_model():
    """Load the NeMo FastConformer model."""
    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict

    print(f"Loading model: {MODEL_NAME}")
    print("(First run downloads ~900 MB from HuggingFace Hub)")
    t0 = time.time()
    m = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(MODEL_NAME)
    with open_dict(m.cfg):
        m.cfg.decoding.greedy.use_cuda_graphs = False
        m.cfg.decoding.greedy.use_cuda_graph_decoder = False
    m.change_decoding_strategy(m.cfg.decoding)
    m.eval()
    try:
        m = m.cuda()
        print(f"Model loaded on GPU in {time.time()-t0:.1f}s")
    except Exception:
        print(f"Model loaded on CPU in {time.time()-t0:.1f}s (GPU not available)")
    return m


def load_wav(path: str) -> np.ndarray:
    """Load a WAV file as float32 numpy array at 16kHz mono."""
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        # Resample if needed
        try:
            import resampy
            wav = resampy.resample(wav, sr, SAMPLE_RATE)
        except ImportError:
            print(f"WARNING: Audio is {sr}Hz, expected 16000Hz. "
                  "Install resampy for automatic resampling.")
    return wav


def generate_silence(duration_s: float = 3.0) -> np.ndarray:
    """Generate silent audio for self-test (model should give low scores)."""
    return np.zeros(int(SAMPLE_RATE * duration_s), dtype=np.float32)


def print_result(result: dict) -> None:
    ref = result.get("reference", "")
    print(f"\n{'─'*65}")
    if ref:
        print(f"  Reference  : {ref}")
    print(f"  Transcript : {result.get('transcription', '')}")
    score = result.get('overall_score', 0)
    passed = result.get('passed', False)
    print(f"  Score      : {score:.3f}  {'✓ PASSED' if passed else '✗ FAILED'}")
    print(f"  Duration   : {result.get('duration_s', 0):.2f}s")
    print()
    for w in result.get("words", []):
        icon = {"correct": "✓", "warning": "⚠", "error": "✗"}.get(w["status"], "?")
        note = f"  ← {w['note']}" if w["note"] else ""
        print(f"    {icon} {w['word']:<22} "
              f"[{w['start_s']:.2f}s–{w['end_s']:.2f}s]  "
              f"score={w['score']:.3f}{note}")
    mistakes = result.get("mistakes", [])
    if mistakes:
        print(f"\n  ⚠ Mistakes ({len(mistakes)}):")
        for m in mistakes:
            print(f"      → {m}")


def main():
    parser = argparse.ArgumentParser(description="Test Quran forced alignment")
    parser.add_argument("wavs", nargs="*", help="WAV file paths to align")
    parser.add_argument("--surah",     type=int, default=1, help="Surah number (default: 1)")
    parser.add_argument("--ayah",      type=int, default=1, help="Ayah number (default: 1)")
    parser.add_argument("--reference", type=str, default=None,
                        help="Custom reference text (overrides --surah/--ayah)")
    parser.add_argument("--selftest",  action="store_true",
                        help="Run self-test with generated silence (no WAV needed)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run 20-iteration benchmark on the first WAV")
    parser.add_argument("--cpu",       action="store_true",
                        help="Force CPU inference (slower)")
    args = parser.parse_args()

    from ctc_aligner import CTCAligner
    from quran_db import QuranDB

    model = load_model()
    if args.cpu:
        model = model.cpu()

    aligner  = CTCAligner(model)
    quran_db = QuranDB()
    print(f"QuranDB: {len(quran_db)} ayat loaded\n")

    # ── Resolve reference text ────────────────────────────────────────────────
    if args.reference:
        reference = args.reference
    else:
        reference = quran_db.get(args.surah, args.ayah)
        if not reference:
            print(f"ERROR: Ayah {args.surah}:{args.ayah} not found in database.")
            sys.exit(1)

    print(f"Reference ({args.surah}:{args.ayah}): {reference}\n")

    # ── Self-test (no WAV needed) ─────────────────────────────────────────────
    if args.selftest:
        print("=" * 65)
        print("SELF-TEST — silence audio (expect low scores, verify pipeline works)")
        print("=" * 65)
        wav = generate_silence(3.0)
        result = aligner.align(wav, reference)
        result["reference"] = reference
        print_result(result)
        print("\n✓ Self-test complete (pipeline works; low scores are expected for silence)")
        return

    # ── No WAV files provided ─────────────────────────────────────────────────
    if not args.wavs:
        print("No WAV files provided. Examples:")
        print(f"  python test_align.py audio.wav")
        print(f"  python test_align.py --surah 1 --ayah 1 basmala.wav")
        print(f"  python test_align.py --selftest")
        sys.exit(0)

    # ── Align each WAV ────────────────────────────────────────────────────────
    print("=" * 65)
    print(f"ALIGNMENT — {len(args.wavs)} file(s)")
    print("=" * 65)

    for wav_path in args.wavs:
        if not Path(wav_path).exists():
            print(f"  SKIP {wav_path} — file not found")
            continue

        wav = load_wav(wav_path)
        t0  = time.time()
        result = aligner.align(wav, reference)
        elapsed = time.time() - t0

        result["reference"] = reference
        print(f"\n  File: {wav_path}  (align_time={elapsed*1000:.0f}ms)")
        print_result(result)

    # ── Benchmark ─────────────────────────────────────────────────────────────
    if args.benchmark and args.wavs:
        wav_path = args.wavs[0]
        if Path(wav_path).exists():
            print("\n" + "=" * 65)
            print("BENCHMARK — 20 iterations")
            print("=" * 65)
            wav   = load_wav(wav_path)
            times = []
            for _ in range(20):
                t0 = time.time()
                aligner.align(wav, reference)
                times.append(time.time() - t0)
            times_ms = [t * 1000 for t in times]
            dur_s    = len(wav) / SAMPLE_RATE
            print(f"  Audio duration : {dur_s:.2f}s")
            print(f"  Mean           : {np.mean(times_ms):.1f}ms")
            print(f"  Min / Max      : {np.min(times_ms):.1f}ms / {np.max(times_ms):.1f}ms")
            print(f"  Real-time factor: {np.mean(times)/dur_s:.3f}× "
                  f"({1/np.mean(times)*dur_s:.0f}× faster than real-time)")

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
