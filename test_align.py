"""
test_align.py
─────────────
End-to-end test of the forced alignment system.
Runs directly (no server needed) against real Minshawi WAV files.

Tests:
  1. Al-Fatiha verses 1-7 (correct recitation, should all pass)
  2. Ayat al-Kursi (2:255, longer verse)
  3. Quran DB lookup
  4. Timing and performance

Run:
  source ~/CollegeX/bin/activate
  CUDA_VISIBLE_DEVICES=0 python test_align.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))

WAV_DIR   = Path(__file__).parent.parent / "wav"
MODEL_NAME = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"


def load_model():
    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict
    print(f"Loading model: {MODEL_NAME}")
    m = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(MODEL_NAME)
    with open_dict(m.cfg):
        m.cfg.decoding.greedy.use_cuda_graphs = False
        m.cfg.decoding.greedy.use_cuda_graph_decoder = False
    m.change_decoding_strategy(m.cfg.decoding)
    m.eval()
    return m.cuda()


def load_wav(path: Path) -> np.ndarray:
    wav, sr = sf.read(str(path), dtype="float32")
    assert sr == 16000, f"Expected 16kHz, got {sr}"
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav


def print_result(result: dict, label: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"  Reference:  {result.get('reference', '?')}")
    print(f"  Transcript: {result.get('transcription', '?')}")
    print(f"  Score:      {result.get('overall_score', 0):.3f}  "
          f"{'✓ PASSED' if result.get('passed') else '✗ FAILED'}")
    print(f"  Duration:   {result.get('duration_s', 0):.2f}s")
    print()

    words = result.get("words", [])
    for w in words:
        icon = "✓" if w["status"] == "correct" else ("⚠" if w["status"] == "warning" else "✗")
        print(f"    {icon} {w['word']:<20} [{w['start_s']:.2f}s–{w['end_s']:.2f}s]  "
              f"score={w['score']:.3f}  {w['note']}")

    mistakes = result.get("mistakes", [])
    if mistakes:
        print(f"\n  Mistakes ({len(mistakes)}):")
        for m in mistakes:
            print(f"    → {m}")


def main():
    from ctc_aligner import CTCAligner
    from quran_db import QuranDB

    # Load model
    t0 = time.time()
    model = load_model()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    aligner  = CTCAligner(model)
    quran_db = QuranDB()
    print(f"QuranDB loaded: {len(quran_db)} ayat\n")

    # ── Test 1: Al-Fatiha (correct recitation, should all pass) ───────────────
    print("=" * 60)
    print("TEST 1 — Al-Fatiha (correct Minshawi recitation, expect PASS)")
    print("=" * 60)

    fatiha_tests = [
        ("001001", 1, 1),
        ("001002", 1, 2),
        ("001003", 1, 3),
        ("001004", 1, 4),
        ("001005", 1, 5),
        ("001006", 1, 6),
        ("001007", 1, 7),
    ]

    scores = []
    for wav_id, surah, ayah in fatiha_tests:
        wav_path = WAV_DIR / f"{wav_id}.wav"
        if not wav_path.exists():
            print(f"  SKIP {wav_id} — file not found")
            continue

        reference = quran_db.get(surah, ayah)
        if not reference:
            print(f"  SKIP {wav_id} — not in QuranDB")
            continue

        wav = load_wav(wav_path)
        t = time.time()
        result = aligner.align(wav, reference)
        elapsed = time.time() - t

        result["reference"] = reference
        label = f"{surah}:{ayah}  [{wav_id}]  align_time={elapsed:.2f}s"
        print_result(result, label)
        scores.append(result.get("overall_score", 0))

    if scores:
        print(f"\nAl-Fatiha average score: {np.mean(scores):.3f}")

    # ── Test 2: Ayat al-Kursi (longer verse) ──────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 2 — Ayat al-Kursi (2:255, longer verse)")
    print("=" * 60)

    kursi_path = WAV_DIR / "002255.wav"
    if kursi_path.exists():
        reference = quran_db.get(2, 255)
        if reference:
            wav = load_wav(kursi_path)
            t = time.time()
            result = aligner.align(wav, reference)
            result["reference"] = reference
            print_result(result, f"2:255  align_time={time.time()-t:.2f}s")

    # ── Test 3: First 5 Al-Baqara verses ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 3 — Al-Baqara 1-5")
    print("=" * 60)

    for ayah in range(1, 6):
        wav_id   = f"002{ayah:03d}"
        wav_path = WAV_DIR / f"{wav_id}.wav"
        if not wav_path.exists():
            continue
        reference = quran_db.get(2, ayah)
        if not reference:
            continue
        wav = load_wav(wav_path)
        t   = time.time()
        result = aligner.align(wav, reference)
        result["reference"] = reference
        print_result(result, f"2:{ayah}  [{wav_id}]  align_time={time.time()-t:.2f}s")

    # ── Test 4: Performance benchmark ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 4 — Performance: 10 alignments back-to-back")
    print("=" * 60)

    wav_path = WAV_DIR / "001001.wav"
    reference = quran_db.get(1, 1)
    if wav_path.exists() and reference:
        wav   = load_wav(wav_path)
        times = []
        for _ in range(10):
            t = time.time()
            aligner.align(wav, reference)
            times.append(time.time() - t)
        print(f"  Mean: {np.mean(times)*1000:.1f}ms  "
              f"Min: {np.min(times)*1000:.1f}ms  "
              f"Max: {np.max(times)*1000:.1f}ms")
        print(f"  Audio duration: {len(wav)/16000:.2f}s  "
              f"→ Real-time factor: {np.mean(times)/(len(wav)/16000):.2f}×")

    print("\n✓ All tests complete.")


if __name__ == "__main__":
    main()
