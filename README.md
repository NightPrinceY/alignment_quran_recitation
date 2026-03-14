# Forced Alignment for Quran Recitation Checking

## What This Is

A system that takes student audio + the correct Ayah text and returns:
- **Which words** the student got wrong
- **Where in time** each word was spoken
- **Confidence score** per word
- **Harakat-level** error detection (Fatha vs Kasra vs Damma)

This is the right tool for a recitation teacher agent because **you already know what the student should say**. You don't need to transcribe — you need to verify.

---

## Why Forced Alignment, Not ASR Transcription

Regular STT: `audio → model → "what did they say?"`
Forced alignment: `audio + reference → model → "how well did they say the reference?"`

For a recitation teacher:
- Teacher assigns Ayah: `بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ`
- Student recites (may make mistakes)
- System scores each word against the reference, reports mistakes

ASR would need to transcribe then compare — two error sources.
Forced alignment has one: how well does the audio match the reference?

---

## How CTC Forced Alignment Works

The FastConformer model has a CTC branch alongside RNNT.
CTC outputs: `log P(token | audio_frame)` for each frame — shape `(T, V+1)`.

Given a reference sequence `[t₁, t₂, ..., tₙ]`, the Viterbi algorithm finds:
- The most likely time boundaries for each token
- The probability of the reference path through the audio

```
Audio frames:   |--bism--|--illah--|--alrahman--|--alrahim--|
Reference:       بِسْمِ    اللَّهِ    الرَّحْمَنِ   الرَّحِيمِ
Score per word:  0.94      0.91       0.73         0.89
                 ✓OK       ✓OK        ⚠ LOW         ✓OK
```

Low score = student mispronounced that word.

### Two-Level Strategy

**Level 1 — Word correctness** (works with plain Arabic, model is trained on this):
- Align using non-diacritized tokens
- Get per-word timing and confidence
- Detects: wrong words, skipped words, added words, word order errors

**Level 2 — Harakat verification** (dual-token comparison):
- At each aligned word segment, compare CTC scores of:
  - `P(correct diacritized token | audio)`  vs  `P(wrong diacritized token | audio)`
- Even though model wasn't trained on diacritics, the encoder IS phonetically aware
- /a/ (Fatha), /i/ (Kasra), /u/ (Damma) are acoustically distinct phonemes
- The encoder representations DO capture this distinction
- The relative probability ratio still carries information

---

## Architecture

```
                        LiveKit Agent
                             │
              ┌──────────────┴──────────────┐
              │                             │
        /transcribe                     /align
     (plain STT)              (reference + audio)
              │                             │
       NeMo STT Server              Alignment Server
              │                             │
    EncDecHybridRNNTCTC           EncDecHybridRNNTCTC
         (RNNT path)                  (CTC path)
                                         │
                               torchaudio.forced_align
                                         │
                              Per-word confidence scores
                                         │
                              Harakat error detection
```

Both servers share the same model (or the alignment server IS the STT server with an extra endpoint).

---

## API

### POST /align

**Request:**
```json
{
  "reference": "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ",
  "surah": 1,
  "ayah": 1
}
```
Body: raw audio bytes (WAV or PCM, same as /transcribe)

**Response:**
```json
{
  "overall_score": 0.87,
  "passed": true,
  "words": [
    {
      "word": "بِسْمِ",
      "word_plain": "بسم",
      "start_s": 0.12,
      "end_s": 0.68,
      "score": 0.94,
      "status": "correct"
    },
    {
      "word": "الرَّحْمَنِ",
      "word_plain": "الرحمن",
      "start_s": 1.45,
      "end_s": 2.31,
      "score": 0.41,
      "status": "error",
      "note": "low confidence — possible harakat error"
    }
  ],
  "mistakes": ["الرَّحْمَنِ at 1.45s (score: 0.41)"],
  "transcription": "بسم الله الرحمن الرحيم"
}
```

---

## Frame Rate Math

FastConformer downsampling: 4× (ConvSubsampling with stride=4 × 2 steps = 8× total)
- Input: 16,000 samples/sec
- Mel: 100 frames/sec (10ms hop)
- After encoder subsampling: ~12.5 frames/sec (80ms per frame)

For a 7-second Basmala: ~87 encoder frames, aligned across 4 words + 18 subword tokens.

---

## Harakat Error Types to Detect

| Error Type | Arabic | Example |
|---|---|---|
| Fatha instead of Kasra | فتحة بدل كسرة | بِسْمِ → بَسْمَ |
| Kasra instead of Damma | كسرة بدل ضمة | الرَّحِيمُ → الرَّحِيمِ |
| Missing Shadda | حذف الشدة | اللَّهِ → اِلَهِ |
| Missing Sukun | حذف السكون | بِسْمِ → بِسَمِ |
| Wrong tanwin | تنوين خاطئ | عَلِيمٌ → عَلِيمًا |

---

## File Structure

```
alignment/
├── README.md               ← this file
├── ctc_aligner.py          ← core CTC forced alignment logic
├── quran_db.py             ← Quran reference text lookup
├── harakat.py              ← harakat-level error analysis
├── align_server.py         ← FastAPI server (/align endpoint)
└── test_align.py           ← end-to-end test against real WAVs
```
