# Quran Recitation Forced Alignment

CTC-based forced alignment system that evaluates a student's Quran recitation word-by-word — including phonetic mistakes such as wrong harakat (tashkeel).

Built on **NVIDIA NeMo FastConformer** (`nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0`) and designed to integrate with a **LiveKit** voice agent.

---

## How It Works

Standard STT produces a transcript but cannot tell you *which word was wrong* or *how wrong it was*. Forced alignment solves this:

```
Student audio  ──►  CTC log-probabilities  ──►  Viterbi alignment
Reference text ──►  tokenize word-by-word  ──►  per-word score + timing
```

Given the known reference (e.g. Al-Fatiha, ayah 1) and the student's recording, the system returns a per-word confidence score and flags mispronounced words.

### Why CTC Forced Alignment for Harakat Detection?

| Approach | Detects wrong letter | Detects wrong haraka |
|---|---|---|
| STT transcript diff | ✓ | ✗ (model never learned diacritics) |
| CTC forced alignment | ✓ | ✓ (frame-level probability drop) |

The FastConformer model was trained on **plain Arabic** (no tashkeel). When the student says *kasra* where *fatha* is expected, the phoneme sequence changes at the acoustic level → the CTC probability for that token drops → score falls below threshold.

---

## Architecture

```
raw audio (16kHz mono float32)
    │
    ▼
preprocessor          mel spectrogram  (80 bins, 10ms hop)
    │
    ▼
ConformerEncoder      (17 layers, 8× downsampling → 12.5 fps)
    │
    ▼
CTC decoder           Conv1d(512 → 1025, kernel=1) → log_softmax
    │                 blank_id = 1024
    ▼
torchaudio.functional.forced_align   (Viterbi)
    │
    ▼
per-token boundaries + scores
    │
    ▼
group tokens → words  (word-by-word tokenization, explicit counts)
    │
    ▼
{overall_score, passed, words:[{word, start_s, end_s, score, status}]}
```

**Frame rate**: 16000 / (160 × 8) = **12.5 frames/sec** → 80ms per encoder frame

---

## File Structure

```
alignment/
├── ctc_aligner.py          # Core alignment engine
├── quran_db.py             # Diacritized Quran reference database
├── align_server.py         # FastAPI server (port 3006)
├── harakat.py              # Harakat error analysis
├── test_align.py           # CLI test tool
├── requirements.txt        # Python dependencies
├── data/
│   └── quran_diacritized.json   # 6,236 ayat, fully diacritized (bundled)
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU inference; CPU works but is ~10× slower)
- `ffmpeg` system binary (for MP3 input support)

```bash
# Ubuntu / Debian
sudo apt install ffmpeg
```

### Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note**: `nemo_toolkit[asr]` installs PyTorch and CUDA-compatible NeMo. If you have a specific CUDA version, install `torch` manually first:
> ```bash
> pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
> pip install nemo_toolkit[asr]>=1.23.0
> ```

### Model download

The model (~900 MB) is downloaded automatically from HuggingFace on first run:

```
nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0
```

No manual download needed.

---

## Quick Start

### Test with your own audio

```bash
# Align a WAV file against Al-Fatiha, ayah 1
python test_align.py basmala.wav --surah 1 --ayah 1

# Align multiple files
python test_align.py audio1.wav audio2.wav --surah 2 --ayah 255

# Custom reference text
python test_align.py recitation.wav --reference "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ"

# Self-test (no audio needed — verifies the pipeline works)
python test_align.py --selftest

# Performance benchmark (20 iterations)
python test_align.py audio.wav --benchmark
```

**Audio requirements**: 16kHz mono WAV. Install `resampy` for automatic resampling from other sample rates.

### Example output

```
─────────────────────────────────────────────────────────────────
  Reference  : بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ
  Transcript : بسم الله الرحمن الرحيم
  Score      : 0.923  ✓ PASSED
  Duration   : 3.12s

    ✓ بِسْمِ                 [0.08s–0.64s]  score=0.941
    ✓ اللَّهِ                [0.72s–1.20s]  score=0.978
    ✓ الرَّحْمَنِ            [1.28s–2.08s]  score=0.889
    ✓ الرَّحِيمِ             [2.16s–3.12s]  score=0.882
```

---

## Running the Server

```bash
python align_server.py
# Listening on http://0.0.0.0:3006
```

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `ALIGN_PORT` | `3006` | Server port |
| `ALIGN_HOST` | `0.0.0.0` | Bind address |
| `NEMO_MODEL_PATH` | HuggingFace | Path to local `.nemo` file (skips download) |

### API

#### `GET /health`

```json
{"status": "ok", "model": "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0", "quran_db_size": 6236}
```

#### `POST /align`

**Headers**: `Content-Type: audio/wav`
**Body**: raw WAV bytes
**Query params** (pick one):
- `?surah=1&ayah=1` — look up from bundled Quran DB
- `?reference=بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ` — custom reference text
- `?threshold=0.55` — optional score threshold (default 0.55)

**Response**:

```json
{
  "overall_score": 0.923,
  "passed": true,
  "reference": "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ",
  "surah": 1,
  "ayah": 1,
  "transcription": "بسم الله الرحمن الرحيم",
  "duration_s": 3.12,
  "words": [
    {
      "word": "بِسْمِ",
      "word_plain": "بسم",
      "start_s": 0.08,
      "end_s": 0.64,
      "score": 0.941,
      "status": "correct",
      "note": ""
    }
  ],
  "mistakes": []
}
```

**Status values**: `correct` (score ≥ 0.55) · `warning` (0.30–0.55) · `error` (< 0.30)

#### `POST /transcribe`

Plain STT — same input format, returns `{"text": "...", "is_final": true}`.

---

## LiveKit Agent Integration

When the student finishes speaking (end of VAD), your LiveKit agent calls:

```python
import httpx

async def check_recitation(audio_bytes: bytes, surah: int, ayah: int):
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "http://localhost:3006/align",
            content=audio_bytes,
            headers={"Content-Type": "audio/wav"},
            params={"surah": surah, "ayah": ayah},
            timeout=30,
        )
    result = r.json()

    if result["passed"]:
        return "Excellent! Your recitation is correct."
    else:
        mistakes = result["mistakes"]
        feedback = "Please review: " + "; ".join(mistakes[:3])
        return feedback
```

---

## Performance

Tested on RTX 2080 Ti (CC 7.5), Al-Fatiha (7 ayat):

| Metric | Value |
|---|---|
| Average latency per ayah | ~75ms |
| Real-time factor | ~50× faster than real-time |
| GPU memory | ~2.1 GB |
| Quran DB load time | <50ms |

---

## Harakat Error Types

| Error | Arabic | Description |
|---|---|---|
| Wrong short vowel | حركة خاطئة | fatha↔kasra↔damma swap |
| Missing sukun | سكون ناقص | Vowel where silence expected |
| Missing shadda | شدة ناقصة | Single consonant instead of doubled |
| Wrong tanwin | تنوين خاطئ | Tanwin fath↔kasr↔damm swap |
| Madd error | خطأ في المد | Wrong elongation length |

---

## Quran Database

Bundled at `data/quran_diacritized.json` — 6,236 ayat in fully diacritized form.

```
Format:  {"SSSAAA": "diacritized text", ...}
Key:     SSS = surah (001–114), AAA = ayah (001–...)
Source:  Husary diacritized transcriptions
```

```python
from quran_db import QuranDB
db = QuranDB()
print(db.get(1, 1))    # بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ
print(len(db))         # 6236
```

---

## License

MIT — see [LICENSE](LICENSE).
