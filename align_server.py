"""
align_server.py
───────────────
FastAPI server that extends the NeMo STT server with a /align endpoint.

Runs on port 3006 (STT server stays on 3005).
Shares the same model instance to avoid loading 900 MB weights twice
— the model is loaded once at startup and used by both /transcribe and /align.

Endpoints:
  GET  /health           → {"status": "ok"}
  POST /transcribe       → same as nemo_stt_server.py (plain text output)
  POST /align            → forced alignment against reference text

/align request:
  Headers:  Content-Type: audio/wav  (or application/octet-stream for PCM)
  Body:     raw audio bytes
  Query params:
    reference=<diacritized Arabic text>
    surah=<int>&ayah=<int>   (alternative: load from Quran DB)
    threshold=<float>        (optional, default 0.55)

/align response:
  {
    "overall_score": 0.87,
    "passed": true,
    "words": [
      {"word": "بِسْمِ", "word_plain": "بسم", "start_s": 0.12, "end_s": 0.68,
       "score": 0.94, "status": "correct", "note": ""},
      ...
    ],
    "mistakes": [],
    "transcription": "بسم الله الرحمن الرحيم",
    "duration_s": 6.4
  }

Integration with LiveKit agent:
  When student finishes speaking (end of VAD), agent calls:
    POST http://localhost:3006/align?surah=1&ayah=1
  with the recorded audio.
  The agent reads `words` and `mistakes` and gives the student feedback.
"""

import logging
import os
import tempfile
import wave

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

# ── configuration ─────────────────────────────────────────────────────────────
MODEL_NAME      = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
_MODEL_FILENAME = "stt_ar_fastconformer_hybrid_large_pcd_v1.0.nemo"
_server_dir     = os.path.dirname(os.path.abspath(__file__))
_local_model    = os.path.join(_server_dir, "..", "fine-tuning", "models", _MODEL_FILENAME)
MODEL_PATH      = os.getenv("NEMO_MODEL_PATH") or (
    _local_model if os.path.isfile(_local_model) else f"/app/{_MODEL_FILENAME}"
)
SAMPLE_RATE = 16000

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── app setup ─────────────────────────────────────────────────────────────────
app      = FastAPI(title="NeMo Alignment Server", version="1.0.0")
asr_model = None
_aligner  = None
_quran_db = None


def load_model():
    global asr_model, _aligner, _quran_db
    if asr_model is not None:
        return

    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict

    if os.path.isfile(MODEL_PATH):
        logger.info("Loading model from %s", MODEL_PATH)
        asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(MODEL_PATH)
    else:
        logger.info("Downloading %s", MODEL_NAME)
        asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(MODEL_NAME)

    # Disable CUDA graphs (broken on RTX 2080 Ti CC 7.5)
    with open_dict(asr_model.cfg):
        asr_model.cfg.decoding.greedy.use_cuda_graphs = False
        asr_model.cfg.decoding.greedy.use_cuda_graph_decoder = False
    asr_model.change_decoding_strategy(asr_model.cfg.decoding)
    asr_model.eval()
    asr_model = asr_model.cuda()

    # Build aligner and Quran DB (share model)
    import sys
    sys.path.insert(0, _server_dir)
    from ctc_aligner import CTCAligner
    from quran_db import QuranDB

    _aligner  = CTCAligner(asr_model)
    _quran_db = QuranDB()
    logger.info("Model + aligner ready. QuranDB: %d ayat", len(_quran_db))


@app.on_event("startup")
async def startup():
    load_model()


# ── /health ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model":  MODEL_NAME,
        "quran_db_size": len(_quran_db) if _quran_db else 0,
    }


# ── /transcribe (plain STT, same as nemo_stt_server) ─────────────────────────
@app.post("/transcribe")
async def transcribe(request: Request):
    if asr_model is None:
        load_model()
    body = await request.body()
    if not body or len(body) < 1000:
        raise HTTPException(400, "Audio too short")
    wav_path = None
    try:
        wav_path = _to_wav(body, request.headers.get("content-type", ""))
        output = asr_model.transcribe([wav_path])
        text = _extract_text(output)
        return JSONResponse({"text": text.strip(), "is_final": True})
    except Exception as e:
        logger.exception("transcribe error")
        raise HTTPException(500, str(e))
    finally:
        _cleanup(wav_path)


# ── /align (forced alignment) ─────────────────────────────────────────────────
@app.post("/align")
async def align(
    request:   Request,
    reference: str   = Query(None,  description="Diacritized Arabic reference text"),
    surah:     int   = Query(None,  description="Surah number (1-114)"),
    ayah:      int   = Query(None,  description="Ayah number"),
    threshold: float = Query(0.55,  description="Score threshold for pass/fail"),
):
    """
    Forced alignment of student audio against a reference ayah.

    Provide EITHER:
      ?reference=بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ
    OR:
      ?surah=1&ayah=1  (looks up from Quran DB)
    """
    if asr_model is None:
        load_model()

    # ── resolve reference text ────────────────────────────────────────────────
    if not reference:
        if surah is None or ayah is None:
            raise HTTPException(
                400,
                "Provide either 'reference' text or 'surah' + 'ayah' query params"
            )
        reference = _quran_db.get(surah, ayah)
        if not reference:
            raise HTTPException(
                404,
                f"Ayah {surah}:{ayah} not found in Quran DB"
            )

    # ── read audio ────────────────────────────────────────────────────────────
    body = await request.body()
    if not body or len(body) < 1000:
        raise HTTPException(400, "Audio too short (min ~1s)")

    wav_path = None
    try:
        wav_path = _to_wav(body, request.headers.get("content-type", ""))
        wav_np, sr = sf.read(wav_path, dtype="float32")
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(axis=1)   # stereo → mono
        if sr != SAMPLE_RATE:
            raise HTTPException(400, f"Audio must be 16kHz (got {sr}Hz)")

        # ── run alignment ─────────────────────────────────────────────────────
        result = _aligner.align(wav_np, reference)
        result["reference"] = reference
        if surah is not None:
            result["surah"] = surah
            result["ayah"]  = ayah

        # Override threshold if caller specified one
        if threshold != 0.55:
            from ctc_aligner import SCORE_THRESHOLD
            result["passed"] = (
                result["overall_score"] >= threshold and
                len([w for w in result["words"] if w["status"] == "error"]) == 0
            )

        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("align error")
        raise HTTPException(500, str(e))
    finally:
        _cleanup(wav_path)


# ── /align/surah — align against all ayat of a surah back-to-back ─────────────
@app.post("/align/batch")
async def align_batch(
    request: Request,
    surah:   int = Query(..., description="Surah number"),
    ayah:    int = Query(..., description="Ayah number"),
):
    """Convenience wrapper: same as /align?surah=X&ayah=Y."""
    return await align(request, reference=None, surah=surah, ayah=ayah)


# ── helpers ───────────────────────────────────────────────────────────────────
def _to_wav(body: bytes, content_type: str) -> str:
    """Write audio bytes to a temp WAV file, converting via ffmpeg if needed."""
    is_wav = "wav" in content_type or body[:4] == b"RIFF"
    is_mp3 = "mp3" in content_type or body[:3] == b"ID3" or body[:2] == b"\xff\xfb"

    if is_wav:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(body)
            return f.name
    elif is_mp3:
        return _ffmpeg_convert(body, ".mp3")
    else:
        # Assume raw PCM 16kHz mono 16-bit
        return _pcm_to_wav(body)


def _ffmpeg_convert(data: bytes, suffix: str) -> str:
    import ffmpeg
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(data)
        tmp = f.name
    out = tempfile.mktemp(suffix=".wav")
    try:
        (
            ffmpeg
            .input(tmp)
            .output(out, acodec="pcm_s16le", ac=1, ar=SAMPLE_RATE, loglevel="error")
            .run(overwrite_output=True)
        )
        return out
    finally:
        _cleanup(tmp)


def _pcm_to_wav(pcm: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return path


def _extract_text(output) -> str:
    if not output:
        return ""
    if isinstance(output, tuple):
        output = output[0]
    first = output[0] if output else None
    if hasattr(first, "text"):
        return first.text or ""
    return str(first) if first else ""


def _cleanup(path):
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("ALIGN_PORT", "3006"))
    host = os.getenv("ALIGN_HOST", "0.0.0.0")
    logger.info("Starting alignment server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)
