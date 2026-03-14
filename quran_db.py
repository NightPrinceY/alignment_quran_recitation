"""
quran_db.py
───────────
Loads the diacritized Quran reference text.

Primary source: quran_transcriptions_clean.json (Husary dataset, fully diacritized)
  Format: list of {"audio_filepath": "audio/001_001.wav", "text": "...", ...}
  Key:    "001001" (surah * 1000 + ayah, zero-padded 6 digits)

Usage:
    from quran_db import QuranDB
    db = QuranDB()
    text = db.get(surah=1, ayah=1)
    # → "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ"

    all_fatiha = db.get_surah(1)
    # → [(1, "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ"), (2, "الْحَمْدُ ..."), ...]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Path relative to this file
_HERE = Path(__file__).parent
_DEFAULT_JSON = _HERE.parent / "fine-tuning" / "quran_transcriptions_clean.json"


class QuranDB:
    """In-memory lookup for diacritized Quran text."""

    def __init__(self, json_path: Optional[Path] = None):
        path = Path(json_path) if json_path else _DEFAULT_JSON
        self._db: dict[str, str] = {}   # "001001" → diacritized text
        self._load(path)

    def _load(self, path: Path) -> None:
        if not path.exists():
            logger.warning("Quran DB not found at %s — DB will be empty", path)
            return

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            # Format: [{"audio_filepath": "audio/001_001.wav", "text": "..."}, ...]
            for entry in data:
                fp = Path(entry["audio_filepath"]).stem   # "001_001"
                key = fp.replace("_", "")                 # "001001"
                self._db[key] = entry["text"]
        elif isinstance(data, dict):
            # Format: {"001001": "text", ...}
            self._db = {k: v for k, v in data.items()}

        logger.info("QuranDB loaded %d ayat", len(self._db))

    def _key(self, surah: int, ayah: int) -> str:
        return f"{surah:03d}{ayah:03d}"

    def get(self, surah: int, ayah: int) -> Optional[str]:
        """Return diacritized text for surah:ayah, or None if not found."""
        return self._db.get(self._key(surah, ayah))

    def get_surah(self, surah: int) -> list[tuple[int, str]]:
        """Return all ayat of a surah as [(ayah_num, text), ...]."""
        results = []
        prefix = f"{surah:03d}"
        for key, text in sorted(self._db.items()):
            if key.startswith(prefix):
                ayah = int(key[3:])
                results.append((ayah, text))
        return results

    def search(self, text_fragment: str) -> list[tuple[int, int, str]]:
        """Search for ayat containing a text fragment. Returns [(surah, ayah, text)]."""
        results = []
        for key, text in self._db.items():
            if text_fragment in text:
                surah = int(key[:3])
                ayah  = int(key[3:])
                results.append((surah, ayah, text))
        return results

    def __len__(self) -> int:
        return len(self._db)

    def __contains__(self, item) -> bool:
        if isinstance(item, tuple) and len(item) == 2:
            return self._key(*item) in self._db
        return item in self._db


# Module-level singleton — load once, reuse everywhere
_instance: Optional[QuranDB] = None

def get_db() -> QuranDB:
    global _instance
    if _instance is None:
        _instance = QuranDB()
    return _instance
