"""
harakat.py
──────────
Harakat-level error analysis.

The ASR model was trained without diacritics, so its CTC output has near-zero
probability for diacritized tokens. However:

  1. The ENCODER is phonetically aware — Fatha (/a/), Kasra (/i/), Damma (/u/)
     are acoustically distinct vowels and ARE captured in encoder features.

  2. Even though absolute probabilities for diacritized tokens are small,
     RELATIVE comparisons (P(بِ) vs P(بَ) vs P(بُ)) still carry signal.

This module analyzes the CTC log-probs at aligned word segments to detect
harakat errors.

Approach:
  For each aligned word segment:
    - Get the encoder frames for that segment
    - For each Arabic letter in the word, identify its haraka
    - Compare P(correct_diacritized_token) vs P(alternate_diacritized_tokens)
    - If the correct haraka has lower relative probability → flag as error

Arabic Harakat Unicode:
  U+064E  FATHA         َ  /a/
  U+064F  DAMMA         ُ  /u/
  U+0650  KASRA         ِ  /i/
  U+0651  SHADDA        ّ  (gemination, not a vowel)
  U+0652  SUKUN         ْ  (no vowel / consonant close)
  U+064B  FATHATAN      ً  /an/
  U+064C  DAMMATAN      ٌ  /un/
  U+064D  KASRATAN      ٍ  /in/
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import torch

# Harakat character constants
FATHA    = '\u064e'
DAMMA    = '\u064f'
KASRA    = '\u0650'
SHADDA   = '\u0651'
SUKUN    = '\u0652'
FATHATAN = '\u064b'
DAMMATAN = '\u064c'
KASRATAN = '\u064d'

HARAKAT_NAMES = {
    FATHA:    "فتحة (Fatha /a/)",
    DAMMA:    "ضمة (Damma /u/)",
    KASRA:    "كسرة (Kasra /i/)",
    SHADDA:   "شدة (Shadda — gemination)",
    SUKUN:    "سكون (Sukun — no vowel)",
    FATHATAN: "تنوين فتح (Fathatan /an/)",
    DAMMATAN: "تنوين ضم (Dammatan /un/)",
    KASRATAN: "تنوين كسر (Kasratan /in/)",
}

# Phonetically similar harakat pairs — these are the most common student errors
COMMON_CONFUSIONS = [
    (FATHA, KASRA,  "said Fatha /a/ instead of Kasra /i/"),
    (KASRA, FATHA,  "said Kasra /i/ instead of Fatha /a/"),
    (FATHA, DAMMA,  "said Fatha /a/ instead of Damma /u/"),
    (DAMMA, FATHA,  "said Damma /u/ instead of Fatha /a/"),
    (KASRA, DAMMA,  "said Kasra /i/ instead of Damma /u/"),
    (DAMMA, KASRA,  "said Damma /u/ instead of Kasra /i/"),
]


@dataclass
class HarakatError:
    word: str                  # the full diacritized word
    letter: str                # the Arabic letter where error occurred
    expected: str              # correct haraka character
    expected_name: str         # human-readable name
    score_correct: float       # P(correct haraka token)
    score_best_alt: float      # P(best alternative haraka)
    best_alt: str              # most likely alternative haraka character
    best_alt_name: str         # human-readable alternative
    confidence: float          # how certain we are about this error (0-1)
    description: str           # teacher-friendly message


def analyze_word_harakat(
    word_diac: str,
    log_probs: torch.Tensor,   # (T_segment, V+1) — frames for this word
    tokenizer,
    blank_id: int = 1024,
) -> list[HarakatError]:
    """
    Compare log-probs for diacritized vs non-diacritized token variants
    within a word's aligned frames.

    For each letter+haraka pair, we:
    1. Tokenize the sub-string with correct haraka
    2. Tokenize the sub-string with each alternative haraka
    3. Compare their average CTC probabilities across the frames
    4. Flag if an alternative scores significantly higher

    Returns a list of HarakatError for detected issues.
    """
    errors: list[HarakatError] = []
    if log_probs.shape[0] == 0:
        return errors

    # Sum log-probs across frames for each token (overall segment score)
    # Shape: (V+1,)
    summed_lp = log_probs.mean(dim=0).cpu()

    # Extract letter-haraka pairs from the diacritized word
    pairs = _extract_letter_haraka_pairs(word_diac)

    for letter, haraka in pairs:
        if not haraka or haraka == SHADDA:
            continue   # no vowel or shadda only — skip

        # Build the 2-char string (letter + haraka) for the correct version
        correct_str = letter + haraka
        correct_ids = tokenizer.text_to_ids(correct_str)
        if not correct_ids:
            continue

        correct_id = correct_ids[0]
        score_correct = float(summed_lp[correct_id].item())

        # Compare against alternative harakat
        best_alt_score = score_correct
        best_alt_char  = haraka
        best_alt_name  = HARAKAT_NAMES.get(haraka, haraka)
        found_error    = False
        description    = ""

        for alt_haraka, _, desc_template in [(h, n, d) for src, h, d in COMMON_CONFUSIONS if src == haraka
                                              for n in [HARAKAT_NAMES.get(h, h)]]:
            alt_str = letter + alt_haraka
            alt_ids = tokenizer.text_to_ids(alt_str)
            if not alt_ids:
                continue
            alt_id = alt_ids[0]
            score_alt = float(summed_lp[alt_id].item())

            if score_alt > best_alt_score:
                best_alt_score = score_alt
                best_alt_char  = alt_haraka
                best_alt_name  = HARAKAT_NAMES.get(alt_haraka, alt_haraka)
                found_error    = True
                description    = desc_template

        if found_error:
            # Confidence = how much higher the alternative scored
            delta = best_alt_score - score_correct
            confidence = float(min(1.0, delta / 3.0))  # normalize: 3 log-units = 100%

            errors.append(HarakatError(
                word            = word_diac,
                letter          = letter,
                expected        = haraka,
                expected_name   = HARAKAT_NAMES.get(haraka, haraka),
                score_correct   = score_correct,
                score_best_alt  = best_alt_score,
                best_alt        = best_alt_char,
                best_alt_name   = best_alt_name,
                confidence      = round(confidence, 2),
                description     = description,
            ))

    return errors


def _extract_letter_haraka_pairs(word: str) -> list[tuple[str, str]]:
    """
    Parse diacritized Arabic word into (letter, haraka) pairs.
    E.g. "بِسْمِ" → [('ب', 'ِ'), ('س', 'ْ'), ('م', 'ِ')]
    """
    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(word):
        ch = word[i]
        # Is this an Arabic letter?
        if '\u0600' <= ch <= '\u06ff' and ch not in HARAKAT_NAMES:
            haraka = ""
            j = i + 1
            # Collect following diacritics
            while j < len(word) and word[j] in HARAKAT_NAMES:
                haraka += word[j]
                j += 1
            pairs.append((ch, haraka))
            i = j
        else:
            i += 1
    return pairs


def format_harakat_report(errors: list[HarakatError]) -> str:
    """Format errors as a teacher-friendly message in Arabic + English."""
    if not errors:
        return "✓ Harakat pronunciation appears correct."

    lines = ["Harakat errors detected:"]
    for e in errors:
        lines.append(
            f"  • Letter «{e.letter}» in «{e.word}»: "
            f"expected {e.expected_name} — {e.description} "
            f"(confidence: {e.confidence:.0%})"
        )
    return "\n".join(lines)
