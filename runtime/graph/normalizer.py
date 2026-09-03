"""
AS Code — Graph Normalizer

Deterministic text and label normalization utility for the Graph Layer.
Provides consistent canonical forms for matching while preserving the original labels.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Tuple


def strip_accents(text: str) -> str:
    """Strip combining diacritical marks (e.g. 'Pérez' -> 'Perez')."""
    nfkd_form = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd_form if not unicodedata.combining(c))


def normalize_label(label: str) -> str:
    """
    Produce a canonical lowercase normalized string:
      - NFC unicode normalization
      - Lowercase
      - Strip leading/trailing whitespace and punctuation
      - Collapse multiple spaces into one
    """
    if not label:
        return ""
    normalized = unicodedata.normalize("NFC", label)
    # Strip common leading/trailing quotes and punctuation
    cleaned = normalized.strip().strip("\"'«»()[]{}.,;:").strip()
    return " ".join(cleaned.lower().split())


def normalize_key(label: str) -> str:
    """
    Produce an accent-free, lowercase key for secondary fuzzy / match comparison:
      - Normalizes whitespace
      - Lowercases
      - Strips accents/diacritics
      - Strips common corporate/legal suffixes (S.A., S.R.L., Inc, etc.)
    """
    canonical = normalize_label(label)
    unaccented = strip_accents(canonical)
    
    # Remove dots and punctuation within legal suffixes only at the end of the string
    cleaned = re.sub(
        r'(?:\s+)(?:s\.?\s*a\.?|s\.?\s*r\.?\s*l\.?|ltd\.?|inc\.?|llc|sociedad\s+anonima|gmbh)\b\.?$',
        '',
        unaccented,
        flags=re.IGNORECASE
    ).strip()
    
    return " ".join(cleaned.split())
