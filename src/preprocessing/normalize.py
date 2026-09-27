"""
normalize.py — Phase 2: Normalisation Engine
Applies priority-ordered noise corrections to business_name and business_address.

Usage:
    from src.preprocessing.normalize import Normalizer
    norm = Normalizer()
    df = norm.normalize_dataframe(df)
"""

import re
import unicodedata
from functools import lru_cache
from src.config import NORM


class Normalizer:
    """
    Applies the full normalisation pipeline derived from Phase 1 EDA.
    Produces multiple representations per record as specified in the docx:
        - original_name / original_address     (raw input — preserved)
        - normalized_name / normalized_address (canonical cleaned form)
        - alphanumeric_name                    (only alphanum + spaces)
        - tokenized_name                       (whitespace-split list)
        - name_without_legal_suffix            (suffix stripped)
        - address_tokens                       (cleaned token list)
        - numbers                              (numeric tokens from address)
    """

    def __init__(self):
        self.legal_map  = NORM["legal_suffix_expansions"]
        self.addr_map   = NORM["address_abbreviation_expansions"]

        # Build regex patterns once
        self._legal_re  = self._build_word_re(self.legal_map)
        self._addr_re   = self._build_word_re(self.addr_map)
        self._amp_re    = re.compile(r"\s*&\s*")
        self._ws_re     = re.compile(r"\s+")
        self._punct_re  = re.compile(r"[^\w\s\-]")   # keep hyphens & digits

    # ── Public API ─────────────────────────────────────────

    def normalize_dataframe(self, df):
        """
        Add normalised representation columns to df in-place.
        df must have: business_name, business_address, country
        """
        import pandas as pd
        df = df.copy()

        # Fill nulls
        df["business_name"]    = df["business_name"].fillna("").astype(str)
        df["business_address"] = df["business_address"].fillna("").astype(str)

        # Name representations (N01→N06 in priority order)
        df["original_name"]           = df["business_name"]
        df["normalized_name"]         = df.apply(
            lambda r: self.normalize_name(r["business_name"], r["country"]), axis=1)
        df["alphanumeric_name"]        = df["normalized_name"].apply(self._alphanumeric)
        df["tokenized_name"]           = df["normalized_name"].str.split()
        df["name_without_legal_suffix"]= df["normalized_name"].apply(self._strip_legal_suffix)

        # Address representations (N07→N09)
        df["original_address"]   = df["business_address"]
        df["normalized_address"] = df.apply(
            lambda r: self.normalize_address(r["business_address"], r["country"]), axis=1)
        df["address_tokens"]     = df["normalized_address"].str.split()
        df["numbers"]            = df["business_address"].apply(self._extract_numbers)

        return df

    @lru_cache(maxsize=128_000)
    def normalize_name(self, name: str, country: str = "") -> str:
        """
        Full name normalisation pipeline (cached per unique value).
        N04 → N01 → N05 → N02 → N03 → N06
        """
        if not name or not name.strip():
            return ""
        text = name
        text = self._transliterate(text)       # N04: non-Latin → Latin
        text = text.lower()                     # N01: lowercase
        text = self._strip_accents(text)        # N05: é→e, ç→c
        text = self._expand_legal(text)         # N02: pvt→private
        text = self._amp_re.sub(" and ", text)  # N03: & → and
        text = self._clean(text)                # N06: strip punct, collapse ws
        return text

    @lru_cache(maxsize=128_000)
    def normalize_address(self, address: str, country: str = "") -> str:
        """
        Full address normalisation pipeline (cached per unique value).
        N01 → N05 → N07 → N06
        """
        if not address or not address.strip():
            return ""
        text = address
        text = text.lower()                     # N01: lowercase
        text = self._strip_accents(text)        # N05: diacritics
        text = self._expand_address(text)       # N07: rd→road, dist→district
        text = self._clean(text)                # N06: strip punct, collapse ws
        return text

    # ── Private helpers ────────────────────────────────────

    def _transliterate(self, text: str) -> str:
        """
        Transliterate Devanagari / non-Latin to Latin.
        Requires: pip install anyascii   (fallback to unidecode if available)
        Falls back to unicodedata NFKD if neither is installed.
        """
        try:
            from anyascii import anyascii
            return anyascii(text)
        except ImportError:
            pass
        try:
            from unidecode import unidecode
            return unidecode(text)
        except ImportError:
            pass
        # NFKD fallback — handles accents but not Devanagari
        return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()

    def _strip_accents(self, text: str) -> str:
        """é→e, ç→c, ô→o etc. — critical for French records."""
        nfkd = unicodedata.normalize("NFKD", text)
        return "".join(c for c in nfkd if not unicodedata.combining(c))

    def _expand_legal(self, text: str) -> str:
        return self._legal_re.sub(
            lambda m: self.legal_map.get(m.group(0).rstrip("."), m.group(0)), text)

    def _expand_address(self, text: str) -> str:
        return self._addr_re.sub(
            lambda m: self.addr_map.get(m.group(0).rstrip("."), m.group(0)), text)

    def _clean(self, text: str) -> str:
        text = self._punct_re.sub(" ", text)
        text = self._ws_re.sub(" ", text)
        return text.strip()

    def _alphanumeric(self, text: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", text).strip()

    def _strip_legal_suffix(self, text: str) -> str:
        """Remove trailing legal suffix tokens."""
        legal_tokens = set(self.legal_map.values()) | set(self.legal_map.keys())
        tokens = text.split()
        while tokens and tokens[-1] in legal_tokens:
            tokens.pop()
        return " ".join(tokens)

    def _extract_numbers(self, text: str) -> list:
        """Extract all numeric tokens from address (house numbers, PINs)."""
        return re.findall(r"\b\d+\b", str(text))

    @staticmethod
    def _build_word_re(mapping: dict) -> re.Pattern:
        """Build a word-boundary regex that matches any key in the mapping."""
        keys = sorted(mapping.keys(), key=len, reverse=True)
        escaped = [re.escape(k) + r"\." + "?" for k in keys]
        pattern = r"\b(?:" + "|".join(escaped) + r")\b"
        return re.compile(pattern, re.IGNORECASE)
