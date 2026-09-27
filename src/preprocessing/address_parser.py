"""
address_parser.py — Structured address component extraction.
Extracts: house_number, street, city, state, postal_code, country_hint
"""

import re


# ── US patterns ────────────────────────────────────────────
_US_STATE_ABBREVS = {
    "al","ak","az","ar","ca","co","ct","de","fl","ga","hi","id","il","in",
    "ia","ks","ky","la","me","md","ma","mi","mn","ms","mo","mt","ne","nv",
    "nh","nj","nm","ny","nc","nd","oh","ok","or","pa","ri","sc","sd","tn",
    "tx","ut","vt","va","wa","wv","wi","wy","dc",
}

_US_ZIP_RE    = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
_US_HOUSE_RE  = re.compile(r"^\s*(\d+[a-zA-Z]?)\s+")

# ── India patterns ─────────────────────────────────────────
_IN_PIN_RE   = re.compile(r"\b([1-9]\d{5})\b")
_IN_STATE_ABBREVS = {
    "ap","ar","as","br","cg","ga","gj","hr","hp","jk","jh","ka","kl",
    "mp","mh","mn","ml","mz","nl","od","pb","rj","sk","tn","tg","tr",
    "up","uk","wb","an","ch","dn","dd","dl","ld","py",
}

# ── France patterns ────────────────────────────────────────
_FR_POSTAL_RE = re.compile(r"\b(\d{5})\b")


def parse_address(address: str, country: str) -> dict:
    """
    Parse a normalised address string into components.
    Returns a dict with keys:
        house_number, street, city, state, postal_code, raw
    """
    result = {
        "house_number": None,
        "street":       None,
        "city":         None,
        "state":        None,
        "postal_code":  None,
        "raw":          address,
    }

    if not address:
        return result

    country = (country or "").upper().strip()

    if country == "US":
        _parse_us(address, result)
    elif country == "INDIA":
        _parse_india(address, result)
    elif country == "FRANCE":
        _parse_france(address, result)
    else:
        # Generic: just extract numbers
        result["postal_code"] = _generic_postal(address)

    return result


def _parse_us(addr: str, r: dict):
    # ZIP
    m = _US_ZIP_RE.search(addr)
    if m:
        r["postal_code"] = m.group(1)

    # House number at start
    m = _US_HOUSE_RE.match(addr)
    if m:
        r["house_number"] = m.group(1)

    # State abbreviation (last 2-letter token before ZIP)
    tokens = addr.lower().replace(",", " ").split()
    for tok in reversed(tokens):
        if tok in _US_STATE_ABBREVS:
            r["state"] = tok.upper()
            break


def _parse_india(addr: str, r: dict):
    # PIN code
    m = _IN_PIN_RE.search(addr)
    if m:
        r["postal_code"] = m.group(1)

    # State abbreviation
    tokens = addr.lower().replace(",", " ").split()
    for tok in reversed(tokens):
        if tok in _IN_STATE_ABBREVS:
            r["state"] = tok.upper()
            break

    # House/plot number
    m = re.search(r"\b(?:no|plot|flat|h\.?no|house)\s*[.#]?\s*(\d+[\w\/\-]*)", addr, re.I)
    if m:
        r["house_number"] = m.group(1)


def _parse_france(addr: str, r: dict):
    # Postal code (5-digit)
    m = _FR_POSTAL_RE.search(addr)
    if m:
        r["postal_code"] = m.group(1)

    # House number
    m = _US_HOUSE_RE.match(addr)
    if m:
        r["house_number"] = m.group(1)

    # City: token after postal code if exists
    if r["postal_code"]:
        after = addr[addr.index(r["postal_code"]) + 5:].strip().strip(",").strip()
        city_tok = after.split(",")[0].strip()
        if city_tok:
            r["city"] = city_tok


def _generic_postal(addr: str) -> str | None:
    m = re.search(r"\b\d{5,6}\b", addr)
    return m.group(0) if m else None
