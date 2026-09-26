"""
preprocess.py — Text normalization for business names and addresses.

Country-agnostic transformations that generalize across US, India, and France.
Handles: legal suffixes, address abbreviations, punctuation, casing, unicode.
"""

import re
import unicodedata
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from pathlib import Path

from src import config
from src.utils import log, timed, load_source, log_memory


# ─────────────────────────────────────────────────────────────
# NORMALIZATION MAPS
# ─────────────────────────────────────────────────────────────

# Legal suffix standardization (multi-language)
LEGAL_SUFFIXES = {
    # English
    r"\binc\b": "incorporated",
    r"\bincorp\b": "incorporated",
    r"\bincorporated\b": "incorporated",
    r"\bcorp\b": "corporation",
    r"\bcorporation\b": "corporation",
    r"\bllc\b": "limited liability company",
    r"\bllp\b": "limited liability partnership",
    r"\bltd\b": "limited",
    r"\blimited\b": "limited",
    r"\bco\b": "company",
    r"\bcompany\b": "company",
    r"\bplc\b": "public limited company",
    r"\bdba\b": "",  # "doing business as" — noise
    r"\bt/?a\b": "",  # "trading as" — noise
    # Indian
    r"\bpvt\b": "private",
    r"\bprivate\b": "private",
    r"\bngo\b": "non governmental organization",
    # French
    r"\bsarl\b": "societe a responsabilite limitee",
    r"\bsa\b": "societe anonyme",
    r"\bsas\b": "societe par actions simplifiee",
    r"\beurl\b": "entreprise unipersonnelle a responsabilite limitee",
    r"\bsci\b": "societe civile immobiliere",
    r"\bgmbh\b": "gesellschaft mit beschrankter haftung",
    r"\bag\b": "aktiengesellschaft",
    r"\bkg\b": "kommanditgesellschaft",
    r"\bev\b": "eingetragener verein",
}

# Address abbreviation standardization (multi-language)
ADDRESS_ABBREVS = {
    # English
    r"\bst\b": "street",
    r"\brd\b": "road",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bpkwy\b": "parkway",
    r"\bhwy\b": "highway",
    r"\bsq\b": "square",
    r"\bapt\b": "apartment",
    r"\bste\b": "suite",
    r"\bfl\b": "floor",
    r"\bbldg\b": "building",
    r"\bnr\b": "near",
    r"\bopp\b": "opposite",
    r"\badj\b": "adjacent",
    r"\bn\b": "north",
    r"\bs\b": "south",
    r"\be\b": "east",
    r"\bw\b": "west",
    r"\bnw\b": "northwest",
    r"\bne\b": "northeast",
    r"\bsw\b": "southwest",
    r"\bse\b": "southeast",
    # French
    r"\br\b": "rue",
    r"\bbd\b": "boulevard",
    r"\bav\b": "avenue",
    r"\bpl\b": "place",
    r"\bimp\b": "impasse",
    r"\bch\b": "chemin",
    r"\brt\b": "route",
    r"\ballée\b": "allee",
    # Indian
    r"\bnagar\b": "nagar",
    r"\bcolony\b": "colony",
    r"\btq\b": "taluk",
    r"\bdist\b": "district",
    r"\btaluka\b": "taluk",
    r"\bmarg\b": "marg",
    r"\bchowk\b": "chowk",
    r"\bgali\b": "gali",
    r"\bmohalla\b": "mohalla",
}

# US state abbreviations → full names
US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut", "de": "delaware",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah",
    "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
}

# Pre-compile regex patterns for speed
_LEGAL_COMPILED = [(re.compile(pat, re.IGNORECASE), repl)
                   for pat, repl in LEGAL_SUFFIXES.items()]
_ADDR_COMPILED = [(re.compile(pat, re.IGNORECASE), repl)
                  for pat, repl in ADDRESS_ABBREVS.items()]
_MULTI_SPACE = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_DIGITS = re.compile(r"\b(\d+)\b")
_POSTAL_US = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
_POSTAL_INDIA = re.compile(r"\b(\d{6})\b")
_POSTAL_FRANCE = re.compile(r"\b(\d{5})\b")


# ─────────────────────────────────────────────────────────────
# CORE NORMALIZATION FUNCTIONS
# ─────────────────────────────────────────────────────────────

def normalize_unicode(text: str) -> str:
    """Normalize unicode to ASCII-compatible form (NFD → strip accents → NFC)."""
    # Decompose, strip combining marks, recompose
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(c for c in nfkd if not unicodedata.combining(c))
    return ascii_text


def normalize_name(name: str) -> str:
    """
    Full normalization pipeline for business names.

    Steps:
    1. Unicode normalization (accents → base characters)
    2. Lowercase
    3. Replace '&' with 'and'
    4. Strip punctuation (keep alphanumeric + spaces)
    5. Standardize legal suffixes
    6. Collapse whitespace
    """
    if not isinstance(name, str) or not name.strip():
        return ""

    text = normalize_unicode(name)
    text = text.lower()
    text = text.replace("&", " and ")
    text = text.replace("@", " at ")
    text = _NON_ALNUM.sub(" ", text)

    for pattern, replacement in _LEGAL_COMPILED:
        text = pattern.sub(replacement, text)

    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


def normalize_address(address: str) -> str:
    """
    Full normalization pipeline for business addresses.

    Steps:
    1. Unicode normalization
    2. Lowercase
    3. Strip punctuation
    4. Standardize address abbreviations
    5. Collapse whitespace
    """
    if not isinstance(address, str) or not address.strip():
        return ""

    text = normalize_unicode(address)
    text = text.lower()
    text = text.replace("&", " and ")
    text = text.replace("#", " number ")
    text = _NON_ALNUM.sub(" ", text)

    for pattern, replacement in _ADDR_COMPILED:
        text = pattern.sub(replacement, text)

    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


# ─────────────────────────────────────────────────────────────
# COMPONENT EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_street_number(address: str) -> str:
    """Extract the first numeric token (likely street number)."""
    if not address:
        return ""
    match = _DIGITS.search(address)
    return match.group(1) if match else ""


def extract_postal_code(address: str, country: str = "") -> str:
    """
    Extract postal code based on country patterns.
    US: 5 digits (optionally +4)
    India: 6 digits
    France: 5 digits
    Falls back to any 5-6 digit sequence.
    """
    if not address:
        return ""

    country = country.lower().strip() if isinstance(country, str) else ""

    if country in ("us", "usa", "united states"):
        m = _POSTAL_US.search(address)
        if m:
            return m.group(1)
    elif country == "india":
        m = _POSTAL_INDIA.search(address)
        if m:
            return m.group(1)
    elif country == "france":
        m = _POSTAL_FRANCE.search(address)
        if m:
            return m.group(1)

    # Fallback: try 6-digit then 5-digit
    m = re.search(r"\b(\d{6})\b", address)
    if m:
        return m.group(1)
    m = re.search(r"\b(\d{5})\b", address)
    if m:
        return m.group(1)
    return ""


def extract_city_state(address: str, country: str = "") -> Tuple[str, str]:
    """
    Heuristic extraction of city and state from normalized address.

    Strategy: split on commas, look for known patterns.
    This is deliberately simple and robust — we extract the last 2-3
    comma-separated tokens as candidate city/state.
    """
    if not address:
        return "", ""

    # Split on commas first, then clean
    raw = address if isinstance(address, str) else ""
    # Normalize first for consistent parsing
    raw = normalize_unicode(raw).lower()
    raw = _NON_ALNUM.sub(" ", raw)
    raw = _MULTI_SPACE.sub(" ", raw).strip()

    parts = [p.strip() for p in raw.split() if p.strip()]

    # Simple heuristic: for addresses, city is often the second-to-last
    # non-numeric token, state is the last non-numeric token
    non_numeric = [p for p in parts if not p.isdigit()]

    city, state = "", ""
    if len(non_numeric) >= 2:
        # Check if last token is a state abbreviation
        last = non_numeric[-1]
        if last in US_STATES:
            state = US_STATES[last]
            city = non_numeric[-2] if len(non_numeric) >= 2 else ""
        else:
            city = non_numeric[-1]
    elif len(non_numeric) == 1:
        city = non_numeric[0]

    return city, state


def extract_numeric_tokens(text: str) -> set:
    """Extract all numeric tokens from text."""
    if not text:
        return set()
    return set(_DIGITS.findall(text))


# ─────────────────────────────────────────────────────────────
# FULL PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full normalization pipeline to a source DataFrame.

    Adds columns:
    - name_clean: normalized business name
    - addr_clean: normalized business address
    - name_addr:  concatenated name + address for embedding
    - street_num: extracted street number
    - postal:     extracted postal code
    - country_clean: lowercased country
    """
    df = df.copy()

    # Fill NaN
    df["business_name"] = df["business_name"].fillna("")
    df["business_address"] = df["business_address"].fillna("")
    df["country"] = df["country"].fillna("")

    # Core normalization
    log.info("  Normalizing names...")
    df["name_clean"] = df["business_name"].apply(normalize_name)

    log.info("  Normalizing addresses...")
    df["addr_clean"] = df["business_address"].apply(normalize_address)

    # Combined field for embeddings
    df["name_addr"] = df["name_clean"] + " " + df["addr_clean"]
    df["name_addr"] = df["name_addr"].str.strip()

    # Component extraction (ultra-fast list comprehension, 10x faster than df.apply)
    log.info("  Extracting components (street number, postal code, city, state)...")
    df["country_clean"] = df["country"].fillna("").astype(str).str.lower().str.strip()
    addr_clean_list = df["addr_clean"].fillna("").astype(str).tolist()
    country_clean_list = df["country_clean"].tolist()

    df["street_num"] = [extract_street_number(a) for a in addr_clean_list]
    df["postal"] = [extract_postal_code(a, c) for a, c in zip(addr_clean_list, country_clean_list)]

    city_state_tuples = [extract_city_state(a, c) for a, c in zip(addr_clean_list, country_clean_list)]
    df["city"] = [t[0] for t in city_state_tuples]
    df["state"] = [t[1] for t in city_state_tuples]
    del city_state_tuples, addr_clean_list, country_clean_list

    # Name tokens for blocking (significant words only)
    df["name_tokens"] = df["name_clean"].apply(
        lambda x: [t for t in x.split() if len(t) > 1]
    )

    # Address tokens for blocking
    df["addr_tokens"] = df["addr_clean"].apply(
        lambda x: [t for t in x.split() if len(t) > 1]
    )

    return df


@timed
def run_preprocessing(split: str = "train"):
    """
    Run full preprocessing on a data split and save to parquet.

    Args:
        split: "train" or "test"
    """
    if split == "train":
        paths = {"s1": config.TRAIN_S1, "s2": config.TRAIN_S2, "s3": config.TRAIN_S3}
    else:
        paths = {"s1": config.TEST_S1, "s2": config.TEST_S2, "s3": config.TEST_S3}

    for label, path in paths.items():
        out_path = config.PREPROCESSED_DIR / f"{split}_{label}.parquet"
        if out_path.exists():
            try:
                import pyarrow.parquet as pq
                schema = pq.read_schema(out_path)
                if "city" in schema.names and "state" in schema.names:
                    log.info(f"Preprocessed file {out_path.name} already exists and has city/state. Skipping.")
                    continue
                else:
                    log.info(f"Preprocessed file {out_path.name} missing city/state columns. Re-generating...")
            except Exception:
                pass

        log.info(f"Processing {split}_{label}...")
        df = load_source(path)
        df = preprocess_dataframe(df)

        out_path = config.PREPROCESSED_DIR / f"{split}_{label}.parquet"
        df.to_parquet(out_path, index=False)
        log.info(f"  Saved to {out_path.name} ({len(df):,} rows)")
        log_memory()

        del df  # free memory


def load_preprocessed(split: str, source: str) -> pd.DataFrame:
    """Load a preprocessed parquet file."""
    path = config.PREPROCESSED_DIR / f"{split}_{source}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessed file not found: {path}. Run preprocessing first."
        )
    return pd.read_parquet(path)


if __name__ == "__main__":
    run_preprocessing("train")
    run_preprocessing("test")
