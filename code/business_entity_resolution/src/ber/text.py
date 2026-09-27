"""Text normalisation, Indic transliteration, learned native->Latin dictionary, address parsing helpers.

Normalisation levels (compared in 01_normalization.py):
  L0  lowercase + whitespace collapse only
  L1  + NFKC, URL/phone/'#123' cleanup, '&'->and, accent folding (Latin only), punctuation split,
        digit/letter split ('1287b' -> '1287 b'), leading zeros dropped ('0137' -> '137'),
        single-letter runs joined ('l l c' -> 'llc'), repeated tokens removed ('llc llc')
  L2  + Indic scripts transliterated to Latin (anyascii by default, 'sanscript' mode optional)
  L3  + learned dictionary: native token -> most likely Latin token (learned from disjoint train pairs),
        applied before the rule-based transliteration fallback
  L4  + canonical forms: legal suffixes (private->pvt, limited->ltd ...), street types (road->rd ...),
        US / Indian state names -> codes (per country), a few city aliases, address marker words dropped
"""
import json
import re
import unicodedata
from collections import Counter, defaultdict

try:
    from anyascii import anyascii
    HAVE_ANY = True
except ImportError:
    HAVE_ANY = False
try:
    from indic_transliteration import sanscript
    HAVE_SAN = True
except ImportError:
    HAVE_SAN = False

INDIC_CHAR = re.compile(r"[ऀ-ൿ]")
INDIC_RUN = re.compile(r"[ऀ-ൿ‌‍]+")
ZW = re.compile(r"[​-‏﻿]")
BLOCKS = [(0x0900, "DEVANAGARI"), (0x0980, "BENGALI"), (0x0A00, "GURMUKHI"), (0x0A80, "GUJARATI"),
          (0x0B00, "ORIYA"), (0x0B80, "TAMIL"), (0x0C00, "TELUGU"), (0x0C80, "KANNADA"), (0x0D00, "MALAYALAM")]

URL_RE = re.compile(r"\b(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:com|in|net|org|co|io|biz|info|fr|us)\b")
PHONE_RE = re.compile(r"\b\d{8,}\b")
HASHNUM_RE = re.compile(r"#\s*\d+")
NULL_RE = re.compile(r"<\s*null\s*>|\bnull\b")
NONWORD = re.compile(r"[^0-9a-zऀ-ൿ]+")
NUMSIGN = re.compile(r"\bn\s*[\u00b0\u00ba]")
SPLIT_AN = re.compile(r"(?<=\d)(?=(?!(?:st|nd|rd|th)\b)[a-z])|(?<=[a-z])(?=\d)")


def has_indic(s):
    return INDIC_CHAR.search(s) is not None


def fold(s):
    """Strip accents from non-Indic characters only (NFKD would damage Indic viramas/nuktas)."""
    if s.isascii():
        return s
    out = []
    for ch in s:
        if "ऀ" <= ch <= "ൿ":
            out.append(ch)
        else:
            out.append("".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c)))
    return "".join(out)


def translit_run(run, mode="anyascii"):
    run = ZW.sub("", run)
    if not run:
        return ""
    if mode == "sanscript" and HAVE_SAN:
        cp = ord(run[0])
        scheme = None
        for start, name in reversed(BLOCKS):
            if cp >= start:
                scheme = name
                break
        try:
            return fold(sanscript.transliterate(run, getattr(sanscript, scheme), sanscript.IAST))
        except Exception:
            pass
    if HAVE_ANY:
        return anyascii(run)
    return run


# ----------------------------------------------------------------------------- canonical maps (L4)
NAME_MAP = {
    "private": "pvt", "pvt": "pvt", "limited": "ltd", "ltd": "ltd", "incorporated": "inc", "corporation": "corp",
    "company": "co", "centre": "center", "services": "service", "brothers": "bros", "enterprises": "enterprise",
    "industries": "industry", "technologies": "technology", "societe": "ste", "compagnie": "cie",
    "etablissements": "ets", "associates": "assoc", "and": "and", "groupe": "group", "et": "and",
}
LEGAL_SKEL = {"prvt": "pvt", "lmtd": "ltd", "lmtt": "ltd", "lmt": "ltd"}
ADDR_MAP = {
    "street": "st", "str": "st", "saint": "st", "road": "rd", "drive": "dr", "avenue": "ave", "av": "ave",
    "avenu": "ave", "lane": "ln", "boulevard": "blvd", "bd": "blvd", "court": "ct", "place": "pl",
    "highway": "hwy", "parkway": "pkwy", "circle": "cir", "apartment": "apt", "suite": "ste", "floor": "fl",
    "flr": "fl", "flat": "flt", "north": "n", "south": "s", "east": "e", "west": "w", "first": "1st",
    "ist": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "chemin": "ch", "route": "rte",
    "impasse": "imp", "faubourg": "fbg", "sainte": "ste", "r": "rue",
    # city aliases (address only)
    "bengaluru": "bangalore", "mysuru": "mysore", "gurugram": "gurgaon", "calcutta": "kolkata",
    "madras": "chennai", "bombay": "mumbai",
}
ADDR_DROP = {"no", "nos", "plot", "door", "house", "hno", "shop", "null", "cedex", "number"}
# France only (seen in the test data): street-type spellings that still disagree after the generic map, house-number
# suffixes (bis/ter/letter), 'N 31' number markers and articles -- 'Rue de la Paix' vs 'R. Paix'
FR_ADDR_MAP = {"allee": "all", "cours": "crs", "quai": "q", "residence": "res", "appt": "apt", "appartement": "apt",
               "chem": "ch", "square": "sq", "passage": "pass", "fg": "fbg", "lotissement": "lot", "hameau": "ham",
               "lieudit": "ld", "promenade": "prom", "esplanade": "esp", "carrefour": "carr", "traverse": "trav"}
FR_ADDR_DROP = {"bis", "ter", "n", "a", "b", "c", "d", "l", "de", "la", "le", "les", "du", "des", "et", "en"}
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "chattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh",
    "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od",
    "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn", "telangana": "tg",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk", "west bengal": "wb",
    "delhi": "dl", "jammu and kashmir": "jk", "ladakh": "la", "chandigarh": "ch", "puducherry": "py",
    "pondicherry": "py", "andaman and nicobar islands": "an", "lakshadweep": "ld",
    "dadra and nagar haveli": "dn", "daman and diu": "dd",
    "ts": "tg", "or": "od", "ct": "cg", "ut": "uk",
}
FR_REGIONS = {  # regions and their departements -> one region code (records use either)
    "hauts de france": "hdf", "nord": "hdf", "pas de calais": "hdf", "somme": "hdf", "aisne": "hdf", "oise": "hdf",
    "nouvelle aquitaine": "naq", "gironde": "naq", "landes": "naq", "pyrenees atlantiques": "naq",
    "dordogne": "naq", "lot et garonne": "naq", "charente": "naq", "charente maritime": "naq", "vienne": "naq",
    "deux sevres": "naq", "haute vienne": "naq", "creuse": "naq", "correze": "naq",
    "pays de la loire": "pdl", "loire atlantique": "pdl", "maine et loire": "pdl", "mayenne": "pdl",
    "sarthe": "pdl", "vendee": "pdl",
    "ile de france": "idf", "paris": "idf", "auvergne rhone alpes": "ara", "rhone": "ara",
    "provence alpes cote d azur": "paca", "provence alpes cote dazur": "paca", "bouches du rhone": "paca",
    "occitanie": "occ", "haute garonne": "occ", "grand est": "ges", "bretagne": "bre", "normandie": "nor",
    "bourgogne franche comte": "bfc", "centre val de loire": "cvl", "corse": "cor",
}
STATE_MAPS = {"US": US_STATES, "India": IN_STATES, "France": FR_REGIONS}
_PHRASE_RE = {}


def _phrase_re(country):
    if country not in _PHRASE_RE:
        m = STATE_MAPS.get(country, {})
        ph = sorted((k for k in m if " " in k), key=len, reverse=True)
        _PHRASE_RE[country] = re.compile(r"\b(" + "|".join(map(re.escape, ph)) + r")\b") if ph else None
    return _PHRASE_RE[country]


GENERIC_NAME = {
    "pvt", "ltd", "llc", "llp", "lp", "inc", "corp", "co", "pc", "pllc", "plc", "the", "and", "of", "group",
    "service", "center", "partners", "holdings", "com", "smt", "dr", "mr", "mrs", "ms", "shri", "sarl", "sas",
    "sa", "eurl", "sasu", "sci", "snc", "ste", "cie", "et", "assoc", "public", "ets", "www", "de", "la", "le",
    "les", "du", "des", "france", "india", "usa",
    # more French legal forms (EI is the 20th most common last name token in the French test records)
    "ei", "eirl", "scp", "selarl", "selas", "scm", "gie", "scop", "sca", "sem", "sarlu",
}

PHON = [("ph", "f"), ("bh", "b"), ("dh", "d"), ("th", "t"), ("kh", "k"), ("gh", "g"), ("sh", "s"), ("ch", "c"),
        ("jh", "j"), ("ck", "k"), ("q", "k"), ("x", "ks"), ("w", "v"), ("z", "j"), ("c", "k")]
VOWELS = re.compile(r"[aeiouy]")
REPEAT = re.compile(r"(.)\1+")


def skel(tok):
    """Consonant skeleton: phonetic merges, vowels dropped (except the first letter), repeats collapsed.
    'private'/'praivet'/'piraivet' -> 'prvt';  'aditya'/'adity' -> 'adty'."""
    if not tok or tok.isdigit():
        return tok
    t = tok
    for a, b in PHON:
        t = t.replace(a, b)
    return REPEAT.sub(r"\1", t[0] + VOWELS.sub("", t[1:]))


def _join_singles(toks):
    out, buf = [], []
    for t in toks:
        if len(t) == 1 and t.isalpha() and t.isascii():
            buf.append(t)
            continue
        if buf:
            out.append("".join(buf) if len(buf) > 1 else buf[0])
            buf = []
        out.append(t)
    if buf:
        out.append("".join(buf) if len(buf) > 1 else buf[0])
    return out


def _canon(toks, field, country):
    s = " ".join(toks)
    if field == "addr":
        rx = _phrase_re(country)
        if rx is not None:
            s = rx.sub(lambda m: STATE_MAPS[country][m.group(0)], s)
        sm = STATE_MAPS.get(country, {})
        fr = country == "France"
        out = []
        for t in s.split():
            if t in ADDR_DROP or (fr and t in FR_ADDR_DROP):
                continue
            t = sm.get(t, ADDR_MAP.get(t, t))
            if fr:
                t = FR_ADDR_MAP.get(t, t)
            out.append(t)
        return out
    out = []
    for t in s.split():
        t2 = NAME_MAP.get(t)
        if t2 is None and len(t) >= 5 and t.isalpha():
            t2 = LEGAL_SKEL.get(skel(t))
        out.append(t2 or t)
    return out


def normalize(s, level=4, field="name", country=None, dct=None, mode="anyascii"):
    if not s:
        return ""
    s = ZW.sub("", s).lower()
    if level <= 0:
        return " ".join(s.split())
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("&", " and ").replace("'", "").replace("’", "")
    s = NUMSIGN.sub(" no ", s)
    s = URL_RE.sub(r" \1 ", s)
    if field == "name":
        s = PHONE_RE.sub(" ", s)
        s = HASHNUM_RE.sub(" ", s)
    s = NULL_RE.sub(" ", s)
    if level >= 2 and has_indic(s):
        d1 = d2 = None
        if level >= 3 and dct:
            d1 = dct.get(field, {})
            d2 = dct.get("addr" if field == "name" else "name", {})

        def repl(m):
            run = ZW.sub("", m.group(0))
            t = (d1.get(run) or d2.get(run)) if d1 is not None else None
            return " " + (t if t else translit_run(run, mode)) + " "

        s = INDIC_RUN.sub(repl, s).lower()
    s = fold(s)
    s = SPLIT_AN.sub(" ", s)
    toks = [t for t in NONWORD.split(s) if t]
    toks = [(t.lstrip("0") or "0") if t.isdigit() else t for t in toks]
    toks = _join_singles(toks)
    toks = [t for i, t in enumerate(toks) if i == 0 or t != toks[i - 1]]
    if level >= 4:
        toks = _canon(toks, field, country)
    return " ".join(toks)


# ----------------------------------------------------------------------------- address helpers
NUM_RE = re.compile(r"\d+")
ZIP_RE = re.compile(r"(?<!\d)(\d{5,6})(?!\d)")


def numbers(addr_raw):
    """All numbers in the address, leading zeros stripped. '1287-B' -> {1287}; '674-678' -> {674, 678}."""
    return frozenset((x.lstrip("0") or "0") for x in NUM_RE.findall(addr_raw))


def first_number(addr_raw):
    m = NUM_RE.search(addr_raw)
    return (m.group(0).lstrip("0") or "0") if m else ""


def zips(addr_raw):
    s = re.sub(r"(?<!\d)(\d{3}) (\d{3})(?!\d)", r"\1\2", addr_raw)
    return frozenset(ZIP_RE.findall(s))


def components(addr_raw, level, country, dct=None):
    out = []
    for c in addr_raw.split(","):
        n = normalize(c, level, "addr", country, dct)
        if n and n not in out:
            out.append(n)
    return out


# ----------------------------------------------------------------------------- learned dictionary
def indic_runs(s):
    return [r for r in (ZW.sub("", x) for x in INDIC_RUN.findall(s)) if r]


def learn_dict(q_names, c_names, q_addrs, c_addrs, min_count=3, min_prob=0.5, log=print):
    """Learn native-script token -> Latin token maps from TRUE pairs (S1 Latin text vs matched native text).

    names: positional alignment when both token lists have equal length (native names are word-by-word
           transliterations), falling back to lift-based association.
    addr : lift-based association  P(l|n) * log(P(l|n) / P(l))  -- picks e.g. native 'maharashtra' over
           generic co-occurring words such as 'road'.
    """
    out = {}
    for field, qs, cs in (("name", q_names, c_names), ("addr", q_addrs, c_addrs)):
        cn, cl, cnl = Counter(), Counter(), Counter()
        pos_n, pos_nl = Counter(), Counter()
        n_pairs = 0
        for q, c in zip(qs, cs):
            if not c or not has_indic(c):
                continue
            n_pairs += 1
            qt = normalize(q, 1, field).split()
            ct = normalize(c, 1, field).split()
            if field == "name" and len(qt) == len(ct):
                for a, b in zip(ct, qt):
                    if has_indic(a) and not has_indic(b) and not b.isdigit():
                        pos_n[a] += 1
                        pos_nl[(a, b)] += 1
            nat = {t for t in ct if has_indic(t)}
            lat = {t for t in qt if not t.isdigit() and not has_indic(t)}
            if field == "addr":  # state names are often two words ('tamil nadu', 'west bengal')
                lat |= {f"{x} {y}" for x, y in zip(qt, qt[1:]) if x.isalpha() and y.isalpha()}
            cn.update(nat)
            cl.update(lat)
            for a in nat:
                for b in lat:
                    cnl[(a, b)] += 1
        best = {}
        for (a, b), c in pos_nl.items():
            if c >= min_count and c / pos_n[a] >= min_prob and (a not in best or c > best[a][1]):
                best[a] = (b, c)
        mapping = {a: b for a, (b, _) in best.items()}
        assoc = {}
        for (a, b), c in cnl.items():
            if a in mapping or c < min_count:
                continue
            p = c / cn[a]
            base = cl[b] / max(n_pairs, 1)
            if p < min_prob or p <= base:
                continue
            score = (p * (p / base if base > 0 else 1e9), b.count(" "))  # ties -> prefer the longer phrase
            if a not in assoc or score > assoc[a][1]:
                assoc[a] = (b, score)
        for a, (b, _) in assoc.items():
            mapping[a] = b
        out[field] = mapping
        log(f"  learned {field} dictionary: {len(mapping):,} native tokens "
            f"({len(best):,} positional, {len(assoc):,} association) from {n_pairs:,} pairs")
    return out


def save_dict(d, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)


def load_dict(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def normalize_series(values, countries, level, field, dct=None, mode="anyascii"):
    """Normalise a column with a cache on (value, country) -- names repeat a lot."""
    cache = {}
    out = []
    for v, c in zip(values, countries):
        key = (v, c)
        r = cache.get(key)
        if r is None:
            r = normalize(v, level, field, c, dct, mode)
            cache[key] = r
        out.append(r)
    return out
