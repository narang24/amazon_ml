"""
Business Entity Resolution Challenge — preprocessing module
============================================================

Dependencies: pandas only (stdlib for everything else) — no external lookups,
no geocoding, fully compliant with the "no external data" rule.

What it does?
------------
1. Loads every *.tsv safely (tab sep, all-string dtype, empty strings kept,
   no quote interpretation — addresses / ID lists contain commas and quotes).
2. Normalises text: Unicode NFKC -> accent stripping (é->e, needed for France),
   lower-case, '&' -> 'and', punctuation cleanup, whitespace collapse.
3. Business names:
     - splits DBA / trade names ("X dba Y", "X d/b/a Y", "X t/a Y", "X (Y)")
     - expands abbreviations (corp->corporation, pvt->private, intl->international …)
     - extracts & canonicalises legal suffixes (US, India, France, generic)
       into their own column and removes them from the "core" name
     - builds word-order-invariant and transliteration-tolerant keys
4. Addresses:
     - extracts postal codes (US ZIP / ZIP+4, India PIN incl. "160 062", FR CP)
     - extracts landmark phrases ("near SBI ATM", "opp. bus stand", "pres de …")
       and removes them from the structural address
     - normalises municipal numbering ("H.No. 12-3/45", "#221B", "Plot No 7")
     - expands street-type abbreviations (st, rd, ave, blvd, marg, nagar, bd, av …)
     - canonicalises US / Indian state names <-> codes
     - builds a transliteration-tolerant token key (bazaar/bazar, gali/galli …)
5. Country: normalised as an OPEN set of labels (aliases collapsed, unseen
   labels such as 'france' pass through untouched — nothing is filtered).
6. Ground truth: exploded into a long (s1_id, match_id) pair table + a
   group-aware train/validation split by Source-1 entity + an F0.5 scorer
   that mirrors the official macro-averaged metric (singletons included).

Usage
-----
    python preprocess.py --data-dir dataset --out-dir processed
    # or from Python:
    from preprocess import load_split
    s1, s2, s3 = load_split("dataset", "train")
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 1. SAFE LOADING
# ─────────────────────────────────────────────────────────────────────────────
EXPECTED_COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path: str | Path) -> pd.DataFrame:
    """Read a challenge TSV without any silent mangling."""
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,                 # IDs like S1-00001 / PIN codes with leading 0 stay strings
        keep_default_na=False,     # "NA", "null", "" stay as literal strings, not NaN
        na_values=[],
        quoting=csv.QUOTE_NONE,    # addresses may contain stray " characters
        encoding="utf-8",
        on_bad_lines="warn",
    )
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]  # BOM / stray spaces
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    if df.shape[1] == 1:
        raise ValueError(f"{path}: only one column — file was not tab-separated?")
    return df


def check_source(df: pd.DataFrame, prefix: str, name: str) -> pd.DataFrame:
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}; got {list(df.columns)}")
    bad_prefix = ~df["entity_id"].str.startswith(prefix)
    if bad_prefix.any():
        print(f"[warn] {name}: {bad_prefix.sum()} ids without prefix {prefix}")
    dups = df["entity_id"].duplicated(keep=False)
    if dups.any():
        print(f"[warn] {name}: {dups.sum()} duplicated entity_id rows — keeping first")
        df = df.drop_duplicates("entity_id", keep="first")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. GENERIC TEXT NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────
_WS = re.compile(r"\s+")


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    # Only remove combining characters in the standard Latin diacritical block (U+0300 - U+036F)
    # This preserves Indic vowel signs (matras) which are also combining characters.
    return "".join(ch for ch in s if not (unicodedata.combining(ch) and 0x0300 <= ord(ch) <= 0x036F))


def base_normalize(s: str) -> str:
    """Lower-case, de-accent, unify quotes/dashes, '&'->'and', collapse spaces."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = strip_accents(s).lower()
    s = (s.replace("’", "'").replace("‘", "'").replace("`", "'")
           .replace("–", "-").replace("—", "-"))
    s = re.sub(r"\s*&\s*", " and ", s)
    s = re.sub(r"\s*\+\s*", " and ", s)
    s = re.sub(r"(\w)'s\b", r"\1s", s)          # mcdonald's -> mcdonalds
    s = re.sub(r"\b((?:[a-z]\.){2,})", lambda m: m.group(1).replace(".", ""), s)  # s.c.o. -> sco
    s = s.replace("'", "")
    return _WS.sub(" ", s).strip()


def remove_punct(s: str, keep: str = "") -> str:
    """Replace punctuation with space; dotted acronyms collapse (p.v.t -> pvt)."""
    s = re.sub(r"\b((?:[a-z]\.){2,})", lambda m: m.group(1).replace(".", ""), s)
    s = re.sub(r"(?<=[a-z])\.(?=\s|$)", "", s)   # trailing dots: "ltd." -> "ltd"
    punct = r"!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~“”‘’…—–•®™"
    if keep:
        punct = re.sub(r"[" + re.escape(keep) + r"]", "", punct)
    s = re.sub(r"[" + punct + r"]", " ", s)
    return _WS.sub(" ", s).strip()


def apply_token_map(tokens: list[str], mapping: dict[str, str]) -> list[str]:
    out = []
    for t in tokens:
        out.extend(mapping.get(t, t).split())
    return out


def translit_token(t: str) -> str:
    """Cheap, deterministic transliteration folding for Indic/French spelling drift.
    bazaar/bazar, gali/galli, chowk/chauk, mohalla/mohala, sharma/sharmaa, ph->f …"""
    if t.isdigit() or len(t) <= 2:
        return t
    t = t.replace("ph", "f").replace("ck", "k").replace("q", "k")
    t = re.sub(r"(aa|ah\b)", "a", t)
    t = t.replace("ee", "i").replace("oo", "u").replace("ou", "u")
    t = t.replace("au", "o").replace("ow", "o")
    t = t.replace("th", "t").replace("dh", "d").replace("bh", "b")
    t = t.replace("kh", "k").replace("gh", "g").replace("sh", "s").replace("ch", "c")
    t = re.sub(r"y\b", "i", t)
    t = re.sub(r"(.)\1+", r"\1", t)             # collapse doubled letters
    return t


# ─────────────────────────────────────────────────────────────────────────────
# 3. BUSINESS NAME NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────
NAME_ABBR = {
    "corp": "corporation", "co": "company", "cos": "companies", "coop": "cooperative",
    "inc": "incorporated", "incorp": "incorporated",
    "ltd": "limited", "ltda": "limited", "lmt": "limited",
    "pvt": "private", "pte": "private", "prv": "private",
    "intl": "international", "int": "international", "natl": "national",
    "mfg": "manufacturing", "mfrs": "manufacturers", "mktg": "marketing",
    "svc": "services", "svcs": "services", "serv": "services", "srvs": "services",
    "tech": "technologies", "techs": "technologies", "technology": "technologies",
    "sys": "systems", "sol": "solutions", "solns": "solutions",
    "ent": "enterprises", "entp": "enterprises", "enterprise": "enterprises",
    "assoc": "associates", "assn": "association", "bros": "brothers",
    "grp": "group", "hldgs": "holdings", "hldg": "holding",
    "dept": "department", "ctr": "center", "centre": "center",
    "mgmt": "management", "mgt": "management", "dev": "development",
    "engg": "engineering", "eng": "engineering", "pharma": "pharmaceuticals",
    "indus": "industries", "ind": "industries", "inds": "industries",
    "univ": "university", "inst": "institute", "hosp": "hospital",
    "restr": "restaurant", "mkt": "market",
    "st": "saint", "mt": "mount", "ft": "fort",
    "n": "and", "et": "and", "und": "and",
    "cie": "compagnie", "ets": "etablissements", "ste": "societe", "soc": "societe",
}

# canonical legal forms (multi-word phrases matched on the token string, longest first)
LEGAL_SUFFIXES = {
    # US / generic
    "limited liability company": "llc", "l l c": "llc", "llc": "llc",
    "limited liability partnership": "llp", "llp": "llp",
    "limited partnership": "lp", "lp": "lp",
    "professional corporation": "pc", "pc": "pc", "pllc": "pllc",
    "incorporated": "inc", "corporation": "corp", "company": "co",
    "limited": "ltd", "plc": "plc", "public limited company": "plc",
    # India
    "private limited": "pvt ltd", "pvt limited": "pvt ltd", "private ltd": "pvt ltd",
    "pvt ltd": "pvt ltd", "opc private limited": "opc pvt ltd", "opc": "opc",
    "one person company": "opc",
    # France
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl",
    "snc": "snc", "sci": "sci", "sca": "sca", "scop": "scop",
    "societe anonyme": "sa", "societe par actions simplifiee": "sas",
    "societe a responsabilite limitee": "sarl",
    # other common
    "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv", "srl": "srl", "spa": "spa",
    "pty": "pty", "pty ltd": "pty ltd",
    # Hindi translations
    "प्राइवेट लिमिटेड": "pvt ltd", "प्राइवेट लिमिटेड कंपनी": "pvt ltd", "लिमिटेड": "ltd",
    "एलएलपी": "llp",
    # non-legal but name-noise words that often wrap the core name
    "and company": "co", "and co": "co", "and sons": "sons", "the": "",
}
# sort multi-word first so "private limited" wins over "limited"
_LEGAL_KEYS = sorted(LEGAL_SUFFIXES, key=lambda k: -len(k.split()))
_LEGAL_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in _LEGAL_KEYS) + r")\b")

# Only strip these when they appear at the *end* (or start for "the"), so that
# "Company Store Inc" keeps "company store".
_DBA_RE = re.compile(
    r"\s+(?:d\s*/\s*b\s*/\s*a|dba|d\.b\.a\.?|t\s*/\s*a|trading as|doing business as|"
    r"a\s*/\s*k\s*/\s*a|aka|formerly|fka|f/k/a)\s+",
    flags=re.I,
)
_PAREN_RE = re.compile(r"\(([^)]*)\)")

# generic words carrying little identity signal — used only for a "core" key
NAME_STOPWORDS = {
    "and", "of", "the", "de", "la", "le", "les", "du", "des", "et", "a", "an",
    "shri", "sri", "shree", "m/s", "ms", "messrs",
}


def split_dba(raw: str) -> tuple[str, str]:
    """'Acme Holdings LLC dba Acme Cafe' -> ('Acme Holdings LLC', 'Acme Cafe').
    Also treats a parenthetical as an alternate name: 'Foo Ltd (Bar Foods)'."""
    parts = _DBA_RE.split(raw, maxsplit=1)
    legal, dba = (parts[0], parts[1]) if len(parts) == 2 else (raw, "")
    if not dba:
        m = _PAREN_RE.search(legal)
        if m and len(m.group(1).split()) >= 1 and not _LEGAL_RE.fullmatch(
                base_normalize(m.group(1))):
            dba = m.group(1)
            legal = _PAREN_RE.sub(" ", legal)
    return legal.strip(), dba.strip()


def extract_legal(tokens_str: str) -> tuple[str, list[str]]:
    """Strip legal-form suffixes from the END of the name (repeatedly) and the
    leading 'the'. Returns (core_name, [canonical_legal_forms])."""
    s = tokens_str
    forms: list[str] = []
    s = re.sub(r"^the\s+", "", s)
    changed = True
    while changed and s:
        changed = False
        for k in _LEGAL_KEYS:
            if s == k:
                break  # never strip the whole name
            if s.endswith(" " + k):
                canon = LEGAL_SUFFIXES[k]
                if canon:
                    forms.insert(0, canon)
                s = s[: -len(k) - 1].strip()
                changed = True
                break
    return s, forms


def normalize_name(raw: str) -> dict:
    raw = re.sub(r"^\s*(?:m\s*/\s*s\.?|messrs\.?)\s+", "", raw or "", flags=re.I)  # M/s Foo
    legal_raw, dba_raw = split_dba(raw)

    def _clean(x: str) -> str:
        x = remove_punct(base_normalize(x), keep="")
        return " ".join(apply_token_map(x.split(), NAME_ABBR))

    full = _clean(legal_raw)
    dba = _clean(dba_raw)
    core, forms = extract_legal(full)
    dba_core, _ = extract_legal(dba) if dba else ("", [])

    core_tokens = [t for t in core.split() if t not in NAME_STOPWORDS] or core.split()
    return {
        "name_clean": full,                                  # full normalised name
        "name_core": " ".join(core_tokens),                  # legal suffix removed
        "name_legal_form": " ".join(forms),                  # e.g. "pvt ltd", "llc", "sarl"
        "name_dba": dba_core,                                # trade / alternate name
        "name_sorted": " ".join(sorted(core_tokens)),        # word-order invariant
        "name_nospace": "".join(core_tokens),                # "mc donalds" == "mcdonalds"
        "name_translit": " ".join(sorted(translit_token(t) for t in core_tokens)),
        "name_acronym": "".join(t[0] for t in core_tokens if t and not t.isdigit()),
        "name_first_token": core_tokens[0] if core_tokens else "",
        "name_n_tokens": len(core_tokens),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. ADDRESS NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────
ADDR_ABBR = {
    # English / US
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "avn": "avenue", "blvd": "boulevard", "bd": "boulevard", "bvd": "boulevard",
    "ln": "lane", "dr": "drive", "ct": "court", "pl": "place", "sq": "square",
    "hwy": "highway", "pkwy": "parkway", "expy": "expressway", "fwy": "freeway",
    "cir": "circle", "trl": "trail", "ter": "terrace", "terr": "terrace",
    "ste": "suite", "apt": "apartment", "bldg": "building", "blg": "building",
    "fl": "floor", "flr": "floor", "rm": "room", "pob": "po box",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "mt": "mount", "ft": "fort", "ctr": "center", "centre": "center",
    "jn": "junction", "jct": "junction", "xing": "crossing",
    # India
    "mg": "mahatma gandhi", "ngr": "nagar", "nr": "near", "opp": "opposite",
    "sec": "sector", "sect": "sector", "secto": "sector", "ph": "phase",
    "extn": "extension", "ext": "extension", "enclv": "enclave",
    "colny": "colony", "col": "colony", "mkt": "market", "bazaar": "bazar",
    "chowk": "chowk", "chk": "chowk", "cross": "cross", "crs": "cross",
    "main": "main", "mn": "main", "stn": "station", "rly": "railway",
    "hsg": "housing", "soc": "society", "apts": "apartments", "twr": "tower",
    "indl": "industrial", "ind": "industrial", "estt": "estate",
    "dist": "district", "distt": "district", "teh": "tehsil", "vill": "village",
    "vpo": "village post office", "ps": "police station",
    # France
    "r": "rue", "bld": "boulevard", "imp": "impasse",
    "rte": "route", "chem": "chemin", "fbg": "faubourg", "qu": "quai",
    "zi": "zone industrielle", "za": "zone artisanale", "zac": "zac",
    "cedex": "", "bp": "bp",
}

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont",
    "va": "virginia", "wa": "washington", "wv": "west virginia", "wi": "wisconsin",
    "wy": "wyoming", "dc": "district of columbia",
}
IN_STATES = {
    "ap": "andhra pradesh", "ar": "arunachal pradesh", "as": "assam", "br": "bihar",
    "cg": "chhattisgarh", "ct": "chhattisgarh", "ga": "goa", "gj": "gujarat",
    "hr": "haryana", "hp": "himachal pradesh", "jh": "jharkhand", "ka": "karnataka",
    "kl": "kerala", "mp": "madhya pradesh", "mh": "maharashtra", "mn": "manipur",
    "ml": "meghalaya", "mz": "mizoram", "nl": "nagaland", "od": "odisha", "or": "odisha",
    "orissa": "odisha", "pb": "punjab", "rj": "rajasthan", "sk": "sikkim",
    "tn": "tamil nadu", "ts": "telangana", "tg": "telangana", "tr": "tripura",
    "up": "uttar pradesh", "uk": "uttarakhand", "ut": "uttarakhand",
    "uttaranchal": "uttarakhand", "wb": "west bengal", "dl": "delhi",
    "nct": "delhi", "ch": "chandigarh", "jk": "jammu and kashmir", "la": "ladakh",
    "py": "puducherry", "pondicherry": "puducherry",
}
# common Indian city alternates (spelling / renamed cities) — in-data knowledge only
CITY_ALIASES = {
    "bombay": "mumbai", "bangalore": "bengaluru", "bengalooru": "bengaluru",
    "calcutta": "kolkata", "madras": "chennai", "gurgaon": "gurugram",
    "poona": "pune", "baroda": "vadodara", "trivandrum": "thiruvananthapuram",
    "cochin": "kochi", "mysore": "mysuru", "benares": "varanasi", "banaras": "varanasi",
    "new delhi": "delhi", "nyc": "new york",
}

CITY_STATES = {"chandigarh", "delhi", "puducherry", "ch", "dl", "dc"}

LANDMARK_RE = re.compile(
    r"\b(?:near(?:by)?|nr|opp(?:osite)?|behind|beside|besides|next to|adjacent to|"
    r"adj(?:acent)?|in front of|infront of|facing|across from|close to|"
    r"pres de|pres du|en face de|a cote de|derriere)\b"
    r"\.?\s+([^,;]+)",
    flags=re.I,
)

# postal codes: IN PIN "160062" / "160 062"; US ZIP "94103" / "94103-1234"; FR "75008"
PIN_RE = re.compile(r"\b([1-9]\d{2})\s?(\d{3})\b")
ZIP_RE = re.compile(r"\b(\d{5})(?:-(\d{4}))?\b")

UNIT_RE = re.compile(
    r"\b(?:h[.\s]*no|house[.\s]*no|h[.\s]*#|plot[.\s]*no|plot|shop[.\s]*no|shop|"
    r"flat[.\s]*no|flat|door[.\s]*no|d[.\s]*no|survey[.\s]*no|sy[.\s]*no|"
    r"khasra[.\s]*no|scf|sco|unit|suite|ste|apt|"
    r"no|number|num|#)(?![a-z])\s*[.:#-]?\s*",
    flags=re.I,
)


def extract_postcode(s: str, country: str) -> tuple[str, tuple[int, int] | None]:
    """Country-aware but open-set. Takes the LAST match (house numbers such as
    '12345 Main St' come first, the postcode comes at the end).
    India -> 6-digit PIN; everything else -> 5-digit, falling back to 6-digit."""
    order = [PIN_RE] if country == "india" else [ZIP_RE, PIN_RE]
    for rx in order:
        # a 5/6-digit number at the very start is a house number (12345 Main St)
        hits = [m for m in rx.finditer(s) if s[: m.start()].strip(" ,#")]
        if hits:
            m = hits[-1]
            code = m.group(1) + m.group(2) if rx is PIN_RE else m.group(1)
            return code, m.span()
    return "", None


def canon_state_tokens(tokens: list[str], country: str) -> list[str]:
    """Expand state codes only when they are plausibly the state (last ~4 tokens),
    so 'in' / 'or' / 'me' inside a street aren't rewritten."""
    table = IN_STATES if country == "india" else US_STATES if country == "us" else {}
    if not table:
        return tokens
    n = len(tokens)
    out = []
    for i, t in enumerate(tokens):
        if i >= n - 4 and t in table:
            out.extend(table[t].split())
        else:
            out.append(t)
    return out


def normalize_address(raw: str, country: str) -> dict:
    s = base_normalize(raw or "")

    # 1) landmarks ("near SBI ATM") -> separate column, removed from structure
    landmarks = [remove_punct(m.group(1)) for m in LANDMARK_RE.finditer(s)]
    s_struct = LANDMARK_RE.sub(" ", s)

    # 2) postal code, then remove it from the structural string
    postcode, span = extract_postcode(s_struct, country)
    if span:
        s_struct = s_struct[: span[0]] + " " + s_struct[span[1]:]

    # 3) municipal numbering: "h.no. 12-3/45" -> "12 3 45"; "#221b" -> "221b"
    s_struct = UNIT_RE.sub(" ", s_struct)
    s_struct = re.sub(r"(\d)\s*[/\\-]\s*(\d)", r"\1 \2", s_struct)
    s_struct = re.sub(r"\b(\d+)(st|nd|rd|th)\b", r"\1", s_struct)     # 5th -> 5
    s_struct = re.sub(r"(\d)([a-z])\b", r"\1 \2", s_struct)           # 221b -> 221 b

    # comma segments (before punctuation removal) — the tail usually holds city/state
    segments = [remove_punct(x) for x in s_struct.split(",") if remove_punct(x)]

    tokens = remove_punct(s_struct).split()
    # 'st' = 'saint' when it starts a name (St Louis, 12 St Marks Pl, St Germain),
    # otherwise 'street' (Main St, Main St Suite 5)
    fixed = []
    for i, t in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        prv = tokens[i - 1] if i else ""
        if t in ("st", "ste") and nxt.isalpha() and nxt not in ADDR_ABBR \
                and (not prv or prv.isdigit()):
            fixed.append("saint" if t == "st" else "sainte")
        else:
            fixed.append(t)
    tokens = apply_token_map(fixed, ADDR_ABBR)
    tokens = canon_state_tokens(tokens, country)
    joined = " ".join(tokens)
    for k, v in CITY_ALIASES.items():
        joined = re.sub(rf"\b{re.escape(k)}\b", v, joined)
    tokens = [t for i, t in enumerate(joined.split())
              if t and (i == 0 or t != joined.split()[i - 1])]   # drop "delhi delhi"

    numbers = [t for t in tokens if t.isdigit()]
    words = [t for t in tokens if not t.isdigit()]
    # city = last comma segment, stepping back one if that segment is only a state
    # (city-states like Chandigarh / Delhi are kept as the city)
    city_guess = segments[-1] if len(segments) >= 2 else ""
    last = segments[-1] if segments else ""
    is_state = (last in US_STATES or last in IN_STATES or last in US_STATES.values()
                or last in IN_STATES.values()) and last not in CITY_STATES
    if is_state:
        city_guess = segments[-2] if len(segments) >= 3 else ""
    city_guess = " ".join(canon_state_tokens(
        apply_token_map(city_guess.split(), ADDR_ABBR), country))
    city_guess = CITY_ALIASES.get(city_guess, city_guess)

    return {
        "addr_clean": joined,
        "addr_sorted": " ".join(sorted(set(tokens))),
        "addr_words": " ".join(words),
        "addr_numbers": " ".join(numbers),            # house / plot / sector numbers
        "addr_translit": " ".join(sorted({translit_token(t) for t in words})),
        "addr_postcode": postcode,
        "addr_landmark": " | ".join(landmarks),
        "addr_city_guess": city_guess,
        "addr_n_tokens": len(tokens),
        "addr_missing": int(len(tokens) == 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. COUNTRY (open set — never filter, never one-hot to a fixed list)
# ─────────────────────────────────────────────────────────────────────────────
COUNTRY_ALIASES = {
    "us": "us", "usa": "us", "u s": "us", "u s a": "us", "united states": "us",
    "united states of america": "us", "america": "us",
    "india": "india", "in": "india", "ind": "india", "bharat": "india",
    "republic of india": "india",
    "france": "france", "fr": "france", "fra": "france", "republique francaise": "france",
}


def normalize_country(raw: str) -> str:
    c = remove_punct(base_normalize(raw or ""))
    return COUNTRY_ALIASES.get(c, c)          # unseen labels pass through unchanged


# ─────────────────────────────────────────────────────────────────────────────
# 6. FULL SOURCE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_source(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["source"] = df["entity_id"].str.split("-").str[0]            # S1/S2/S3
    df["country_norm"] = df["country"].map(normalize_country)

    name_feats = pd.DataFrame([normalize_name(x) for x in df["business_name"]],
                              index=df.index)
    addr_feats = pd.DataFrame(
        [normalize_address(a, c) for a, c in zip(df["business_address"], df["country_norm"])],
        index=df.index,
    )
    out = pd.concat([df, name_feats, addr_feats], axis=1)

    # handy blocking keys (use several in union for recall)
    out["bk_name_prefix"] = out["name_nospace"].str[:4]
    out["bk_country_postcode"] = out["country_norm"] + "|" + out["addr_postcode"]
    out["bk_country_city"] = out["country_norm"] + "|" + out["addr_city_guess"]
    out["bk_name_translit"] = out["country_norm"] + "|" + out["name_translit"]
    out["name_missing"] = (out["name_core"] == "").astype(int)
    return out


def load_split(data_dir: str | Path, split: str = "train"):
    d = Path(data_dir) / split
    frames = []
    for i in (1, 2, 3):
        raw = read_tsv(d / f"{split}_source{i}.tsv")
        raw = check_source(raw, f"S{i}-", f"{split}_source{i}")
        frames.append(preprocess_source(raw))
    return tuple(frames)


# ─────────────────────────────────────────────────────────────────────────────
# 7. GROUND TRUTH → PAIRS, VALIDATION SPLIT, OFFICIAL-STYLE METRIC
# ─────────────────────────────────────────────────────────────────────────────
def parse_id_list(x: str) -> list[str]:
    if not x:
        return []
    seen, out = set(), []
    for t in x.split(","):
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def load_ground_truth(path: str | Path) -> dict[str, list[str]]:
    gt = read_tsv(path)
    if "matched_entity_ids" not in gt.columns:           # trailing empty col edge case
        gt["matched_entity_ids"] = ""
    return {r.source1_entity_id: parse_id_list(r.matched_entity_ids)
            for r in gt.itertuples(index=False)}


def gt_to_pairs(gt: dict[str, list[str]]) -> pd.DataFrame:
    rows = [(s1, m, m.split("-")[0]) for s1, ms in gt.items() for m in ms]
    return pd.DataFrame(rows, columns=["s1_id", "match_id", "match_source"])


def split_s1_ids(s1_ids, val_frac: float = 0.2, seed: int = 42, strat=None):
    """Group-aware split: every candidate pair of an S1 entity lands on one side.
    `strat` (optional Series indexed by S1 id, e.g. country|has_match) keeps
    the country / singleton mix balanced across train and val."""
    ids = pd.Series(sorted(set(s1_ids)))
    if strat is None:
        val = ids.sample(frac=val_frac, random_state=seed)
    else:
        key = ids.map(strat).fillna("na")
        val = ids.groupby(key, group_keys=False).apply(
            lambda g: g.sample(frac=val_frac, random_state=seed))
    val_set = set(val)
    return [i for i in ids if i not in val_set], sorted(val_set)


def f05_macro(pred: dict[str, list[str]], gt: dict[str, list[str]]) -> float:
    """Exactly mirrors the brief: per-S1 F0.5, singletons score 1/0, macro mean."""
    scores = []
    for s1, truth in gt.items():
        p, t = set(pred.get(s1, [])), set(truth)
        if not t:
            scores.append(1.0 if not p else 0.0)
            continue
        if not p:
            scores.append(0.0)
            continue
        tp = len(p & t)
        prec, rec = tp / len(p), tp / len(t)
        scores.append(0.0 if tp == 0 else 1.25 * prec * rec / (0.25 * prec + rec))
    return sum(scores) / len(scores) if scores else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 8. QUICK DATA PROFILE (run this first on the real data)
# ─────────────────────────────────────────────────────────────────────────────
def profile(df: pd.DataFrame, name: str) -> None:
    print(f"\n=== {name}: {len(df):,} rows ===")
    print("country:", df["country_norm"].value_counts().to_dict())
    print("empty name:", int((df["business_name"] == "").sum()),
          "| empty address:", int((df["business_address"] == "").sum()))
    print("postcode found: {:.1%}".format((df["addr_postcode"] != "").mean()))
    print("has landmark:   {:.1%}".format((df["addr_landmark"] != "").mean()))
    print("has DBA name:   {:.1%}".format((df["name_dba"] != "").mean()))
    print("legal forms:", df["name_legal_form"].value_counts().head(8).to_dict())
    print(df[["entity_id", "business_name", "name_core", "addr_clean",
              "addr_postcode"]].head(5).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--out-dir", default="processed")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        try:
            srcs = load_split(args.data_dir, split)
        except FileNotFoundError as e:
            print(f"[skip] {split}: {e}")
            continue
        for i, df in enumerate(srcs, 1):
            profile(df, f"{split}_source{i}")
            if _has_parquet():
                df.to_parquet(out / f"{split}_source{i}.parquet", index=False)
            else:
                df.to_csv(out / f"{split}_source{i}.tsv", sep="\t", index=False)

        if split == "train":
            gt = load_ground_truth(Path(args.data_dir) / "train" / "train_ground_truth.tsv")
            pairs = gt_to_pairs(gt)
            s1 = srcs[0].set_index("entity_id")
            n_single = sum(1 for v in gt.values() if not v)
            print(f"\nGT: {len(gt):,} S1 entities | {len(pairs):,} positive pairs | "
                  f"{n_single:,} singletons ({n_single / max(len(gt), 1):.1%})")
            print("matches per S1:", pd.Series({k: len(v) for k, v in gt.items()})
                  .value_counts().sort_index().to_dict())
            known = set(srcs[1]["entity_id"]) | set(srcs[2]["entity_id"])
            orphan = pairs.loc[~pairs["match_id"].isin(known)]
            if len(orphan):
                print(f"[warn] {len(orphan)} GT ids not present in S2/S3 files")
            strat = (s1["country_norm"] + "|" +
                     pd.Series({k: str(min(len(v), 2)) for k, v in gt.items()}))
            tr, va = split_s1_ids(list(gt), 0.2, 42, strat)
            pairs.to_csv(out / "train_positive_pairs.tsv", sep="\t", index=False)
            pd.Series(tr, name="s1_id").to_csv(out / "split_train_ids.tsv", index=False)
            pd.Series(va, name="s1_id").to_csv(out / "split_val_ids.tsv", index=False)
            print(f"split: {len(tr):,} train / {len(va):,} val S1 entities")
            print(f"sanity — empty predictions F0.5 on val = "
                  f"{f05_macro({}, {k: gt[k] for k in va}):.3f} (singleton share)")
    print(f"\nWrote processed files to {out.resolve()}")


def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()