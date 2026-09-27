"""
Business Entity Resolution
Module: Normalization

Purpose
-------
Create robust normalized representations of business names,
addresses and countries before candidate generation.

Design principles
-----------------
1. Raw input data is never modified.
2. Country is treated as an open-set string.
3. Business-name normalization handles legal suffixes and
   "&" / "and" variations.
4. Address normalization preserves useful numbers and postal
   information.
5. Postal extraction supports:
      - US ZIP
      - US ZIP+4
      - India 6-digit PIN
      - France 5-digit postal codes
      - Canadian-style postal codes
6. Locality extraction is conservative.
7. Soundex is used only as an auxiliary signal.
8. No external APIs, databases, registries or internet data.
9. The module is deterministic and suitable for large-scale
   downstream blocking.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List

import pandas as pd


# ============================================================
# REGULAR EXPRESSIONS
# ============================================================

WHITESPACE_RE = re.compile(r"\s+")

PUNCTUATION_RE = re.compile(
    r"[^\w\s]",
    flags=re.UNICODE,
)

# Supports:
#
#   411001       -> India PIN
#   400001       -> India PIN
#   75001        -> France postal code
#   10001        -> US ZIP
#   10001-1234   -> US ZIP+4
#
# IMPORTANT:
# The 6-digit alternative comes first so a 6-digit India PIN
# cannot be partially interpreted as a 5-digit postal code.
NUMERIC_POSTAL_RE = re.compile(
    r"(?<!\d)(\d{6}|\d{5}-\d{4}|\d{5})(?!\d)"
)

# Canadian postal-code format:
#
# M5V 3A8
# M5V-3A8
# M5V3A8
#
CANADIAN_POSTAL_RE = re.compile(
    r"(?<![A-Z0-9])"
    r"([A-Z]\d[A-Z][ -]?\d[A-Z]\d)"
    r"(?![A-Z0-9])",
    flags=re.IGNORECASE,
)

# Handles:
#
#   Acme & Co.
#   Acme & Company
#   Acme and Co.
#   Acme and Company
#
# ONLY when this occurs at the end.
TRAILING_COMPANY_RE = re.compile(
    r"""
    \s*
    (?:
        &
        |
        \band\b
    )
    \s*
    (?:
        co
        |
        company
    )
    \.?
    \s*$
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


# ============================================================
# CONSTANTS
# ============================================================

LEGAL_SUFFIXES = frozenset(
    {
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "co",
        "company",
        "ltd",
        "limited",
        "llc",
        "llp",
        "plc",
        "pvt",
        "private",
        "pte",
        "gmbh",
        "ag",
        "sa",
        "sarl",
        "sas",
        "bv",
        "nv",
        "oy",
        "ab",
        "as",
        "spa",
        "srl",
        "kg",
        "lp",
        "holdings",
    }
)


ADDRESS_ABBREVIATIONS = {
    "street": "st",
    "st.": "st",
    "road": "rd",
    "rd.": "rd",
    "avenue": "ave",
    "ave.": "ave",
    "boulevard": "blvd",
    "blvd.": "blvd",
    "drive": "dr",
    "dr.": "dr",
    "lane": "ln",
    "ln.": "ln",
    "parkway": "pkwy",
    "pkwy.": "pkwy",
    "highway": "hwy",
    "hwy.": "hwy",
    "court": "ct",
    "ct.": "ct",
    "place": "pl",
    "pl.": "pl",
    "terrace": "ter",
    "trail": "trl",
    "north": "n",
    "south": "s",
    "east": "e",
    "west": "w",
    "northeast": "ne",
    "northwest": "nw",
    "southeast": "se",
    "southwest": "sw",
}


LOCALITY_STOPWORDS = frozenset(
    {
        "street",
        "st",
        "road",
        "rd",
        "avenue",
        "ave",
        "boulevard",
        "blvd",
        "drive",
        "dr",
        "lane",
        "ln",
        "highway",
        "hwy",
        "parkway",
        "pkwy",
        "place",
        "pl",
        "court",
        "ct",
        "unit",
        "apt",
        "apartment",
        "suite",
        "ste",
        "floor",
        "fl",
        "building",
        "bldg",
    }
)


# Used only to prevent state codes such as "NY" from becoming
# the locality. This does NOT restrict supported countries.
US_STATE_CODES = frozenset(
    {
        "al",
        "ak",
        "az",
        "ar",
        "ca",
        "co",
        "ct",
        "de",
        "fl",
        "ga",
        "hi",
        "id",
        "il",
        "in",
        "ia",
        "ks",
        "ky",
        "la",
        "me",
        "md",
        "ma",
        "mi",
        "mn",
        "ms",
        "mo",
        "mt",
        "ne",
        "nv",
        "nh",
        "nj",
        "nm",
        "ny",
        "nc",
        "nd",
        "oh",
        "ok",
        "or",
        "pa",
        "ri",
        "sc",
        "sd",
        "tn",
        "tx",
        "ut",
        "vt",
        "va",
        "wa",
        "wv",
        "wi",
        "wy",
        "dc",
    }
)


# ============================================================
# BASIC TEXT NORMALIZATION
# ============================================================

def unicode_normalize(value: object) -> str:
    """
    Apply Unicode NFKC normalization.

    None and NaN are represented as an empty string.
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return unicodedata.normalize(
        "NFKC",
        str(value),
    )


def normalize_text(value: object) -> str:
    """
    General-purpose normalization.

    Steps:
        1. Unicode NFKC
        2. lowercase
        3. & -> and
        4. punctuation -> whitespace
        5. whitespace collapse
    """

    text = unicode_normalize(value)

    if not text:
        return ""

    text = text.lower()

    # Canonical representation of ampersand.
    text = text.replace("&", " and ")

    # Replace punctuation with spaces.
    text = PUNCTUATION_RE.sub(" ", text)

    # Collapse whitespace.
    text = WHITESPACE_RE.sub(" ", text)

    return text.strip()


def tokenize(value: object) -> List[str]:
    """
    Tokenize normalized text.
    """

    text = normalize_text(value)

    if not text:
        return []

    return text.split()


# ============================================================
# BUSINESS NAME NORMALIZATION
# ============================================================

def _remove_trailing_legal_suffixes(
    tokens: List[str],
) -> List[str]:
    """
    Remove one or more legal suffixes from the end.

    Example:

        ["abc", "private", "limited"]

        becomes:

        ["abc"]
    """

    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()

    return tokens


def normalize_business_name(value: object) -> str:
    """
    Normalize a business name.

    Important cases:

        Acme Corporation
            -> acme

        Acme Corp.
            -> acme

        Acme & Co.
            -> acme

        Procter & Gamble
            -> procter and gamble

        Rock and Roll
            -> rock and roll
    """

    raw = unicode_normalize(value)

    if not raw:
        return ""

    raw = raw.lower().strip()

    # Remove trailing "& Co"/"and Co" constructions before
    # converting ampersand to "and".
    raw = TRAILING_COMPANY_RE.sub("", raw)

    # General normalization.
    text = normalize_text(raw)

    if not text:
        return ""

    tokens = text.split()

    # Remove legal suffix chain.
    tokens = _remove_trailing_legal_suffixes(tokens)

    return " ".join(tokens)


def remove_trailing_legal_suffixes(
    value: object,
) -> str:
    """
    Remove legal suffixes from a normalized representation.
    """

    text = normalize_text(value)

    if not text:
        return ""

    tokens = text.split()

    tokens = _remove_trailing_legal_suffixes(tokens)

    return " ".join(tokens)


# ============================================================
# ADDRESS NORMALIZATION
# ============================================================

def normalize_address_tokens(
    tokens: List[str],
) -> List[str]:
    """
    Apply conservative address abbreviation mapping.
    """

    return [
        ADDRESS_ABBREVIATIONS.get(
            token,
            token,
        )
        for token in tokens
        if token
    ]


def normalize_address(value: object) -> str:
    """
    Normalize an address.

    Numbers are retained because they can carry strong
    entity-resolution information.

    Example:

        10 Main Street, New York, NY 10001

        becomes:

        10 main st new york ny 10001
    """

    text = unicode_normalize(value)

    if not text:
        return ""

    text = text.lower()

    # Canonical ampersand.
    text = text.replace("&", " and ")

    # Punctuation -> spaces.
    text = PUNCTUATION_RE.sub(" ", text)

    # Collapse whitespace.
    text = WHITESPACE_RE.sub(
        " ",
        text,
    ).strip()

    if not text:
        return ""

    tokens = text.split()

    tokens = normalize_address_tokens(tokens)

    return " ".join(tokens)


# ============================================================
# COUNTRY NORMALIZATION
# ============================================================

def normalize_country(value: object) -> str:
    """
    Normalize country as an OPEN-SET string.

    We intentionally do not restrict this to:
        US
        India

    Therefore:
        France
        Germany
        Canada
        etc.

    remain valid values.
    """

    return normalize_text(value)


# ============================================================
# POSTAL CODE EXTRACTION
# ============================================================

def extract_postal_code(value: object) -> str:
    """
    Extract a postal code from an address.

    Supported formats:

        US:
            10001
            10001-1234

        India:
            411001

        France:
            75001

        Canada:
            M5V 3A8
    """

    text = unicode_normalize(value)

    if not text:
        return ""

    # --------------------------------------------------------
    # Numeric postal code
    # --------------------------------------------------------
    match = NUMERIC_POSTAL_RE.search(text)

    if match:
        return match.group(1).lower()

    # --------------------------------------------------------
    # Canadian-style postal code
    # --------------------------------------------------------
    match = CANADIAN_POSTAL_RE.search(text)

    if match:
        return (
            match.group(1)
            .replace(" ", "")
            .replace("-", "")
            .lower()
        )

    return ""


# ============================================================
# LOCALITY EXTRACTION
# ============================================================

def _clean_locality_segment(
    segment: str,
) -> str:
    """
    Clean one possible locality component.
    """

    text = normalize_text(segment)

    if not text:
        return ""

    # Remove postal code.
    text = NUMERIC_POSTAL_RE.sub(
        " ",
        text,
    )

    # Remove Canadian postal code.
    text = CANADIAN_POSTAL_RE.sub(
        " ",
        text,
    )

    text = WHITESPACE_RE.sub(
        " ",
        text,
    )

    return text.strip()


def _is_state_like(
    value: str,
) -> bool:
    """
    Detect a state/administrative-code-like segment.
    """

    tokens = value.split()

    if not tokens:
        return True

    if len(tokens) == 1:

        token = tokens[0]

        if token in US_STATE_CODES:
            return True

        # Short alphabetic region codes.
        if len(token) <= 3 and token.isalpha():
            return True

    return False


def _looks_like_street(
    value: str,
) -> bool:
    """
    Detect whether a segment resembles a street address.
    """

    tokens = value.split()

    if not tokens:
        return False

    # Street terminology.
    if any(
        token in LOCALITY_STOPWORDS
        for token in tokens
    ):
        return True

    # Presence of digits usually means address/building
    # information rather than a locality.
    if any(
        any(char.isdigit() for char in token)
        for token in tokens
    ):
        return True

    return False


def extract_locality(
    value: object,
) -> str:
    """
    Extract a conservative locality/city-like component.

    Examples:

        10 Main Street, New York, NY 10001
            -> new york

        123 Main Road, Pune 411001
            -> pune

        123 MG Road, Mumbai, Maharashtra 400001
            -> mumbai

        1600 Pennsylvania Ave NW, Washington, DC 20500
            -> washington
    """

    raw = unicode_normalize(value)

    if not raw:
        return ""

    # --------------------------------------------------------
    # Address components separated by commas/semicolons.
    # --------------------------------------------------------
    segments = [
        segment.strip()
        for segment in re.split(
            r"[,;]",
            raw,
        )
        if segment.strip()
    ]

    postal_segment_index = None

    for index, segment in enumerate(segments):

        if extract_postal_code(segment):

            postal_segment_index = index
            break

    # --------------------------------------------------------
    # Postal-containing segment exists.
    # --------------------------------------------------------
    if postal_segment_index is not None:

        # Example:
        #
        # 10 Main Street
        # New York
        # NY 10001
        #
        # -> New York
        if postal_segment_index > 0:

            candidate = _clean_locality_segment(
                segments[
                    postal_segment_index - 1
                ]
            )

            if (
                candidate
                and not _is_state_like(candidate)
                and not _looks_like_street(candidate)
            ):
                return candidate

        # Example:
        #
        # Pune 411001
        #
        # -> Pune
        candidate = _clean_locality_segment(
            segments[postal_segment_index]
        )

        if (
            candidate
            and not _is_state_like(candidate)
            and not _looks_like_street(candidate)
        ):
            return candidate

    # --------------------------------------------------------
    # No postal code, but multiple address segments exist.
    # --------------------------------------------------------
    if len(segments) >= 2:

        for segment in reversed(segments):

            candidate = _clean_locality_segment(
                segment
            )

            if not candidate:
                continue

            if _is_state_like(candidate):
                continue

            if _looks_like_street(candidate):
                continue

            return candidate

    # --------------------------------------------------------
    # Non-comma address.
    #
    # Example:
    #
    # 123 Main Street Pune 411001
    # --------------------------------------------------------
    normalized = normalize_text(raw)

    if not normalized:
        return ""

    postal = extract_postal_code(raw)

    if postal:

        normalized_without_postal = normalized.replace(
            postal.lower(),
            " ",
        )

        normalized_without_postal = WHITESPACE_RE.sub(
            " ",
            normalized_without_postal,
        ).strip()

        tokens = normalized_without_postal.split()

        for token in reversed(tokens):

            if not token:
                continue

            if token in LOCALITY_STOPWORDS:
                continue

            if token in US_STATE_CODES:
                continue

            if token.isdigit():
                continue

            if len(token) <= 1:
                continue

            return token

    return ""


# ============================================================
# SOUNDEX
# ============================================================

def soundex(
    value: object,
) -> str:
    """
    Lightweight Soundex implementation.

    This is an auxiliary blocking/feature signal only.
    It must never be treated as a definitive match.
    """

    text = normalize_text(value)

    if not text:
        return ""

    # Keep Latin alphabetic characters for this phonetic signal.
    letters = [
        char
        for char in text
        if "a" <= char <= "z"
    ]

    if not letters:
        return ""

    first = letters[0]

    mapping = {
        **dict.fromkeys(
            "bfpv",
            "1",
        ),
        **dict.fromkeys(
            "cgjkqsxz",
            "2",
        ),
        **dict.fromkeys(
            "dt",
            "3",
        ),
        **dict.fromkeys(
            "l",
            "4",
        ),
        **dict.fromkeys(
            "mn",
            "5",
        ),
        **dict.fromkeys(
            "r",
            "6",
        ),
    }

    result = [
        first.upper()
    ]

    previous = mapping.get(
        first,
        "",
    )

    for char in letters[1:]:

        code = mapping.get(
            char,
            "",
        )

        # Vowels / h / w / y reset duplicate suppression.
        if not code:
            previous = ""
            continue

        if code != previous:
            result.append(code)

        previous = code

        if len(result) == 4:
            break

    while len(result) < 4:
        result.append("0")

    return "".join(result)


# ============================================================
# SINGLE RECORD NORMALIZATION
# ============================================================

def normalize_record(
    business_name: object,
    business_address: object,
    country: object,
) -> Dict[str, str]:
    """
    Normalize one business record.
    """

    name_norm = normalize_business_name(
        business_name
    )

    address_norm = normalize_address(
        business_address
    )

    return {
        "business_name_norm": name_norm,
        "business_address_norm": address_norm,
        "country_norm": normalize_country(
            country
        ),
        "postal_code": extract_postal_code(
            business_address
        ),
        "locality": extract_locality(
            business_address
        ),
        "name_soundex": soundex(
            name_norm
        ),
    }


# ============================================================
# DATAFRAME NORMALIZATION
# ============================================================

def normalize_dataframe(
    df: pd.DataFrame,
    include_phonetic: bool = True,
) -> pd.DataFrame:
    """
    Normalize a DataFrame.

    Original columns remain untouched.

    Added columns:

        business_name_norm
        business_address_norm
        country_norm
        postal_code
        locality
        name_soundex
    """

    required_columns = {
        "business_name",
        "business_address",
        "country",
    }

    missing = required_columns.difference(
        df.columns
    )

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    result = df.copy()

    # --------------------------------------------------------
    # Business name
    # --------------------------------------------------------
    result["business_name_norm"] = (
        result["business_name"]
        .map(normalize_business_name)
    )

    # --------------------------------------------------------
    # Address
    # --------------------------------------------------------
    result["business_address_norm"] = (
        result["business_address"]
        .map(normalize_address)
    )

    # --------------------------------------------------------
    # Country
    # --------------------------------------------------------
    result["country_norm"] = (
        result["country"]
        .map(normalize_country)
    )

    # --------------------------------------------------------
    # Postal code
    # --------------------------------------------------------
    result["postal_code"] = (
        result["business_address"]
        .map(extract_postal_code)
    )

    # --------------------------------------------------------
    # Locality
    # --------------------------------------------------------
    result["locality"] = (
        result["business_address"]
        .map(extract_locality)
    )

    # --------------------------------------------------------
    # Phonetic name
    # --------------------------------------------------------
    if include_phonetic:

        result["name_soundex"] = (
            result["business_name_norm"]
            .map(soundex)
        )

    return result


# ============================================================
# UNIT TESTS
# ============================================================

def run_unit_tests() -> None:
    """
    Run comprehensive deterministic tests.
    """

    # ========================================================
    # GENERAL TEXT
    # ========================================================

    assert (
        normalize_text(
            "ABC & Sons, Ltd."
        )
        == "abc and sons ltd"
    )

    assert (
        normalize_text(
            "  HELLO   WORLD!!! "
        )
        == "hello world"
    )

    # ========================================================
    # BUSINESS NAMES
    # ========================================================

    assert (
        normalize_business_name(
            "Acme Corporation"
        )
        == "acme"
    )

    assert (
        normalize_business_name(
            "ACME CORP."
        )
        == "acme"
    )

    assert (
        normalize_business_name(
            "Acme & Co."
        )
        == "acme"
    )

    assert (
        normalize_business_name(
            "Acme & Company"
        )
        == "acme"
    )

    assert (
        normalize_business_name(
            "Acme and Co."
        )
        == "acme"
    )

    assert (
        normalize_business_name(
            "ABC Pvt. Ltd."
        )
        == "abc"
    )

    assert (
        normalize_business_name(
            "Procter & Gamble"
        )
        == "procter and gamble"
    )

    assert (
        normalize_business_name(
            "Rock and Roll"
        )
        == "rock and roll"
    )

    # ========================================================
    # ADDRESS
    # ========================================================

    assert (
        normalize_address(
            "10 Main Street, New York, NY 10001"
        )
        == "10 main st new york ny 10001"
    )

    assert (
        normalize_address(
            "123 Main Road"
        )
        == "123 main rd"
    )

    # ========================================================
    # COUNTRY
    # ========================================================

    assert (
        normalize_country(
            "United States"
        )
        == "united states"
    )

    assert (
        normalize_country(
            "INDIA"
        )
        == "india"
    )

    # Open-set requirement.
    assert (
        normalize_country(
            "France"
        )
        == "france"
    )

    # ========================================================
    # POSTAL CODES
    # ========================================================

    postal_tests = {
        # US
        "10 Main Street, New York, NY 10001":
            "10001",

        # India
        "Pune Maharashtra 411001":
            "411001",

        "Mumbai Maharashtra 400001":
            "400001",

        # France
        "Paris 75001 France":
            "75001",

        # US ZIP+4
        "New York NY 10001-1234":
            "10001-1234",

        # Canada
        "Toronto ON M5V 3A8 Canada":
            "m5v3a8",
    }

    for address, expected in postal_tests.items():

        actual = extract_postal_code(
            address
        )

        assert actual == expected, (
            "\nPostal extraction failed:\n"
            f"Input:    {address!r}\n"
            f"Expected: {expected!r}\n"
            f"Actual:   {actual!r}"
        )

    # ========================================================
    # LOCALITY
    # ========================================================

    locality_tests = {
        "10 Main Street, New York, NY 10001":
            "new york",

        "1600 Pennsylvania Ave NW, Washington, DC 20500":
            "washington",

        "123 Main Road, Pune 411001":
            "pune",

        "123 MG Road, Mumbai, Maharashtra 400001":
            "mumbai",

        "10 Rue de Paris, Paris 75001":
            "paris",
    }

    for address, expected in locality_tests.items():

        actual = extract_locality(
            address
        )

        assert actual == expected, (
            "\nLocality extraction failed:\n"
            f"Input:    {address!r}\n"
            f"Expected: {expected!r}\n"
            f"Actual:   {actual!r}"
        )

    # ========================================================
    # SOUNDEX
    # ========================================================

    assert soundex("Robert") == "R163"
    assert soundex("Rupert") == "R163"

    # ========================================================
    # COMPLETE RECORD
    # ========================================================

    record = normalize_record(
        "Acme & Co.",
        "10 Main Street, New York, NY 10001",
        "United States",
    )

    assert (
        record["business_name_norm"]
        == "acme"
    )

    assert (
        record["business_address_norm"]
        == "10 main st new york ny 10001"
    )

    assert (
        record["country_norm"]
        == "united states"
    )

    assert (
        record["postal_code"]
        == "10001"
    )

    assert (
        record["locality"]
        == "new york"
    )

    assert (
        record["name_soundex"]
        == soundex("acme")
    )

    print()
    print("=" * 100)
    print("ALL NORMALIZATION UNIT TESTS PASSED")
    print("=" * 100)


# ============================================================
# DEMO
# ============================================================

def run_demo() -> None:
    """
    Run a small synthetic demonstration.
    """

    examples = [
        {
            "business_name": "Acme & Co.",
            "business_address":
                "10 Main Street, New York, NY 10001",
            "country": "United States",
        },
        {
            "business_name": "ACME CORPORATION",
            "business_address":
                "10 Main St., New York, NY 10001",
            "country": "USA",
        },
        {
            "business_name": "Procter & Gamble",
            "business_address":
                "1600 Pennsylvania Ave NW, Washington, DC 20500",
            "country": "United States",
        },
        {
            "business_name": "ABC Pvt. Ltd.",
            "business_address":
                "123 MG Road, Mumbai, Maharashtra 400001",
            "country": "India",
        },
        {
            "business_name": "Example SAS",
            "business_address":
                "10 Rue de Paris, Paris 75001",
            "country": "France",
        },
    ]

    df = pd.DataFrame(examples)

    normalized = normalize_dataframe(df)

    columns = [
        "business_name",
        "business_name_norm",
        "business_address",
        "business_address_norm",
        "country",
        "country_norm",
        "postal_code",
        "locality",
        "name_soundex",
    ]

    print()
    print("=" * 100)
    print("NORMALIZATION DEMONSTRATION")
    print("=" * 100)

    print(
        normalized[columns].to_string(
            index=False
        )
    )

    print("=" * 100)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 100)
    print(
        "BUSINESS ENTITY RESOLUTION - NORMALIZATION"
    )
    print("=" * 100)

    run_unit_tests()

    run_demo()

    print()
    print(
        "Normalization module is ready."
    )
    