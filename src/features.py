from __future__ import annotations

import re

import numpy as np
from rapidfuzz.fuzz import (
    ratio,
    token_set_ratio,
    token_sort_ratio,
    partial_ratio,
)

from src.text_utils import (
    transliterate_text,
    text_tokens,
)


FEATURE_NAMES = [
    "name_exact",
    "name_ratio",
    "name_token_set_ratio",
    "name_token_sort_ratio",
    "name_partial_ratio",
    "name_token_jaccard",
    "name_char3_jaccard",
    "name_length_ratio",
    "name_phonetic_match",
    "address_exact",
    "address_ratio",
    "address_token_jaccard",
    "address_char3_jaccard",
    "address_length_ratio",
    "postal_exact",
    "locality_exact",
    "country_exact",
    "address_missing",
    "evidence_count",
    "evidence_name_first",
    "evidence_name_first_country",
    "evidence_name_token_country",
    "evidence_phonetic",
    "evidence_phonetic_country",
    "evidence_postal",
    "evidence_postal_country",
    "evidence_locality_country",
]


def _clean(value):
    if value is None:
        return ""

    return str(value).strip()


def _ascii(value):
    return transliterate_text(value)


def _tokens(value):
    return set(text_tokens(value))


def _char_ngrams(value, n=3):
    text = _ascii(value).replace(" ", "")

    if len(text) < n:
        return {text} if text else set()

    return {
        text[i:i + n]
        for i in range(len(text) - n + 1)
    }


def _jaccard(a, b):
    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def _length_ratio(a, b):
    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return min(len(a), len(b)) / max(
        len(a),
        len(b),
    )


def _get(record, *names):
    for name in names:
        if name in record:
            return record[name]

    return ""


def _evidence_count(
    evidence_keys,
):
    return len(set(evidence_keys or []))


def make_feature_vector(
    left,
    right,
    evidence_keys=None,
):
    if hasattr(left, "_asdict"):
        left = left._asdict()

    if hasattr(right, "_asdict"):
        right = right._asdict()

    name_a = _ascii(
        _get(
            left,
            "business_name_norm",
            "name_norm",
            "business_name",
        )
    )

    name_b = _ascii(
        _get(
            right,
            "business_name_norm",
            "name_norm",
            "business_name",
        )
    )

    addr_a = _ascii(
        _get(
            left,
            "business_address_norm",
            "address_norm",
            "business_address",
        )
    )

    addr_b = _ascii(
        _get(
            right,
            "business_address_norm",
            "address_norm",
            "business_address",
        )
    )

    country_a = _clean(
        _get(
            left,
            "country_norm",
            "country",
        )
    ).lower()

    country_b = _clean(
        _get(
            right,
            "country_norm",
            "country",
        )
    ).lower()

    postal_a = _clean(
        _get(left, "postal_code")
    )

    postal_b = _clean(
        _get(right, "postal_code")
    )

    locality_a = _ascii(
        _get(
            left,
            "locality",
            "locality_norm",
        )
    )

    locality_b = _ascii(
        _get(
            right,
            "locality",
            "locality_norm",
        )
    )

    phonetic_a = _clean(
        _get(
            left,
            "name_soundex",
            "name_phonetic",
        )
    )

    phonetic_b = _clean(
        _get(
            right,
            "name_soundex",
            "name_phonetic",
        )
    )

    evidence_keys = set(
        evidence_keys or []
    )

    name_tokens_a = _tokens(name_a)
    name_tokens_b = _tokens(name_b)

    addr_tokens_a = _tokens(addr_a)
    addr_tokens_b = _tokens(addr_b)

    features = [
        float(
            bool(name_a)
            and name_a == name_b
        ),

        ratio(name_a, name_b),

        token_set_ratio(
            name_a,
            name_b,
        ),

        token_sort_ratio(
            name_a,
            name_b,
        ),

        partial_ratio(
            name_a,
            name_b,
        ),

        _jaccard(
            name_tokens_a,
            name_tokens_b,
        ),

        _jaccard(
            _char_ngrams(name_a),
            _char_ngrams(name_b),
        ),

        _length_ratio(
            name_a,
            name_b,
        ),

        float(
            bool(phonetic_a)
            and phonetic_a == phonetic_b
        ),

        float(
            bool(addr_a)
            and addr_a == addr_b
        ),

        ratio(addr_a, addr_b),

        _jaccard(
            addr_tokens_a,
            addr_tokens_b,
        ),

        _jaccard(
            _char_ngrams(addr_a),
            _char_ngrams(addr_b),
        ),

        _length_ratio(
            addr_a,
            addr_b,
        ),

        float(
            bool(postal_a)
            and postal_a == postal_b
        ),

        float(
            bool(locality_a)
            and locality_a == locality_b
        ),

        float(
            bool(country_a)
            and country_a == country_b
        ),

        float(
            not addr_a or not addr_b
        ),

        float(
            len(evidence_keys)
        ),

        float(
            "name_first" in evidence_keys
        ),

        float(
            "name_first_country"
            in evidence_keys
        ),

        float(
            "name_token_country"
            in evidence_keys
        ),

        float(
            "phonetic" in evidence_keys
        ),

        float(
            "phonetic_country"
            in evidence_keys
        ),

        float(
            "postal" in evidence_keys
        ),

        float(
            "postal_country"
            in evidence_keys
        ),

        float(
            "locality_country"
            in evidence_keys
        ),
    ]

    return np.asarray(
        features,
        dtype=np.float32,
    )


def feature_names():
    return FEATURE_NAMES.copy()


def build_feature_matrix(
    pairs,
    left_lookup,
    right_lookup,
):
    matrix = []

    for pair in pairs:
        left_id = pair["left_id"]
        right_id = pair["right_id"]

        evidence = pair.get(
            "evidence_keys",
            (),
        )

        matrix.append(
            make_feature_vector(
                left_lookup[left_id],
                right_lookup[right_id],
                evidence,
            )
        )

    return np.vstack(matrix)



def feature_columns():
    return FEATURE_NAMES.copy()


def build_pair_features(
    left,
    right,
    evidence_keys=None,
):
    return make_feature_vector(
        left,
        right,
        evidence_keys=evidence_keys,
    )