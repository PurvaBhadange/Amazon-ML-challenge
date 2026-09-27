from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd
from rapidfuzz.fuzz import ratio, token_set_ratio

from src.normalize import normalize_dataframe, normalize_record
from src.text_utils import (
    transliterate_text,
    text_tokens,
    address_anchors,
)


@dataclass(frozen=True)
class BlockingConfig:
    min_token_length: int = 2
    max_name_block_tokens: int = 3
    max_bucket_size: int = 5000
    broad_bucket_top_k: int = 150
    name_prefilter_threshold: float = 45.0
    address_prefilter_threshold: float = 35.0
    max_final_candidates: int = 250


def _canonical_record(record) -> dict:
    if hasattr(record, "_asdict"):
        record = record._asdict()

    if isinstance(record, pd.Series):
        record = record.to_dict()

    if not isinstance(record, Mapping):
        record = dict(record)

    entity_id = str(record.get("entity_id", ""))

    name_norm = str(
        record.get(
            "business_name_norm",
            record.get("name_norm", record.get("business_name", "")),
        )
        or ""
    )

    address_norm = str(
        record.get(
            "business_address_norm",
            record.get("address_norm", record.get("business_address", "")),
        )
        or ""
    )

    country_norm = str(
        record.get(
            "country_norm",
            record.get("country", ""),
        )
        or ""
    )

    postal_code = str(record.get("postal_code", "") or "")

    locality = str(
        record.get(
            "locality",
            record.get("locality_norm", ""),
        )
        or ""
    )

    name_phonetic = str(
        record.get(
            "name_soundex",
            record.get("name_phonetic", ""),
        )
        or ""
    )

    return {
        "entity_id": entity_id,
        "name_norm": name_norm,
        "address_norm": address_norm,
        "country_norm": country_norm,
        "postal_code": postal_code,
        "locality_norm": locality,
        "name_phonetic": name_phonetic,

        "name_ascii": transliterate_text(name_norm),
        "address_ascii": transliterate_text(address_norm),
        "locality_ascii": transliterate_text(locality),

        "address_anchors": address_anchors(address_norm),
    }


def make_blocking_keys(
    record,
    config: BlockingConfig | None = None,
) -> List[Tuple[str, str]]:
    config = config or BlockingConfig()
    r = _canonical_record(record)

    name_tokens = [
        t for t in r["name_ascii"].split()
        if len(t) >= config.min_token_length
    ]

    keys = []

    if name_tokens:
        first = name_tokens[0]

        keys.append(("name_first", first))

        if r["country_norm"]:
            keys.append(
                ("name_first_country",
                 f"{first}|{r['country_norm']}")
            )

        for token in name_tokens[:config.max_name_block_tokens]:
            if len(token) >= config.min_token_length:
                if r["country_norm"]:
                    keys.append(
                        (
                            "name_token_country",
                            f"{token}|{r['country_norm']}",
                        )
                    )

    if r["name_phonetic"]:
        keys.append(
            ("phonetic", r["name_phonetic"])
        )

        if r["country_norm"]:
            keys.append(
                (
                    "phonetic_country",
                    f"{r['name_phonetic']}|{r['country_norm']}",
                )
            )

    if r["postal_code"]:
        keys.append(
            ("postal", r["postal_code"])
        )

        if r["country_norm"]:
            keys.append(
                (
                    "postal_country",
                    f"{r['postal_code']}|{r['country_norm']}",
                )
            )

    if r["locality_ascii"] and r["country_norm"]:
        keys.append(
            (
                "locality_country",
                f"{r['locality_ascii']}|{r['country_norm']}",
            )
        )

    # New: strong address anchors.
    for anchor in r["address_anchors"]:
        if r["country_norm"]:
            keys.append(
                (
                    "address_anchor_country",
                    f"{anchor}|{r['country_norm']}",
                )
            )

    # New: transliterated locality first token.
    locality_tokens = r["locality_ascii"].split()

    if locality_tokens and r["country_norm"]:
        for token in locality_tokens[:2]:
            if len(token) >= config.min_token_length:
                keys.append(
                    (
                        "locality_token_country",
                        f"{token}|{r['country_norm']}",
                    )
                )

    return list(dict.fromkeys(keys))


def build_blocking_index(
    records,
    config: BlockingConfig | None = None,
):
    config = config or BlockingConfig()

    if isinstance(records, pd.DataFrame):
        records = records.to_dict("records")

    index = defaultdict(list)

    for record in records:
        record = _canonical_record(record)

        for key in make_blocking_keys(record, config):
            index[key].append(record["entity_id"])

    return dict(index)


def build_source_lookup(records):
    if isinstance(records, pd.DataFrame):
        records = records.to_dict("records")

    lookup = {}

    for record in records:
        r = _canonical_record(record)
        lookup[r["entity_id"]] = r

    return lookup


def _name_similarity(a, b):
    a = a or ""
    b = b or ""

    if not a or not b:
        return 0.0

    return max(
        ratio(a, b),
        token_set_ratio(a, b),
    )


def _address_similarity(a, b):
    a = a or ""
    b = b or ""

    if not a or not b:
        return 0.0

    return max(
        ratio(a, b),
        token_set_ratio(a, b),
    )


def _candidate_score(
    query,
    candidate,
    evidence_keys=None,
):
    q = _canonical_record(query)
    c = _canonical_record(candidate)

    name_score = _name_similarity(
        q["name_ascii"],
        c["name_ascii"],
    )

    address_score = _address_similarity(
        q["address_ascii"],
        c["address_ascii"],
    )

    score = max(
        name_score,
        address_score * 0.92,
    )

    if (
        q["name_ascii"]
        and c["name_ascii"]
        and q["name_ascii"] == c["name_ascii"]
    ):
        score += 45

    if (
        q["country_norm"]
        and q["country_norm"] == c["country_norm"]
    ):
        score += 15

    if (
        q["postal_code"]
        and q["postal_code"] == c["postal_code"]
    ):
        score += 35

    if (
        q["locality_ascii"]
        and q["locality_ascii"] == c["locality_ascii"]
    ):
        score += 25

    if (
        q["name_phonetic"]
        and q["name_phonetic"] == c["name_phonetic"]
    ):
        score += 10

    q_anchors = set(q["address_anchors"])
    c_anchors = set(c["address_anchors"])

    anchor_overlap = len(q_anchors & c_anchors)

    if anchor_overlap:
        score += min(30, anchor_overlap * 15)

    evidence_weights = {
        "name_first": 2,
        "name_first_country": 5,
        "name_token_country": 7,
        "phonetic": 4,
        "phonetic_country": 6,
        "postal": 12,
        "postal_country": 15,
        "locality_country": 18,
        "locality_token_country": 12,
        "address_anchor_country": 20,
    }

    evidence_score = 0.0

    for key_type in evidence_keys or ():
        evidence_score += evidence_weights.get(key_type, 0)

    score += min(45, evidence_score)

    return float(score)


def _query_index(
    query,
    index,
    lookup,
    config,
):
    query = _canonical_record(query)

    grouped = defaultdict(set)

    for key_type, key_value in make_blocking_keys(query, config):
        ids = index.get((key_type, key_value), [])

        for entity_id in ids:
            grouped[entity_id].add(key_type)

    if not grouped:
        return []

    scored = []

    for entity_id, evidence in grouped.items():
        candidate = lookup.get(entity_id)

        if candidate is None:
            continue

        score = _candidate_score(
            query,
            candidate,
            evidence,
        )

        name_score = _name_similarity(
            query["name_ascii"],
            candidate["name_ascii"],
        )

        address_score = _address_similarity(
            query["address_ascii"],
            candidate["address_ascii"],
        )

        strong_address = (
            address_score >= config.address_prefilter_threshold
            and bool(
                set(query["address_anchors"])
                & set(candidate["address_anchors"])
            )
        )

        strong_name = name_score >= config.name_prefilter_threshold

        if (
            strong_name
            or strong_address
            or len(evidence) >= 2
            or query["name_ascii"] == candidate["name_ascii"]
            or query["postal_code"] == candidate["postal_code"]
        ):
            scored.append(
                (
                    score,
                    len(evidence),
                    entity_id,
                )
            )

    scored.sort(
        key=lambda x: (x[0], x[1]),
        reverse=True,
    )

    if len(scored) > config.max_bucket_size:
        scored = scored[:config.broad_bucket_top_k]

    return [
        (entity_id, score)
        for score, _, entity_id in scored
    ]


def filter_candidates(
    query,
    candidate_ids,
    lookup,
    config=None,
):
    config = config or BlockingConfig()

    q = _canonical_record(query)

    scored = []

    for entity_id in candidate_ids:
        candidate = lookup.get(entity_id)

        if candidate is None:
            continue

        score = _candidate_score(
            q,
            candidate,
        )

        if score >= config.name_prefilter_threshold:
            scored.append((entity_id, score))

    scored.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        entity_id
        for entity_id, _ in scored[:config.max_final_candidates]
    ]


def _global_candidate_selection(
    query,
    source_candidates,
    source_lookups,
    config,
):
    all_candidates = {}

    for source_name, candidates in source_candidates.items():
        lookup = source_lookups[source_name]

        for entity_id, score in candidates:
            candidate = lookup.get(entity_id)

            if candidate is None:
                continue

            key = (source_name, entity_id)

            if key not in all_candidates:
                all_candidates[key] = {
                    "source": source_name,
                    "entity_id": entity_id,
                    "score": score,
                    "candidate": candidate,
                }

    ranked = []

    q = _canonical_record(query)

    for item in all_candidates.values():
        c = item["candidate"]

        evidence = []

        for key_type, key_value in make_blocking_keys(q, config):
            if c["entity_id"] in source_lookups[item["source"]]:
                pass

        score = _candidate_score(q, c)

        ranked.append(
            (
                score,
                item["source"],
                item["entity_id"],
            )
        )

    ranked.sort(reverse=True)

    return [
        entity_id
        for _, _, entity_id in ranked[
            :config.max_final_candidates
        ]
    ]

def generate_candidates_for_record(
    record,
    indexes,
    lookups,
    config=None,
):
    config = config or BlockingConfig()

    combined = {}

    for source_name, index in indexes.items():
        lookup = lookups[source_name]

        query = _canonical_record(record)
        grouped = defaultdict(set)

        for key_type, key_value in make_blocking_keys(
            query,
            config,
        ):
            for entity_id in index.get(
                (key_type, key_value),
                [],
            ):
                grouped[entity_id].add(
                    key_type
                )

        for entity_id, evidence in grouped.items():
            candidate = lookup.get(entity_id)

            if candidate is None:
                continue

            name_score = _name_similarity(
                query["name_ascii"],
                candidate["name_ascii"],
            )

            address_score = _address_similarity(
                query["address_ascii"],
                candidate["address_ascii"],
            )

            strong_name = (
                name_score
                >= config.name_prefilter_threshold
            )

            strong_address = (
                address_score
                >= config.address_prefilter_threshold
                and bool(
                    set(query["address_anchors"])
                    & set(candidate["address_anchors"])
                )
            )

            if not (
                strong_name
                or strong_address
                or len(evidence) >= 2
                or query["name_ascii"]
                == candidate["name_ascii"]
                or (
                    query["postal_code"]
                    and query["postal_code"]
                    == candidate["postal_code"]
                )
            ):
                continue

            score = _candidate_score(
                query,
                candidate,
                evidence,
            )

            key = (
                source_name,
                entity_id,
            )

            combined[key] = (
                score,
                evidence,
            )

    ranked = sorted(
        combined.items(),
        key=lambda item: item[1][0],
        reverse=True,
    )

    return [
        entity_id
        for (
            _,
            entity_id
        ), (
            _score,
            _evidence,
        ) in ranked[
            :config.max_final_candidates
        ]
    ]

def _normalize_dataframe(
    df,
    include_phonetic=True,
):
    return normalize_dataframe(
        df,
        include_phonetic=include_phonetic,
    )


def generate_candidate_pairs(
    source1_df,
    source2_df,
    source3_df,
    config=None,
):
    config = config or BlockingConfig()

    s1 = normalize_dataframe(
        source1_df,
        include_phonetic=True,
    )

    s2 = normalize_dataframe(
        source2_df,
        include_phonetic=True,
    )

    s3 = normalize_dataframe(
        source3_df,
        include_phonetic=True,
    )

    index2 = build_blocking_index(s2, config)
    index3 = build_blocking_index(s3, config)

    lookup2 = build_source_lookup(s2)
    lookup3 = build_source_lookup(s3)

    indexes = {
        "source2": index2,
        "source3": index3,
    }

    lookups = {
        "source2": lookup2,
        "source3": lookup3,
    }

    rows = []

    for record in s1.to_dict("records"):
        candidates = generate_candidates_for_record(
            record,
            indexes,
            lookups,
            config,
        )

        rows.append(
            {
                "source1_entity_id": record["entity_id"],
                "candidate_entity_ids": ",".join(
                    candidates
                ),
            }
        )

    return pd.DataFrame(rows)


def candidate_statistics(candidate_pairs):
    counts = candidate_pairs[
        "candidate_entity_ids"
    ].fillna("").map(
        lambda x: 0 if not x else len(
            [v for v in x.split(",") if v]
        )
    )

    return {
        "average": float(counts.mean()),
        "median": float(counts.median()),
        "maximum": int(counts.max()),
        "minimum": int(counts.min()),
        "zero_candidates": int((counts == 0).sum()),
        "at_cap": int((counts >= 250).sum()),
    }


def save_candidate_pairs(
    candidate_pairs,
    path,
):
    candidate_pairs.to_csv(
        path,
        sep="\t",
        index=False,
    )


def parse_ground_truth(
    ground_truth_df,
):
    result = {}

    for row in ground_truth_df.itertuples(index=False):
        value = str(
            getattr(
                row,
                "matched_entity_ids",
                "",
            )
            or ""
        ).strip()

        if not value:
            result[row.source1_entity_id] = set()
        else:
            result[row.source1_entity_id] = {
                x.strip()
                for x in value.split(",")
                if x.strip()
            }

    return result


def evaluate_blocking_recall(
    candidate_pairs,
    ground_truth,
):
    recovered = 0
    total = 0

    entity_hits = 0
    entity_total = len(ground_truth)

    candidate_map = {
        row.source1_entity_id:
            set(
                x for x in str(
                    row.candidate_entity_ids
                ).split(",")
                if x
            )
        for row in candidate_pairs.itertuples()
    }

    for s1_id, true_ids in ground_truth.items():
        candidates = candidate_map.get(
            s1_id,
            set(),
        )

        if true_ids:
            total += len(true_ids)
            recovered += len(
                true_ids & candidates
            )

            if true_ids <= candidates:
                entity_hits += 1
        else:
            entity_hits += 1

    return {
        "pair_recall": (
            recovered / total
            if total
            else 1.0
        ),
        "entity_recall": (
            entity_hits / entity_total
            if entity_total
            else 1.0
        ),
        "true_pairs": total,
        "recovered_pairs": recovered,
        "fully_recovered_entities": entity_hits,
        "total_entities": entity_total,
    }


if __name__ == "__main__":
    print("Blocking module loaded.")
    print(BlockingConfig())