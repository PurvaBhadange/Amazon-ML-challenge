from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.blocking import (
    BlockingConfig,
    build_blocking_index,
    build_source_lookup,
    generate_candidates_for_record,
)
from src.features import build_pair_features, feature_columns
from src.normalize import normalize_dataframe


RANDOM_STATE = 42

TRAIN_ENTITY_FRACTION = 0.80

MAX_NEGATIVES_PER_ENTITY = 5

MIN_POSITIVES_REQUIRED = 1


def load_ground_truth(path: str | Path) -> dict[str, set[str]]:
    gt = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    result: dict[str, set[str]] = {}

    for row in gt.itertuples(index=False):
        source1_id = str(row.source1_entity_id).strip()
        matched = str(row.matched_entity_ids).strip()

        if not matched:
            result[source1_id] = set()
            continue

        result[source1_id] = {
            value.strip()
            for value in matched.split(",")
            if value.strip()
        }

    return result


def split_source1_entities(
    source1_df: pd.DataFrame,
    ground_truth: dict[str, set[str]],
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    eligible_ids = [
        entity_id
        for entity_id in source1_df["entity_id"].astype(str)
        if entity_id in ground_truth
    ]

    rng = np.random.default_rng(random_state)

    eligible_ids = np.asarray(eligible_ids)

    rng.shuffle(eligible_ids)

    split_index = int(
        len(eligible_ids) * TRAIN_ENTITY_FRACTION
    )

    train_ids = set(
        eligible_ids[:split_index]
    )

    validation_ids = set(
        eligible_ids[split_index:]
    )

    train_df = source1_df[
        source1_df["entity_id"].astype(str).isin(train_ids)
    ].copy()

    validation_df = source1_df[
        source1_df["entity_id"].astype(str).isin(validation_ids)
    ].copy()

    return train_df, validation_df


def _candidate_ids_by_source(
    candidates,
) -> dict[str, list[str]]:

    result = defaultdict(list)

    for candidate in candidates:
        result[candidate.source].append(
            candidate.entity_id
        )

    return dict(result)


def _build_positive_pairs(
    source1_id: str,
    truth_ids: set[str],
    candidate_ids: dict[str, list[str]],
) -> list[tuple[str, str, int]]:

    pairs = []

    candidate_set = {
        entity_id
        for ids in candidate_ids.values()
        for entity_id in ids
    }

    for matched_id in truth_ids:

        if matched_id in candidate_set:
            source = (
                "source2"
                if matched_id in candidate_ids.get("source2", [])
                else "source3"
            )

            pairs.append(
                (
                    source1_id,
                    matched_id,
                    1,
                )
            )

    return pairs


def _build_negative_pairs(
    source1_id: str,
    truth_ids: set[str],
    candidate_ids: dict[str, list[str]],
    max_negatives: int,
    rng: np.random.Generator,
) -> list[tuple[str, str, int]]:

    negatives = []

    for source in ("source2", "source3"):

        candidates = candidate_ids.get(
            source,
            [],
        )

        available = [
            entity_id
            for entity_id in candidates
            if entity_id not in truth_ids
        ]

        if not available:
            continue

        if len(available) > max_negatives:
            selected = rng.choice(
                available,
                size=max_negatives,
                replace=False,
            )
            available = selected.tolist()

        for entity_id in available:
            negatives.append(
                (
                    source1_id,
                    entity_id,
                    0,
                )
            )

    return negatives


def _candidate_evidence(
    query_record: dict,
    candidate_record: dict,
    config: BlockingConfig,
) -> set[str]:

    query_keys = {}

    from src.blocking import make_blocking_keys

    query_keys = make_blocking_keys(
        query_record,
        config,
    )

    candidate_keys = make_blocking_keys(
        candidate_record,
        config,
    )

    evidence = set()

    for key_type, query_values in query_keys.items():

        candidate_values = set(
            candidate_keys.get(
                key_type,
                [],
            )
        )

        if candidate_values.intersection(
            query_values
        ):
            evidence.add(key_type)

    return evidence


def generate_training_pairs(
    source1_df: pd.DataFrame,
    source2_df: pd.DataFrame,
    source3_df: pd.DataFrame,
    ground_truth: dict[str, set[str]],
    max_negatives_per_entity: int = MAX_NEGATIVES_PER_ENTITY,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    config = BlockingConfig()

    print("Normalizing Source 1...")
    source1_norm = normalize_dataframe(
        source1_df,
        include_phonetic=True,
    )

    print("Normalizing Source 2...")
    source2_norm = normalize_dataframe(
        source2_df,
        include_phonetic=True,
    )

    print("Normalizing Source 3...")
    source3_norm = normalize_dataframe(
        source3_df,
        include_phonetic=True,
    )

    print("Building Source 2 index...")
    source2_index = build_blocking_index(
        source2_norm,
        config,
    )

    print("Building Source 3 index...")
    source3_index = build_blocking_index(
        source3_norm,
        config,
    )

    indexes = {
        "source2": source2_index,
        "source3": source3_index,
    }

    print("Building source lookups...")

    source2_lookup = build_source_lookup(
        source2_norm
    )

    source3_lookup = build_source_lookup(
        source3_norm
    )

    lookups = {
        "source2": source2_lookup,
        "source3": source3_lookup,
    }

    rng = np.random.default_rng(
        random_state
    )

    pair_rows = []

    source1_lookup = build_source_lookup(
        source1_norm
    )

    total_entities = len(source1_norm)

    for position, row in enumerate(
        source1_norm.itertuples(index=False),
        start=1,
    ):

        source1_id = str(row.entity_id)

        truth_ids = ground_truth.get(
            source1_id,
            set(),
        )

        query_record = source1_lookup.get(
            source1_id
        )

        if query_record is None:
            continue

        candidates = generate_candidates_for_record(
            query_record,
            indexes,
            lookups,
            config,
        )

        candidate_ids = _candidate_ids_by_source(
            candidates
        )

        positives = _build_positive_pairs(
            source1_id,
            truth_ids,
            candidate_ids,
        )

        negatives = _build_negative_pairs(
            source1_id,
            truth_ids,
            candidate_ids,
            max_negatives_per_entity,
            rng,
        )

        for left_id, right_id, label in (
            positives + negatives
        ):

            source = (
                "source2"
                if right_id in source2_lookup
                else "source3"
            )

            right_record = lookups[source].get(
                right_id
            )

            if right_record is None:
                continue

            evidence = _candidate_evidence(
                query_record,
                right_record,
                config,
            )

            features = build_pair_features(
                query_record,
                right_record,
                evidence,
            )

            features["source1_entity_id"] = source1_id
            features["candidate_entity_id"] = right_id
            features["candidate_source"] = source
            features["label"] = label

            pair_rows.append(features)

        if position % 1000 == 0:
            print(
                f"Processed {position:,}/{total_entities:,} "
                f"Source-1 entities | "
                f"pairs={len(pair_rows):,}"
            )

    pair_df = pd.DataFrame(
        pair_rows
    )

    if pair_df.empty:
        raise RuntimeError(
            "No training pairs were generated."
        )

    feature_cols = feature_columns()

    metadata_cols = [
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
        "label",
    ]

    pair_df = pair_df[
        feature_cols + metadata_cols
    ]

    train_mask = pair_df[
        "source1_entity_id"
    ].isin(
        set(
            source1_df["entity_id"].astype(str)
        )
    )

    train_pairs = pair_df[
        train_mask
    ].reset_index(drop=True)

    return train_pairs, pair_df


def print_pair_statistics(
    pair_df: pd.DataFrame,
) -> None:

    print()
    print("=" * 80)
    print("TRAINING PAIR STATISTICS")
    print("=" * 80)

    print(
        f"Total pairs : {len(pair_df):,}"
    )

    positives = int(
        pair_df["label"].sum()
    )

    negatives = len(pair_df) - positives

    print(
        f"Positive pairs : {positives:,}"
    )

    print(
        f"Negative pairs : {negatives:,}"
    )

    if len(pair_df):
        print(
            f"Positive ratio : "
            f"{positives / len(pair_df):.6f}"
        )

    print()
    print("Pairs by candidate source:")

    print(
        pair_df[
            "candidate_source"
        ].value_counts()
    )

    print()
    print("Feature matrix shape:")

    print(
        pair_df[
            feature_columns()
        ].shape
    )


def validate_training_module() -> None:

    source1 = pd.DataFrame(
        [
            {
                "entity_id": "s1",
                "business_name": "Acme Corporation",
                "business_address": "Mumbai 400001",
                "country": "India",
            },
            {
                "entity_id": "s2",
                "business_name": "Beta Limited",
                "business_address": "Pune 411001",
                "country": "India",
            },
        ]
    )

    source2 = pd.DataFrame(
        [
            {
                "entity_id": "a1",
                "business_name": "Acme Corp",
                "business_address": "Mumbai 400001",
                "country": "India",
            },
            {
                "entity_id": "a2",
                "business_name": "Random Store",
                "business_address": "Delhi 110001",
                "country": "India",
            },
        ]
    )

    source3 = pd.DataFrame(
        [
            {
                "entity_id": "b1",
                "business_name": "Beta Ltd",
                "business_address": "Pune 411001",
                "country": "India",
            },
        ]
    )

    ground_truth = {
        "s1": {"a1"},
        "s2": {"b1"},
    }

    train, full = generate_training_pairs(
        source1,
        source2,
        source3,
        ground_truth,
        max_negatives_per_entity=1,
    )

    assert not train.empty

    assert "label" in train.columns

    assert int(train["label"].sum()) >= 1

    assert set(
        feature_columns()
    ).issubset(train.columns)

    print()
    print("=" * 80)
    print("TRAINING DATA UNIT TESTS PASSED")
    print("=" * 80)

    print(
        f"Generated pairs: {len(train)}"
    )

    print(
        f"Positive pairs: {int(train['label'].sum())}"
    )

    print(
        f"Negative pairs: "
        f"{int((train['label'] == 0).sum())}"
    )


if __name__ == "__main__":
    validate_training_module()