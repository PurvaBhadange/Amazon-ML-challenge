from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Set

import pandas as pd

from src.blocking import (
    BlockingConfig,
    _canonical_record,
    build_blocking_index,
    build_source_lookup,
    generate_candidates_for_record,
    make_blocking_keys,
)
from src.normalize import normalize_dataframe


RANDOM_STATE = 42

VALIDATION_ENTITY_COUNT = 2_000

BACKGROUND_SOURCE2_SIZE = 50_000
BACKGROUND_SOURCE3_SIZE = 50_000

OUTPUT_PATH = Path(
    "output/candidate_loss_diagnostic.tsv"
)


def resolve_dataset_files():
    paths = {
        "source1": Path(
            "dataset/train/train_source1.tsv"
        ),
        "source2": Path(
            "dataset/train/train_source2.tsv"
        ),
        "source3": Path(
            "dataset/train/train_source3.tsv"
        ),
        "ground_truth": Path(
            "dataset/train/train_ground_truth.tsv"
        ),
    }

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{name} not found: {path}"
            )

    return paths


def parse_ground_truth(
    ground_truth: pd.DataFrame,
) -> Dict[str, Set[str]]:
    result = {}

    for row in ground_truth.itertuples(
        index=False
    ):
        source1_id = str(
            row.source1_entity_id
        )

        raw = getattr(
            row,
            "matched_entity_ids",
            "",
        )

        if pd.isna(raw):
            raw = ""

        raw = str(raw).strip()

        if not raw:
            result[source1_id] = set()
        else:
            result[source1_id] = {
                x.strip()
                for x in raw.split(",")
                if x.strip()
            }

    return result


def load_data():
    paths = resolve_dataset_files()

    print("Dataset files:")
    print(
        f"Source 1      : {paths['source1']}"
    )
    print(
        f"Source 2      : {paths['source2']}"
    )
    print(
        f"Source 3      : {paths['source3']}"
    )
    print(
        f"Ground truth  : {paths['ground_truth']}"
    )
    print()

    source1 = pd.read_csv(
        paths["source1"],
        sep="\t",
        dtype=str,
    )

    source2 = pd.read_csv(
        paths["source2"],
        sep="\t",
        dtype=str,
    )

    source3 = pd.read_csv(
        paths["source3"],
        sep="\t",
        dtype=str,
    )

    ground_truth = pd.read_csv(
        paths["ground_truth"],
        sep="\t",
        dtype=str,
    )

    return (
        source1,
        source2,
        source3,
        ground_truth,
    )


def select_validation_entities(
    source1: pd.DataFrame,
    ground_truth: pd.DataFrame,
):
    valid_ids = set(
        ground_truth[
            "source1_entity_id"
        ].astype(str)
    )

    available = source1[
        source1["entity_id"]
        .astype(str)
        .isin(valid_ids)
    ].copy()

    if len(available) <= VALIDATION_ENTITY_COUNT:
        return (
            available["entity_id"]
            .astype(str)
            .tolist()
        )

    return (
        available.sample(
            n=VALIDATION_ENTITY_COUNT,
            random_state=RANDOM_STATE,
        )["entity_id"]
        .astype(str)
        .tolist()
    )


def collect_required_ids(
    validation_ids,
    ground_truth_map,
):
    required = set()

    for entity_id in validation_ids:
        required.update(
            ground_truth_map.get(
                entity_id,
                set(),
            )
        )

    return required


def collect_source_rows(
    source_df,
    required_ids,
    background_size,
):
    required_ids = {
        str(x)
        for x in required_ids
    }

    required = source_df[
        source_df["entity_id"]
        .astype(str)
        .isin(required_ids)
    ].copy()

    present = set(
        required["entity_id"]
        .astype(str)
    )

    missing = required_ids - present

    if missing:
        raise RuntimeError(
            "Missing required IDs: "
            f"{len(missing)}"
        )

    remaining = source_df[
        ~source_df["entity_id"]
        .astype(str)
        .isin(present)
    ]

    if len(remaining) > background_size:
        background = remaining.sample(
            n=background_size,
            random_state=RANDOM_STATE,
        )
    else:
        background = remaining

    return pd.concat(
        [required, background],
        ignore_index=True,
    ).drop_duplicates(
        subset=["entity_id"]
    )


def evidence_between(
    query,
    candidate,
    config,
):
    query_keys = set(
        make_blocking_keys(
            query,
            config,
        )
    )

    candidate_keys = set(
        make_blocking_keys(
            candidate,
            config,
        )
    )

    return {
        key_type
        for key_type, key_value in query_keys
        if (key_type, key_value)
        in candidate_keys
    }


def analyze_entity(
    source1_record,
    source2_lookup,
    source3_lookup,
    indexes,
    ground_truth_ids,
    config,
):
    query = _canonical_record(
        source1_record
    )

    candidates = generate_candidates_for_record(
        query,
        indexes,
        {
            "source2": source2_lookup,
            "source3": source3_lookup,
        },
        config,
    )

    candidate_set = set(candidates)

    recovered = (
        candidate_set
        & ground_truth_ids
    )

    missed = (
        ground_truth_ids
        - candidate_set
    )

    rows = []

    for true_id in missed:
        if true_id in source2_lookup:
            candidate = source2_lookup[
                true_id
            ]
            source = "source2"

        elif true_id in source3_lookup:
            candidate = source3_lookup[
                true_id
            ]
            source = "source3"

        else:
            continue

        evidence = evidence_between(
            query,
            candidate,
            config,
        )

        rows.append(
            {
                "source1_entity_id":
                    query["entity_id"],
                "missed_entity_id":
                    true_id,
                "source":
                    source,
                "exact_name":
                    int(
                        bool(
                            query["name_ascii"]
                        )
                        and
                        query["name_ascii"]
                        ==
                        candidate["name_ascii"]
                    ),
                "postal_exact":
                    int(
                        bool(
                            query["postal_code"]
                        )
                        and
                        query["postal_code"]
                        ==
                        candidate["postal_code"]
                    ),
                "locality_exact":
                    int(
                        bool(
                            query["locality_ascii"]
                        )
                        and
                        query["locality_ascii"]
                        ==
                        candidate["locality_ascii"]
                    ),
                "phonetic_exact":
                    int(
                        bool(
                            query["name_phonetic"]
                        )
                        and
                        query["name_phonetic"]
                        ==
                        candidate["name_phonetic"]
                    ),
                "evidence_count":
                    len(evidence),
                "evidence":
                    ",".join(
                        sorted(evidence)
                    ),
                "candidate_count":
                    len(candidate_set),
            }
        )

    return {
        "candidate_count":
            len(candidate_set),
        "true_count":
            len(ground_truth_ids),
        "recovered_count":
            len(recovered),
        "missed_count":
            len(missed),
        "missed_rows":
            rows,
    }


def main():
    random.seed(RANDOM_STATE)

    print("=" * 80)
    print("CANDIDATE LOSS DIAGNOSTIC")
    print("=" * 80)
    print()

    (
        source1,
        source2,
        source3,
        ground_truth,
    ) = load_data()

    ground_truth_map = parse_ground_truth(
        ground_truth
    )

    validation_ids = (
        select_validation_entities(
            source1,
            ground_truth,
        )
    )

    required_ids = collect_required_ids(
        validation_ids,
        ground_truth_map,
    )

    required_source2 = {
        x for x in required_ids
        if x.startswith("S2-")
    }

    required_source3 = {
        x for x in required_ids
        if x.startswith("S3-")
    }

    print(
        f"Validation entities : "
        f"{len(validation_ids):,}"
    )

    print(
        f"Required match IDs : "
        f"{len(required_ids):,}"
    )

    eval_source2 = collect_source_rows(
        source2,
        required_source2,
        BACKGROUND_SOURCE2_SIZE,
    )

    eval_source3 = collect_source_rows(
        source3,
        required_source3,
        BACKGROUND_SOURCE3_SIZE,
    )

    print(
        f"Source 2 evaluation rows : "
        f"{len(eval_source2):,}"
    )

    print(
        f"Source 3 evaluation rows : "
        f"{len(eval_source3):,}"
    )

    print()
    print("Normalizing...")

    source1_norm = normalize_dataframe(
        source1[
            source1["entity_id"]
            .isin(validation_ids)
        ].copy(),
        include_phonetic=True,
    )

    source2_norm = normalize_dataframe(
        eval_source2,
        include_phonetic=True,
    )

    source3_norm = normalize_dataframe(
        eval_source3,
        include_phonetic=True,
    )

    print("Building indexes...")

    config = BlockingConfig(
        max_final_candidates=250
    )

    source2_index = build_blocking_index(
        source2_norm,
        config,
    )

    source3_index = build_blocking_index(
        source3_norm,
        config,
    )

    source2_lookup = build_source_lookup(
        source2_norm
    )

    source3_lookup = build_source_lookup(
        source3_norm
    )

    indexes = {
        "source2": source2_index,
        "source3": source3_index,
    }

    print()
    print("Analyzing candidate loss...")

    total_true = 0
    total_recovered = 0
    total_candidates = 0

    zero_candidates = 0
    at_cap = 0

    missed_rows = []

    entity_count = len(
        source1_norm
    )

    for i, record in enumerate(
        source1_norm.to_dict("records"),
        start=1,
    ):
        if i % 250 == 0:
            print(
                f"  processed "
                f"{i:,}/{entity_count:,}"
            )

        source1_id = str(
            record["entity_id"]
        )

        true_ids = ground_truth_map.get(
            source1_id,
            set(),
        )

        result = analyze_entity(
            record,
            source2_lookup,
            source3_lookup,
            indexes,
            true_ids,
            config,
        )

        total_true += result[
            "true_count"
        ]

        total_recovered += result[
            "recovered_count"
        ]

        total_candidates += result[
            "candidate_count"
        ]

        if result["candidate_count"] == 0:
            zero_candidates += 1

        if (
            result["candidate_count"]
            >= config.max_final_candidates
        ):
            at_cap += 1

        missed_rows.extend(
            result["missed_rows"]
        )

    pair_recall = (
        total_recovered / total_true
        if total_true
        else 1.0
    )

    average_candidates = (
        total_candidates / entity_count
        if entity_count
        else 0.0
    )

    diagnostic = pd.DataFrame(
        missed_rows
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostic.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False,
    )

    print()
    print("=" * 80)
    print("CANDIDATE LOSS RESULTS")
    print("=" * 80)

    print(
        f"True pairs          : "
        f"{total_true:,}"
    )

    print(
        f"Recovered pairs     : "
        f"{total_recovered:,}"
    )

    print(
        f"Missed pairs        : "
        f"{total_true - total_recovered:,}"
    )

    print(
        f"Pair recall         : "
        f"{pair_recall:.6f}"
    )

    print()

    print(
        f"Entities hitting cap: "
        f"{at_cap:,}/{entity_count:,}"
    )

    print(
        f"Cap percentage      : "
        f"{100 * at_cap / entity_count:.2f}%"
    )

    print(
        f"Zero-candidate entities: "
        f"{zero_candidates:,}"
    )

    print()

    print("Candidate count statistics")

    print(
        f"Average candidates : "
        f"{average_candidates:.2f}"
    )

    if diagnostic.empty:
        median_candidates = 0
        maximum_candidates = 0
    else:
        # candidate_count is repeated once per missed pair,
        # so use the original per-entity values separately below.
        counts = []

        for record in source1_norm.to_dict(
            "records"
        ):
            candidates = generate_candidates_for_record(
                record,
                indexes,
                {
                    "source2": source2_lookup,
                    "source3": source3_lookup,
                },
                config,
            )

            counts.append(len(candidates))

        median_candidates = (
            pd.Series(counts).median()
        )

        maximum_candidates = max(
            counts
        )

    print(
        f"Median candidates  : "
        f"{median_candidates:.0f}"
    )

    print(
        f"Maximum candidates : "
        f"{maximum_candidates}"
    )

    print()

    if diagnostic.empty:
        print(
            "No missed pairs."
        )
    else:
        print(
            "Evidence among missed true matches"
        )

        print(
            f"Exact name          : "
            f"{diagnostic['exact_name'].sum()}"
        )

        print(
            f"Exact postal        : "
            f"{diagnostic['postal_exact'].sum()}"
        )

        print(
            f"Exact locality      : "
            f"{diagnostic['locality_exact'].sum()}"
        )

        print(
            f"Exact phonetic      : "
            f"{diagnostic['phonetic_exact'].sum()}"
        )

        print(
            "Any strong evidence : "f"{int((diagnostic['evidence_count'] > 0).sum())}"
        )

        print()
        print(
            "Blocking evidence among "
            "missed pairs:"
        )

        evidence_counts = {}

        for value in diagnostic[
            "evidence"
        ].fillna(""):
            if not value:
                continue

            for key in value.split(","):
                evidence_counts[key] = (
                    evidence_counts.get(
                        key,
                        0,
                    )
                    + 1
                )

        for key, count in sorted(
            evidence_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            print(
                f"  {key:<28} "
                f"{count}"
            )

        print()

        print(
            "Missed-pair evidence summary:"
        )

        print(
            diagnostic[
                [
                    "exact_name",
                    "postal_exact",
                    "locality_exact",
                    "phonetic_exact",
                    "evidence_count",
                ]
            ].describe()
        )

    print()
    print(
        "Saved detailed diagnostic to:"
    )
    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()