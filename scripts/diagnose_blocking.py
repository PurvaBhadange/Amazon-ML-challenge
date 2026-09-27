from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.blocking import (
    BlockingConfig,
    build_blocking_index,
    build_source_lookup,
    generate_candidates_for_record,
    make_blocking_keys,
)
from src.config import (
    TRAIN_GROUND_TRUTH,
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
)
from src.normalize import normalize_dataframe


RANDOM_STATE = 42
S1_SAMPLE_SIZE = 5_000
CHUNK_SIZE = 250_000

RANDOM_SOURCE_SIZE = 100_000
MAX_HARD_NEGATIVES_PER_SOURCE = 100_000

CONFIG = BlockingConfig(
    max_bucket_size=5000,
    broad_bucket_top_k=100,
    name_prefilter_threshold=55,
    max_final_candidates=200,
)


def cheap_name_country(df: pd.DataFrame) -> pd.DataFrame:
    name = (
        df["business_name"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace("&", " and ", regex=False)
    )

    name = name.str.replace(
        r"[^a-z0-9]+",
        " ",
        regex=True,
    )

    name = (
        name.str.replace(
            r"\s+",
            " ",
            regex=True,
        )
        .str.strip()
    )

    return pd.DataFrame(
        {
            "name_first": name.str.split().str[0].fillna(""),
            "country": (
                df["country"]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.strip()
            ),
        },
        index=df.index,
    )


def load_sample():
    print("Loading Source 1 and ground truth...")

    source1 = pd.read_csv(
        TRAIN_SOURCE1,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    ground_truth = pd.read_csv(
        TRAIN_GROUND_TRUTH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    source1 = (
        source1.sample(
            n=min(
                S1_SAMPLE_SIZE,
                len(source1),
            ),
            random_state=RANDOM_STATE,
        )
        .reset_index(drop=True)
    )

    ground_truth_map = dict(
        zip(
            ground_truth["source1_entity_id"].astype(str),
            ground_truth["matched_entity_ids"]
            .fillna("")
            .astype(str),
        )
    )

    required = {}

    for entity_id in source1["entity_id"].astype(str):

        raw = ground_truth_map.get(
            entity_id,
            "",
        )

        if not raw:
            required[entity_id] = []
            continue

        required[entity_id] = [
            value.strip()
            for value in raw.split(",")
            if value.strip()
        ]

    return source1, required


def collect_validation_source(
    path: Path,
    sampled_source1: pd.DataFrame,
    required: dict[str, list[str]],
    source_name: str,
) -> pd.DataFrame:

    all_true_ids = set()

    for ids in required.values():
        all_true_ids.update(ids)

    sampled_keys = cheap_name_country(
        sampled_source1
    )

    lookup_keys = set(
        zip(
            sampled_keys["name_first"],
            sampled_keys["country"],
        )
    )

    true_rows = []
    hard_rows = []
    random_rows = []

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    print(f"\nScanning {path.name}...")

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):

        chunk["entity_id"] = (
            chunk["entity_id"]
            .astype(str)
        )

        true_mask = chunk["entity_id"].isin(
            all_true_ids
        )

        if true_mask.any():
            true_rows.append(
                chunk.loc[
                    true_mask
                ].copy()
            )

        remaining = chunk.loc[
            ~true_mask
        ]

        if remaining.empty:
            continue

        cheap = cheap_name_country(
            remaining
        )

        remaining_keys = pd.MultiIndex.from_arrays(
            [
                cheap["name_first"],
                cheap["country"],
            ]
        )

        hard_mask = remaining_keys.isin(
            lookup_keys
        )

        if hard_mask.any():
            hard_rows.append(
                remaining.loc[
                    hard_mask
                ].copy()
            )

        sample_count = min(
            len(remaining),
            max(
                1,
                int(len(remaining) * 0.02),
            ),
        )

        if sample_count:
            sampled_indices = rng.choice(
                remaining.index.to_numpy(),
                size=sample_count,
                replace=False,
            )

            random_rows.append(
                remaining.loc[
                    sampled_indices
                ].copy()
            )

    true_df = (
        pd.concat(
            true_rows,
            ignore_index=True,
        )
        if true_rows
        else pd.DataFrame()
    )

    hard_df = (
        pd.concat(
            hard_rows,
            ignore_index=True,
        )
        if hard_rows
        else pd.DataFrame()
    )

    random_df = (
        pd.concat(
            random_rows,
            ignore_index=True,
        )
        if random_rows
        else pd.DataFrame()
    )

    if not hard_df.empty:
        hard_df = (
            hard_df
            .drop_duplicates(
                "entity_id"
            )
        )

        hard_df = hard_df[
            ~hard_df["entity_id"].isin(
                all_true_ids
            )
        ].head(
            MAX_HARD_NEGATIVES_PER_SOURCE
        )

    if not random_df.empty:
        random_df = (
            random_df
            .drop_duplicates(
                "entity_id"
            )
        )

        random_df = random_df[
            ~random_df["entity_id"].isin(
                all_true_ids
            )
        ].head(
            RANDOM_SOURCE_SIZE
        )

    frames = [
        frame
        for frame in (
            true_df,
            hard_df,
            random_df,
        )
        if not frame.empty
    ]

    if not frames:
        return pd.DataFrame()

    result = (
        pd.concat(
            frames,
            ignore_index=True,
        )
        .drop_duplicates(
            "entity_id"
        )
    )

    print(
        f"{source_name} validation rows: "
        f"{len(result):,}"
    )

    print(
        f"  True matches  : "
        f"{len(true_df):,}"
    )

    print(
        f"  Hard negatives: "
        f"{len(hard_df):,}"
    )

    print(
        f"  Random        : "
        f"{len(random_df):,}"
    )

    return result


def get_source_key(source_name: str) -> str:
    if source_name == "SOURCE 2":
        return "source2"

    if source_name == "SOURCE 3":
        return "source3"

    raise ValueError(
        f"Unknown source name: {source_name}"
    )


def candidate_ids_for_record(
    record: dict,
    index: dict,
    lookup: dict,
    source_key: str,
):
    return generate_candidates_for_record(
        record,
        {
            source_key: index,
        },
        {
            source_key: lookup,
        },
        CONFIG,
    )


def analyze_source(
    source_name: str,
    source1_normalized: pd.DataFrame,
    source_normalized: pd.DataFrame,
    required: dict[str, list[str]],
):

    print("\n" + "=" * 80)
    print(f"ANALYZING {source_name}")
    print("=" * 80)

    source_index = build_blocking_index(
        source_normalized,
        CONFIG,
    )

    source_lookup = build_source_lookup(
        source_normalized
    )

    source_ids = set(
        source_normalized[
            "entity_id"
        ]
        .astype(str)
    )

    source_key = get_source_key(
        source_name
    )

    true_pairs = {
        (
            s1_id,
            target_id,
        )
        for s1_id, ids in required.items()
        for target_id in ids
        if target_id in source_ids
    }

    print(
        f"True pairs in validation source: "
        f"{len(true_pairs):,}"
    )

    source_by_id = (
        source_normalized
        .set_index("entity_id")
    )

    stage_counts = Counter()
    key_recovery = Counter()

    missing_examples = []

    for _, row in source1_normalized.iterrows():

        s1_id = str(
            row["entity_id"]
        )

        if s1_id not in required:
            continue

        record = row.to_dict()

        candidates = candidate_ids_for_record(
            record,
            source_index,
            source_lookup,
            source_key,
        )

        candidate_ids = {
            str(candidate.entity_id)
            for candidate in candidates
        }

        true_ids = {
            target_id
            for target_id in required[s1_id]
            if target_id in source_ids
        }

        for target_id in true_ids:

            if target_id in candidate_ids:
                stage_counts[
                    "recovered"
                ] += 1
                continue

            stage_counts[
                "missed"
            ] += 1

            target = source_by_id.loc[
                target_id
            ].to_dict()

            target_keys = make_blocking_keys(
                target,
                CONFIG,
            )

            shared_keys = []

            for key_name, values in target_keys.items():

                bucket_map = source_index.get(
                    key_name,
                    {},
                )

                for value in values:

                    bucket = bucket_map.get(
                        value,
                        [],
                    )

                    if target_id in bucket:
                        shared_keys.append(
                            f"{key_name}={value}"
                        )

            if shared_keys:

                stage_counts[
                    "shared_blocking_key"
                ] += 1

                for key in shared_keys:

                    key_name = key.split(
                        "=",
                        1,
                    )[0]

                    key_recovery[
                        key_name
                    ] += 1

            else:
                stage_counts[
                    "no_shared_blocking_key"
                ] += 1

            if len(missing_examples) < 30:

                missing_examples.append(
                    {
                        "s1_id": s1_id,
                        "target_id": target_id,
                        "shared_keys": shared_keys,
                        "candidate_count": len(
                            candidate_ids
                        ),
                    }
                )

    total = len(true_pairs)

    recovered = stage_counts[
        "recovered"
    ]

    missed = stage_counts[
        "missed"
    ]

    shared = stage_counts[
        "shared_blocking_key"
    ]

    no_key = stage_counts[
        "no_shared_blocking_key"
    ]

    print("\nStage diagnosis")
    print("-" * 80)

    print(
        f"True pairs                    : "
        f"{total:,}"
    )

    print(
        f"Recovered by current blocker  : "
        f"{recovered:,}"
    )

    print(
        f"Missed                        : "
        f"{missed:,}"
    )

    print(
        f"Missed but shared blocking key: "
        f"{shared:,}"
    )

    print(
        f"Missed with NO blocking key   : "
        f"{no_key:,}"
    )

    if total:

        print(
            f"\nCurrent pair recall           : "
            f"{recovered / total:.6f}"
        )

        print(
            f"No-key loss rate              : "
            f"{no_key / total:.6f}"
        )

    print(
        "\nBlocking-key recovery among "
        "missed pairs"
    )

    print("-" * 80)

    if key_recovery:

        for key_name, count in (
            key_recovery.most_common()
        ):
            print(
                f"{key_name:<30}"
                f"{count:>8,}"
            )

    else:
        print(
            "No missed pair shared any "
            "blocking key."
        )

    print(
        "\nMissing-pair examples"
    )

    print("-" * 80)

    for example in missing_examples:

        print(
            f"S1={example['s1_id']} | "
            f"TRUE={example['target_id']} | "
            f"candidates="
            f"{example['candidate_count']}"
        )

        if example["shared_keys"]:

            print(
                "  shared keys: "
                + ", ".join(
                    example[
                        "shared_keys"
                    ][:10]
                )
            )

        else:
            print(
                "  shared keys: NONE"
            )

    return {
        "true_pairs": total,
        "recovered": recovered,
        "missed": missed,
        "shared_blocking_key": shared,
        "no_shared_blocking_key": no_key,
        "key_recovery": key_recovery,
    }


def main():

    start = time.time()

    source1, required = load_sample()

    print(
        f"\nSource 1 sample: "
        f"{len(source1):,}"
    )

    print(
        f"True matched IDs: "
        f"{sum(len(ids) for ids in required.values()):,}"
    )

    source2 = collect_validation_source(
        TRAIN_SOURCE2,
        source1,
        required,
        "Source 2",
    )

    source3 = collect_validation_source(
        TRAIN_SOURCE3,
        source1,
        required,
        "Source 3",
    )

    print(
        "\nNormalizing validation data..."
    )

    source1_normalized = normalize_dataframe(
        source1,
        include_phonetic=True,
    )

    source2_normalized = normalize_dataframe(
        source2,
        include_phonetic=True,
    )

    source3_normalized = normalize_dataframe(
        source3,
        include_phonetic=True,
    )

    source2_result = analyze_source(
        "SOURCE 2",
        source1_normalized,
        source2_normalized,
        required,
    )

    source3_result = analyze_source(
        "SOURCE 3",
        source1_normalized,
        source3_normalized,
        required,
    )

    total_true = (
        source2_result["true_pairs"]
        + source3_result["true_pairs"]
    )

    total_recovered = (
        source2_result["recovered"]
        + source3_result["recovered"]
    )

    total_missed = (
        total_true
        - total_recovered
    )

    print("\n" + "=" * 80)
    print("COMBINED DIAGNOSTIC")
    print("=" * 80)

    print(
        f"True pairs     : "
        f"{total_true:,}"
    )

    print(
        f"Recovered      : "
        f"{total_recovered:,}"
    )

    print(
        f"Missed         : "
        f"{total_missed:,}"
    )

    if total_true:

        print(
            f"Pair recall    : "
            f"{total_recovered / total_true:.6f}"
        )

    print(
        f"\nTotal runtime  : "
        f"{time.time() - start:.2f} seconds"
    )


if __name__ == "__main__":
    main()