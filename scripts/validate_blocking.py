from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.blocking import (
    BlockingConfig,
    build_blocking_index,
    build_source_lookup,
    candidate_statistics,
    evaluate_blocking_recall,
    generate_candidate_pairs,
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

RANDOM_SOURCE2_SIZE = 100_000
RANDOM_SOURCE3_SIZE = 100_000

CHUNK_SIZE = 250_000

MAX_HARD_NEGATIVES_PER_SOURCE = 100_000


def load_source1_and_ground_truth():
    print("Loading Source 1 and ground truth...")

    source1 = pd.read_csv(
        TRAIN_SOURCE1,
        sep="\t",
        dtype="string",
    )

    ground_truth = pd.read_csv(
        TRAIN_GROUND_TRUTH,
        sep="\t",
        dtype="string",
    )

    source1_sample = source1.sample(
        n=min(
            S1_SAMPLE_SIZE,
            len(source1),
        ),
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    sample_ids = set(
        source1_sample["entity_id"]
        .astype(str)
    )

    ground_truth_sample = ground_truth[
        ground_truth["source1_entity_id"]
        .astype(str)
        .isin(sample_ids)
    ].copy()

    return (
        source1_sample,
        ground_truth_sample,
    )


def get_true_match_ids(
    ground_truth_sample: pd.DataFrame,
) -> set[str]:
    true_ids = set()

    for value in ground_truth_sample[
        "matched_entity_ids"
    ]:
        if pd.isna(value):
            continue

        text = str(value).strip()

        if not text:
            continue

        true_ids.update(
            entity_id.strip()
            for entity_id in text.split(",")
            if entity_id.strip()
        )

    return true_ids


def cheap_name_country(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df[
        [
            "entity_id",
            "business_name",
            "country",
        ]
    ].copy()

    result["business_name"] = (
        result["business_name"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace("&", " and ", regex=False)
        .str.replace(
            r"[^\w\s]",
            " ",
            regex=True,
        )
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
        .str.strip()
    )

    result["name_first"] = (
        result["business_name"]
        .str.split()
        .str[0]
        .fillna("")
    )

    result["country"] = (
        result["country"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    return result


def collect_source_subset(
    source_path: Path,
    true_match_ids: set[str],
    source1_sample: pd.DataFrame,
    random_size: int,
    random_state: int,
) -> pd.DataFrame:
    required_columns = [
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ]

    print()
    print(
        f"Scanning {source_path.name} in chunks..."
    )

    source1_keys = cheap_name_country(
        source1_sample
    )

    source1_key_set = set(
        zip(
            source1_keys["name_first"],
            source1_keys["country"],
        )
    )

    source1_key_set.discard(
        ("", "")
    )

    true_rows = []
    hard_negative_rows = []

    sampled_chunks = []

    rng = RANDOM_STATE + random_state

    for chunk in pd.read_csv(
        source_path,
        sep="\t",
        dtype="string",
        usecols=required_columns,
        chunksize=CHUNK_SIZE,
    ):
        ids = (
            chunk["entity_id"]
            .fillna("")
            .astype(str)
        )

        true_mask = ids.isin(
            true_match_ids
        )

        if true_mask.any():
            true_rows.append(
                chunk.loc[true_mask]
            )

        cheap = cheap_name_country(
            chunk
        )

        hard_mask = [
            (
                first,
                country,
            ) in source1_key_set
            for first, country in zip(
                cheap["name_first"],
                cheap["country"],
            )
        ]

        hard_mask = pd.Series(
            hard_mask,
            index=chunk.index,
        )

        hard_mask &= ~true_mask

        if hard_mask.any():
            hard = chunk.loc[
                hard_mask
            ]

            hard_negative_rows.append(
                hard
            )

        if len(sampled_chunks) < 20:
            sampled_chunks.append(
                chunk.sample(
                    frac=min(
                        1.0,
                        random_size
                        / max(
                            len(chunk),
                            1,
                        ),
                    ),
                    random_state=rng,
                )
            )

    true_df = (
        pd.concat(
            true_rows,
            ignore_index=True,
        )
        if true_rows
        else pd.DataFrame(
            columns=required_columns
        )
    )

    hard_df = (
        pd.concat(
            hard_negative_rows,
            ignore_index=True,
        )
        if hard_negative_rows
        else pd.DataFrame(
            columns=required_columns
        )
    )

    random_df = (
        pd.concat(
            sampled_chunks,
            ignore_index=True,
        )
        if sampled_chunks
        else pd.DataFrame(
            columns=required_columns
        )
    )

    if len(random_df) > random_size:
        random_df = random_df.sample(
            n=random_size,
            random_state=rng,
        )

    if len(hard_df) > MAX_HARD_NEGATIVES_PER_SOURCE:
        hard_df = hard_df.sample(
            n=MAX_HARD_NEGATIVES_PER_SOURCE,
            random_state=rng,
        )

    result = pd.concat(
        [
            true_df,
            hard_df,
            random_df,
        ],
        ignore_index=True,
    )

    result = result.drop_duplicates(
        subset=["entity_id"]
    )

    print(
        f"  Ground-truth rows : {len(true_df):,}"
    )
    print(
        f"  Hard-negative rows : {len(hard_df):,}"
    )
    print(
        f"  Random rows        : {len(random_df):,}"
    )
    print(
        f"  Final rows         : {len(result):,}"
    )

    return result


def main() -> None:
    total_start = time.perf_counter()

    source1_sample, ground_truth_sample = (
        load_source1_and_ground_truth()
    )

    true_match_ids = get_true_match_ids(
        ground_truth_sample
    )

    print()
    print(
        f"Source 1 validation sample : "
        f"{len(source1_sample):,}"
    )
    print(
        f"True matched IDs required  : "
        f"{len(true_match_ids):,}"
    )

    source2 = collect_source_subset(
        TRAIN_SOURCE2,
        true_match_ids,
        source1_sample,
        RANDOM_SOURCE2_SIZE,
        101,
    )

    source3 = collect_source_subset(
        TRAIN_SOURCE3,
        true_match_ids,
        source1_sample,
        RANDOM_SOURCE3_SIZE,
        202,
    )

    print()
    print("Normalizing validation Source 1...")

    source1_norm = normalize_dataframe(
        source1_sample,
        include_phonetic=True,
    )

    print("Normalizing validation Source 2...")

    source2_norm = normalize_dataframe(
        source2,
        include_phonetic=True,
    )

    print("Normalizing validation Source 3...")

    source3_norm = normalize_dataframe(
        source3,
        include_phonetic=True,
    )

    print()
    print("Building Source 2 blocking index...")

    index2 = build_blocking_index(
        source2_norm
    )

    print("Building Source 3 blocking index...")

    index3 = build_blocking_index(
        source3_norm
    )

    print("Building Source 2 lookup...")

    lookup2 = build_source_lookup(
        source2_norm
    )

    print("Building Source 3 lookup...")

    lookup3 = build_source_lookup(
        source3_norm
    )

    indexes = {
        "source2": index2,
        "source3": index3,
    }

    lookups = {
        "source2": lookup2,
        "source3": lookup3,
    }

    config = BlockingConfig(
        max_bucket_size=5000,
        broad_bucket_top_k=100,
        name_prefilter_threshold=55,
        max_final_candidates=200,
    )

    print()
    print("=" * 80)
    print("BLOCKING CONFIGURATION")
    print("=" * 80)

    print(
        f"max_bucket_size           = "
        f"{config.max_bucket_size}"
    )

    print(
        f"broad_bucket_top_k        = "
        f"{config.broad_bucket_top_k}"
    )

    print(
        f"name_prefilter_threshold  = "
        f"{config.name_prefilter_threshold}"
    )

    print(
        f"max_final_candidates      = "
        f"{config.max_final_candidates}"
    )

    print()
    print("Generating final candidate sets...")

    blocking_start = time.perf_counter()

    candidate_pairs = generate_candidate_pairs(
        source1_norm,
        indexes,
        lookups,
        config,
    )

    blocking_time = (
        time.perf_counter()
        - blocking_start
    )

    stats = candidate_statistics(
        candidate_pairs
    )

    recall = evaluate_blocking_recall(
        candidate_pairs,
        ground_truth_sample,
    )

    print()
    print("=" * 80)
    print("BLOCKING VALIDATION RESULTS")
    print("=" * 80)

    print()
    print("Candidate statistics")

    print(
        f"Entities evaluated      : "
        f"{stats['entities']:,}"
    )

    print(
        f"Average candidates      : "
        f"{stats['average']:.2f}"
    )

    print(
        f"Median candidates       : "
        f"{stats['median']:.0f}"
    )

    print(
        f"Maximum candidates     : "
        f"{stats['maximum']:,}"
    )

    print(
        f"Zero-candidate entities : "
        f"{stats['zero']:,}"
    )

    print()
    print("Recall")

    print(
        f"True matching pairs      : "
        f"{recall['true_pairs']:,}"
    )

    print(
        f"Recovered matching pairs : "
        f"{recall['recovered_pairs']:,}"
    )

    print(
        f"Pair recall              : "
        f"{recall['pair_recall']:.6f}"
    )

    print(
        f"Matched entities         : "
        f"{recall['matched_entities']:,}"
    )

    print(
        f"Fully recovered entities : "
        f"{recall['fully_recovered_entities']:,}"
    )

    print(
        f"Entity recall            : "
        f"{recall['entity_recall']:.6f}"
    )

    print()
    print(
        f"Blocking runtime         : "
        f"{blocking_time:.2f} seconds"
    )

    print(
        f"Total runtime            : "
        f"{time.perf_counter() - total_start:.2f} seconds"
    )

    print("=" * 80)

    if recall["pair_recall"] < 0.99:
        print()
        print(
            "WARNING: pair recall is below 99%."
        )

    if recall["entity_recall"] < 0.99:
        print()
        print(
            "WARNING: entity recall is below 99%."
        )


if __name__ == "__main__":
    main()