from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.blocking import (
    BlockingConfig,
    build_blocking_index,
    build_source_lookup,
    generate_candidates_for_record,
    make_blocking_keys,
)
from src.features import build_pair_features, feature_columns
from src.normalize import normalize_dataframe


ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "dataset" / "train"
OUTPUT = ROOT / "output"
MODELS = ROOT / "models"

SOURCE1_PATH = DATASET / "train_source1.tsv"
SOURCE2_PATH = DATASET / "train_source2.tsv"
SOURCE3_PATH = DATASET / "train_source3.tsv"
GT_PATH = DATASET / "train_ground_truth.tsv"

VALIDATION_PAIRS = OUTPUT / "validation_pairs.tsv"
MODEL_PATH = MODELS / "matching_model.joblib"

MAX_SOURCE1_ENTITIES = 2_000
RANDOM_STATE = 42


def load_ground_truth() -> dict[str, set[str]]:
    gt = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    result = {}

    for row in gt.itertuples(index=False):
        s1_id = str(row.source1_entity_id).strip()
        matched = str(row.matched_entity_ids).strip()

        if matched:
            result[s1_id] = {
                x.strip()
                for x in matched.split(",")
                if x.strip()
            }
        else:
            result[s1_id] = set()

    return result


def get_validation_ids() -> set[str]:
    validation = pd.read_csv(
        VALIDATION_PAIRS,
        sep="\t",
        dtype=str,
        usecols=["source1_entity_id"],
    )

    ids = (
        validation["source1_entity_id"]
        .drop_duplicates()
        .astype(str)
        .tolist()
    )

    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(ids)

    return set(
        ids[:MAX_SOURCE1_ENTITIES]
    )


def load_source1(
    validation_ids: set[str],
) -> pd.DataFrame:

    df = pd.read_csv(
        SOURCE1_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    return df[
        df["entity_id"].astype(str).isin(
            validation_ids
        )
    ].copy()


def collect_required_ids(
    source1: pd.DataFrame,
    ground_truth: dict[str, set[str]],
) -> set[str]:

    required = set()

    for s1_id in source1["entity_id"].astype(str):
        required.update(
            ground_truth.get(
                s1_id,
                set(),
            )
        )

    return required


def load_required_source_rows(
    path: Path,
    required_ids: set[str],
) -> pd.DataFrame:

    frames = []

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=250_000,
    ):

        mask = chunk["entity_id"].astype(str).isin(
            required_ids
        )

        if mask.any():
            frames.append(
                chunk.loc[mask].copy()
            )

    if not frames:
        return pd.DataFrame(
            columns=[
                "entity_id",
                "business_name",
                "business_address",
                "country",
            ]
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def build_candidate_feature_rows(
    source1_norm: pd.DataFrame,
    source2_norm: pd.DataFrame,
    source3_norm: pd.DataFrame,
    ground_truth: dict[str, set[str]],
    model,
    threshold: float,
):
    config = BlockingConfig()

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

    print("Building lookups...")

    source1_lookup = build_source_lookup(
        source1_norm
    )

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

    feature_rows = []

    candidate_counts = []

    required_pairs = 0
    recovered_pairs = 0

    entity_results = []

    for position, row in enumerate(
        source1_norm.itertuples(index=False),
        start=1,
    ):

        s1_id = str(row.entity_id)

        query = source1_lookup[s1_id]

        truth = ground_truth.get(
            s1_id,
            set(),
        )

        candidates = generate_candidates_for_record(
            query,
            indexes,
            lookups,
            config,
        )

        candidate_counts.append(
            len(candidates)
        )

        candidate_ids = {
            candidate.entity_id
            for candidate in candidates
        }

        recovered = truth.intersection(
            candidate_ids
        )

        required_pairs += len(truth)
        recovered_pairs += len(recovered)

        feature_records = []

        for candidate in candidates:

            if candidate.source == "source2":
                candidate_record = source2_lookup.get(
                    candidate.entity_id
                )
            else:
                candidate_record = source3_lookup.get(
                    candidate.entity_id
                )

            if candidate_record is None:
                continue

            query_keys = make_blocking_keys(
                query,
                config,
            )

            candidate_keys = make_blocking_keys(
                candidate_record,
                config,
            )

            evidence = set()

            for key_type, keys in query_keys.items():

                if set(keys).intersection(
                    candidate_keys.get(
                        key_type,
                        [],
                    )
                ):
                    evidence.add(key_type)

            features = build_pair_features(
                query,
                candidate_record,
                evidence,
            )

            feature_records.append(
                {
                    "candidate_id": candidate.entity_id,
                    "candidate_source": candidate.source,
                    "features": features,
                }
            )

        if feature_records:

            X = pd.DataFrame(
                [
                    item["features"]
                    for item in feature_records
                ]
            )[feature_columns()]

            probabilities = model.predict_proba(
                X
            )[:, 1]

            predicted_ids = {
                item["candidate_id"]
                for item, probability in zip(
                    feature_records,
                    probabilities,
                )
                if probability >= threshold
            }

        else:
            predicted_ids = set()

        true_ids = truth

        if not predicted_ids and not true_ids:
            f05 = 1.0

        elif not predicted_ids and true_ids:
            f05 = 0.0

        else:
            tp = len(
                predicted_ids.intersection(
                    true_ids
                )
            )

            fp = len(
                predicted_ids - true_ids
            )

            fn = len(
                true_ids - predicted_ids
            )

            precision = (
                tp / (tp + fp)
                if tp + fp
                else 0.0
            )

            recall = (
                tp / (tp + fn)
                if tp + fn
                else 0.0
            )

            if precision == 0 and recall == 0:
                f05 = 0.0
            else:
                f05 = (
                    1.25
                    * precision
                    * recall
                    / (
                        0.25 * precision
                        + recall
                    )
                )

        entity_results.append(
            {
                "source1_entity_id": s1_id,
                "true_count": len(true_ids),
                "predicted_count": len(predicted_ids),
                "candidate_count": len(candidates),
                "recovered_count": len(recovered),
                "f0.5": f05,
                "singleton": int(
                    len(true_ids) == 0
                ),
                "singleton_correct": int(
                    len(true_ids) == 0
                    and not predicted_ids
                ),
            }
        )

        if position % 500 == 0:
            print(
                f"Evaluated "
                f"{position:,}/"
                f"{len(source1_norm):,} entities"
            )

    entity_df = pd.DataFrame(
        entity_results
    )

    return (
        entity_df,
        required_pairs,
        recovered_pairs,
        candidate_counts,
    )


def main():

    print("=" * 80)
    print("END-TO-END PIPELINE VALIDATION")
    print("=" * 80)

    model = joblib.load(
        MODEL_PATH
    )

    metadata_path = (
        MODELS / "model_metadata.tsv"
    )

    metadata = pd.read_csv(
        metadata_path,
        sep="\t",
    )

    threshold = float(
        metadata.iloc[0]["threshold"]
    )

    print(
        f"Model threshold : {threshold:.3f}"
    )

    ground_truth = load_ground_truth()

    validation_ids = get_validation_ids()

    print(
        f"Validation entities : "
        f"{len(validation_ids):,}"
    )

    source1 = load_source1(
        validation_ids
    )

    required_ids = collect_required_ids(
        source1,
        ground_truth,
    )

    print(
        f"Required matched IDs : "
        f"{len(required_ids):,}"
    )

    print()
    print("Loading required Source 2 rows...")

    source2 = load_required_source_rows(
        SOURCE2_PATH,
        required_ids,
    )

    print(
        f"Source 2 rows : "
        f"{len(source2):,}"
    )

    print("Loading required Source 3 rows...")

    source3 = load_required_source_rows(
        SOURCE3_PATH,
        required_ids,
    )

    print(
        f"Source 3 rows : "
        f"{len(source3):,}"
    )

    # The blocker needs enough non-matching records to
    # create realistic candidate competition.
    #
    # Add a controlled sample from each source.
    rng = np.random.default_rng(
        RANDOM_STATE
    )

    def add_background(
        path: Path,
        current: pd.DataFrame,
        target_extra: int = 50_000,
    ):

        frames = [current]

        current_ids = set(
            current["entity_id"].astype(str)
        )

        collected = 0

        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=250_000,
        ):

            remaining = chunk[
                ~chunk["entity_id"].astype(str).isin(
                    current_ids
                )
            ]

            if remaining.empty:
                continue

            take = min(
                target_extra - collected,
                len(remaining),
            )

            if take <= 0:
                break

            indices = rng.choice(
                len(remaining),
                size=take,
                replace=False,
            )

            selected = remaining.iloc[
                indices
            ]

            frames.append(selected)

            collected += len(selected)

            if collected >= target_extra:
                break

        return pd.concat(
            frames,
            ignore_index=True,
        )

    source2 = add_background(
        SOURCE2_PATH,
        source2,
    )

    source3 = add_background(
        SOURCE3_PATH,
        source3,
    )

    print()
    print("Normalizing validation data...")

    source1_norm = normalize_dataframe(
        source1,
        include_phonetic=True,
    )

    source2_norm = normalize_dataframe(
        source2,
        include_phonetic=True,
    )

    source3_norm = normalize_dataframe(
        source3,
        include_phonetic=True,
    )

    (
        entity_df,
        required_pairs,
        recovered_pairs,
        candidate_counts,
    ) = build_candidate_feature_rows(
        source1_norm,
        source2_norm,
        source3_norm,
        ground_truth,
        model,
        threshold,
    )

    candidate_recall = (
        recovered_pairs / required_pairs
        if required_pairs
        else 1.0
    )

    macro_f05 = entity_df[
        "f0.5"
    ].mean()

    singleton_df = entity_df[
        entity_df["singleton"] == 1
    ]

    singleton_accuracy = (
        singleton_df["singleton_correct"].mean()
        if len(singleton_df)
        else 1.0
    )

    predicted_total = entity_df[
        "predicted_count"
    ].sum()

    true_total = entity_df[
        "true_count"
    ].sum()

    recovered_total = entity_df[
        "recovered_count"
    ].sum()

    print()
    print("=" * 80)
    print("END-TO-END RESULTS")
    print("=" * 80)

    print(
        f"Entities evaluated       : "
        f"{len(entity_df):,}"
    )

    print(
        f"True matching pairs      : "
        f"{true_total:,}"
    )

    print(
        f"Recovered candidate pairs: "
        f"{recovered_total:,}"
    )

    print(
        f"Candidate pair recall    : "
        f"{candidate_recall:.6f}"
    )

    print()

    print(
        f"Predicted matches        : "
        f"{predicted_total:,}"
    )

    print(
        f"Entity macro F0.5        : "
        f"{macro_f05:.6f}"
    )

    print()

    print(
        f"Average candidates       : "
        f"{np.mean(candidate_counts):.2f}"
    )

    print(
        f"Median candidates        : "
        f"{np.median(candidate_counts):.0f}"
    )

    print(
        f"Maximum candidates       : "
        f"{np.max(candidate_counts):,}"
    )

    print()

    print(
        f"Singleton entities       : "
        f"{len(singleton_df):,}"
    )

    print(
        f"Singleton empty accuracy : "
        f"{singleton_accuracy:.6f}"
    )

    print()

    print("=" * 80)

    entity_df.to_csv(
        OUTPUT / "pipeline_validation_entities.tsv",
        sep="\t",
        index=False,
    )

    print(
        "Saved detailed results:"
    )

    print(
        OUTPUT
        / "pipeline_validation_entities.tsv"
    )


if __name__ == "__main__":
    main()