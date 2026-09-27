from __future__ import annotations

from pathlib import Path
from typing import Dict, Set

import joblib
import numpy as np
import pandas as pd

from src.normalize import normalize_dataframe
from src.blocking import (
    BlockingConfig,
    build_blocking_index,
    build_source_lookup,
    generate_candidates_for_record,
    make_blocking_keys,
)
from src.features import (
    build_pair_features,
    feature_columns,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TEST_DIR = ROOT / "dataset" / "test"
OUTPUT_DIR = ROOT / "output"
MODELS_DIR = ROOT / "models"

S1_PATH = TEST_DIR / "test_source1.tsv"
S2_PATH = TEST_DIR / "test_source2.tsv"
S3_PATH = TEST_DIR / "test_source3.tsv"

MODEL_PATH = MODELS_DIR / "matching_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.tsv"

FINAL_MATCHING_PATH = OUTPUT_DIR / "matching_results_ml.tsv"
FINAL_CANDIDATE_PATH = OUTPUT_DIR / "candidate_pairs_ml.tsv"


# ============================================================
# CONFIG
# ============================================================

CHUNK_SIZE = 5_000

# This is the threshold already selected during validation.
THRESHOLD = 0.775

# Same blocking family used by the training pipeline.
CONFIG = BlockingConfig(
    max_bucket_size=5000,
    broad_bucket_top_k=100,
    name_prefilter_threshold=55.0,
    max_final_candidates=200,
)


# ============================================================
# FEATURE HELPERS
# ============================================================

def normalize_blocking_keys(keys) -> Dict[str, Set[str]]:
    """
    Convert either blocking API format into:

        {
            "name_first": {"abc"},
            "postal_country": {"12345|usa"}
        }
    """

    if keys is None:
        return {}

    result: Dict[str, Set[str]] = {}

    # Current/older implementation may return dict.
    if isinstance(keys, dict):

        for key_type, values in keys.items():

            if isinstance(
                values,
                (list, tuple, set),
            ):

                result[str(key_type)] = {
                    str(value)
                    for value in values
                    if value is not None
                    and str(value)
                }

            elif values is not None:

                value = str(values)

                if value:

                    result.setdefault(
                        str(key_type),
                        set(),
                    ).add(value)

        return result

    # Other implementation returns:
    #
    # [
    #     ("name_first", "abc"),
    #     ("postal", "12345")
    # ]

    for item in keys:

        if (
            isinstance(item, (list, tuple))
            and len(item) == 2
        ):

            key_type, value = item

            if value is None:
                continue

            key_type = str(key_type)
            value = str(value)

            if not value:
                continue

            result.setdefault(
                key_type,
                set(),
            ).add(value)

    return result


def evidence_between(
    query,
    candidate,
    config,
):
    """
    Calculate exactly the blocking evidence shared by
    query and candidate.

    This is important because evidence features were part
    of model training.
    """

    query_keys = normalize_blocking_keys(
        make_blocking_keys(
            query,
            config,
        )
    )

    candidate_keys = normalize_blocking_keys(
        make_blocking_keys(
            candidate,
            config,
        )
    )

    evidence = set()

    common_types = (
        set(query_keys)
        & set(candidate_keys)
    )

    for key_type in common_types:

        if (
            query_keys[key_type]
            &
            candidate_keys[key_type]
        ):

            evidence.add(key_type)

    return evidence


def feature_vector(
    query,
    candidate,
    evidence,
):
    """
    Build the exact 27-feature vector expected by
    matching_model.joblib.
    """

    vector = build_pair_features(
        query,
        candidate,
        evidence,
    )

    columns = feature_columns()

    if len(vector) != len(columns):

        raise RuntimeError(
            "Feature mismatch: "
            f"model expects {len(columns)} "
            f"features but feature builder "
            f"returned {len(vector)}"
        )

    return np.asarray(
        vector,
        dtype=np.float64,
    )


# ============================================================
# MODEL
# ============================================================

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Missing model: {MODEL_PATH}"
        )

    print()
    print("=" * 80)
    print("LOADING MATCHING MODEL")
    print("=" * 80)

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Model:",
        type(model).__name__,
    )

    print(
        "Threshold:",
        THRESHOLD,
    )

    if METADATA_PATH.exists():

        print()
        print(
            "Model metadata:"
        )

        print(
            METADATA_PATH.read_text(
                encoding="utf-8"
            )
        )

    return model


# ============================================================
# DATA LOADING
# ============================================================

def load_source(
    path,
    name,
):

    print()
    print(
        f"Loading {name}..."
    )

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    )

    print(
        f"{name} rows: "
        f"{len(df):,}"
    )

    print(
        f"Normalizing {name}..."
    )

    normalized = normalize_dataframe(
        df,
        include_phonetic=True,
    )

    del df

    return normalized


# ============================================================
# BUILD INDEXES
# ============================================================

def build_indexes(
    source2,
    source3,
):

    print()
    print("=" * 80)
    print("BUILDING BLOCKING INDEXES")
    print("=" * 80)

    print(
        "Building Source 2 index..."
    )

    source2_index = build_blocking_index(
        source2,
        CONFIG,
    )

    print(
        "Building Source 3 index..."
    )

    source3_index = build_blocking_index(
        source3,
        CONFIG,
    )

    print(
        "Building Source 2 lookup..."
    )

    source2_lookup = build_source_lookup(
        source2
    )

    print(
        "Building Source 3 lookup..."
    )

    source3_lookup = build_source_lookup(
        source3
    )

    indexes = {
        "source2": source2_index,
        "source3": source3_index,
    }

    lookups = {
        "source2": source2_lookup,
        "source3": source3_lookup,
    }

    return indexes, lookups


# ============================================================
# SCORE ONE ENTITY
# ============================================================

def score_entity(
    query,
    candidates,
    lookups,
    model,
):
    """
    Score every final blocking candidate.

    Returns:
        candidate_ids
        probabilities
        accepted_matches
    """

    candidate_ids = []

    candidate_records = []

    candidate_sources = []

    # --------------------------------------------------------
    # Preserve Source 2 + Source 3 candidate set
    # --------------------------------------------------------

    seen = set()

    for source_name in (
        "source2",
        "source3",
    ):

        source_candidates = candidates.get(
            source_name,
            [],
        )

        lookup = lookups[source_name]

        for candidate_id in source_candidates:

            candidate_id = str(
                candidate_id
            )

            if candidate_id in seen:
                continue

            candidate = lookup.get(
                candidate_id
            )

            if candidate is None:
                continue

            seen.add(candidate_id)

            candidate_ids.append(
                candidate_id
            )

            candidate_records.append(
                candidate
            )

            candidate_sources.append(
                source_name
            )

    if not candidate_records:

        return [], [], []

    # --------------------------------------------------------
    # Build feature matrix
    # --------------------------------------------------------

    vectors = []

    for candidate in candidate_records:

        evidence = evidence_between(
            query,
            candidate,
            CONFIG,
        )

        vector = feature_vector(
            query,
            candidate,
            evidence,
        )

        vectors.append(vector)

    X = np.vstack(vectors)

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    probabilities = model.predict_proba(X)[:, 1]

    # --------------------------------------------------------
    # Final threshold
    # --------------------------------------------------------

    accepted = []

    for candidate_id, probability in zip(
        candidate_ids,
        probabilities,
    ):

        if float(probability) >= THRESHOLD:

            accepted.append(
                candidate_id
            )

    return (
        candidate_ids,
        probabilities,
        accepted,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 80)
    print("MODEL-BASED FULL TEST INFERENCE")
    print("=" * 80)

    # --------------------------------------------------------
    # Validate input files
    # --------------------------------------------------------

    for path in (
        S1_PATH,
        S2_PATH,
        S3_PATH,
        MODEL_PATH,
    ):

        if not path.exists():

            raise FileNotFoundError(
                f"Missing required file: {path}"
            )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Load Source 2
    # --------------------------------------------------------

    source2 = load_source(
        S2_PATH,
        "Source 2",
    )

    # --------------------------------------------------------
    # Load Source 3
    # --------------------------------------------------------

    source3 = load_source(
        S3_PATH,
        "Source 3",
    )

    # --------------------------------------------------------
    # Build indexes
    # --------------------------------------------------------

    indexes, lookups = build_indexes(
        source2,
        source3,
    )

    # --------------------------------------------------------
    # Prepare output files
    # --------------------------------------------------------

    if FINAL_MATCHING_PATH.exists():
        FINAL_MATCHING_PATH.unlink()

    if FINAL_CANDIDATE_PATH.exists():
        FINAL_CANDIDATE_PATH.unlink()

    with open(
        FINAL_MATCHING_PATH,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        f.write(
            "source1_entity_id\t"
            "matched_entity_ids\n"
        )

    with open(
        FINAL_CANDIDATE_PATH,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        f.write(
            "source1_entity_id\t"
            "candidate_entity_ids\n"
        )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    total = 0

    entities_with_candidates = 0
    entities_with_matches = 0

    total_candidates = 0
    total_matches = 0

    probability_sum = 0.0
    probability_count = 0

    # --------------------------------------------------------
    # Stream Source 1
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("STARTING FULL SOURCE-1 INFERENCE")
    print("=" * 80)

    with open(
        FINAL_MATCHING_PATH,
        "a",
        encoding="utf-8",
        newline="",
    ) as matching_file:

        with open(
            FINAL_CANDIDATE_PATH,
            "a",
            encoding="utf-8",
            newline="",
        ) as candidate_file:

            for chunk in pd.read_csv(
                S1_PATH,
                sep="\t",
                dtype=str,
                keep_default_na=False,
                chunksize=CHUNK_SIZE,
            ):

                # --------------------------------------------
                # Normalize only current S1 chunk
                # --------------------------------------------

                normalized_chunk = normalize_dataframe(
                    chunk,
                    include_phonetic=True,
                )

                matching_lines = []
                candidate_lines = []

                # --------------------------------------------
                # Process each S1 entity
                # --------------------------------------------

                for query in normalized_chunk.to_dict(
                    orient="records"
                ):

                    s1_id = str(
                        query["entity_id"]
                    )

                    # ----------------------------------------
                    # Candidate generation
                    # ----------------------------------------

                    candidates = (
                        generate_candidates_for_record(
                            query,
                            indexes,
                            lookups,
                            CONFIG,
                        )
                    )

                    # ----------------------------------------
                    # Flatten candidate set
                    # ----------------------------------------

                    candidate_ids = []

                    seen = set()

                    for source_name in (
                        "source2",
                        "source3",
                    ):

                        for candidate_id in candidates.get(
                            source_name,
                            [],
                        ):

                            candidate_id = str(
                                candidate_id
                            )

                            if candidate_id in seen:
                                continue

                            seen.add(
                                candidate_id
                            )

                            candidate_ids.append(
                                candidate_id
                            )

                    # ------------------------------------------------
                    # IMPORTANT:
                    #
                    # candidate_pairs is EXACTLY the candidate
                    # set sent into the model.
                    # ------------------------------------------------

                    if candidate_ids:

                        entities_with_candidates += 1

                    total_candidates += len(
                        candidate_ids
                    )

                    # ----------------------------------------
                    # Model scoring
                    # ----------------------------------------

                    (
                        scored_ids,
                        probabilities,
                        matches,
                    ) = score_entity(
                        query,
                        candidates,
                        lookups,
                        model,
                    )

                    # ----------------------------------------
                    # Safety:
                    #
                    # final matches MUST be candidates.
                    # ----------------------------------------

                    candidate_set = set(
                        candidate_ids
                    )

                    matches = [
                        entity_id
                        for entity_id in matches
                        if entity_id in candidate_set
                    ]

                    # Remove duplicates.
                    matches = list(
                        dict.fromkeys(
                            matches
                        )
                    )

                    # ----------------------------------------
                    # Statistics
                    # ----------------------------------------

                    if matches:

                        entities_with_matches += 1

                    total_matches += len(
                        matches
                    )

                    if probabilities:

                        probability_sum += float(
                            np.sum(
                                probabilities
                            )
                        )

                        probability_count += len(
                            probabilities
                        )

                    total += 1

                    # ----------------------------------------
                    # Output
                    # ----------------------------------------

                    matching_lines.append(
                        s1_id
                        + "\t"
                        + ",".join(matches)
                        + "\n"
                    )

                    candidate_lines.append(
                        s1_id
                        + "\t"
                        + ",".join(candidate_ids)
                        + "\n"
                    )

                # --------------------------------------------
                # Write current chunk
                # --------------------------------------------

                matching_file.writelines(
                    matching_lines
                )

                candidate_file.writelines(
                    candidate_lines
                )

                # --------------------------------------------
                # Progress
                # --------------------------------------------

                if total % 25_000 == 0:

                    avg_candidates = (
                        total_candidates / total
                        if total
                        else 0.0
                    )

                    avg_matches = (
                        total_matches / total
                        if total
                        else 0.0
                    )

                    avg_probability = (
                        probability_sum
                        / probability_count
                        if probability_count
                        else 0.0
                    )

                    print(
                        f"S1: {total:,} | "
                        f"candidate_entities: "
                        f"{entities_with_candidates:,} | "
                        f"matched_entities: "
                        f"{entities_with_matches:,} | "
                        f"avg_candidates: "
                        f"{avg_candidates:.2f} | "
                        f"avg_matches: "
                        f"{avg_matches:.2f} | "
                        f"avg_probability: "
                        f"{avg_probability:.4f}"
                    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    print()
    print("=" * 80)
    print("FINAL VALIDATION")
    print("=" * 80)

    # Test S1 exact count.
    with open(
        S1_PATH,
        "rb",
    ) as f:

        s1_count = sum(
            1
            for _ in f
        ) - 1

    # Output counts.
    with open(
        FINAL_MATCHING_PATH,
        "rb",
    ) as f:

        matching_count = (
            sum(
                1
                for _ in f
            )
            - 1
        )

    with open(
        FINAL_CANDIDATE_PATH,
        "rb",
    ) as f:

        candidate_count = (
            sum(
                1
                for _ in f
            )
            - 1
        )

    print(
        f"Expected S1 rows:       "
        f"{s1_count:,}"
    )

    print(
        f"Matching output rows:   "
        f"{matching_count:,}"
    )

    print(
        f"Candidate output rows:  "
        f"{candidate_count:,}"
    )

    print()
    print(
        f"Entities with candidates: "
        f"{entities_with_candidates:,}"
    )

    print(
        f"Entities with matches:    "
        f"{entities_with_matches:,}"
    )

    print(
        f"Total predicted matches:  "
        f"{total_matches:,}"
    )

    print(
        f"Average candidates/entity: "
        f"{total_candidates / total:.3f}"
        if total
        else
        "Average candidates/entity: 0"
    )

    print(
        f"Average matches/entity:    "
        f"{total_matches / total:.3f}"
        if total
        else
        "Average matches/entity: 0"
    )

    # --------------------------------------------------------
    # Required structural checks
    # --------------------------------------------------------

    if matching_count != s1_count:

        raise RuntimeError(
            "ERROR: matching_results_ml.tsv "
            "does not contain exactly one row "
            "for every test Source-1 entity."
        )

    if candidate_count != s1_count:

        raise RuntimeError(
            "ERROR: candidate_pairs_ml.tsv "
            "does not contain exactly one row "
            "for every test Source-1 entity."
        )

    print()
    print("=" * 80)
    print("MODEL INFERENCE COMPLETE")
    print("=" * 80)

    print()
    print(
        "Matching file:"
    )

    print(
        FINAL_MATCHING_PATH
    )

    print()
    print(
        "Candidate file:"
    )

    print(
        FINAL_CANDIDATE_PATH
    )

    print()
    print(
        "DO NOT overwrite the old "
        "matching_results.tsv yet."
    )


if __name__ == "__main__":
    main()