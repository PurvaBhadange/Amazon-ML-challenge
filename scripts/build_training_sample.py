from pathlib import Path
import hashlib

import pandas as pd

from src.blocking import (
    BlockingConfig,
    build_blocking_index,
    build_source_lookup,
    generate_candidates_for_record,
    make_blocking_keys,
    parse_ground_truth,
)
from src.normalize import normalize_dataframe
from src.features import (
    build_pair_features,
    feature_columns,
)


ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = ROOT / "dataset" / "train"
OUTPUT_DIR = ROOT / "output"

SOURCE1_PATH = TRAIN_DIR / "train_source1.tsv"
SOURCE2_PATH = TRAIN_DIR / "train_source2.tsv"
SOURCE3_PATH = TRAIN_DIR / "train_source3.tsv"
GROUND_TRUTH_PATH = TRAIN_DIR / "train_ground_truth.tsv"

TRAINING_OUTPUT = OUTPUT_DIR / "training_pairs.tsv"
VALIDATION_OUTPUT = OUTPUT_DIR / "validation_pairs.tsv"

S1_SAMPLE_SIZE = 10_000

MAX_HARD_NEGATIVES_PER_ENTITY = 75

RANDOM_NEGATIVES_SOURCE2 = 10_000
RANDOM_NEGATIVES_SOURCE3 = 10_000

RANDOM_STATE = 42


# ============================================================
# GENERAL HELPERS
# ============================================================

def stable_seed(value, offset=0):
    digest = hashlib.md5(
        str(value).encode("utf-8")
    ).hexdigest()

    return (
        int(digest[:8], 16)
        + RANDOM_STATE
        + offset
    ) % (2**32 - 1)


def vector_to_feature_dict(vector):
    columns = feature_columns()

    if len(vector) != len(columns):
        raise ValueError(
            f"Feature count mismatch: "
            f"vector={len(vector)}, "
            f"columns={len(columns)}"
        )

    return {
        column: float(value)
        for column, value in zip(
            columns,
            vector,
        )
    }


# ============================================================
# BLOCKING KEY ADAPTER
# ============================================================

def normalize_blocking_keys(keys):
    """
    Convert supported blocking-key formats into:

        {
            key_type: {key_value, ...}
        }

    Supported:

        [
            ("name_first", "acme"),
            ("postal", "400001")
        ]

    and:

        {
            "name_first": ["acme"],
            "postal": ["400001"]
        }
    """

    if keys is None:
        return {}

    result = {}

    if isinstance(keys, dict):
        for key_type, values in keys.items():

            if values is None:
                continue

            if isinstance(
                values,
                (list, tuple, set),
            ):
                clean_values = {
                    str(value)
                    for value in values
                    if value is not None
                    and str(value)
                }

                if clean_values:
                    result.setdefault(
                        str(key_type),
                        set(),
                    ).update(
                        clean_values
                    )

            else:
                value = str(values)

                if value:
                    result.setdefault(
                        str(key_type),
                        set(),
                    ).add(value)

        return result

    if isinstance(
        keys,
        (list, tuple, set),
    ):
        for item in keys:

            if not isinstance(
                item,
                (list, tuple),
            ):
                continue

            if len(item) != 2:
                continue

            key_type = str(item[0])
            value = item[1]

            if value is None:
                continue

            value = str(value)

            if not value:
                continue

            result.setdefault(
                key_type,
                set(),
            ).add(value)

    return result


# ============================================================
# CANDIDATE RESULT ADAPTER
# ============================================================

def _extract_candidate_id(candidate):
    """
    Extract entity ID from all candidate representations
    encountered in the project.
    """

    if candidate is None:
        return None

    if isinstance(
        candidate,
        str,
    ):
        return candidate.strip()

    if isinstance(
        candidate,
        dict,
    ):
        for key in (
            "entity_id",
            "candidate_entity_id",
            "candidate_id",
            "id",
        ):
            value = candidate.get(key)

            if value is not None:
                value = str(value).strip()

                if value:
                    return value

        return None

    for attr in (
        "entity_id",
        "candidate_entity_id",
        "candidate_id",
        "id",
    ):
        value = getattr(
            candidate,
            attr,
            None,
        )

        if value is not None:
            value = str(value).strip()

            if value:
                return value

    if hasattr(
        candidate,
        "_asdict",
    ):
        try:
            data = candidate._asdict()

            for key in (
                "entity_id",
                "candidate_entity_id",
                "candidate_id",
                "id",
            ):
                value = data.get(key)

                if value is not None:
                    value = str(value).strip()

                    if value:
                        return value
        except Exception:
            pass

    return None


def _extract_candidate_source(candidate):
    """
    Extract source if the candidate object provides one.
    """

    if candidate is None:
        return None

    if isinstance(
        candidate,
        dict,
    ):
        for key in (
            "source",
            "candidate_source",
            "source_name",
        ):
            value = candidate.get(key)

            if value is not None:
                return str(value).strip()

        return None

    for attr in (
        "source",
        "candidate_source",
        "source_name",
    ):
        value = getattr(
            candidate,
            attr,
            None,
        )

        if value is not None:
            return str(value).strip()

    if hasattr(
        candidate,
        "_asdict",
    ):
        try:
            data = candidate._asdict()

            for key in (
                "source",
                "candidate_source",
                "source_name",
            ):
                value = data.get(key)

                if value is not None:
                    return str(value).strip()
        except Exception:
            pass

    return None


def _resolve_candidate_source(
    candidate_id,
    candidate_source,
    lookups,
):
    """
    Resolve source using actual lookup membership.

    Lookup membership is authoritative.
    """

    candidate_id = str(
        candidate_id
    ).strip()

    source2_lookup = lookups.get(
        "source2",
        {},
    )

    source3_lookup = lookups.get(
        "source3",
        {},
    )

    in_source2 = (
        candidate_id
        in source2_lookup
    )

    in_source3 = (
        candidate_id
        in source3_lookup
    )

    # The actual lookup wins.
    if in_source2 and not in_source3:
        return "source2"

    if in_source3 and not in_source2:
        return "source3"

    # If both exist, use the supplied source.
    if in_source2 and in_source3:
        if candidate_source in (
            "source2",
            "source3",
        ):
            return candidate_source

        if candidate_id.startswith(
            ("S2-", "S2_")
        ):
            return "source2"

        if candidate_id.startswith(
            ("S3-", "S3_")
        ):
            return "source3"

    # Normalize source labels.
    if candidate_source:
        source = str(
            candidate_source
        ).lower().strip()

        if source in (
            "source2",
            "source_2",
            "s2",
            "2",
        ):
            if in_source2:
                return "source2"

        if source in (
            "source3",
            "source_3",
            "s3",
            "3",
        ):
            if in_source3:
                return "source3"

    # Last fallback: ID prefix.
    if candidate_id.startswith(
        ("S2-", "S2_")
    ):
        if in_source2:
            return "source2"

    if candidate_id.startswith(
        ("S3-", "S3_")
    ):
        if in_source3:
            return "source3"

    return None


def normalize_candidate_result(
    candidates,
    lookups,
):
    """
    Convert ANY supported blocker output into:

        [
            ("source2", "S2-..."),
            ("source3", "S3-...")
        ]

    This deliberately does not assume that
    generate_candidates_for_record() returns a particular type.
    """

    normalized = []

    def add_candidate(
        candidate,
        explicit_source=None,
    ):
        candidate_id = _extract_candidate_id(
            candidate
        )

        if not candidate_id:
            return

        candidate_source = (
            explicit_source
            or
            _extract_candidate_source(
                candidate
            )
        )

        resolved_source = (
            _resolve_candidate_source(
                candidate_id,
                candidate_source,
                lookups,
            )
        )

        if resolved_source is None:
            return

        normalized.append(
            (
                resolved_source,
                candidate_id,
            )
        )

    if candidates is None:
        return []

    # --------------------------------------------------------
    # Dictionary API
    # --------------------------------------------------------

    if isinstance(
        candidates,
        dict,
    ):

        # Source-separated dictionary.
        if (
            "source2" in candidates
            or
            "source3" in candidates
        ):

            for source_name in (
                "source2",
                "source3",
            ):

                values = candidates.get(
                    source_name,
                    [],
                )

                if values is None:
                    continue

                if not isinstance(
                    values,
                    (list, tuple, set),
                ):
                    values = [values]

                for value in values:
                    add_candidate(
                        value,
                        source_name,
                    )

            return list(
                dict.fromkeys(
                    normalized
                )
            )

        # Single candidate dictionary.
        add_candidate(
            candidates
        )

        return list(
            dict.fromkeys(
                normalized
            )
        )

    # --------------------------------------------------------
    # List / tuple / set API
    # --------------------------------------------------------

    if isinstance(
        candidates,
        (list, tuple, set),
    ):

        for candidate in candidates:

            if candidate is None:
                continue

            # Tuple of:
            # (entity_id, source)
            if isinstance(
                candidate,
                (tuple, list),
            ):
                if len(candidate) >= 1:

                    entity_id = candidate[0]

                    source = (
                        candidate[1]
                        if len(candidate) >= 2
                        else None
                    )

                    add_candidate(
                        entity_id,
                        source,
                    )

                continue

            add_candidate(
                candidate
            )

        return list(
            dict.fromkeys(
                normalized
            )
        )

    # --------------------------------------------------------
    # Single Candidate object
    # --------------------------------------------------------

    add_candidate(
        candidates
    )

    return list(
        dict.fromkeys(
            normalized
        )
    )


# ============================================================
# DATA LOADING
# ============================================================

def load_data():

    print("Loading Source 1...")

    source1 = pd.read_csv(
        SOURCE1_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print("Loading Source 2...")

    source2 = pd.read_csv(
        SOURCE2_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print("Loading Source 3...")

    source3 = pd.read_csv(
        SOURCE3_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print("Loading ground truth...")

    ground_truth_df = pd.read_csv(
        GROUND_TRUTH_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    ground_truth = parse_ground_truth(
        ground_truth_df
    )

    return (
        source1,
        source2,
        source3,
        ground_truth,
    )


# ============================================================
# SOURCE 1 SAMPLING
# ============================================================

def sample_source1(source1):

    if len(source1) <= S1_SAMPLE_SIZE:
        return source1.copy()

    return (
        source1
        .sample(
            n=S1_SAMPLE_SIZE,
            random_state=RANDOM_STATE,
        )
        .reset_index(drop=True)
    )


# ============================================================
# REQUIRED MATCH RECORDS
# ============================================================

def collect_required_records(
    source1_sample,
    source2,
    source3,
    ground_truth,
):

    required_source2 = set()
    required_source3 = set()

    for s1_id in source1_sample[
        "entity_id"
    ].astype(str):

        truth = ground_truth.get(
            s1_id,
            set(),
        )

        for entity_id in truth:

            entity_id = str(
                entity_id
            )

            if entity_id.startswith(
                ("S2-", "S2_")
            ):
                required_source2.add(
                    entity_id
                )

            elif entity_id.startswith(
                ("S3-", "S3_")
            ):
                required_source3.add(
                    entity_id
                )

    print()
    print(
        "Required Source2 IDs:",
        len(required_source2),
    )

    print(
        "Required Source3 IDs:",
        len(required_source3),
    )

    source2_ids = set(
        source2[
            "entity_id"
        ].astype(str)
    )

    source3_ids = set(
        source3[
            "entity_id"
        ].astype(str)
    )

    missing_s2 = (
        required_source2
        - source2_ids
    )

    missing_s3 = (
        required_source3
        - source3_ids
    )

    if missing_s2:
        raise RuntimeError(
            f"{len(missing_s2):,} required "
            "Source2 IDs are absent from "
            "train_source2.tsv."
        )

    if missing_s3:
        raise RuntimeError(
            f"{len(missing_s3):,} required "
            "Source3 IDs are absent from "
            "train_source3.tsv."
        )

    source2_required = source2[
        source2[
            "entity_id"
        ].astype(str).isin(
            required_source2
        )
    ].copy()

    source3_required = source3[
        source3[
            "entity_id"
        ].astype(str).isin(
            required_source3
        )
    ].copy()

    return (
        source2_required,
        source3_required,
    )


# ============================================================
# RANDOM BACKGROUND RECORDS
# ============================================================

def collect_random_rows(
    source,
    required_ids,
    n,
    random_state,
):

    required_ids = {
        str(x)
        for x in required_ids
    }

    available = source[
        ~source[
            "entity_id"
        ].astype(str).isin(
            required_ids
        )
    ]

    if len(available) <= n:
        return available.copy()

    return (
        available
        .sample(
            n=n,
            random_state=random_state,
        )
        .copy()
    )


# ============================================================
# NORMALIZATION
# ============================================================

def prepare_normalized_source(
    source,
):

    return normalize_dataframe(
        source,
        include_phonetic=True,
    )


# ============================================================
# BLOCKING EVIDENCE
# ============================================================

def evidence_between(
    query,
    candidate,
    config,
):

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
        &
        set(candidate_keys)
    )

    for key_type in common_types:

        if (
            query_keys[key_type]
            &
            candidate_keys[key_type]
        ):
            evidence.add(
                key_type
            )

    return evidence


# ============================================================
# BUILD HARD-NEGATIVE TRAINING ROWS
# ============================================================

def build_candidate_training_rows(
    source1_norm,
    source2_norm,
    source3_norm,
    ground_truth,
    config,
):

    print()
    print(
        "Building blocking indexes..."
    )

    indexes = {
        "source2": build_blocking_index(
            source2_norm,
            config,
        ),
        "source3": build_blocking_index(
            source3_norm,
            config,
        ),
    }

    lookups = {
        "source2": build_source_lookup(
            source2_norm
        ),
        "source3": build_source_lookup(
            source3_norm
        ),
    }

    source1_lookup = build_source_lookup(
        source1_norm
    )

    rows = []

    print()
    print(
        "Generating candidate pairs..."
    )

    total_entities = len(
        source1_norm
    )

    diagnostic_candidates = 0
    diagnostic_resolved = 0
    diagnostic_positive = 0

    for position, row in enumerate(
        source1_norm.itertuples(
            index=False
        ),
        start=1,
    ):

        s1_id = str(
            row.entity_id
        )

        query = source1_lookup.get(
            s1_id
        )

        if query is None:
            continue

        truth = {
            str(x)
            for x in ground_truth.get(
                s1_id,
                set(),
            )
        }

        candidates = (
            generate_candidates_for_record(
                query,
                indexes,
                lookups,
                config,
            )
        )

        normalized_candidates = (
            normalize_candidate_result(
                candidates,
                lookups,
            )
        )

        diagnostic_candidates += (
            len(normalized_candidates)
        )

        negative_count = 0

        for (
            candidate_source,
            candidate_id,
        ) in normalized_candidates:

            candidate_lookup = (
                lookups[
                    candidate_source
                ]
            )

            candidate_record = (
                candidate_lookup.get(
                    candidate_id
                )
            )

            if candidate_record is None:
                continue

            diagnostic_resolved += 1

            label = int(
                candidate_id in truth
            )

            if label == 1:
                diagnostic_positive += 1

            else:
                if (
                    negative_count
                    >= MAX_HARD_NEGATIVES_PER_ENTITY
                ):
                    continue

                negative_count += 1

            evidence = evidence_between(
                query,
                candidate_record,
                config,
            )

            vector = build_pair_features(
                query,
                candidate_record,
                evidence,
            )

            features = (
                vector_to_feature_dict(
                    vector
                )
            )

            features[
                "source1_entity_id"
            ] = s1_id

            features[
                "candidate_entity_id"
            ] = candidate_id

            features[
                "candidate_source"
            ] = candidate_source

            features[
                "label"
            ] = label

            rows.append(
                features
            )

        if position % 500 == 0:

            print(
                f"Candidate pairs generated: "
                f"{position:,}/"
                f"{total_entities:,}"
                f" | rows={len(rows):,}"
            )

    print()
    print(
        "Blocking diagnostic:"
    )

    print(
        "  Candidate objects resolved:",
        f"{diagnostic_candidates:,}",
    )

    print(
        "  Candidate records used:",
        f"{diagnostic_resolved:,}",
    )

    print(
        "  Positive candidates:",
        f"{diagnostic_positive:,}",
    )

    if diagnostic_candidates == 0:

        raise RuntimeError(
            "\n"
            "BLOCKING RETURNED ZERO USABLE "
            "CANDIDATES.\n"
            "The training script and blocking "
            "module are still using incompatible "
            "candidate representations.\n"
            "\n"
            "Do NOT continue to model training."
        )

    if not rows:

        raise RuntimeError(
            "Blocking returned candidates, "
            "but no training rows were produced."
        )

    return pd.DataFrame(rows)


# ============================================================
# CONTROLLED RANDOM NEGATIVES
# ============================================================

def add_random_negatives(
    pairs,
    source1_sample,
    source2_norm,
    source3_norm,
    ground_truth,
):

    if pairs.empty:
        raise RuntimeError(
            "Cannot add random negatives "
            "to an empty training set."
        )

    existing_pairs = set(
        zip(
            pairs[
                "source1_entity_id"
            ].astype(str),
            pairs[
                "candidate_entity_id"
            ].astype(str),
        )
    )

    source1_records = {
        str(row.entity_id): row
        for row in source1_sample.itertuples(
            index=False
        )
    }

    source2_records = {
        str(row.entity_id): row
        for row in source2_norm.itertuples(
            index=False
        )
    }

    source3_records = {
        str(row.entity_id): row
        for row in source3_norm.itertuples(
            index=False
        )
    }

    random_rows = []

    def generate_random_pairs(
        source_records,
        source_name,
        target_count,
        seed_offset,
    ):

        if not source_records:
            return

        source_ids = list(
            source_records.keys()
        )

        s1_ids = list(
            source1_records.keys()
        )

        if not s1_ids:
            return

        pair_candidates = []

        for index in range(
            target_count
        ):

            s1_id = s1_ids[
                (
                    index
                    + seed_offset
                )
                % len(s1_ids)
            ]

            candidate_id = source_ids[
                (
                    index * 104729
                    + seed_offset
                )
                % len(source_ids)
            ]

            truth = {
                str(x)
                for x in ground_truth.get(
                    s1_id,
                    set(),
                )
            }

            if candidate_id in truth:
                continue

            pair_key = (
                s1_id,
                candidate_id,
            )

            if pair_key in existing_pairs:
                continue

            pair_candidates.append(
                pair_key
            )

            existing_pairs.add(
                pair_key
            )

        for (
            s1_id,
            candidate_id,
        ) in pair_candidates:

            query = source1_records[
                s1_id
            ]

            candidate = source_records[
                candidate_id
            ]

            vector = build_pair_features(
                query,
                candidate,
                set(),
            )

            features = (
                vector_to_feature_dict(
                    vector
                )
            )

            features[
                "source1_entity_id"
            ] = s1_id

            features[
                "candidate_entity_id"
            ] = candidate_id

            features[
                "candidate_source"
            ] = source_name

            features[
                "label"
            ] = 0

            random_rows.append(
                features
            )

    generate_random_pairs(
        source2_records,
        "source2",
        RANDOM_NEGATIVES_SOURCE2,
        101,
    )

    generate_random_pairs(
        source3_records,
        "source3",
        RANDOM_NEGATIVES_SOURCE3,
        202,
    )

    if not random_rows:
        return pairs

    return pd.concat(
        [
            pairs,
            pd.DataFrame(
                random_rows
            ),
        ],
        ignore_index=True,
    )


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

def split_by_entity(pairs):

    entities = (
        pairs[
            "source1_entity_id"
        ]
        .astype(str)
        .drop_duplicates()
        .sample(
            frac=1.0,
            random_state=RANDOM_STATE,
        )
        .tolist()
    )

    split_index = int(
        len(entities) * 0.80
    )

    train_entities = set(
        entities[:split_index]
    )

    validation_entities = set(
        entities[split_index:]
    )

    train = pairs[
        pairs[
            "source1_entity_id"
        ].astype(str).isin(
            train_entities
        )
    ].copy()

    validation = pairs[
        pairs[
            "source1_entity_id"
        ].astype(str).isin(
            validation_entities
        )
    ].copy()

    overlap = (
        set(
            train[
                "source1_entity_id"
            ]
        )
        &
        set(
            validation[
                "source1_entity_id"
            ]
        )
    )

    if overlap:
        raise RuntimeError(
            "Entity leakage detected: "
            f"{len(overlap):,} entities."
        )

    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_training_pairs(
    pairs,
    ground_truth,
):

    feature_cols = feature_columns()

    required_columns = (
        feature_cols
        + [
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
            "label",
        ]
    )

    missing = [
        column
        for column in required_columns
        if column not in pairs.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing training columns: "
            f"{missing}"
        )

    if pairs.empty:
        raise RuntimeError(
            "Training pairs are empty."
        )

    if pairs[
        feature_cols
    ].isna().any().any():

        raise RuntimeError(
            "NaN values found in "
            "feature matrix."
        )

    duplicate_count = (
        pairs
        .duplicated(
            subset=[
                "source1_entity_id",
                "candidate_entity_id",
            ]
        )
        .sum()
    )

    if duplicate_count:
        raise RuntimeError(
            "Duplicate candidate pairs: "
            f"{duplicate_count:,}"
        )

    labels = set(
        pairs[
            "label"
        ].unique()
    )

    if not labels.issubset(
        {0, 1}
    ):
        raise RuntimeError(
            f"Invalid labels: {labels}"
        )

    positive_pairs = pairs[
        pairs["label"] == 1
    ]

    invalid_positive_count = 0

    for row in positive_pairs.itertuples(
        index=False
    ):

        truth = {
            str(x)
            for x in ground_truth.get(
                str(
                    row.source1_entity_id
                ),
                set(),
            )
        }

        if (
            str(
                row.candidate_entity_id
            )
            not in truth
        ):
            invalid_positive_count += 1

    if invalid_positive_count:
        raise RuntimeError(
            "Invalid positive rows: "
            f"{invalid_positive_count:,}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        source1,
        source2,
        source3,
        ground_truth,
    ) = load_data()

    source1_sample = sample_source1(
        source1
    )

    print()
    print(
        "Source-1 training sample:",
        f"{len(source1_sample):,}",
    )

    (
        source2_required,
        source3_required,
    ) = collect_required_records(
        source1_sample,
        source2,
        source3,
        ground_truth,
    )

    print(
        "Source2 required rows:",
        f"{len(source2_required):,}",
    )

    print(
        "Source3 required rows:",
        f"{len(source3_required):,}",
    )

    source2_random = collect_random_rows(
        source2,
        source2_required[
            "entity_id"
        ],
        50_000,
        RANDOM_STATE,
    )

    source3_random = collect_random_rows(
        source3,
        source3_required[
            "entity_id"
        ],
        50_000,
        RANDOM_STATE + 1,
    )

    source2_working = (
        pd.concat(
            [
                source2_required,
                source2_random,
            ],
            ignore_index=True,
        )
        .drop_duplicates(
            subset=["entity_id"]
        )
        .reset_index(drop=True)
    )

    source3_working = (
        pd.concat(
            [
                source3_required,
                source3_random,
            ],
            ignore_index=True,
        )
        .drop_duplicates(
            subset=["entity_id"]
        )
        .reset_index(drop=True)
    )

    print()
    print(
        "Source2 working rows:",
        f"{len(source2_working):,}",
    )

    print(
        "Source3 working rows:",
        f"{len(source3_working):,}",
    )

    print()
    print(
        "Normalizing Source 1..."
    )

    source1_norm = (
        prepare_normalized_source(
            source1_sample
        )
    )

    print(
        "Normalizing Source 2..."
    )

    source2_norm = (
        prepare_normalized_source(
            source2_working
        )
    )

    print(
        "Normalizing Source 3..."
    )

    source3_norm = (
        prepare_normalized_source(
            source3_working
        )
    )

    config = BlockingConfig(
        max_final_candidates=250
    )

    pairs = (
        build_candidate_training_rows(
            source1_norm,
            source2_norm,
            source3_norm,
            ground_truth,
            config,
        )
    )

    print()
    print(
        "Blocking candidate rows:",
        f"{len(pairs):,}",
    )

    print(
        "Positive rows:",
        f"{int(pairs['label'].sum()):,}",
    )

    print(
        "Negative rows:",
        f"{int((pairs['label'] == 0).sum()):,}",
    )

    print()
    print(
        "Adding controlled random negatives..."
    )

    pairs = add_random_negatives(
        pairs,
        source1_sample,
        source2_norm,
        source3_norm,
        ground_truth,
    )

    pairs = (
        pairs
        .drop_duplicates(
            subset=[
                "source1_entity_id",
                "candidate_entity_id",
            ]
        )
        .reset_index(drop=True)
    )

    feature_cols = feature_columns()

    metadata_cols = [
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
        "label",
    ]

    pairs = pairs[
        feature_cols
        + metadata_cols
    ]

    validate_training_pairs(
        pairs,
        ground_truth,
    )

    pairs = (
        pairs
        .sample(
            frac=1.0,
            random_state=RANDOM_STATE,
        )
        .reset_index(drop=True)
    )

    train, validation = (
        split_by_entity(
            pairs
        )
    )

    # Final sanity checks.
    if len(
        set(
            train[
                "source1_entity_id"
            ]
        )
        &
        set(
            validation[
                "source1_entity_id"
            ]
        )
    ):
        raise RuntimeError(
            "Fatal train/validation "
            "entity leakage."
        )

    train.to_csv(
        TRAINING_OUTPUT,
        sep="\t",
        index=False,
    )

    validation.to_csv(
        VALIDATION_OUTPUT,
        sep="\t",
        index=False,
    )

    print()
    print("=" * 80)
    print(
        "TRAINING SAMPLE COMPLETE"
    )
    print("=" * 80)

    print(
        "Total entities:",
        pairs[
            "source1_entity_id"
        ].nunique(),
    )

    print(
        "Total pairs:",
        f"{len(pairs):,}",
    )

    print(
        "Total positives:",
        f"{int(pairs['label'].sum()):,}",
    )

    print(
        "Total negatives:",
        f"{int((pairs['label'] == 0).sum()):,}",
    )

    print()

    print(
        "Training entities:",
        train[
            "source1_entity_id"
        ].nunique(),
    )

    print(
        "Training pairs:",
        f"{len(train):,}",
    )

    print(
        "Training positives:",
        f"{int(train['label'].sum()):,}",
    )

    print()

    print(
        "Validation entities:",
        validation[
            "source1_entity_id"
        ].nunique(),
    )

    print(
        "Validation pairs:",
        f"{len(validation):,}",
    )

    print(
        "Validation positives:",
        f"{int(validation['label'].sum()):,}",
    )

    print()

    print(
        "Feature columns:",
        len(feature_cols),
    )

    print(
        "Saved:",
        TRAINING_OUTPUT,
    )

    print(
        "Saved:",
        VALIDATION_OUTPUT,
    )


if __name__ == "__main__":
    main()