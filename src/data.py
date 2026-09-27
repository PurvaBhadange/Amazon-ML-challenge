"""
Data loading and dataset auditing utilities.

Responsibilities:
    1. Validate dataset files.
    2. Read TSV files safely.
    3. Stream large files using chunks.
    4. Validate source schemas.
    5. Validate ground-truth schema.
    6. Audit missing/empty values.
    7. Detect duplicate entity IDs.
    8. Audit country distributions.

This module does NOT perform:
    - normalization
    - blocking
    - feature engineering
    - model training
    - matching

Those will be implemented in later modules.
"""

from pathlib import Path
from typing import Iterator

import pandas as pd

from src.config import (
    SOURCE_COLUMNS,
    GROUND_TRUTH_COLUMNS,
    TSV_SEPARATOR,
    ENCODING,
    CHUNK_SIZE,
    SOURCE_DTYPES,
    GROUND_TRUTH_DTYPES,
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
)


# ============================================================
# 1. FILE VALIDATION
# ============================================================

def validate_file_exists(path: Path) -> None:
    """
    Verify that a dataset file exists and is actually a file.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file does not exist:\n{path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Expected a file but found something else:\n{path}"
        )


# ============================================================
# 2. SOURCE SCHEMA VALIDATION
# ============================================================

def validate_source_schema(
    columns,
    path: Path,
) -> None:
    """
    Validate the schema of Source 1, Source 2 or Source 3.

    Required columns:
        entity_id
        business_name
        business_address
        country
    """

    columns = list(columns)

    missing_columns = [
        column
        for column in SOURCE_COLUMNS
        if column not in columns
    ]

    if missing_columns:
        raise ValueError(
            f"\nInvalid source schema: {path}\n"
            f"Missing columns: {missing_columns}\n"
            f"Available columns: {columns}"
        )


# ============================================================
# 3. GROUND-TRUTH SCHEMA VALIDATION
# ============================================================

def validate_ground_truth_schema(
    columns,
    path: Path,
) -> None:
    """
    Validate the ground-truth file schema.

    Required columns:
        source1_entity_id
        matched_entity_ids
    """

    columns = list(columns)

    missing_columns = [
        column
        for column in GROUND_TRUTH_COLUMNS
        if column not in columns
    ]

    if missing_columns:
        raise ValueError(
            f"\nInvalid ground-truth schema: {path}\n"
            f"Missing columns: {missing_columns}\n"
            f"Available columns: {columns}"
        )


# ============================================================
# 4. READ SOURCE HEADER
# ============================================================

def read_source_header(path: Path) -> list[str]:
    """
    Read only the header of a source TSV.

    This does not load the actual dataset into memory.
    """

    validate_file_exists(path)

    df = pd.read_csv(
        path,
        sep=TSV_SEPARATOR,
        encoding=ENCODING,
        nrows=0,
    )

    validate_source_schema(df.columns, path)

    return list(df.columns)


# ============================================================
# 5. READ GROUND-TRUTH HEADER
# ============================================================

def read_ground_truth_header(path: Path) -> list[str]:
    """
    Read only the header of the ground-truth TSV.
    """

    validate_file_exists(path)

    df = pd.read_csv(
        path,
        sep=TSV_SEPARATOR,
        encoding=ENCODING,
        nrows=0,
    )

    validate_ground_truth_schema(df.columns, path)

    return list(df.columns)


# ============================================================
# 6. READ SMALL SOURCE SAMPLE
# ============================================================

def read_source_sample(
    path: Path,
    nrows: int = 10,
) -> pd.DataFrame:
    """
    Read a small sample from a source dataset.

    Useful for notebook inspection.
    """

    validate_file_exists(path)

    df = pd.read_csv(
        path,
        sep=TSV_SEPARATOR,
        encoding=ENCODING,
        dtype=SOURCE_DTYPES,
        keep_default_na=False,
        nrows=nrows,
    )

    validate_source_schema(df.columns, path)

    return df


# ============================================================
# 7. READ SMALL GROUND-TRUTH SAMPLE
# ============================================================

def read_ground_truth_sample(
    path: Path,
    nrows: int = 10,
) -> pd.DataFrame:
    """
    Read a small sample from the ground-truth dataset.
    """

    validate_file_exists(path)

    df = pd.read_csv(
        path,
        sep=TSV_SEPARATOR,
        encoding=ENCODING,
        dtype=GROUND_TRUTH_DTYPES,
        keep_default_na=False,
        nrows=nrows,
    )

    validate_ground_truth_schema(df.columns, path)

    return df


# ============================================================
# 8. STREAM SOURCE DATA
# ============================================================

def iter_source_chunks(
    path: Path,
    chunksize: int = CHUNK_SIZE,
) -> Iterator[pd.DataFrame]:
    """
    Stream a source TSV chunk-by-chunk.

    This is the primary mechanism we will use for
    very large datasets.
    """

    validate_file_exists(path)

    reader = pd.read_csv(
        path,
        sep=TSV_SEPARATOR,
        encoding=ENCODING,
        dtype=SOURCE_DTYPES,
        keep_default_na=False,
        chunksize=chunksize,
    )

    first_chunk = True

    for chunk in reader:

        if first_chunk:
            validate_source_schema(chunk.columns, path)
            first_chunk = False

        yield chunk


# ============================================================
# 9. STREAM GROUND TRUTH
# ============================================================

def iter_ground_truth_chunks(
    path: Path,
    chunksize: int = CHUNK_SIZE,
) -> Iterator[pd.DataFrame]:
    """
    Stream ground-truth TSV chunk-by-chunk.
    """

    validate_file_exists(path)

    reader = pd.read_csv(
        path,
        sep=TSV_SEPARATOR,
        encoding=ENCODING,
        dtype=GROUND_TRUTH_DTYPES,
        keep_default_na=False,
        chunksize=chunksize,
    )

    first_chunk = True

    for chunk in reader:

        if first_chunk:
            validate_ground_truth_schema(
                chunk.columns,
                path,
            )
            first_chunk = False

        yield chunk


# ============================================================
# 10. SOURCE DATASET AUDIT
# ============================================================

def audit_source(
    path: Path,
    chunksize: int = CHUNK_SIZE,
) -> dict:
    """
    Perform a memory-efficient audit of a source dataset.

    The complete file is NOT loaded into memory.

    Returns:
        rows
        columns
        missing counts
        empty-string counts
        duplicate IDs
        unique ID count
        country distribution
    """

    validate_file_exists(path)

    total_rows = 0

    columns = None

    missing_counts = {
        column: 0
        for column in SOURCE_COLUMNS
    }

    empty_counts = {
        column: 0
        for column in SOURCE_COLUMNS
    }

    unique_entity_ids = set()
    duplicate_entity_ids = set()

    country_counts = {}

    for chunk in iter_source_chunks(
        path,
        chunksize=chunksize,
    ):

        if columns is None:
            columns = list(chunk.columns)

        total_rows += len(chunk)

        # ----------------------------------------------------
        # Missing / empty values
        # ----------------------------------------------------

        for column in SOURCE_COLUMNS:

            values = chunk[column]

            missing_counts[column] += int(
                values.isna().sum()
            )

            empty_counts[column] += int(
                values.astype("string")
                .str.strip()
                .eq("")
                .sum()
            )

        # ----------------------------------------------------
        # Duplicate entity IDs
        # ----------------------------------------------------

        entity_ids = (
            chunk["entity_id"]
            .astype("string")
            .str.strip()
        )

        for entity_id in entity_ids:

            if not entity_id:
                continue

            if entity_id in unique_entity_ids:
                duplicate_entity_ids.add(entity_id)
            else:
                unique_entity_ids.add(entity_id)

        # ----------------------------------------------------
        # Country distribution
        # ----------------------------------------------------

        countries = (
            chunk["country"]
            .astype("string")
            .str.strip()
        )

        counts = countries.value_counts()

        for country, count in counts.items():

            country = str(country)

            country_counts[country] = (
                country_counts.get(country, 0)
                + int(count)
            )

    return {
        "file": str(path),
        "rows": total_rows,
        "columns": columns,
        "missing_values": missing_counts,
        "empty_strings": empty_counts,
        "unique_entity_ids": len(unique_entity_ids),
        "duplicate_entity_ids": len(duplicate_entity_ids),
        "duplicate_id_examples": sorted(
            duplicate_entity_ids
        )[:10],
        "country_distribution": country_counts,
    }


# ============================================================
# 11. GROUND-TRUTH AUDIT
# ============================================================

def audit_ground_truth(
    path: Path,
    chunksize: int = CHUNK_SIZE,
) -> dict:
    """
    Audit the ground-truth file.

    We count:
        - total rows
        - empty match lists
        - non-empty match lists
        - duplicate Source-1 IDs
    """

    validate_file_exists(path)

    total_rows = 0
    empty_matches = 0
    non_empty_matches = 0

    unique_source1_ids = set()
    duplicate_source1_ids = set()

    for chunk in iter_ground_truth_chunks(
        path,
        chunksize=chunksize,
    ):

        total_rows += len(chunk)

        source1_ids = (
            chunk["source1_entity_id"]
            .astype("string")
            .str.strip()
        )

        matched_ids = (
            chunk["matched_entity_ids"]
            .astype("string")
            .str.strip()
        )

        # ----------------------------------------------------
        # Empty / non-empty match lists
        # ----------------------------------------------------

        empty_mask = matched_ids.eq("")

        empty_matches += int(empty_mask.sum())
        non_empty_matches += int((~empty_mask).sum())

        # ----------------------------------------------------
        # Duplicate Source-1 IDs
        # ----------------------------------------------------

        for source1_id in source1_ids:

            if not source1_id:
                continue

            if source1_id in unique_source1_ids:
                duplicate_source1_ids.add(source1_id)
            else:
                unique_source1_ids.add(source1_id)

    return {
        "file": str(path),
        "rows": total_rows,
        "unique_source1_ids": len(unique_source1_ids),
        "duplicate_source1_ids": len(
            duplicate_source1_ids
        ),
        "duplicate_source1_id_examples": sorted(
            duplicate_source1_ids
        )[:10],
        "empty_matches": empty_matches,
        "non_empty_matches": non_empty_matches,
    }


# ============================================================
# 12. PRINT SOURCE AUDIT
# ============================================================

def print_source_audit(result: dict) -> None:
    """
    Pretty-print a source audit result.
    """

    print("=" * 75)
    print("SOURCE DATASET AUDIT")
    print("=" * 75)

    print(f"File:")
    print(f"  {result['file']}")

    print(f"\nRows:")
    print(f"  {result['rows']:,}")

    print(f"\nColumns:")
    for column in result["columns"]:
        print(f"  - {column}")

    print(f"\nUnique entity IDs:")
    print(f"  {result['unique_entity_ids']:,}")

    print(f"\nDuplicate entity IDs:")
    print(f"  {result['duplicate_entity_ids']:,}")

    if result["duplicate_id_examples"]:
        print("\nDuplicate ID examples:")
        for entity_id in result["duplicate_id_examples"]:
            print(f"  - {entity_id}")

    print("\nMissing values:")
    for column, count in result["missing_values"].items():
        print(f"  {column:20}: {count:,}")

    print("\nEmpty strings:")
    for column, count in result["empty_strings"].items():
        print(f"  {column:20}: {count:,}")

    print("\nTop countries:")

    countries = sorted(
        result["country_distribution"].items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for country, count in countries[:20]:
        print(f"  {country:20}: {count:,}")

    print("=" * 75)


# ============================================================
# 13. PRINT GROUND-TRUTH AUDIT
# ============================================================

def print_ground_truth_audit(result: dict) -> None:
    """
    Pretty-print a ground-truth audit result.
    """

    print("=" * 75)
    print("GROUND-TRUTH AUDIT")
    print("=" * 75)

    print(f"File:")
    print(f"  {result['file']}")

    print(f"\nRows:")
    print(f"  {result['rows']:,}")

    print(f"\nUnique Source-1 IDs:")
    print(f"  {result['unique_source1_ids']:,}")

    print(f"\nDuplicate Source-1 IDs:")
    print(f"  {result['duplicate_source1_ids']:,}")

    if result["duplicate_source1_id_examples"]:
        print("\nDuplicate Source-1 ID examples:")

        for entity_id in result[
            "duplicate_source1_id_examples"
        ]:
            print(f"  - {entity_id}")

    print("\nMatch lists:")
    print(
        f"  Empty     : {result['empty_matches']:,}"
    )
    print(
        f"  Non-empty : {result['non_empty_matches']:,}"
    )

    print("=" * 75)


# ============================================================
# 14. DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 75)
    print("DATA MODULE TEST")
    print("=" * 75)
    

    datasets = {
        "TRAIN SOURCE 1": TRAIN_SOURCE1,
        "TRAIN SOURCE 2": TRAIN_SOURCE2,
        "TRAIN SOURCE 3": TRAIN_SOURCE3,
    }

    # --------------------------------------------------------
    # Audit source datasets
    # --------------------------------------------------------

    for name, path in datasets.items():

        print(f"\n\n{name}")

        result = audit_source(path)

        print_source_audit(result)

    # --------------------------------------------------------
    # Audit ground truth
    # --------------------------------------------------------

    print("\n\nTRAIN GROUND TRUTH")

    ground_truth_result = audit_ground_truth(
        TRAIN_GROUND_TRUTH
    )

    print_ground_truth_audit(
        ground_truth_result
    )

    print("\n")
    print("=" * 75)
    print("DATA MODULE TEST COMPLETED")
    print("=" * 75)