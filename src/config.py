"""
Configuration module for the Business Entity Resolution project.

This module contains:
- Project paths
- Dataset paths
- Required schema
- File-reading configuration
- Output/model paths
- Basic project configuration

It can also be executed directly to verify that the project
structure and dataset files are correctly detected.
"""

from pathlib import Path


# ============================================================
# 1. PROJECT ROOT
# ============================================================

# config.py is located at:
#
# business-entity-resolution/
# └── src/
#     └── config.py
#
# Therefore:
# parent      -> src/
# parent.parent -> project root

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# 2. PROJECT DIRECTORIES
# ============================================================

DATASET_DIR = PROJECT_ROOT / "dataset"

TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


# ============================================================
# 3. CREATE GENERATED DIRECTORIES IF REQUIRED
# ============================================================

# These directories are created by our code if they don't exist.
#
# We deliberately do NOT create dataset/train or dataset/test
# because those are input directories and should already exist.

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 4. DATASET FILES — TRAIN
# ============================================================

TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"

TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"


# ============================================================
# 5. DATASET FILES — TEST
# ============================================================

TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"


# ============================================================
# 6. FINAL OUTPUT FILES
# ============================================================

MATCHING_RESULTS = OUTPUT_DIR / "matching_results.tsv"

CANDIDATE_PAIRS = OUTPUT_DIR / "candidate_pairs.tsv"


# ============================================================
# 7. REQUIRED SOURCE SCHEMA
# ============================================================

# The challenge source files contain these fields.

SOURCE_COLUMNS = (
    "entity_id",
    "business_name",
    "business_address",
    "country",
)


# Ground-truth columns are different from source columns.

GROUND_TRUTH_COLUMNS = (
    "source1_entity_id",
    "matched_entity_ids",
)


# ============================================================
# 8. TSV READING CONFIGURATION
# ============================================================

TSV_SEPARATOR = "\t"

ENCODING = "utf-8"

# We will use chunked reading for the large datasets.
#
# 100,000 rows is a starting point.
# We can change this later after measuring memory/performance.

CHUNK_SIZE = 100_000


# ============================================================
# 9. DATA TYPES
# ============================================================

# Entity IDs and text fields should be treated as strings.
#
# This prevents pandas from accidentally converting IDs,
# postal codes, PIN codes, etc. into numeric values.

SOURCE_DTYPES = {
    "entity_id": "string",
    "business_name": "string",
    "business_address": "string",
    "country": "string",
}


GROUND_TRUTH_DTYPES = {
    "source1_entity_id": "string",
    "matched_entity_ids": "string",
}


# ============================================================
# 10. PROJECT CONSTANTS
# ============================================================

# Source identifiers used throughout the pipeline.

SOURCE1 = "source1"
SOURCE2 = "source2"
SOURCE3 = "source3"


# ============================================================
# 11. DATASET REGISTRY
# ============================================================

# Keeping the paths in one dictionary makes it easier to
# iterate through datasets during auditing/training.

TRAIN_FILES = {
    SOURCE1: TRAIN_SOURCE1,
    SOURCE2: TRAIN_SOURCE2,
    SOURCE3: TRAIN_SOURCE3,
    "ground_truth": TRAIN_GROUND_TRUTH,
}


TEST_FILES = {
    SOURCE1: TEST_SOURCE1,
    SOURCE2: TEST_SOURCE2,
    SOURCE3: TEST_SOURCE3,
}


# ============================================================
# 12. HELPER FUNCTIONS
# ============================================================

def get_train_file(source_name: str) -> Path:
    """
    Return the path of a training file.

    Example:
        get_train_file("source1")
    """

    if source_name not in TRAIN_FILES:
        valid = ", ".join(TRAIN_FILES.keys())

        raise ValueError(
            f"Unknown training source: {source_name!r}. "
            f"Valid values: {valid}"
        )

    return TRAIN_FILES[source_name]


def get_test_file(source_name: str) -> Path:
    """
    Return the path of a test file.

    Example:
        get_test_file("source2")
    """

    if source_name not in TEST_FILES:
        valid = ", ".join(TEST_FILES.keys())

        raise ValueError(
            f"Unknown test source: {source_name!r}. "
            f"Valid values: {valid}"
        )

    return TEST_FILES[source_name]


def get_all_input_files() -> dict[str, Path]:
    """
    Return all expected input files.
    """

    return {
        **TRAIN_FILES,
        **{
            f"test_{name}": path
            for name, path in TEST_FILES.items()
        },
    }


# ============================================================
# 13. PATH VALIDATION
# ============================================================

def check_project_structure() -> dict[str, bool]:
    """
    Check whether the important project directories exist.
    """

    return {
        "project_root": PROJECT_ROOT.exists(),
        "dataset": DATASET_DIR.exists(),
        "train": TRAIN_DIR.exists(),
        "test": TEST_DIR.exists(),
        "models": MODELS_DIR.exists(),
        "output": OUTPUT_DIR.exists(),
        "notebooks": NOTEBOOKS_DIR.exists(),
        "scripts": SCRIPTS_DIR.exists(),
    }


def check_dataset_files() -> dict[str, bool]:
    """
    Check whether all expected dataset files exist.
    """

    return {
        name: path.exists()
        for name, path in get_all_input_files().items()
    }


# ============================================================
# 14. CONFIGURATION SUMMARY
# ============================================================

def print_configuration() -> None:
    """
    Print the current project configuration.
    """

    print("=" * 70)
    print("BUSINESS ENTITY RESOLUTION")
    print("PROJECT CONFIGURATION")
    print("=" * 70)

    print(f"\nProject root:")
    print(f"  {PROJECT_ROOT}")

    print(f"\nDataset directory:")
    print(f"  {DATASET_DIR}")

    print(f"\nTraining directory:")
    print(f"  {TRAIN_DIR}")

    print(f"\nTest directory:")
    print(f"  {TEST_DIR}")

    print(f"\nModels directory:")
    print(f"  {MODELS_DIR}")

    print(f"\nOutput directory:")
    print(f"  {OUTPUT_DIR}")

    print("\n" + "-" * 70)
    print("TRAIN FILES")
    print("-" * 70)

    for name, path in TRAIN_FILES.items():
        status = "FOUND" if path.exists() else "MISSING"
        print(f"{name:15} {status:8} {path.name}")

    print("\n" + "-" * 70)
    print("TEST FILES")
    print("-" * 70)

    for name, path in TEST_FILES.items():
        status = "FOUND" if path.exists() else "MISSING"
        print(f"{name:15} {status:8} {path.name}")

    print("\n" + "-" * 70)
    print("OUTPUT FILES")
    print("-" * 70)

    print(f"matching_results : {MATCHING_RESULTS}")
    print(f"candidate_pairs  : {CANDIDATE_PAIRS}")

    print("\n" + "-" * 70)
    print("SCHEMA")
    print("-" * 70)

    print("Source columns:")
    for column in SOURCE_COLUMNS:
        print(f"  - {column}")

    print("\nGround-truth columns:")
    for column in GROUND_TRUTH_COLUMNS:
        print(f"  - {column}")

    print("\n" + "-" * 70)
    print("READING CONFIGURATION")
    print("-" * 70)

    print(f"Separator : {TSV_SEPARATOR!r}")
    print(f"Encoding  : {ENCODING}")
    print(f"Chunk size: {CHUNK_SIZE}")

    print("\n" + "=" * 70)


# ============================================================
# 15. MAIN — ALLOWS DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    print_configuration()

    print("\n" + "=" * 70)
    print("PROJECT STRUCTURE CHECK")
    print("=" * 70)

    structure = check_project_structure()

    for name, exists in structure.items():
        status = "OK" if exists else "MISSING"
        print(f"{name:15} : {status}")

    print("\n" + "=" * 70)
    print("DATASET FILE CHECK")
    print("=" * 70)

    dataset_files = check_dataset_files()

    all_files_found = True

    for name, exists in dataset_files.items():

        status = "OK" if exists else "MISSING"

        print(f"{name:15} : {status}")

        if not exists:
            all_files_found = False

    print("\n" + "=" * 70)

    if all_files_found:
        print("CONFIGURATION CHECK PASSED")
        print("All expected dataset files were found.")
    else:
        print("CONFIGURATION CHECK FAILED")
        print("One or more expected dataset files are missing.")

    print("=" * 70)