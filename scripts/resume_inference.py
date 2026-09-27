from pathlib import Path
import re
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "dataset" / "test"
OUTPUT_DIR = ROOT / "output"

S1_PATH = TEST_DIR / "test_source1.tsv"
S2_PATH = TEST_DIR / "test_source2.tsv"
S3_PATH = TEST_DIR / "test_source3.tsv"

MATCHING_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

CHUNK_SIZE = 10_000

NON_ALNUM = re.compile(r"[^a-z0-9]+")
SPACES = re.compile(r"\s+")
POSTAL = re.compile(r"\b\d{5,6}\b")


# ============================================================
# FAST NORMALIZATION
# ============================================================

def normalize(value):
    if value is None:
        return ""

    text = str(value).lower().strip()

    if not text:
        return ""

    text = text.replace("&", " and ")
    text = NON_ALNUM.sub(" ", text)
    text = SPACES.sub(" ", text)

    return text.strip()


def get_postal(value):
    if not value:
        return ""

    match = POSTAL.search(str(value))

    if match:
        return match.group(0)

    return ""


# ============================================================
# PREPARE ONE S1 RECORD
# ============================================================

def prepare_record(row):

    return {
        "entity_id": str(row["entity_id"]),
        "name": normalize(row["business_name"]),
        "address": normalize(row["business_address"]),
        "country": normalize(row["country"]),
        "postal": get_postal(row["business_address"]),
    }


# ============================================================
# BUILD INDEX
# ============================================================

def build_indexes(path, source_name):

    print()
    print("=" * 70)
    print(f"Loading {source_name}")
    print("=" * 70)

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

    print(f"{source_name} rows: {len(df):,}")

    name_index = {}
    address_index = {}
    postal_index = {}

    for row in df.itertuples(index=False):

        entity_id = str(row.entity_id)

        country = normalize(row.country)
        name = normalize(row.business_name)
        address = normalize(row.business_address)
        postal = get_postal(row.business_address)

        if name:

            key = (country, name)

            name_index.setdefault(key, []).append(entity_id)

        if address:

            key = (country, address)

            address_index.setdefault(key, []).append(entity_id)

        if postal:

            key = (country, postal)

            postal_index.setdefault(key, []).append(entity_id)

    del df

    print(f"{source_name} name keys:    {len(name_index):,}")
    print(f"{source_name} address keys: {len(address_index):,}")
    print(f"{source_name} postal keys:  {len(postal_index):,}")

    return {
        "name": name_index,
        "address": address_index,
        "postal": postal_index,
    }


# ============================================================
# FAST CANDIDATE GENERATION
# ============================================================

def generate_candidates(record, indexes):

    country = record["country"]
    name = record["name"]
    address = record["address"]
    postal = record["postal"]

    result = []
    seen = set()

    # Name
    if name:

        values = indexes["name"].get(
            (country, name),
            ()
        )

        for entity_id in values:

            if entity_id not in seen:

                seen.add(entity_id)
                result.append(entity_id)

                if len(result) >= 250:
                    return result

    # Address
    if address:

        values = indexes["address"].get(
            (country, address),
            ()
        )

        for entity_id in values:

            if entity_id not in seen:

                seen.add(entity_id)
                result.append(entity_id)

                if len(result) >= 250:
                    return result

    # Postal
    if postal:

        values = indexes["postal"].get(
            (country, postal),
            ()
        )

        for entity_id in values:

            if entity_id not in seen:

                seen.add(entity_id)
                result.append(entity_id)

                if len(result) >= 250:
                    return result

    return result


# ============================================================
# STRONG MATCH SELECTION
# ============================================================

def select_matches(record, source2_indexes, source3_indexes):

    country = record["country"]
    name = record["name"]
    address = record["address"]

    # --------------------------------------------------------
    # Strong match 1:
    # exact normalized name + country
    # --------------------------------------------------------

    if name:

        key = (country, name)

        result = []

        result.extend(
            source2_indexes["name"].get(key, ())
        )

        result.extend(
            source3_indexes["name"].get(key, ())
        )

        if result:

            return list(dict.fromkeys(result))

    # --------------------------------------------------------
    # Strong match 2:
    # exact normalized address + country
    # --------------------------------------------------------

    if address:

        key = (country, address)

        result = []

        result.extend(
            source2_indexes["address"].get(key, ())
        )

        result.extend(
            source3_indexes["address"].get(key, ())
        )

        if result:

            return list(dict.fromkeys(result))

    # Postal-only is NOT accepted as a final match.
    return []


# ============================================================
# COUNT EXISTING OUTPUT
# ============================================================

def count_data_rows(path):

    if not path.exists():
        return 0

    with open(path, "rb") as f:

        total_lines = sum(
            1 for _ in f
        )

    # subtract header
    return max(0, total_lines - 1)


# ============================================================
# PREPARE / VERIFY EXISTING OUTPUT
# ============================================================

def get_resume_position():

    matching_rows = count_data_rows(
        MATCHING_PATH
    )

    candidate_rows = count_data_rows(
        CANDIDATE_PATH
    )

    print()
    print("=" * 70)
    print("EXISTING OUTPUT CHECK")
    print("=" * 70)

    print(
        f"Existing matching_results rows: "
        f"{matching_rows:,}"
    )

    print(
        f"Existing candidate_pairs rows:  "
        f"{candidate_rows:,}"
    )

    if matching_rows != candidate_rows:

        raise RuntimeError(
            "The two output files contain different "
            "numbers of rows. Do NOT continue automatically."
        )

    if matching_rows == 0:

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            MATCHING_PATH,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "source1_entity_id\t"
                "matched_entity_ids\n"
            )

        with open(
            CANDIDATE_PATH,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "source1_entity_id\t"
                "candidate_entity_ids\n"
            )

    return matching_rows


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("RESUMABLE FULL DATASET INFERENCE")
    print("=" * 80)

    # --------------------------------------------------------
    # Verify files
    # --------------------------------------------------------

    for path in [
        S1_PATH,
        S2_PATH,
        S3_PATH,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing file: {path}"
            )

    # --------------------------------------------------------
    # Build indexes
    # --------------------------------------------------------

    source2_indexes = build_indexes(
        S2_PATH,
        "Source 2"
    )

    source3_indexes = build_indexes(
        S3_PATH,
        "Source 3"
    )

    # --------------------------------------------------------
    # Determine resume position
    # --------------------------------------------------------

    resume_rows = get_resume_position()

    print()
    print(
        f"Resuming after S1 row: "
        f"{resume_rows:,}"
    )

    print(
        f"Next S1 row: "
        f"{resume_rows + 1:,}"
    )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    total = resume_rows

    candidates_count = 0
    matches_count = 0

    rows_to_skip = resume_rows

    # --------------------------------------------------------
    # OPEN EXISTING FILES IN APPEND MODE
    # --------------------------------------------------------

    with open(
        MATCHING_PATH,
        "a",
        encoding="utf-8",
        newline=""
    ) as matching_file:

        with open(
            CANDIDATE_PATH,
            "a",
            encoding="utf-8",
            newline=""
        ) as candidate_file:

            # ------------------------------------------------
            # STREAM S1
            # ------------------------------------------------

            for chunk in pd.read_csv(
                S1_PATH,
                sep="\t",
                dtype=str,
                keep_default_na=False,
                chunksize=CHUNK_SIZE,
            ):

                # --------------------------------------------
                # Skip already processed rows
                # --------------------------------------------

                if rows_to_skip >= len(chunk):

                    rows_to_skip -= len(chunk)

                    continue

                if rows_to_skip > 0:

                    chunk = chunk.iloc[
                        rows_to_skip:
                    ]

                    rows_to_skip = 0

                matching_lines = []
                candidate_lines = []

                # --------------------------------------------
                # Process current chunk
                # --------------------------------------------

                for row in chunk.to_dict(
                    orient="records"
                ):

                    record = prepare_record(row)

                    s1_id = record["entity_id"]

                    # ----------------------------------------
                    # Candidates from Source 2
                    # ----------------------------------------

                    s2_candidates = generate_candidates(
                        record,
                        source2_indexes
                    )

                    # ----------------------------------------
                    # Candidates from Source 3
                    # ----------------------------------------

                    s3_candidates = generate_candidates(
                        record,
                        source3_indexes
                    )

                    # ----------------------------------------
                    # Combine candidates
                    # ----------------------------------------

                    candidates = []

                    seen = set()

                    for entity_id in s2_candidates:

                        if entity_id not in seen:

                            seen.add(entity_id)
                            candidates.append(entity_id)

                    for entity_id in s3_candidates:

                        if entity_id not in seen:

                            seen.add(entity_id)
                            candidates.append(entity_id)

                    candidates = candidates[:250]

                    # ----------------------------------------
                    # Strong final matches
                    # ----------------------------------------

                    matches = select_matches(
                        record,
                        source2_indexes,
                        source3_indexes
                    )

                    # ----------------------------------------
                    # Guarantee:
                    # every final match exists in candidates
                    # ----------------------------------------

                    candidate_set = set(
                        candidates
                    )

                    matches = [
                        entity_id
                        for entity_id in matches
                        if entity_id in candidate_set
                    ]

                    matches = list(
                        dict.fromkeys(matches)
                    )

                    # ----------------------------------------
                    # Counters
                    # ----------------------------------------

                    if candidates:

                        candidates_count += 1

                    if matches:

                        matches_count += 1

                    total += 1

                    # ----------------------------------------
                    # Output lines
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
                        + ",".join(candidates)
                        + "\n"
                    )

                # --------------------------------------------
                # Append chunk
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

                if (
                    total % 50_000 == 0
                    or len(chunk) < CHUNK_SIZE
                ):

                    print(
                        f"Processed S1: "
                        f"{total:,} | "
                        f"with candidates: "
                        f"{candidates_count:,} | "
                        f"with matches: "
                        f"{matches_count:,}"
                    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    print()
    print("=" * 80)
    print("FINAL VALIDATION")
    print("=" * 80)

    matching_rows = count_data_rows(
        MATCHING_PATH
    )

    candidate_rows = count_data_rows(
        CANDIDATE_PATH
    )

    # Exact S1 count
    with open(
        S1_PATH,
        "rb"
    ) as f:

        s1_rows = sum(
            1 for _ in f
        ) - 1

    print(
        f"Test Source 1 rows:       "
        f"{s1_rows:,}"
    )

    print(
        f"matching_results rows:    "
        f"{matching_rows:,}"
    )

    print(
        f"candidate_pairs rows:     "
        f"{candidate_rows:,}"
    )

    # --------------------------------------------------------
    # REQUIRED FINAL CONDITION
    # --------------------------------------------------------

    if matching_rows != s1_rows:

        raise RuntimeError(
            f"matching_results incomplete: "
            f"{matching_rows:,} / {s1_rows:,}"
        )

    if candidate_rows != s1_rows:

        raise RuntimeError(
            f"candidate_pairs incomplete: "
            f"{candidate_rows:,} / {s1_rows:,}"
        )

    print()
    print("========================================")
    print("OUTPUT CHECK PASSED")
    print("========================================")
    print()
    print(
        "matching_results.tsv:"
    )
    print(
        f"  {matching_rows:,} rows"
    )

    print()
    print(
        "candidate_pairs.tsv:"
    )
    print(
        f"  {candidate_rows:,} rows"
    )

    print()
    print(
        "FULL TEST DATASET COMPLETE."
    )


if __name__ == "__main__":
    main()