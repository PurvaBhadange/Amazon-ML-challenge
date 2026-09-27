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


def normalize(value):
    if value is None:
        return ""

    text = str(value).lower().strip()

    text = text.replace("&", " and ")

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def normalize_name(value):
    return normalize(value)


def normalize_address(value):
    return normalize(value)


def normalize_country(value):
    return normalize(value)


def get_postal(value):
    if not value:
        return ""

    matches = re.findall(
        r"\b\d{5,6}\b",
        str(value),
    )

    if matches:
        return matches[0]

    return ""


def build_indexes(path, source_name):

    print()
    print(
        f"Loading {source_name}..."
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
        f"{source_name} rows: "
        f"{len(df):,}"
    )

    name_index = {}
    address_index = {}
    postal_index = {}

    for row in df.itertuples(
        index=False
    ):

        entity_id = str(
            row.entity_id
        )

        name = normalize_name(
            row.business_name
        )

        address = normalize_address(
            row.business_address
        )

        country = normalize_country(
            row.country
        )

        postal = get_postal(
            row.business_address
        )

        if name:
            key = (
                country,
                name,
            )

            name_index.setdefault(
                key,
                [],
            ).append(entity_id)

        if address:
            key = (
                country,
                address,
            )

            address_index.setdefault(
                key,
                [],
            ).append(entity_id)

        if postal:
            key = (
                country,
                postal,
            )

            postal_index.setdefault(
                key,
                [],
            ).append(entity_id)

    del df

    print(
        f"{source_name} name keys: "
        f"{len(name_index):,}"
    )

    print(
        f"{source_name} address keys: "
        f"{len(address_index):,}"
    )

    print(
        f"{source_name} postal keys: "
        f"{len(postal_index):,}"
    )

    return {
        "name": name_index,
        "address": address_index,
        "postal": postal_index,
    }


def generate_candidates(
    row,
    indexes,
):

    country = normalize_country(
        row["country"]
    )

    name = normalize_name(
        row["business_name"]
    )

    address = normalize_address(
        row["business_address"]
    )

    postal = get_postal(
        row["business_address"]
    )

    candidates = []

    # --------------------------------------------------
    # 1. Exact normalized business name + country
    # --------------------------------------------------

    if name:

        key = (
            country,
            name,
        )

        candidates.extend(
            indexes["name"].get(
                key,
                [],
            )
        )

    # --------------------------------------------------
    # 2. Exact normalized address + country
    # --------------------------------------------------

    if address:

        key = (
            country,
            address,
        )

        candidates.extend(
            indexes["address"].get(
                key,
                [],
            )
        )

    # --------------------------------------------------
    # 3. Postal + country
    # --------------------------------------------------

    if postal:

        key = (
            country,
            postal,
        )

        candidates.extend(
            indexes["postal"].get(
                key,
                [],
            )
        )

    return list(
        dict.fromkeys(
            candidates
        )
    )


def select_matches(
    row,
    source2_indexes,
    source3_indexes,
):

    country = normalize_country(
        row["country"]
    )

    name = normalize_name(
        row["business_name"]
    )

    address = normalize_address(
        row["business_address"]
    )

    postal = get_postal(
        row["business_address"]
    )

    # --------------------------------------------------
    # STRONG MATCH 1
    # exact normalized name + country
    # --------------------------------------------------

    name_matches = []

    if name:

        key = (
            country,
            name,
        )

        name_matches.extend(
            source2_indexes[
                "name"
            ].get(
                key,
                [],
            )
        )

        name_matches.extend(
            source3_indexes[
                "name"
            ].get(
                key,
                [],
            )
        )

    if name_matches:

        return list(
            dict.fromkeys(
                name_matches
            )
        )

    # --------------------------------------------------
    # STRONG MATCH 2
    # exact normalized address + country
    # --------------------------------------------------

    address_matches = []

    if address:

        key = (
            country,
            address,
        )

        address_matches.extend(
            source2_indexes[
                "address"
            ].get(
                key,
                [],
            )
        )

        address_matches.extend(
            source3_indexes[
                "address"
            ].get(
                key,
                [],
            )
        )

    if address_matches:

        return list(
            dict.fromkeys(
                address_matches
            )
        )

    # --------------------------------------------------
    # STRONG MATCH 3
    # postal + country
    #
    # Only accept a postal match when the
    # business name has some overlap with
    # the candidate set.
    # --------------------------------------------------

    postal_candidates = []

    if postal:

        key = (
            country,
            postal,
        )

        postal_candidates.extend(
            source2_indexes[
                "postal"
            ].get(
                key,
                [],
            )
        )

        postal_candidates.extend(
            source3_indexes[
                "postal"
            ].get(
                key,
                [],
            )
        )

    # Do NOT blindly output postal-only matches.
    # Precision is more important for F0.5.
    return []


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("FINAL FULL-DATASET INFERENCE")
    print("=" * 80)

    if not S1_PATH.exists():
        raise FileNotFoundError(
            f"Missing: {S1_PATH}"
        )

    if not S2_PATH.exists():
        raise FileNotFoundError(
            f"Missing: {S2_PATH}"
        )

    if not S3_PATH.exists():
        raise FileNotFoundError(
            f"Missing: {S3_PATH}"
        )

    # --------------------------------------------------
    # Build Source 2 index
    # --------------------------------------------------

    source2_indexes = build_indexes(
        S2_PATH,
        "Source 2",
    )

    # --------------------------------------------------
    # Build Source 3 index
    # --------------------------------------------------

    source3_indexes = build_indexes(
        S3_PATH,
        "Source 3",
    )

    print()
    print(
        "Starting Source 1 inference..."
    )

    # --------------------------------------------------
    # Fresh output files
    # --------------------------------------------------

    with open(
        MATCHING_PATH,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        f.write(
            "source1_entity_id\t"
            "matched_entity_ids\n"
        )

    with open(
        CANDIDATE_PATH,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        f.write(
            "source1_entity_id\t"
            "candidate_entity_ids\n"
        )

    total = 0
    total_matches = 0
    total_candidates = 0

    # --------------------------------------------------
    # STREAM Source 1
    # --------------------------------------------------

    for chunk in pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):

        matching_lines = []
        candidate_lines = []

        for row in chunk.to_dict(
            orient="records"
        ):

            s1_id = str(
                row["entity_id"]
            )

            # Candidate set = exactly the
            # records considered by inference.
            s2_candidates = generate_candidates(
                row,
                source2_indexes,
            )

            s3_candidates = generate_candidates(
                row,
                source3_indexes,
            )

            candidates = list(
                dict.fromkeys(
                    s2_candidates
                    + s3_candidates
                )
            )

            # Keep candidate set bounded.
            candidates = candidates[:250]

            matches = select_matches(
                row,
                source2_indexes,
                source3_indexes,
            )

            # Every predicted ID MUST be
            # present in candidate_pairs.
            candidate_set = set(
                candidates
            )

            matches = [
                x
                for x in matches
                if x in candidate_set
            ]

            matches = list(
                dict.fromkeys(
                    matches
                )
            )

            if candidates:
                total_candidates += 1

            if matches:
                total_matches += 1

            total += 1

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

        with open(
            MATCHING_PATH,
            "a",
            encoding="utf-8",
            newline="",
        ) as f:

            f.writelines(
                matching_lines
            )

        with open(
            CANDIDATE_PATH,
            "a",
            encoding="utf-8",
            newline="",
        ) as f:

            f.writelines(
                candidate_lines
            )

        if (
            total % 50_000 == 0
            or len(chunk) < CHUNK_SIZE
        ):

            print(
                f"Processed S1: "
                f"{total:,} | "
                f"entities with candidates: "
                f"{total_candidates:,} | "
                f"entities with matches: "
                f"{total_matches:,}"
            )

    # --------------------------------------------------
    # FINAL SANITY CHECK
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL OUTPUT CREATED")
    print("=" * 80)

    print(
        "Source 1 entities processed:",
        f"{total:,}",
    )

    print(
        "Entities with candidates:",
        f"{total_candidates:,}",
    )

    print(
        "Entities with matches:",
        f"{total_matches:,}",
    )

    print()
    print(
        "Matching:",
        MATCHING_PATH,
    )

    print(
        "Candidates:",
        CANDIDATE_PATH,
    )

    # Count output rows.
    matching_count = sum(
        1
        for _ in open(
            MATCHING_PATH,
            encoding="utf-8",
        )
    ) - 1

    candidate_count = sum(
        1
        for _ in open(
            CANDIDATE_PATH,
            encoding="utf-8",
        )
    ) - 1

    print()
    print(
        "matching_results rows:",
        f"{matching_count:,}",
    )

    print(
        "candidate_pairs rows:",
        f"{candidate_count:,}",
    )

    if matching_count != total:
        raise RuntimeError(
            "matching_results row count "
            "does not equal processed "
            "Source-1 count."
        )

    if candidate_count != total:
        raise RuntimeError(
            "candidate_pairs row count "
            "does not equal processed "
            "Source-1 count."
        )

    print()
    print("OUTPUT CHECK PASSED")


if __name__ == "__main__":
    main()