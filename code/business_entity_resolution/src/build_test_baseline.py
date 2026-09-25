"""Build a simple exact-name + address-overlap test submission baseline.

The address thresholds are selected on the held-out validation split. The global
threshold (0.4) is used for countries unseen in training, such as France.
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import unicodedata
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
PROJECT_ROOT = REPO_ROOT.parent
DEFAULT_TEST_DIR = PROJECT_ROOT / "student_resource" / "dataset" / "test"
DEFAULT_WORK_DIR = REPO_ROOT / "work"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output"
CHUNK_SIZE = 50_000
GLOBAL_THRESHOLD = 0.4
COUNTRY_THRESHOLDS = {"india": 0.5, "us": 0.4}

ALIASES = {
    "st": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "dr": "drive", "blvd": "boulevard", "hwy": "highway",
    "ste": "suite", "apt": "apartment", "fl": "floor",
}
STOP = {
    "street", "road", "avenue", "drive", "boulevard", "lane", "highway",
    "suite", "apartment", "floor", "building", "city", "county", "district",
    "colony", "sector", "phase", "block", "near", "opposite",
}


def normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKC", value or "").casefold().replace("&", " and ")
    chars = []
    for char in text:
        if char.isspace() or unicodedata.category(char)[0] in {"L", "M", "N"}:
            chars.append(char)
        else:
            chars.append(" ")
    return " ".join("".join(chars).split())


def address_tokens(value: str) -> frozenset[str]:
    if not value:
        return frozenset()
    text = unicodedata.normalize("NFKC", value).casefold().replace("&", " and ")
    chars = []
    for char in text:
        if unicodedata.category(char)[0] in {"L", "M", "N"}:
            chars.append(char)
        else:
            chars.append(" ")
    result = []
    for token in "".join(chars).split():
        token = ALIASES.get(token, token)
        if token not in STOP and len(token) > 1:
            result.append(token)
    return frozenset(result)


def overlap_score(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return -1.0
    return len(left & right) / min(len(left), len(right))


def build_index(db_path: Path, data_dir: Path, rebuild: bool) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if rebuild and db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA cache_size=-262144")
        con.execute(
            "CREATE TABLE IF NOT EXISTS entities ("
            "country TEXT NOT NULL, name TEXT NOT NULL, entity_id TEXT NOT NULL, "
            "address_tokens TEXT NOT NULL, "
            "PRIMARY KEY(country, name, entity_id)) WITHOUT ROWID"
        )
        for filename in ("test_source2.tsv", "test_source3.tsv"):
            path = data_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Expected test file: {path}")
            print(f"Indexing {filename} ...", flush=True)
            rows_read = 0
            for chunk in pd.read_csv(
                path,
                sep="\t",
                usecols=["entity_id", "business_name", "business_address", "country"],
                dtype=str,
                keep_default_na=False,
                chunksize=CHUNK_SIZE,
            ):
                rows = []
                for entity_id, name, address, country in chunk.itertuples(
                    index=False, name=None
                ):
                    name_key = normalize_name(name)
                    country_key = country.strip().casefold()
                    if entity_id and name_key and country_key:
                        tokens = address_tokens(address)
                        rows.append(
                            (country_key, name_key, entity_id, " ".join(sorted(tokens)))
                        )
                con.executemany(
                    "INSERT OR IGNORE INTO entities "
                    "(country, name, entity_id, address_tokens) VALUES (?, ?, ?, ?)",
                    rows,
                )
                con.commit()
                rows_read += len(chunk)
                print(f"  {rows_read:,} rows scanned", flush=True)
        print(f"Test index ready: {db_path}", flush=True)
    finally:
        con.close()


def generate(db_path: Path, data_dir: Path, output_dir: Path) -> None:
    source1_path = data_dir / "test_source1.tsv"
    if not source1_path.exists():
        raise FileNotFoundError(f"Expected test file: {source1_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    matching_path = output_dir / "matching_results.tsv"
    candidates_path = output_dir / "candidate_pairs.tsv"

    con = sqlite3.connect(db_path)
    source1_rows = 0
    candidate_pairs = 0
    matched_pairs = 0
    empty_matches = 0
    try:
        with matching_path.open("w", encoding="utf-8", newline="") as match_file, \
             candidates_path.open("w", encoding="utf-8", newline="") as cand_file:
            match_writer = csv.writer(match_file, delimiter="\t", lineterminator="\n")
            cand_writer = csv.writer(cand_file, delimiter="\t", lineterminator="\n")
            match_writer.writerow(["source1_entity_id", "matched_entity_ids"])
            cand_writer.writerow(["source1_entity_id", "candidate_entity_ids"])

            for chunk in pd.read_csv(
                source1_path,
                sep="\t",
                usecols=["entity_id", "business_name", "business_address", "country"],
                dtype=str,
                keep_default_na=False,
                chunksize=CHUNK_SIZE,
            ):
                for entity_id, name, address, country in chunk.itertuples(
                    index=False, name=None
                ):
                    source1_rows += 1
                    name_key = normalize_name(name)
                    country_key = country.strip().casefold()
                    source_tokens = address_tokens(address)
                    candidates = []
                    matches = []
                    if name_key and country_key:
                        rows = con.execute(
                            "SELECT entity_id, address_tokens FROM entities "
                            "WHERE country=? AND name=?",
                            (country_key, name_key),
                        )
                        threshold = COUNTRY_THRESHOLDS.get(country_key, GLOBAL_THRESHOLD)
                        for candidate_id, target_tokens_text in rows:
                            candidates.append(candidate_id)
                            target_tokens = frozenset(target_tokens_text.split())
                            if overlap_score(source_tokens, target_tokens) >= threshold:
                                matches.append(candidate_id)
                    cand_writer.writerow([entity_id, ",".join(candidates)])
                    match_writer.writerow([entity_id, ",".join(matches)])
                    candidate_pairs += len(candidates)
                    matched_pairs += len(matches)
                    empty_matches += not matches
                if source1_rows % 100_000 < len(chunk):
                    print(f"  {source1_rows:,} Source 1 rows scored", flush=True)
    finally:
        con.close()

    print(f"Test Source 1 rows: {source1_rows:,}")
    print(f"Exact-name candidate pairs: {candidate_pairs:,}")
    print(f"Address-filtered matches: {matched_pairs:,}")
    print(f"Rows with empty match list: {empty_matches:,}")
    print(f"Matching output: {matching_path}")
    print(f"Grouped candidate output: {candidates_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    db_path = args.work_dir / "test_exact_name_index.sqlite"
    if args.rebuild or not db_path.exists():
        build_index(db_path, args.test_dir, rebuild=args.rebuild)
    generate(db_path, args.test_dir, args.output_dir)


if __name__ == "__main__":
    main()

