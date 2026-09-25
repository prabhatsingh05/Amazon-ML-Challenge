"""Generate an exact normalized-name candidate baseline for validation.

The SQLite index keeps Source 2/3 off RAM. Matching is same normalized
business name and same country. This is a comparison baseline only.
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parents[2]
PROJECT_ROOT = REPO_ROOT.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "student_resource" / "dataset" / "train"
DEFAULT_WORK_DIR = REPO_ROOT / "work" / "exact_name"
DEFAULT_VALID_LABELS = REPO_ROOT / "work" / "splits" / "valid_labels.tsv"
CHUNK_SIZE = 50_000

sys.path.insert(0, str(SRC_DIR))
from make_split import is_validation_id  # noqa: E402
from normalize import normalize_text  # noqa: E402


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
            "PRIMARY KEY(country, name, entity_id)) WITHOUT ROWID"
        )

        for filename in ("train_source2.tsv", "train_source3.tsv"):
            path = data_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Expected dataset file: {path}")
            print(f"Indexing {filename} ...", flush=True)
            rows_read = 0
            for chunk in pd.read_csv(
                path,
                sep="\t",
                usecols=["entity_id", "business_name", "country"],
                dtype=str,
                keep_default_na=False,
                chunksize=CHUNK_SIZE,
            ):
                rows = []
                for entity_id, business_name, country in chunk.itertuples(
                    index=False, name=None
                ):
                    name_key = normalize_text(business_name)
                    country_key = country.strip().casefold()
                    if entity_id and name_key and country_key:
                        rows.append((country_key, name_key, entity_id))
                con.executemany(
                    "INSERT OR IGNORE INTO entities(country, name, entity_id) "
                    "VALUES (?, ?, ?)",
                    rows,
                )
                con.commit()
                rows_read += len(chunk)
                print(f"  {rows_read:,} rows scanned", flush=True)
        print(f"SQLite index ready: {db_path}", flush=True)
    finally:
        con.close()


def validate_labels_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Validation labels not found: {path}. Run make_split.py first."
        )
    header = pd.read_csv(path, sep="\t", nrows=0).columns
    required = {"source1_entity_id", "matched_entity_ids"}
    if not required.issubset(header):
        raise ValueError(f"{path} needs columns {sorted(required)}")


def generate_candidates(
    db_path: Path, data_dir: Path, valid_labels: Path, out_dir: Path
) -> None:
    validate_labels_file(valid_labels)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / "validation_candidate_pairs.tsv"
    predictions_path = out_dir / "validation_predictions_exact_name.tsv"

    validation_count = 0
    pair_count = 0
    no_candidate_count = 0
    con = sqlite3.connect(db_path)
    try:
        with pairs_path.open("w", encoding="utf-8", newline="") as pairs_file, \
             predictions_path.open("w", encoding="utf-8", newline="") as pred_file:
            pair_writer = csv.writer(pairs_file, delimiter="\t", lineterminator="\n")
            pred_writer = csv.writer(pred_file, delimiter="\t", lineterminator="\n")
            pair_writer.writerow(["source1_entity_id", "candidate_entity_id"])
            pred_writer.writerow(["source1_entity_id", "matched_entity_ids"])

            source1_path = data_dir / "train_source1.tsv"
            if not source1_path.exists():
                raise FileNotFoundError(f"Expected dataset file: {source1_path}")
            for chunk in pd.read_csv(
                source1_path,
                sep="\t",
                usecols=["entity_id", "business_name", "country"],
                dtype=str,
                keep_default_na=False,
                chunksize=CHUNK_SIZE,
            ):
                for entity_id, business_name, country in chunk.itertuples(
                    index=False, name=None
                ):
                    if not is_validation_id(entity_id):
                        continue
                    validation_count += 1
                    name_key = normalize_text(business_name)
                    country_key = country.strip().casefold()
                    if name_key and country_key:
                        cursor = con.execute(
                            "SELECT entity_id FROM entities WHERE country=? AND name=?",
                            (country_key, name_key),
                        )
                        candidate_ids = [row[0] for row in cursor]
                    else:
                        candidate_ids = []

                    if not candidate_ids:
                        no_candidate_count += 1
                    for candidate_id in candidate_ids:
                        pair_writer.writerow([entity_id, candidate_id])
                    pair_count += len(candidate_ids)
                    pred_writer.writerow([entity_id, ",".join(candidate_ids)])
    finally:
        con.close()

    print(f"Validation Source 1 rows: {validation_count:,}")
    print(f"Exact-name candidate pairs: {pair_count:,}")
    print(f"Validation rows with no candidates: {no_candidate_count:,}")
    print(f"Candidate pairs file: {pairs_path}")
    print(f"Predictions file: {predictions_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--valid-labels", type=Path, default=DEFAULT_VALID_LABELS)
    parser.add_argument(
        "--rebuild", action="store_true", help="Rebuild the SQLite index from scratch"
    )
    args = parser.parse_args()

    db_path = args.work_dir / "index.sqlite"
    if args.rebuild or not db_path.exists():
        build_index(db_path, args.data_dir, rebuild=args.rebuild)
    generate_candidates(db_path, args.data_dir, args.valid_labels, args.work_dir)


if __name__ == "__main__":
    main()

