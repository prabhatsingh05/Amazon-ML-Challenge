"""Build test submission files using the validated name/address blocking rules."""
from __future__ import annotations

import argparse
import csv
import sqlite3
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
PROJECT_ROOT = REPO_ROOT.parent
DEFAULT_TEST_DIR = PROJECT_ROOT / "student_resource" / "dataset" / "test"
DEFAULT_WORK_DIR = REPO_ROOT / "work" / "test_name_address_block"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "name_address_block"
CHUNK_SIZE = 50_000
MAX_NAME_DF = 250
MAX_ADDRESS_DF = 500
MIN_ADDRESS_SHARED = 2
MAX_EXACT_PER_NAME = 5_000
GLOBAL_THRESHOLD = 0.6
COUNTRY_THRESHOLDS = {"india": 0.8, "us": 0.5}

NAME_STOP = {
    "and", "the", "of", "for", "at", "a", "an", "llc", "inc", "incorporated",
    "ltd", "limited", "co", "company", "corp", "corporation", "private", "pvt",
    "llp", "plc", "gmbh", "group", "holdings", "holding", "services", "service",
    "solutions", "enterprises", "enterprise", "industries", "international",
}
ADDRESS_STOP = {
    "street", "road", "avenue", "drive", "boulevard", "lane", "highway", "suite",
    "apartment", "floor", "building", "city", "county", "district", "colony",
    "sector", "phase", "block", "near", "opposite", "usa", "india",
}
ADDRESS_ALIASES = {
    "st": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "dr": "drive", "blvd": "boulevard", "hwy": "highway", "ste": "suite",
    "apt": "apartment", "fl": "floor",
}


def tokenize(value: str) -> list[str]:
    text = unicodedata.normalize("NFKC", value or "").casefold().replace("&", " and ")
    chars = []
    for char in text:
        chars.append(char if unicodedata.category(char)[0] in {"L", "M", "N"} else " ")
    return "".join(chars).split()


def normalize_name(value: str) -> str:
    return " ".join(tokenize(value))


def name_tokens(value: str) -> set[str]:
    return {t for t in tokenize(value) if len(t) >= 2 and t not in NAME_STOP}


def address_tokens(value: str) -> set[str]:
    result = set()
    for token in tokenize(value):
        token = ADDRESS_ALIASES.get(token, token)
        if len(token) >= 2 and token not in ADDRESS_STOP:
            result.add(token)
    return result


def build_index(db_path: Path, test_dir: Path, rebuild: bool) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if rebuild and db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA temp_store=FILE")
        con.execute("PRAGMA cache_size=-262144")
        con.execute(
            "CREATE TABLE IF NOT EXISTS entities ("
            "country TEXT NOT NULL, name TEXT NOT NULL, entity_id TEXT NOT NULL, "
            "address_tokens TEXT NOT NULL, PRIMARY KEY(country, name, entity_id)) WITHOUT ROWID"
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS postings ("
            "country TEXT NOT NULL, kind TEXT NOT NULL, token TEXT NOT NULL, "
            "entity_id TEXT NOT NULL, PRIMARY KEY(country, kind, token, entity_id)) WITHOUT ROWID"
        )
        con.execute("DROP TABLE IF EXISTS token_freq")

        for filename in ("test_source2.tsv", "test_source3.tsv"):
            path = test_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Expected test file: {path}")
            print(f"Indexing {filename} ...", flush=True)
            rows_read = 0
            for chunk in pd.read_csv(
                path, sep="\t", usecols=["entity_id", "business_name", "business_address", "country"],
                dtype=str, keep_default_na=False, chunksize=CHUNK_SIZE,
            ):
                entity_rows = []
                posting_rows = []
                for entity_id, business_name, address, country in chunk.itertuples(index=False, name=None):
                    country_key = country.strip().casefold()
                    if not entity_id or not country_key:
                        continue
                    name_key = normalize_name(business_name)
                    a_tokens = address_tokens(address)
                    entity_rows.append((country_key, name_key, entity_id, " ".join(sorted(a_tokens))))
                    posting_rows.extend((country_key, "n", token, entity_id) for token in name_tokens(business_name))
                    posting_rows.extend((country_key, "a", token, entity_id) for token in a_tokens)
                con.executemany(
                    "INSERT OR IGNORE INTO entities(country, name, entity_id, address_tokens) VALUES (?, ?, ?, ?)",
                    entity_rows,
                )
                con.executemany(
                    "INSERT OR IGNORE INTO postings(country, kind, token, entity_id) VALUES (?, ?, ?, ?)",
                    posting_rows,
                )
                con.commit()
                rows_read += len(chunk)
                print(f"  {rows_read:,} rows scanned", flush=True)

        print("Counting token frequencies (one-time index step) ...", flush=True)
        con.execute(
            "CREATE TABLE token_freq AS SELECT country, kind, token, COUNT(*) AS df "
            "FROM postings GROUP BY country, kind, token"
        )
        con.execute("CREATE UNIQUE INDEX token_freq_key ON token_freq(country, kind, token)")
        con.execute("CREATE INDEX entities_by_id ON entities(entity_id)")
        con.commit()
        print(f"Test token index ready: {db_path}", flush=True)
    finally:
        con.close()


def allowed_tokens(con, country: str, kind: str, tokens: set[str], max_df: int) -> list[str]:
    if not tokens:
        return []
    ordered = sorted(tokens)
    marks = ",".join("?" for _ in ordered)
    query = (
        f"SELECT token FROM token_freq WHERE country=? AND kind=? AND df<=? "
        f"AND token IN ({marks})"
    )
    return [row[0] for row in con.execute(query, (country, kind, max_df, *ordered))]


def get_candidates(con, country: str, name_key: str, n_tokens: list[str], a_tokens: list[str]) -> set[str]:
    exact_rows = list(con.execute(
        "SELECT entity_id FROM entities WHERE country=? AND name=? LIMIT ?",
        (country, name_key, MAX_EXACT_PER_NAME + 1),
    )) if name_key else []
    exact_ids = {row[0] for row in exact_rows} if len(exact_rows) <= MAX_EXACT_PER_NAME else set()

    queries = []
    params: list[str] = []
    if n_tokens:
        marks = ",".join("?" for _ in n_tokens)
        queries.append(
            "SELECT entity_id, 'n' AS kind, token FROM postings "
            f"WHERE country=? AND kind='n' AND token IN ({marks})"
        )
        params.extend([country, *n_tokens])
    if a_tokens:
        marks = ",".join("?" for _ in a_tokens)
        queries.append(
            "SELECT entity_id, 'a' AS kind, token FROM postings "
            f"WHERE country=? AND kind='a' AND token IN ({marks})"
        )
        params.extend([country, *a_tokens])

    name_hits: set[str] = set()
    address_hits: dict[str, set[str]] = defaultdict(set)
    if queries:
        for entity_id, kind, token in con.execute(" UNION ALL ".join(queries), params):
            if kind == "n":
                name_hits.add(entity_id)
            else:
                address_hits[entity_id].add(token)
    address_ids = {
        entity_id for entity_id, shared in address_hits.items()
        if len(shared) >= MIN_ADDRESS_SHARED
    }
    return exact_ids | name_hits | address_ids


def overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return -1.0
    return len(left & right) / min(len(left), len(right))


def target_address_tokens(con, candidate_ids: list[str]) -> dict[str, set[str]]:
    result = {}
    for start in range(0, len(candidate_ids), 800):
        batch = candidate_ids[start:start + 800]
        marks = ",".join("?" for _ in batch)
        query = f"SELECT entity_id, address_tokens FROM entities WHERE entity_id IN ({marks})"
        for entity_id, tokens in con.execute(query, batch):
            result[entity_id] = set(tokens.split()) if tokens else set()
    return result


def generate(db_path: Path, test_dir: Path, output_dir: Path) -> None:
    source_path = test_dir / "test_source1.tsv"
    if not source_path.exists():
        raise FileNotFoundError(f"Expected test file: {source_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    matching_path = output_dir / "matching_results.tsv"
    candidates_path = output_dir / "candidate_pairs.tsv"
    con = sqlite3.connect(db_path)
    source_rows = candidate_pairs = matched_pairs = no_candidate_rows = empty_matches = 0
    try:
        with matching_path.open("w", encoding="utf-8", newline="") as match_file, \
             candidates_path.open("w", encoding="utf-8", newline="") as candidate_file:
            match_writer = csv.writer(match_file, delimiter="\t", lineterminator="\n")
            candidate_writer = csv.writer(candidate_file, delimiter="\t", lineterminator="\n")
            match_writer.writerow(["source1_entity_id", "matched_entity_ids"])
            candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"])
            for chunk in pd.read_csv(
                source_path, sep="\t", usecols=["entity_id", "business_name", "business_address", "country"],
                dtype=str, keep_default_na=False, chunksize=CHUNK_SIZE,
            ):
                for source_id, business_name, address, country in chunk.itertuples(index=False, name=None):
                    source_rows += 1
                    country_key = country.strip().casefold()
                    n_tokens = allowed_tokens(con, country_key, "n", name_tokens(business_name), MAX_NAME_DF)
                    a_tokens = allowed_tokens(con, country_key, "a", address_tokens(address), MAX_ADDRESS_DF)
                    candidates = sorted(get_candidates(
                        con, country_key, normalize_name(business_name), n_tokens, a_tokens
                    ))
                    source_address = address_tokens(address)
                    addresses = target_address_tokens(con, candidates)
                    threshold = COUNTRY_THRESHOLDS.get(country_key, GLOBAL_THRESHOLD)
                    matches = [
                        candidate_id for candidate_id in candidates
                        if overlap(source_address, addresses.get(candidate_id, set())) >= threshold
                    ]
                    candidate_writer.writerow([source_id, ",".join(candidates)])
                    match_writer.writerow([source_id, ",".join(matches)])
                    candidate_pairs += len(candidates)
                    matched_pairs += len(matches)
                    no_candidate_rows += not candidates
                    empty_matches += not matches
                if source_rows % 100_000 < len(chunk):
                    print(f"  {source_rows:,} test Source 1 rows processed", flush=True)
    finally:
        con.close()
    print(f"Test Source 1 rows: {source_rows:,}")
    print(f"Candidate pairs: {candidate_pairs:,}")
    print(f"Address-filtered matches: {matched_pairs:,}")
    print(f"Rows with no candidates: {no_candidate_rows:,}")
    print(f"Rows with no matches: {empty_matches:,}")
    print(f"Thresholds: India={COUNTRY_THRESHOLDS['india']}, US={COUNTRY_THRESHOLDS['us']}, other={GLOBAL_THRESHOLD}")
    print(f"Matching output: {matching_path}")
    print(f"Grouped candidate output: {candidates_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    db_path = args.work_dir / "test_token_index.sqlite"
    if args.rebuild or not db_path.exists():
        build_index(db_path, args.test_dir, args.rebuild)
    generate(db_path, args.test_dir, args.output_dir)


if __name__ == "__main__":
    main()

