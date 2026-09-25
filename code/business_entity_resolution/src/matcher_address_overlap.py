"""Score exact-name candidate pairs with address token overlap and tune F0.5.

The script compares one global address threshold with country-specific thresholds.
For unseen test countries, the global threshold is the fallback. Outputs belong in
the ignored work directory, not in Git.
"""
from __future__ import annotations

import argparse
import csv
import unicodedata
from collections import defaultdict
from itertools import groupby
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
PROJECT_ROOT = REPO_ROOT.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "student_resource" / "dataset" / "train"
DEFAULT_WORK_DIR = REPO_ROOT / "work" / "exact_name"
DEFAULT_LABELS = REPO_ROOT / "work" / "splits" / "valid_labels.tsv"
DEFAULT_CANDIDATES = DEFAULT_WORK_DIR / "validation_candidate_pairs.tsv"
CHUNK_SIZE = 100_000

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
THRESHOLDS = (-1.0, 0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def address_tokens(value: str) -> frozenset[str]:
    if not value:
        return frozenset()
    text = unicodedata.normalize("NFKC", str(value)).casefold().replace("&", " and ")
    chars = []
    for char in text:
        if unicodedata.category(char)[0] in {"L", "M", "N"}:
            chars.append(char)
        else:
            chars.append(" ")
    tokens = []
    for token in "".join(chars).split():
        token = ALIASES.get(token, token)
        if token not in STOP and len(token) > 1:
            tokens.append(token)
    return frozenset(tokens)


def address_overlap(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return -1.0
    return len(left & right) / min(len(left), len(right))


def f05(true_ids: set[str], predicted_ids: set[str]) -> float:
    if not true_ids:
        return 1.0 if not predicted_ids else 0.0
    correct = len(true_ids & predicted_ids)
    precision = correct / len(predicted_ids) if predicted_ids else 0.0
    recall = correct / len(true_ids)
    denominator = 0.25 * precision + recall
    return 1.25 * precision * recall / denominator if denominator else 0.0


def read_labels(path: Path) -> dict[str, set[str]]:
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            result[row["source1_entity_id"]] = {
                part.strip() for part in row["matched_entity_ids"].split(",") if part.strip()
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--valid-labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_WORK_DIR)
    args = parser.parse_args()

    truth = read_labels(args.valid_labels)
    print(f"Validation labels: {len(truth):,}", flush=True)

    candidate_ids: set[str] = set()
    with args.candidates.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            candidate_ids.add(row["candidate_entity_id"])
    print(f"Unique target candidates: {len(candidate_ids):,}", flush=True)

    target_addresses: dict[str, frozenset[str]] = {}
    for filename in ("train_source2.tsv", "train_source3.tsv"):
        path = args.data_dir / filename
        print(f"Reading {filename} addresses ...", flush=True)
        found = 0
        for chunk in pd.read_csv(
            path, sep="\t", usecols=["entity_id", "business_address"],
            dtype=str, keep_default_na=False, chunksize=CHUNK_SIZE,
        ):
            subset = chunk[chunk["entity_id"].isin(candidate_ids)]
            for entity_id, address in subset.itertuples(index=False, name=None):
                target_addresses[entity_id] = address_tokens(address)
            found += len(subset)
        print(f"  candidate addresses found: {found:,}", flush=True)
    del candidate_ids

    source_addresses: dict[str, frozenset[str]] = {}
    countries: dict[str, str] = {}
    source1_path = args.data_dir / "train_source1.tsv"
    for chunk in pd.read_csv(
        source1_path, sep="\t",
        usecols=["entity_id", "business_address", "country"],
        dtype=str, keep_default_na=False, chunksize=CHUNK_SIZE,
    ):
        for entity_id, address, country in chunk.itertuples(index=False, name=None):
            if entity_id in truth:
                source_addresses[entity_id] = address_tokens(address)
                countries[entity_id] = country.strip().casefold() or "unknown"
    print(f"Validation Source 1 addresses: {len(source_addresses):,}", flush=True)

    # Start with the score for predicting an empty set for every validation row.
    global_totals = {
        threshold: sum(1.0 for matches in truth.values() if not matches)
        for threshold in THRESHOLDS
    }
    country_counts: dict[str, int] = defaultdict(int)
    country_baselines: dict[str, float] = defaultdict(float)
    for source_id, true_ids in truth.items():
        country = countries.get(source_id, "unknown")
        country_counts[country] += 1
        country_baselines[country] += f05(true_ids, set())
    country_totals = {
        country: {threshold: country_baselines[country] for threshold in THRESHOLDS}
        for country in country_counts
    }
    pair_count = 0

    with args.candidates.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for source_id, group in groupby(reader, key=lambda row: row["source1_entity_id"]):
            country = countries.get(source_id, "unknown")
            source_tokens = source_addresses.get(source_id, frozenset())
            candidates = []
            for row in group:
                candidate_id = row["candidate_entity_id"]
                score = address_overlap(
                    source_tokens, target_addresses.get(candidate_id, frozenset())
                )
                candidates.append((candidate_id, score))
                pair_count += 1

            true_ids = truth[source_id]
            empty_score = f05(true_ids, set())
            for threshold in THRESHOLDS:
                predictions = {
                    candidate_id for candidate_id, score in candidates if score >= threshold
                }
                score = f05(true_ids, predictions)
                global_totals[threshold] += score - empty_score
                country_totals[country][threshold] += score - empty_score

    total_rows = len(truth)
    global_scores = {t: global_totals[t] / total_rows for t in THRESHOLDS}
    best_global = max(THRESHOLDS, key=lambda t: (global_scores[t], t))
    best_by_country = {
        country: max(
            THRESHOLDS,
            key=lambda t: (
                country_totals[country][t] / country_counts[country],
                t,
            ),
        )
        for country in country_counts
    }
    country_specific_score = sum(
        country_totals[country][threshold]
        for country, threshold in best_by_country.items()
    ) / total_rows

    print(f"Candidate pairs scored: {pair_count:,}", flush=True)
    print("Global threshold results:")
    for threshold in THRESHOLDS:
        print(f"  {threshold:>4.1f}: {global_scores[threshold]:.6f}")
    print(
        f"Best global threshold: {best_global:.1f}; "
        f"macro F0.5={global_scores[best_global]:.6f}"
    )
    print("Best validation thresholds by country:")
    for country in sorted(best_by_country):
        threshold = best_by_country[country]
        score = country_totals[country][threshold] / country_counts[country]
        print(f"  {country}: threshold={threshold:.1f}, macro F0.5={score:.6f}")
    print(f"Country-specific validation macro F0.5: {country_specific_score:.6f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    threshold_path = args.output_dir / "address_threshold_report.tsv"
    with threshold_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["scope", "threshold", "macro_f05"])
        for threshold in THRESHOLDS:
            writer.writerow(["global", threshold, f"{global_scores[threshold]:.6f}"])
        for country, threshold in sorted(best_by_country.items()):
            writer.writerow([
                country, threshold,
                f"{country_totals[country][threshold] / country_counts[country]:.6f}",
            ])

    # Write best country-specific predictions; unseen countries use the global
    # threshold, which is also the fallback for the later test set.
    predicted: dict[str, str] = {}
    with args.candidates.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for source_id, group in groupby(reader, key=lambda row: row["source1_entity_id"]):
            threshold = best_by_country.get(countries.get(source_id, "unknown"), best_global)
            source_tokens = source_addresses.get(source_id, frozenset())
            kept = []
            for row in group:
                candidate_id = row["candidate_entity_id"]
                score = address_overlap(
                    source_tokens, target_addresses.get(candidate_id, frozenset())
                )
                if score >= threshold:
                    kept.append(candidate_id)
            if kept:
                predicted[source_id] = ",".join(kept)

    prediction_path = args.output_dir / "validation_predictions_address_overlap.tsv"
    with prediction_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for source_id in truth:
            writer.writerow([source_id, predicted.get(source_id, "")])

    print(f"Threshold report: {threshold_path}")
    print(f"Predictions: {prediction_path}")


if __name__ == "__main__":
    main()

