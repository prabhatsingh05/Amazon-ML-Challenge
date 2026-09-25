import argparse
from collections import Counter, defaultdict

import pandas as pd

from metrics import parse_id_list


def load_ground_truth(path, chunksize=100_000):
    ground_truth = {}
    for chunk in pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunksize
    ):
        required = {"source1_entity_id", "matched_entity_ids"}
        missing = required.difference(chunk.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for source1_id, matched_ids in chunk[
            ["source1_entity_id", "matched_entity_ids"]
        ].itertuples(index=False, name=None):
            if source1_id in ground_truth:
                raise ValueError(f"Duplicate Source 1 label: {source1_id}")
            ground_truth[source1_id] = parse_id_list(matched_ids)
    return ground_truth


def main():
    parser = argparse.ArgumentParser(
        description="Measure candidate recall on held-out Source 1 labels."
    )
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument(
        "--candidates",
        required=True,
        help="Pair-level TSV with source1_entity_id and candidate_entity_id.",
    )
    args = parser.parse_args()
    ground_truth = load_ground_truth(args.ground_truth)
    candidate_counts = Counter()
    hits = defaultdict(int)
    total_candidate_rows = 0
    extra_source1_rows = 0

    for chunk in pd.read_csv(
        args.candidates, sep="\t", dtype=str, keep_default_na=False,
        chunksize=200_000
    ):
        required = {"source1_entity_id", "candidate_entity_id"}
        missing = required.difference(chunk.columns)
        if missing:
            raise ValueError(f"{args.candidates} is missing columns: {sorted(missing)}")
        for source1_id, candidate_id in chunk[
            ["source1_entity_id", "candidate_entity_id"]
        ].itertuples(index=False, name=None):
            total_candidate_rows += 1
            if source1_id not in ground_truth:
                extra_source1_rows += 1
                continue
            candidate_counts[source1_id] += 1
            if candidate_id in ground_truth[source1_id]:
                hits[source1_id] = min(
                    len(ground_truth[source1_id]), hits[source1_id] + 1
                )

    total_true_links = sum(len(ids) for ids in ground_truth.values())
    found_true_links = sum(hits.values())
    positive_ids = [sid for sid, ids in ground_truth.items() if ids]
    singleton_ids = [sid for sid, ids in ground_truth.items() if not ids]
    macro_recall = (
        sum(hits[sid] / len(ground_truth[sid]) for sid in positive_ids)
        / len(positive_ids)
        if positive_ids else 0.0
    )
    singleton_candidate_rate = (
        sum(candidate_counts[sid] > 0 for sid in singleton_ids) / len(singleton_ids)
        if singleton_ids else 0.0
    )

    print(f"Validation Source 1 rows: {len(ground_truth):,}")
    print(f"Candidate pair rows: {total_candidate_rows:,}")
    print(f"Candidate rows for unknown Source 1 IDs: {extra_source1_rows:,}")
    print(f"True links found: {found_true_links:,} / {total_true_links:,}")
    print(
        "Micro candidate recall: "
        f"{found_true_links / total_true_links:.6f}"
        if total_true_links else "Micro candidate recall: n/a"
    )
    print(f"Macro recall over non-singletons: {macro_recall:.6f}")
    print(
        "Average candidates per validation Source 1: "
        f"{sum(candidate_counts.values()) / len(ground_truth):.2f}"
    )
    print(f"Singletons with any candidate: {singleton_candidate_rate:.4%}")


if __name__ == "__main__":
    main()
