import argparse
from pathlib import Path

import pandas as pd

from metrics import macro_f05, parse_id_list


def load_id_map(path, id_column, values_column, chunksize=100_000):
    result = {}

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
    ):
        required = {id_column, values_column}
        missing = required.difference(chunk.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        for source1_id, values in chunk[
            [id_column, values_column]
        ].itertuples(index=False, name=None):
            if source1_id in result:
                raise ValueError(f"Duplicate Source 1 row: {source1_id}")
            result[source1_id] = parse_id_list(values)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Score validation predictions using macro F0.5."
    )
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()

    ground_truth = load_id_map(
        args.ground_truth, "source1_entity_id", "matched_entity_ids"
    )
    predictions = load_id_map(
        args.predictions, "source1_entity_id", "matched_entity_ids"
    )

    missing = len(set(ground_truth) - set(predictions))
    extra = len(set(predictions) - set(ground_truth))
    score = macro_f05(ground_truth, predictions)

    print(f"Validation Source 1 rows: {len(ground_truth):,}")
    print(f"Missing prediction rows (scored as empty): {missing:,}")
    print(f"Extra prediction rows (not scored): {extra:,}")
    print(f"Macro F0.5: {score:.6f}")


if __name__ == "__main__":
    main()
