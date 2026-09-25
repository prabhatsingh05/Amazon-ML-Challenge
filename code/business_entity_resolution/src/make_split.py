import argparse
import hashlib
from pathlib import Path

import pandas as pd


def is_validation_id(source1_id):
    digest = hashlib.blake2b(
        str(source1_id).encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big") % 5 == 0


def main():
    parser = argparse.ArgumentParser(
        description="Create a stable 80/20 split by Source 1 entity ID."
    )
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_labels.tsv"
    valid_path = output_dir / "valid_labels.tsv"

    first_chunk = True
    train_rows = 0
    valid_rows = 0

    for chunk in pd.read_csv(
        args.ground_truth,
        sep="\t",
        chunksize=100_000,
        dtype=str,
        keep_default_na=False,
    ):
        required = {"source1_entity_id", "matched_entity_ids"}
        missing = required.difference(chunk.columns)
        if missing:
            raise ValueError(f"Ground-truth file missing columns: {sorted(missing)}")

        validation_mask = chunk["source1_entity_id"].map(is_validation_id)
        train_chunk = chunk.loc[~validation_mask]
        valid_chunk = chunk.loc[validation_mask]
        mode = "w" if first_chunk else "a"

        train_chunk.to_csv(
            train_path,
            sep="\t",
            index=False,
            mode=mode,
            header=first_chunk,
            encoding="utf-8",
        )
        valid_chunk.to_csv(
            valid_path,
            sep="\t",
            index=False,
            mode=mode,
            header=first_chunk,
            encoding="utf-8",
        )

        train_rows += len(train_chunk)
        valid_rows += len(valid_chunk)
        first_chunk = False

    print(f"Train label rows: {train_rows:,}")
    print(f"Validation label rows: {valid_rows:,}")
    print(f"Split files: {output_dir}")


if __name__ == "__main__":
    main()
