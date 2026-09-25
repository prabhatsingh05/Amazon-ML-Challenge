# Amazon ML Challenge: Business Entity Resolution

Team solution code for matching each Source 1 business to zero or more records from Source 2 and Source 3.

## Local data

Keep the supplied dataset outside this Git repository. The scripts below expect the extracted data beside this repository:

- ../student_resource/dataset/train/
- ../student_resource/dataset/test/

Do not commit the dataset or virtual environment.

## Environment

From the repository root on Windows:

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r .\code\business_entity_resolution\requirements.txt

The data and generated split files are local. The repository .gitignore excludes work/, datasets, and virtual environments.

## Create the fixed validation split

The split is deterministic by Source 1 ID, so all team members get the same 80/20 split:

    .\.venv\Scripts\python.exe .\code\business_entity_resolution\src\make_split.py --ground-truth ..\student_resource\dataset\train\train_ground_truth.tsv --output-dir .\work\splits

This writes train_labels.tsv and valid_labels.tsv under work/splits/. Do not upload these data-derived files to GitHub.

## Score validation predictions

When a validation prediction file is available in the submission column format:

    .\.venv\Scripts\python.exe .\code\business_entity_resolution\src\evaluate.py --ground-truth .\work\splits\valid_labels.tsv --predictions .\work\validation_predictions.tsv

The evaluator reports macro F0.5 and missing or extra prediction rows.

## Current components

- src/normalize.py: Unicode-safe text normalization and RapidFuzz text similarity.
- src/metrics.py: per-entity and macro F0.5 helpers, including singleton scoring.
- src/make_split.py: chunked, deterministic split by Source 1 entity ID.
- src/evaluate.py: score predictions against the held-out labels.

Candidate generation and the matching model are developed as separate components and will be documented here when integrated.

## Reproducibility

Python package versions are pinned in requirements.txt. The full pipeline should run using only the supplied challenge files and these code artefacts; do not use external business lookup, geocoding, or enrichment services.
