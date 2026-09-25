import math


def parse_id_list(value):
    if value is None:
        return set()
    if isinstance(value, float) and math.isnan(value):
        return set()

    value = str(value).strip()
    if not value or value.casefold() == "nan":
        return set()

    return {item.strip() for item in value.split(",") if item.strip()}


def f05_for_entity(true_ids, predicted_ids):
    true_ids = set(true_ids)
    predicted_ids = set(predicted_ids)

    if not true_ids:
        return 1.0 if not predicted_ids else 0.0

    correct = len(true_ids & predicted_ids)
    precision = correct / len(predicted_ids) if predicted_ids else 0.0
    recall = correct / len(true_ids)

    denominator = 0.25 * precision + recall
    if denominator == 0:
        return 0.0

    return 1.25 * precision * recall / denominator


def macro_f05(ground_truth, predictions):
    if not ground_truth:
        raise ValueError("Ground truth is empty")

    scores = [
        f05_for_entity(true_ids, predictions.get(source1_id, set()))
        for source1_id, true_ids in ground_truth.items()
    ]
    return sum(scores) / len(scores)
