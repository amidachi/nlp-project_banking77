"""Fit BANKING77 intent models and evaluate selective routing on paired test sets."""

import argparse
import csv
import json
import platform
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from prepare import ROOT, embedding_path, load_split, with_wrappers

SEED = 42
RISK_TARGET = 0.05
NEIGHBORS = 8


def group_key(text):
    return re.sub(r"[^\w]+", " ", text.casefold()).strip()


def load_embeddings(split, wrapped=False):
    path = embedding_path(split, wrapped)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run prepare.py first.")
    return np.load(path, allow_pickle=False)


def lexical_model():
    features = FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("characters", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True,
        )),
    ])
    return make_pipeline(features, LogisticRegression(C=20, max_iter=500, tol=1e-4))


def semantic_model():
    return LogisticRegression(C=100, max_iter=800, tol=1e-5)


def knn_support(query, bank, labels, prediction):
    """Fraction of the eight nearest fit examples that support the prediction."""
    values = []
    for start in range(0, len(query), 256):
        stop = start + 256
        similarity = query[start:stop] @ bank.T
        nearest = np.argpartition(similarity, -NEIGHBORS, axis=1)[:, -NEIGHBORS:]
        values.extend(np.mean(labels[nearest] == prediction[start:stop, None], axis=1))
    return np.asarray(values)


def model_scores(probabilities, embeddings, bank, bank_labels):
    top = np.sort(probabilities, axis=1)[:, -2:]
    margin = top[:, 1] - top[:, 0]
    support = knn_support(embeddings, bank, bank_labels, probabilities.argmax(axis=1))
    return np.column_stack((top[:, 1], margin, support))


def threshold_at_risk(score, errors):
    """Choose the largest eligible calibration prefix, preserving complete ties."""
    order = np.argsort(-score, kind="stable")
    sorted_scores = score[order]
    sizes = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True]) + 1
    acceptable = sizes[np.cumsum(errors[order])[sizes - 1] / sizes <= RISK_TARGET]
    return float(sorted_scores[acceptable[-1] - 1]) if len(acceptable) else None


def risk_at_threshold(score, errors, threshold):
    accepted = score >= threshold if threshold is not None else np.zeros(len(score), dtype=bool)
    count = int(accepted.sum())
    mistakes = int(np.sum(accepted & errors))
    interval = binomtest(mistakes, count).proportion_ci(method="wilson") if count else None
    return {
        "accepted": count, "errors": mistakes, "coverage": count / len(errors),
        "risk": mistakes / count if count else None,
        "risk_wilson_95": [interval.low, interval.high] if interval else None,
    }


def curve(score, errors):
    order = np.argsort(-score, kind="stable")
    sorted_scores = score[order]
    bounds = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1, len(score)]
    sizes = np.diff(bounds)
    # Use the expected cumulative errors under random ordering within tied scores.
    errors_at_bounds = np.r_[0, np.cumsum(errors[order])[bounds[1:] - 1]]
    tie_mean = np.diff(errors_at_bounds) / sizes
    ranks = np.arange(1, len(score) + 1)
    preceding = np.repeat(bounds[:-1], sizes)
    expected = np.repeat(errors_at_bounds[:-1], sizes)
    expected = expected + (ranks - preceding) * np.repeat(tie_mean, sizes)
    return expected / ranks


def classify(probabilities, target):
    prediction = probabilities.argmax(axis=1)
    return {
        "accuracy": accuracy_score(target, prediction),
        "macro_f1": f1_score(target, prediction, average="macro", zero_division=0),
        "correct": int((prediction == target).sum()),
    }


def paired(first, second):
    first_only = int(np.sum(first & ~second))
    second_only = int(np.sum(second & ~first))
    total = first_only + second_only
    return {
        "first_only": first_only, "second_only": second_only,
        "delta": float(np.mean(first) - np.mean(second)),
        "p_value": binomtest(first_only, total).pvalue if total else 1.0,
    }


def dataset_stats(texts, targets):
    lengths = np.asarray([len(text.split()) for text in texts])
    counts = Counter(targets)
    return {
        "rows": len(texts), "classes": len(counts), "min_per_class": min(counts.values()),
        "max_per_class": max(counts.values()), "median_words": float(np.median(lengths)),
        "p90_words": float(np.percentile(lengths, 90)),
        "duplicate_rows": len(texts) - len(set(map(group_key, texts))),
    }


def choose_weight(lexical, semantic, labels):
    weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    scores = [
        accuracy_score(labels, (weight * lexical + (1 - weight) * semantic).argmax(axis=1))
        for weight in weights
    ]
    return weights[int(np.argmax(scores))], dict(zip(weights, scores))


def write_csv(path, rows):
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output):
    train, raw_train = load_split("train")
    test, raw_test = load_split("test")
    variants = {"original": test, "wrapped": with_wrappers(test)}
    classes = np.unique(raw_train)
    targets = np.searchsorted(classes, raw_train)
    gold = np.searchsorted(classes, raw_test)
    groups = np.asarray([group_key(text) for text in train])
    dev_split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fit, held = next(dev_split.split(train, targets, groups))
    held_split = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=SEED)
    dev_idx, cal_idx = next(held_split.split(held, targets[held], groups[held]))
    dev, cal = held[dev_idx], held[cal_idx]
    assert not (set(groups[fit]) & set(groups[held]) or set(groups[dev]) & set(groups[cal]))
    for split in (fit, dev, cal):
        assert len(np.unique(targets[split])) == len(classes)
    train_vectors = load_embeddings("train")
    test_vectors = {key: load_embeddings("test", key == "wrapped") for key in variants}
    train_text = np.asarray(train)
    overlap = np.isin(list(map(group_key, test)), groups)
    train_splits = {"fit": fit, "development": dev, "calibration": cal, "train": np.arange(len(train))}
    result = {
        "seed": SEED,
        "platform": platform.platform(),
        "risk_target": RISK_TARGET,
        "data": {
            name: dataset_stats(train_text[index], targets[index])
            for name, index in train_splits.items()
        },
    }
    result["data"].update({key: dataset_stats(value, gold) for key, value in variants.items()})
    result["data"]["test_rows_matching_train"] = int(overlap.sum())

    print("Fitting models on the grouped fit split", flush=True)
    lexical = lexical_model().fit(train_text[fit], targets[fit])
    semantic = semantic_model().fit(train_vectors[fit], targets[fit])
    dummy = DummyClassifier(strategy="most_frequent").fit(train_vectors[fit], targets[fit])

    def predict(texts, vectors):
        sparse = lexical.predict_proba(texts)
        dense = semantic.predict_proba(vectors)
        return sparse, dense

    dev_sparse, dev_dense = predict(train_text[dev], train_vectors[dev])
    weight, grid = choose_weight(dev_sparse, dev_dense, targets[dev])
    result["fusion_weight"] = weight
    result["development_fusion_accuracy"] = grid

    def mix(sparse, dense):
        return weight * sparse + (1 - weight) * dense

    def gate_features(probabilities, vectors):
        return model_scores(probabilities, vectors, train_vectors[fit], targets[fit])

    dev_prob = mix(dev_sparse, dev_dense)
    dev_features = gate_features(dev_prob, train_vectors[dev])
    gate = make_pipeline(StandardScaler(), LogisticRegression(C=1))
    dev_correct = dev_prob.argmax(axis=1) == targets[dev]
    gate.fit(dev_features, dev_correct)
    score_only = make_pipeline(StandardScaler(), LogisticRegression(C=1))
    score_only.fit(dev_features[:, :2], dev_correct)

    def gate_scores(features):
        return {
            "max_prob": features[:, 0], "prob_margin": features[:, 1],
            "neighbors": features[:, 2],
            "score_only_gate": score_only.predict_proba(features[:, :2])[:, 1],
            "learned_gate": gate.predict_proba(features)[:, 1],
        }

    cal_prob = mix(*predict(train_text[cal], train_vectors[cal]))
    cal_errors = cal_prob.argmax(axis=1) != targets[cal]
    cal_scores = gate_scores(gate_features(cal_prob, train_vectors[cal]))
    thresholds = {name: threshold_at_risk(values, cal_errors) for name, values in cal_scores.items()}
    result["gate_standardized_weights"] = dict(zip(
        ("max_prob", "prob_margin", "neighbors"), gate[-1].coef_[0].tolist(),
    ))
    result["thresholds"] = thresholds
    result["calibration"] = {
        name: risk_at_threshold(values, cal_errors, thresholds[name]) for name, values in cal_scores.items()
    }

    output.mkdir(parents=True, exist_ok=True)
    predictions = []
    curves = []
    decisions = {}
    result["test"] = {}
    for name, texts in variants.items():
        sparse, dense = predict(texts, test_vectors[name])
        probs = mix(sparse, dense)
        predicted = probs.argmax(axis=1)
        errors = predicted != gold
        decisions[name] = ~errors
        scores = gate_scores(gate_features(probs, test_vectors[name]))
        classifiers = {"majority": dummy.predict_proba(test_vectors[name]),
                       "tfidf": sparse, "bge": dense, "fusion": probs}
        result["test"][name] = {
            "classification": {key: classify(value, gold) for key, value in classifiers.items()},
            "fusion_no_overlap_accuracy": float(np.mean(~errors[~overlap])),
            "fusion_vs_bge_mcnemar": paired(~errors, dense.argmax(axis=1) == gold),
            "routing": {},
            "wrapper_accuracy": {
                str(i + 1): float(np.mean(~errors[i::6])) for i in range(6)
            } if name == "wrapped" else {},
        }
        for key, values in scores.items():
            risks = curve(values, errors)
            operating_point = risk_at_threshold(values, errors, thresholds[key])
            operating_point.update({
                "aurc": float(risks.mean()), "risk_at_80pct": float(risks[int(0.8 * len(test)) - 1]),
            })
            if key == "learned_gate":
                accepted = values >= thresholds[key] if thresholds[key] is not None else np.zeros(len(test), bool)
                operating_point["coverage_by_intent"] = {
                    label: float(accepted[gold == i].mean()) for i, label in enumerate(classes)
                }
            result["test"][name]["routing"][key] = operating_point
            curves.extend({"variant": name, "policy": key, "coverage": i / len(test), "risk": float(risk)}
                          for i, risk in enumerate(risks, 1))
        counts = Counter(zip(raw_test[errors], classes[predicted[errors]]))
        result["test"][name]["top_confusions"] = [
            {"gold": expected, "predicted": got, "count": count}
            for (expected, got), count in counts.most_common(10)
        ]
        for index in range(len(test)):
            row = {
                "variant": name, "row": index, "gold": raw_test[index],
                "tfidf": classes[sparse[index].argmax()], "bge": classes[dense[index].argmax()],
                "fusion": classes[predicted[index]], "in_train": int(overlap[index]),
            }
            for key, values in scores.items():
                row[key] = float(values[index])
                row[f"{key}_auto"] = int(thresholds[key] is not None and values[index] >= thresholds[key])
            predictions.append(row)

    result["original_vs_wrapped_mcnemar"] = paired(decisions["original"], decisions["wrapped"])

    print("Refitting classification baselines on all official train rows", flush=True)
    # This separate run permits direct comparison to published full-training accuracy.
    lexical = lexical_model().fit(train, targets)
    semantic = semantic_model().fit(train_vectors, targets)
    sparse, dense = predict(test, test_vectors["original"])
    result["full_train_original"] = {
        key: classify(value, gold) for key, value in
        {"tfidf": sparse, "bge": dense, "fusion": mix(sparse, dense)}.items()
    }
    result["full_train_original"]["fusion_vs_bge_mcnemar"] = paired(
        mix(sparse, dense).argmax(axis=1) == gold, dense.argmax(axis=1) == gold,
    )
    write_csv(output / "predictions.csv", predictions)
    write_csv(output / "risk_coverage.csv", curves)
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved results to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args.output)


if __name__ == "__main__":
    main()
