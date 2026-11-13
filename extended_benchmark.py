"""Compare larger frozen encoders with the original BANKING77 benchmark."""

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from prepare import CACHE, MAX_TOKENS, ROOT, ensure_asset, load_split
from run_experiment import (
    SEED, classify, group_key, lexical_model, load_embeddings, paired, semantic_model,
    write_csv,
)

ENCODERS = {
    "bge-base": (
        "Xenova/bge-base-en-v1.5", "4d6cd88e18e51a5e020c2c305726d76ada9c03cf",
        "c9729cc84cbd0e9fecc759505d2be65916c9fe05222d7ea26c65fcb3382af38d",
        "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    ),
    "e5-base": (
        "Xenova/e5-base-v2", "21f8d0e36fdfe76e6a023802dfb293fc6d750ad1",
        "d0a7aadf8227f5212695488d99de54523678f5f9ecfb070f2b58cc76cd84d882",
        "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    ),
}
BATCH = 32


def embeddings(name, splits, threads):
    import onnxruntime as ort
    from tokenizers import Tokenizer

    repo, revision, model_hash, tokenizer_hash = ENCODERS[name]
    folder = CACHE / name
    paths = {split: folder / f"{split}-{revision[:12]}-{MAX_TOKENS}.npy" for split in splits}
    missing = [split for split, path in paths.items() if not path.is_file()]
    if missing:
        url = f"https://huggingface.co/{repo}/resolve/{revision}"
        ensure_asset(f"{url}/tokenizer.json?download=true", folder / "tokenizer.json", tokenizer_hash)
        model_path = folder / "model_quantized.onnx"
        ensure_asset(f"{url}/onnx/model_quantized.onnx?download=true", model_path, model_hash)
        tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        tokenizer.enable_truncation(max_length=MAX_TOKENS)
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            model_path.read_bytes(), sess_options=options, providers=["CPUExecutionProvider"],
        )
        required = {item.name for item in session.get_inputs()}
        for split in missing:
            texts = splits[split]
            if name == "e5-base":
                texts = [f"query: {text}" for text in texts]
            output = np.empty((len(texts), 768), dtype=np.float32)
            order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
            for offset in range(0, len(order), BATCH):
                indices = order[offset:offset + BATCH]
                tokens = tokenizer.encode_batch([texts[index] for index in indices])
                fields = {
                    "input_ids": [item.ids for item in tokens],
                    "attention_mask": [item.attention_mask for item in tokens],
                    "token_type_ids": [item.type_ids for item in tokens],
                }
                feed = {key: np.asarray(fields[key], dtype=np.int64) for key in sorted(required)}
                hidden = session.run(["last_hidden_state"], feed)[0]
                if name == "e5-base":
                    mask = feed["attention_mask"][..., None]
                    pooled = (hidden * mask).sum(axis=1)
                else:
                    pooled = hidden[:, 0]
                output[indices] = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
                if offset % 2048 == 0:
                    print(f"{name}/{split}: {min(offset + BATCH, len(texts))}/{len(texts)}", flush=True)
            with NamedTemporaryFile(dir=folder, suffix=".part", delete=False) as stream:
                np.save(stream, output, allow_pickle=False)
            Path(stream.name).replace(paths[split])
    return {split: np.load(path, allow_pickle=False) for split, path in paths.items()}


def selection_split(train, targets):
    groups = [group_key(text) for text in train]
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    return next(folds.split(train, targets, groups))


def accuracy(probabilities, targets):
    return float(np.mean(probabilities.argmax(axis=1) == targets))


def mixture(scores, weights):
    return sum(scores[name] * weight for name, weight in weights.items() if weight)


def fit_models(train, target, vectors, train_rows, query_text, query_vectors):
    lexical = lexical_model().fit(train[train_rows], target[train_rows])
    scores = {"tfidf": lexical.predict_proba(query_text)}
    for name in vectors:
        print(f"Fitting {name} on {len(train_rows)} rows", flush=True)
        linear = semantic_model().fit(vectors[name][train_rows], target[train_rows])
        scores[name] = linear.predict_proba(query_vectors[name])

    features = np.concatenate([vectors[name] / np.sqrt(len(vectors)) for name in vectors], axis=1)
    query = np.concatenate([query_vectors[name] / np.sqrt(len(vectors)) for name in vectors], axis=1)
    nonlinear = make_pipeline(
        Nystroem(kernel="rbf", gamma=2.0, n_components=512, random_state=SEED),
        LogisticRegression(C=30, max_iter=500, tol=1e-4),
    )
    print(f"Fitting RBF features on {len(train_rows)} rows", flush=True)
    nonlinear.fit(features[train_rows], target[train_rows])
    scores["rbf"] = nonlinear.predict_proba(query)
    return scores


def run(threads, encode_only):
    train, train_labels = load_split("train")
    test, test_labels = load_split("test")
    splits = {"train": train, "test": test}
    vectors = {name: embeddings(name, splits, threads) for name in ENCODERS}
    if encode_only:
        return
    vectors["bge-small"] = {split: load_embeddings(split) for split in splits}
    train_vectors = {key: value["train"] for key, value in vectors.items()}
    test_vectors = {key: value["test"] for key, value in vectors.items()}
    classes = np.unique(train_labels)
    targets, gold = np.searchsorted(classes, train_labels), np.searchsorted(classes, test_labels)
    train = np.asarray(train)
    fit, dev = selection_split(train, targets)
    dev_scores = fit_models(train, targets, train_vectors, fit, train[dev],
                            {key: value[dev] for key, value in train_vectors.items()})
    names = tuple(vectors)
    uniform = {name: 1 / len(names) for name in names}
    candidates = {name: {name: 1.0} for name in dev_scores}
    candidates["dense-average"] = uniform
    candidates["original-fusion"] = {"tfidf": 0.2, "bge-small": 0.8}
    for lexical in (0.0, 0.1, 0.2):
        for rbf in (0.0, 0.25, 0.50, 0.75):
            key = f"mix-word-{lexical:.1f}-rbf-{rbf:.2f}"
            weights = {name: weight * (1 - lexical - rbf) for name, weight in uniform.items()}
            candidates[key] = {**weights, "tfidf": lexical, "rbf": rbf}
    development = {
        name: accuracy(mixture(dev_scores, weights), targets[dev])
        for name, weights in candidates.items()
    }
    chosen = max(development, key=development.get)
    print(f"Validation winner: {chosen} ({development[chosen]:.4f})", flush=True)
    full_scores = fit_models(train, targets, train_vectors, np.arange(len(train)), test, test_vectors)
    predictions = {name: mixture(full_scores, weights) for name, weights in candidates.items()
                   if name in (*full_scores, "dense-average", "original-fusion", chosen)}
    result = {
        "encoders": {name: {"repo": values[0], "revision": values[1]} for name, values in ENCODERS.items()},
        "seed": SEED, "fit_rows": len(fit), "development_rows": len(dev),
        "development_accuracy": development, "selected_model": chosen, "weights": candidates[chosen],
        "official_test": {name: classify(scores, gold) for name, scores in predictions.items()},
        "paired_vs_original": paired(
            predictions[chosen].argmax(axis=1) == gold,
            predictions["original-fusion"].argmax(axis=1) == gold,
        ),
    }
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    (output / "extended_benchmark.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    columns = {name: classes[scores.argmax(axis=1)] for name, scores in predictions.items()}
    write_csv(output / "extended_predictions.csv", (
        {"row": index, "gold": label, **{name: column[index] for name, column in columns.items()}}
        for index, label in enumerate(test_labels)
    ))
    print(json.dumps(result["official_test"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--encode-only", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    with threadpool_limits(limits=4):
        run(args.threads, args.encode_only)


if __name__ == "__main__":
    main()
