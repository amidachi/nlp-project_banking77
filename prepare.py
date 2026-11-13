"""Verify BANKING77 and fetch/cache frozen sentence embeddings."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from shutil import copyfileobj
from time import perf_counter
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = ROOT / "cache"
DATA_REVISION = "9d081458ff52e53cf7e848f414e6e9344e4e6696"
DATA_URL = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{DATA_REVISION}/banking_data"
MODEL_REVISION = "ea104dacec62c0de699686887e3f920caeb4f3e3"
MODEL_URL = f"https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/{MODEL_REVISION}"
DATA_FILES = {
    "train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
    "categories.json": "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63",
}
MODEL_FILES = {
    "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    "onnx/model_quantized.onnx": "6c9c6101a956d62dfb5e7190c538226c0c5bb9cb27b651234b6df063ee7dbfe4",
}
MAX_TOKENS = 128
BATCH_SIZE = 32
WRAPPERS = (
    ("Hi there! ", " Thank you for your help."),
    ("Hello, I hope you are doing well. ", " I appreciate your time."),
    ("Good day! I have a quick question. ", " Thanks in advance!"),
    ("Hi support team. Could you help me with this? ", " Many thanks."),
    ("Hello! Sorry to bother you. ", " I would really appreciate your help."),
    ("Hi, thanks for being available. ", " Have a nice day!"),
)
WRAPPER_VERSION = "polite-v1"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ensure_asset(url, path, expected):
    if path.is_file() and sha256(path) == expected:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    try:
        with urlopen(url, timeout=120) as response, partial.open("wb") as stream:
            copyfileobj(response, stream)
        if sha256(partial) != expected:
            raise ValueError(f"SHA-256 mismatch for {url}")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def load_split(split):
    path = DATA / f"{split}.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected_rows = {"train": 10_003, "test": 3_080}
    labels = json.loads((DATA / "categories.json").read_text(encoding="utf-8"))
    texts = [row["text"] for row in rows]
    targets = np.asarray([row["category"] for row in rows])
    if len(rows) != expected_rows[split] or len(labels) != 77 or set(targets) != set(labels):
        raise ValueError(f"Unexpected rows or labels in {path}")
    if any(not text.strip() for text in texts):
        raise ValueError(f"Empty utterance in {path}")
    return texts, targets


def with_wrappers(texts):
    return [
        WRAPPERS[index % len(WRAPPERS)][0] + text + WRAPPERS[index % len(WRAPPERS)][1]
        for index, text in enumerate(texts)
    ]


def embedding_path(split, wrapped=False):
    data_hash = DATA_FILES[f"{split}.csv"][:12]
    variant = f"-{WRAPPER_VERSION}" if wrapped else ""
    return CACHE / f"{split}-{data_hash}-{MODEL_REVISION[:12]}-{MAX_TOKENS}{variant}.npy"


def encode(split, tokenizer, model, wrapped=False):
    texts, _ = load_split(split)
    output_path = embedding_path(split, wrapped)
    if output_path.is_file():
        return
    if wrapped:
        texts = with_wrappers(texts)
    start = perf_counter()
    output = np.empty((len(texts), 384), dtype=np.float32)
    inputs = {item.name for item in model.get_inputs()}
    # Length ordering reduces padding. Restore the original CSV row order.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    for offset in range(0, len(texts), BATCH_SIZE):
        indices = order[offset : offset + BATCH_SIZE]
        batch = tokenizer.encode_batch([texts[i] for i in indices])
        encoded = {
            "input_ids": [item.ids for item in batch],
            "attention_mask": [item.attention_mask for item in batch],
            "token_type_ids": [item.type_ids for item in batch],
        }
        feed = {name: np.asarray(encoded[name], dtype=np.int64) for name in sorted(inputs)}
        vectors = model.run(["last_hidden_state"], feed)[0][:, 0, :]
        output[indices] = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        if offset % 2048 == 0:
            print(f"{split}: encoded {min(offset + BATCH_SIZE, len(texts))}/{len(texts)}", flush=True)
    # Replace the destination only after all vectors have been written.
    partial = output_path.with_suffix(".part")
    with partial.open("wb") as stream:
        np.save(stream, output, allow_pickle=False)
    partial.replace(output_path)
    print(f"{split}: {len(texts)} embeddings in {perf_counter() - start:.1f} s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4, help="ONNX CPU threads")
    parser.add_argument("--data-only", action="store_true", help="Verify or fetch only CSVs and labels")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")

    for name, expected in DATA_FILES.items():
        ensure_asset(f"{DATA_URL}/{name}", DATA / name, expected)
    if args.data_only:
        return

    import onnxruntime as ort
    from tokenizers import Tokenizer

    for name, expected in MODEL_FILES.items():
        ensure_asset(f"{MODEL_URL}/{name}?download=true", CACHE / Path(name).name, expected)
    tokenizer = Tokenizer.from_file(str(CACHE / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=MAX_TOKENS)
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    # The byte API also supports checkout paths with non-ASCII characters on Windows.
    model = ort.InferenceSession(
        (CACHE / "model_quantized.onnx").read_bytes(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    for split in ("train", "test"):
        encode(split, tokenizer, model)
    encode("test", tokenizer, model, wrapped=True)


if __name__ == "__main__":
    main()
