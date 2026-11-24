"""Compare 2025–2026 embedding models using the original BANKING77 routing protocol."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from prepare import (
    CACHE, DATA, DATA_FILES, ROOT, WRAPPER_VERSION, ensure_asset, load_split, sha256,
    with_wrappers,
)
from run_experiment import run

MODELS = {
    "qwen3": {
        "name": "Qwen3-Embedding-0.6B",
        "source": "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B",
        "release": "2025-06-05",
        "repo": "onnx-community/Qwen3-Embedding-0.6B-ONNX",
        "revision": "c25a394dd583836952667c12f008335071b3f43d",
        "files": {
            "tokenizer.json": "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a",
            "onnx/model_quantized.onnx": "87cd124e0ef1fd1f223ebc283efccbaeac386d0b08344701c46975d0657b591f",
        },
        "prefix": "Instruct: Classify the intent of a banking customer support request.\nQuery: ",
        "dimensions": 1024,
        "pad_id": 151643,
        "pad_token": "<|endoftext|>",
        "output": "last_hidden_state",
        # This dynamic-int8 export changes vectors when unrelated texts share a batch.
        # Independent encoding avoids that batch-composition artifact.
        "batch": 1,
    },
    "embeddinggemma": {
        "name": "EmbeddingGemma-300M",
        "source": "https://huggingface.co/google/embeddinggemma-300m",
        "release": "2025-09-04",
        "repo": "onnx-community/embeddinggemma-300m-ONNX",
        "revision": "5090578d9565bb06545b4552f76e6bc2c93e4a66",
        "files": {
            "tokenizer.json": "4dda02faaf32bc91031dc8c88457ac272b00c1016cc679757d1c441b248b9c47",
            "onnx/model_quantized.onnx": "172efde319fe1542dc41f31be6154910b05b78f7a861c265c4600eec906bd6d8",
            "onnx/model_quantized.onnx_data": "705626e28e4c23c82ade34566b4197d97f534c12275fa406dfb71e9937d388c0",
        },
        "prefix": "task: classification | query: ",
        "dimensions": 768,
        "pad_id": 0,
        "pad_token": "<pad>",
        "output": "sentence_embedding",
        "batch": 16,
    },
    "jina-v5": {
        "name": "Jina v5 Nano (classification)",
        "source": "https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-classification",
        "release": "2026-02-18",
        "repo": "jinaai/jina-embeddings-v5-text-nano-classification",
        "revision": "a0129e9b8ea4c54f3dfd250793380d8d69058da3",
        "files": {
            "tokenizer.json": "98d4a1d32152d6cedf85b5e88f3b205106dca1fe72aaab34e0ac13c238421069",
            "onnx/model_quantized.onnx": "ac93a7417c216e5076e37da2b3599f7ef16513934098a477680440c09f735a08",
            "onnx/model_quantized.onnx_data": "cd7b10e9d82dc3c2f27cfca902e15807653c8c1d3445632dcbb62c6d49514cc0",
        },
        "prefix": "Document: ",
        "dimensions": 768,
        "pad_id": 128004,
        "pad_token": "<|pad|>",
        "output": "last_hidden_state",
        "batch": 16,
    },
}
TOKEN_LIMIT = 128


def save_array(path, value):
    with NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as stream:
        temporary = Path(stream.name)
        np.save(stream, value, allow_pickle=False)
    temporary.replace(path)


def empty_kv_cache(inputs, rows):
    """The decoder export takes explicit empty past states for independent texts."""
    values = {}
    for item in inputs:
        if not item.name.startswith("past_key_values."):
            continue
        shape = item.shape
        if len(shape) != 4 or not all(isinstance(shape[i], int) for i in (1, 3)):
            raise ValueError(f"Unsupported cache shape for {item.name}: {shape}")
        dtype = {"tensor(float)": np.float32, "tensor(float16)": np.float16}[item.type]
        values[item.name] = np.empty((rows, shape[1], 0, shape[3]), dtype=dtype)
    return values


def encode(name, threads):
    import onnxruntime as ort
    from tokenizers import Tokenizer

    spec = MODELS[name]
    folder = CACHE / name
    for filename, expected in DATA_FILES.items():
        if sha256(DATA / filename) != expected:
            raise ValueError(f"Data checksum mismatch for {filename}")
    for filename, expected in spec["files"].items():
        url = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{filename}?download=true"
        ensure_asset(url, folder / filename, expected)
    config = {**spec, "max_tokens": TOKEN_LIMIT, "data": DATA_FILES, "wrappers": WRAPPER_VERSION}
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    destination = folder / signature
    destination.mkdir(exist_ok=True)
    train, _ = load_split("train")
    original, _ = load_split("test")
    texts_by_split = {"train": train, "original": original, "wrapped": with_wrappers(original)}
    tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    statistics = {}
    for split, texts in texts_by_split.items():
        counts = [len(item.ids) for item in tokenizer.encode_batch(
            [spec["prefix"] + text for text in texts],
        )]
        statistics[split] = {
            "rows": len(texts), "maximum_tokens": max(counts),
            "rows_over_token_limit": sum(count > TOKEN_LIMIT for count in counts),
        }
    metadata = {**spec, "max_tokens": TOKEN_LIMIT, "token_statistics": statistics,
                "pooling": "exported sentence_embedding" if spec["output"] == "sentence_embedding"
                else "last non-padding token", "l2_normalized": True,
                "onnx_precision": "int8 quantized", "cache_signature": signature}
    if name == "qwen3":
        metadata["kv_cache_initialization"] = "empty for every independent batch"
    (destination / "encoding.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if all((destination / f"{key}.npy").is_file() for key in texts_by_split):
        return {key: np.load(destination / f"{key}.npy", allow_pickle=False) for key in texts_by_split}, metadata

    tokenizer.enable_truncation(max_length=TOKEN_LIMIT)
    tokenizer.enable_padding(pad_id=spec["pad_id"], pad_token=spec["pad_token"], direction="right")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    # File loading resolves EmbeddingGemma's adjacent ONNX external-data file.
    session = ort.InferenceSession(
        str(folder / "onnx/model_quantized.onnx"),
        sess_options=options, providers=["CPUExecutionProvider"],
    )
    inputs = session.get_inputs()
    required = {item.name for item in inputs}
    cache_names = set(empty_kv_cache(inputs, 1))
    supported = {"input_ids", "attention_mask", "position_ids", "token_type_ids"}
    if not required.issubset(supported | cache_names):
        raise ValueError(f"Unexpected ONNX inputs {required - supported - cache_names}")
    if spec["output"] not in {item.name for item in session.get_outputs()}:
        raise ValueError(f"Expected ONNX output {spec['output']}")
    for split, texts in texts_by_split.items():
        final_path = destination / f"{split}.npy"
        if final_path.is_file():
            continue
        started = perf_counter()
        chunks = destination / split
        chunks.mkdir(exist_ok=True)
        output = np.empty((len(texts), spec["dimensions"]), dtype=np.float32)
        order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
        for offset in range(0, len(texts), spec["batch"]):
            indices = order[offset:offset + spec["batch"]]
            chunk = chunks / f"{offset:05d}.npy"
            if chunk.is_file():
                vectors = np.load(chunk, allow_pickle=False)
            else:
                batch = tokenizer.encode_batch([spec["prefix"] + texts[index] for index in indices])
                ids = np.asarray([item.ids for item in batch], dtype=np.int64)
                mask = np.asarray([item.attention_mask for item in batch], dtype=np.int64)
                fields = {
                    "input_ids": ids, "attention_mask": mask,
                    "token_type_ids": np.asarray([item.type_ids for item in batch], dtype=np.int64),
                    "position_ids": np.maximum(mask.cumsum(axis=1) - 1, 0),
                }
                fields.update(empty_kv_cache(inputs, len(indices)))
                hidden = session.run([spec["output"]], {key: fields[key] for key in required})[0]
                vectors = hidden if spec["output"] == "sentence_embedding" else hidden[
                    np.arange(len(indices)), mask.sum(axis=1) - 1,
                ]
                vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
                save_array(chunk, vectors.astype(np.float32))
            if vectors.shape != (len(indices), spec["dimensions"]) or not np.isfinite(vectors).all():
                raise ValueError(f"Invalid vectors in {chunk}")
            output[indices] = vectors
            if offset % 512 == 0:
                elapsed = perf_counter() - started
                print(f"{name}/{split}: {min(offset + len(indices), len(texts))}/{len(texts)} "
                      f"({elapsed:.1f} s)", flush=True)
        save_array(final_path, output)
    return {key: np.load(destination / f"{key}.npy", allow_pickle=False) for key in texts_by_split}, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--encode-only", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    vectors, metadata = encode(args.model, args.threads)
    if not args.encode_only:
        output = ROOT / "results/modern" / args.model
        with threadpool_limits(limits=4):
            run(output, vectors=vectors, semantic_key=args.model)
        (output / "encoder.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
