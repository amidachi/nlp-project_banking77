# When should a banking chatbot defer to a human?

NLP Course Final Project · Autumn 2026 · Karatorgayeva Dina, Alisheva Anel, Lain Adema

This project studies **selective intent routing**. Given an English banking support request, a classifier proposes one of 77 intents. A separate score determines whether to accept that prediction or defer the request to an operator. The experiment tests whether agreement with nearby training examples improves routing when a customer adds greetings and polite phrases.

[Read the project report (PDF)](output/pdf/banking77-report.pdf)

## Results

The official test has 3,080 examples. The second condition wraps the same messages in six deterministic greeting and closing patterns with inherited labels.

| Classification method | Original accuracy | With polite wrappers |
| --- | ---: | ---: |
| Majority class | 1.30% | 1.30% |
| Word + character TF-IDF | 90.45% | 88.12% |
| Frozen BGE-small | 93.05% | 84.64% |
| BGE-small cosine 1-nearest-neighbor | 92.34% | 81.59% |
| TF-IDF + BGE probability fusion | **93.57%** | 86.59% |

Within each encoder experiment, all routing variants use the same fusion predictions. The nominal error target is 5% on the independent calibration set. The two tables above and below describe the BGE-small baseline.

| Routing policy | Original coverage | Original accepted error | With wrappers coverage | With wrappers accepted error |
| --- | ---: | ---: | ---: | ---: |
| Maximum probability | 93.70% | 3.78% | 77.05% | 4.55% |
| Probability margin | 93.99% | 3.94% | 80.52% | 5.04% |
| Neighbor agreement | 93.70% | 3.47% | 81.85% | 4.20% |
| Learned gate without neighbors | 93.90% | 3.94% | 80.52% | 5.04% |
| Learned gate with neighbors | **95.06%** | 3.76% | **84.58%** | 4.72% |

On wrapped inputs, the gate with neighbors had the lowest AURC of the five policies (0.02006) and accepted 2,605 of 3,080 requests with 123 errors. The 95% Wilson interval for accepted error was 3.97% to 5.60%. Compared with the learned gate without neighbors, coverage increased by 4.06 percentage points (paired bootstrap 95% CI 2.99 to 5.16) and AURC decreased by 0.00561 (95% CI for the change −0.00697 to −0.00433). The intervals describe test-row variation with the fitted models and thresholds held fixed.

Cheng et al. report 89.63% coverage, 2.65% accepted error and AURC 0.01044 on the original BANKING77 test. Their MiniLM classifier, ten-run averaging and simultaneous binomial calibration limit direct comparison with this project's single-split result. [Cheng et al., Tables II and III](https://arxiv.org/html/2610.00262v1)

With all 10,003 official training rows, the original fusion reaches **93.60%** on the official test. The development-selected follow-up combines 10% TF-IDF with 30% each BGE-small, BGE-base, and E5-base-v2. It reaches **94.51%** (2,911 / 3,080), with 28 additional correct predictions. Published comparisons include 94.10% for [RoBERTa-base](https://aclanthology.org/2021.naacl-industry.38/) and 94.94% for [SPACE-3](https://arxiv.org/abs/2209.06664).

## Comparison with 2025-2026 encoders

Every row uses the same 8,006 fit, 999 development, 998 calibration and 3,080 paired test examples. Each encoder has independently fitted heads, fusion weights and gates. Models remain frozen.

| Encoder | Linear original | Linear wrapped | Cosine 1-NN wrapped | Fusion wrapped |
| --- | ---: | ---: | ---: | ---: |
| BGE-small | 93.05% | 84.64% | 81.59% | 86.59% |
| Qwen3-Embedding-0.6B | 88.93% | 85.36% | 82.56% | 89.35% |
| EmbeddingGemma-300M | 94.16% | 91.88% | 93.18% | 92.56% |
| Jina v5 Nano (classification) | 91.62% | 89.09% | 88.90% | 90.88% |

| Fusion encoder / gate with neighbors | Wrapped coverage | Accepted error | Wrapped AURC |
| --- | ---: | ---: | ---: |
| BGE-small | 84.58% | 4.72% | 0.02006 |
| Qwen3-Embedding-0.6B | 87.47% | 5.20% | 0.01678 |
| EmbeddingGemma-300M | 99.19% | 6.87% | 0.01098 |
| Jina v5 Nano (classification) | 97.60% | 7.98% | 0.01391 |

The 5% constraint is empirical calibration error. It does not guarantee a 5% test error rate. Coverage must be assessed alongside accepted error. The report compares all five routing policies per encoder and includes 10,000-draw paired bootstrap intervals for the neighbor ablations.

The preset uniform average of six linear semantic heads reaches **95.26%** on the full-training official test (2,934 / 3,080). It exceeds the cited SPACE-3 point score of 94.94% by 0.32 percentage points. This predefined baseline is evaluated separately from the development-selected candidate.

The development-selected six-encoder classification follow-up reaches **94.29%** on the official test after refitting on all training rows (2,904 / 3,080). Its nonzero probability weights (rounded to three decimals) are: embeddinggemma=1.000.

These extensions were added after the initial test evaluation. Fusion weights and ensemble components are selected on grouped development data. The report documents the independent routing and full-training classification protocols.

## Reproduce on CPU

Run from the repository root with Python 3.12.

```sh
python -m pip install -r requirements.txt
python prepare.py
python run_experiment.py
python routing_analysis.py
python extended_benchmark.py
python modern_benchmark.py --model qwen3
python routing_analysis.py --results results/modern/qwen3
python modern_benchmark.py --model embeddinggemma
python routing_analysis.py --results results/modern/embeddinggemma
python modern_benchmark.py --model jina-v5
python routing_analysis.py --results results/modern/jina-v5
python extended_benchmark.py --modern --threads 4
```

`prepare.py` checks the data hashes, downloads the pinned 34 MB ONNX encoder and its tokenizer, and caches normalized embeddings in `cache/`. The files in `data/` are included, so the first run needs network access for the model only. `python prepare.py --data-only` checks just the dataset. `--threads` controls ONNX CPU threads.

`run_experiment.py` uses a fixed seed (42) and groups normalized duplicate texts within the official training file. It fits the heads on 8,006 examples, selects the fusion weight and fits correctness models on 999 development examples, and selects thresholds on 998 separate calibration examples. The official test stays paired by row index. A separate full-training run supplies the benchmark comparison.

`extended_benchmark.py` adds BGE-base and E5-base-v2 (110 MB of int8 weights each). It selects a classifier on 1,997 grouped development rows and refits the heads on the full training set for an additional official-test comparison. This follow-up was planned after the first official-test evaluation.

`modern_benchmark.py` compares Qwen3-Embedding-0.6B (June 2025), EmbeddingGemma-300M (September 2025), and the classification-specific Jina v5 Nano checkpoint (February 2026). Every encoder uses the same fit/development/calibration/test rows. Each experiment compares a learned linear classifier, cosine 1-nearest-neighbor classification, lexical/semantic probability fusion, and five routing policies. Fusion weights and gates are fitted independently per encoder; test results do not select their settings. This extension was added after feedback on the initial report.

The script downloads about 614 MB, 309 MB, and 247 MB of quantized weights respectively, plus tokenizers. It checks pinned revisions and SHA-256 hashes in `modern_benchmark.py`. Encoding uses CPU ONNX Runtime with four threads and resumable batch caches. `--threads` adjusts inference threads; `--encode-only` prepares embeddings without fitting classifiers. Run the encoders sequentially on a machine with limited RAM. The output includes `encoder.json` with source, revision, hashes, task prefix, pooling, token limit and truncation counts.

Qwen receives an intent-classification instruction, EmbeddingGemma receives `task: classification | query: `, and Jina receives `Document: `. Qwen and Jina use the final non-padding token; EmbeddingGemma's export supplies its projected sentence embedding. All vectors are L2 normalized and inputs are capped at 128 tokens. Different architectures, dimensions, sizes, instructions and quantization mean this is a comparison of practical encoder configurations.

Qwen's dynamic-int8 export uses batch size 1. A controlled probe found that the same query had cosine similarity 0.914 when encoded alone versus with unrelated queries. Single-query padding and repeated identical queries gave cosine 1.0; EmbeddingGemma and Jina also matched their single-query vectors in the mixed-batch probe. [The saved inference audit](results/modern/qwen3/inference_audit.json) records the inputs and decision. Gemma and Jina use batches of 16.

`extended_benchmark.py --modern` adds the three newer encoders to the original three-encoder classification experiment. It compares the six linear heads, a Nyström RBF classifier over their concatenated representations, uniform probability mixtures of nonempty score subsets, and lexical weights from 0 to 0.5 in steps of 0.1. It selects on 1,997 grouped development rows and refits on all 10,003 training rows. Ties favor fewer nonzero score components, then lower lexical weight. This classification follow-up has its own saved results and does not alter the single-encoder routing experiments.

| Path | Contents |
| --- | --- |
| [data/](data/) | Upstream CSVs, label names and dataset license |
| [prepare.py](prepare.py) | Verified assets, transformation templates and CPU embedding generation |
| [run_experiment.py](run_experiment.py) | Models, data splits, threshold selection and evaluation |
| [routing_analysis.py](routing_analysis.py) | Routing count audit and paired test-row bootstrap |
| [extended_benchmark.py](extended_benchmark.py) | Follow-up with larger encoders and model selection |
| [modern_benchmark.py](modern_benchmark.py) | Pinned 2025-2026 encoders and the same classification/routing protocol |
| [results/modern/](results/modern/) | Per-encoder metrics, paired predictions, risk-coverage curves, manifests and bootstrap intervals |
| [results/metrics.json](results/metrics.json) | Metrics, thresholds, confidence intervals, tests and per-intent analysis |
| [results/predictions.csv](results/predictions.csv) | Paired predictions, scores and accept/defer decisions, indexed by zero-based test row |
| [results/risk_coverage.csv](results/risk_coverage.csv) | Error across coverage levels for the five routing policies |
| [results/routing_analysis.json](results/routing_analysis.json) | Accepted counts, policy overlap and confidence intervals |
| [results/extended_benchmark.json](results/extended_benchmark.json) | Follow-up model selection and official-test comparison |
| [results/extended_predictions.csv](results/extended_predictions.csv) | Follow-up predictions indexed by official test row |
| [results/modern_ensemble.json](results/modern_ensemble.json) | Development selection and full-training test scores for the six-encoder comparison |
| [results/modern_ensemble_predictions.csv](results/modern_ensemble_predictions.csv) | Development-selected candidate and baseline predictions on the official test |
| [report.py](report.py) | Report source with tables and graph built from the saved results |

To build the PDF, install `reportlab==4.4.9` and run `python report.py`. Windows Times New Roman and common Linux Liberation/DejaVu serif fonts are supported. Set `--font-dir` if using a custom font location.

## Data and model attribution

The original BANKING77 files are attributed to PolyAI and distributed under [CC BY 4.0](data/LICENSE.txt). They come from the [pinned upstream revision](https://github.com/PolyAI-LDN/task-specific-datasets/tree/9d081458ff52e53cf7e848f414e6e9344e4e6696/banking_data), introduced by [Casanueva et al. (2020)](https://aclanthology.org/2020.nlp4convai-1.5/). The synthetic wording changes are generated by this project. The base intent labels come from the upstream dataset.

`.gitattributes` preserves the upstream CSV and category-file bytes when cloning on Windows, so checkout line-ending conversion does not invalidate the pinned data hashes.

Embeddings use [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) and its [pinned Xenova ONNX port](https://huggingface.co/Xenova/bge-small-en-v1.5/tree/ea104dacec62c0de699686887e3f920caeb4f3e3), with an MIT model license. This benchmark concerns English, single-intent, in-scope queries. It measures routing labels and assumes deferred requests reach a human. Production outcomes and unfamiliar intents are outside its evaluation.

The newer models are [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (Apache 2.0; [ONNX export](https://huggingface.co/onnx-community/Qwen3-Embedding-0.6B-ONNX)), [EmbeddingGemma-300M](https://huggingface.co/google/embeddinggemma-300m) ([Gemma terms](https://ai.google.dev/gemma/terms); [ONNX export](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX)), and [Jina v5 Nano classification](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-classification) (CC BY-NC 4.0). Model weights are downloaded locally and are not included in this repository.
