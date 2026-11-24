"""Build the course report from the saved experiment results."""

import argparse
import csv
import json
import os
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, PolyLine, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Table, TableStyle

ROOT = Path(__file__).resolve().parent
REPO = "https://github.com/amidachi/nlp-project_banking77"
DATA_URL = "https://github.com/PolyAI-LDN/task-specific-datasets/tree/9d081458ff52e53cf7e848f414e6e9344e4e6696/banking_data"
TITLE = "When Should a Banking Chatbot Defer to a Human?"
INK = colors.HexColor("#18212A")
MUTED = colors.HexColor("#505B65")
RULE = colors.HexColor("#CDD5DB")


def register_fonts(custom):
    windows = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    candidates = [
        (windows, ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf")),
        (Path("/usr/share/fonts/truetype/liberation2"), tuple(
            f"LiberationSerif-{suffix}.ttf" for suffix in ("Regular", "Bold", "Italic", "BoldItalic")
        )),
        (Path("/usr/share/fonts/truetype/dejavu"), tuple(
            f"DejaVuSerif{suffix}.ttf" for suffix in ("", "-Bold", "-Italic", "-BoldItalic")
        )),
    ]
    for directory, names in candidates:
        directory = custom or directory
        if all((directory / name).is_file() for name in names):
            for suffix, name in zip(("", "Bold", "Italic", "BoldItalic"), names):
                pdfmetrics.registerFont(TTFont("Text" + suffix, str(directory / name)))
            pdfmetrics.registerFontFamily("Text", "Text", "TextBold", "TextItalic", "TextBoldItalic")
            return
    raise FileNotFoundError("Use --font-dir with Times New Roman, Liberation Serif or DejaVu Serif.")


def styles():
    body = ParagraphStyle("Body", fontName="Text", fontSize=10.6, leading=13.65,
                          textColor=INK, spaceAfter=6)
    return {
        "body": body,
        "title": ParagraphStyle("Title", body, fontName="TextBold", fontSize=25,
                                leading=28, alignment=TA_CENTER, spaceAfter=10),
        "author": ParagraphStyle("Author", body, fontSize=12, leading=16,
                                 alignment=TA_CENTER, spaceAfter=2),
        "subtitle": ParagraphStyle("Subtitle", body, fontSize=10.3, leading=14,
                                   alignment=TA_CENTER, textColor=MUTED, spaceAfter=13),
        "heading": ParagraphStyle("Heading", body, fontName="TextBold", fontSize=14,
                                  leading=17, spaceBefore=7, spaceAfter=5, keepWithNext=True),
        "subhead": ParagraphStyle("Subheading", body, fontName="TextBold", fontSize=11.6,
                                  leading=14, spaceBefore=2, spaceAfter=4, keepWithNext=True),
        "caption": ParagraphStyle("Caption", body, fontSize=9.1, leading=11.2,
                                  textColor=MUTED, spaceBefore=4, spaceAfter=6),
        "cell": ParagraphStyle("Cell", body, fontSize=9.5, leading=11.4, spaceAfter=0),
        "reference": ParagraphStyle("Reference", body, fontSize=9.7, leading=12.4,
                                    leftIndent=15, firstLineIndent=-15, spaceAfter=5),
        "formula": ParagraphStyle("Formula", body, fontSize=11.1, leading=15,
                                  alignment=TA_CENTER, spaceBefore=2, spaceAfter=8),
    }


def pipeline_figure():
    drawing = Drawing(480, 112)

    def box(x, y, width, height, heading, details):
        drawing.add(Rect(x, y, width, height, fillColor=colors.white, strokeColor=MUTED,
                         strokeWidth=0.8, rx=4, ry=4))
        for i, line in enumerate([heading, *details]):
            drawing.add(String(x + width / 2, y + height - 14 - 12 * i, line,
                               textAnchor="middle", fontName="TextBold" if i == 0 else "Text",
                               fontSize=9.5, fillColor=INK))

    def arrow(x1, y1, x2, y2):
        drawing.add(Line(x1, y1, x2, y2, strokeColor=MUTED))
        end = [x2 - 4, y2 + 2, x2, y2, x2 - 4, y2 - 2] if x1 != x2 else [
            x2 - 2, y2 - 4, x2, y2, x2 + 2, y2 - 4,
        ]
        drawing.add(PolyLine(end, strokeColor=MUTED))

    box(0, 46, 99, 59, "Classifier", ["Banking message", "TF-IDF + encoder", "Fused intent"])
    box(119, 46, 112, 59, "Three signals", ["Top probability", "Top-two gap", "Neighbor agreement"])
    box(251, 46, 99, 59, "Logistic gate", ["Standardize features", "Score correctness"])
    box(370, 46, 109, 59, "Threshold τ", ["Accept fused intent", "or defer to an operator"])
    box(119, 1, 112, 31, "Fit index", ["8,006 fit vectors"])
    for end in (119, 251, 370):
        arrow(end - 20, 75, end, 75)
    arrow(175, 32, 175, 46)
    return drawing


def risk_figure(path):
    policies = {"max_prob": "Maximum probability", "score_only_gate": "Gate without neighbors",
                "learned_gate": "Gate with neighbors"}
    dashes = {"max_prob": [1, 2.3], "score_only_gate": [5, 2.5], "learned_gate": None}
    series = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            coverage = float(row["coverage"])
            if row["policy"] in policies and coverage >= 0.50:
                series.setdefault((row["variant"], row["policy"]), []).append(
                    (coverage, float(row["risk"])),
                )
    drawing = Drawing(480, 184)
    bottom, height, width = 54, 100, 185
    ymax = 0.15
    drawing.add(String(12, 171, "Selective risk (%)", fontName="Text", fontSize=9, fillColor=INK))
    for left, variant, title in ((49, "original", "Original test"), (279, "wrapped", "With polite wrappers")):
        drawing.add(String(left + width / 2, 171, title, fontName="TextBold",
                           fontSize=11, textAnchor="middle", fillColor=INK))
        for tick in (0, 5, 10, 15):
            y = bottom + tick * height / 15
            drawing.add(Line(left, y, left + width, y, strokeColor=RULE, strokeWidth=0.5))
            drawing.add(String(left - 6, y - 3, f"{tick}%", fontName="Text",
                               fontSize=8.5, textAnchor="end", fillColor=MUTED))
        for tick in (50, 60, 70, 80, 90, 100):
            x = left + width * (tick - 50) / 50
            drawing.add(String(x, bottom - 13, f"{tick}%", fontName="Text",
                               fontSize=8.5, textAnchor="middle", fillColor=MUTED))
        drawing.add(Line(left, bottom, left, bottom + height, strokeColor=MUTED, strokeWidth=0.6))
        drawing.add(String(left + width / 2, 23, "Coverage (%)", fontName="Text", fontSize=9,
                           textAnchor="middle", fillColor=INK))
        for policy in policies:
            values = series[(variant, policy)]
            points = values[::6] + [values[-1]]
            drawing.add(PolyLine(
                [coordinate for cx, ry in points for coordinate in
                 (left + width * (cx - 0.5) * 2, bottom + height * min(ry, ymax) / ymax)],
                strokeColor=colors.black, strokeWidth=1.35, strokeDashArray=dashes[policy],
            ))
    for x, (key, label) in zip((42, 181, 325), policies.items()):
        drawing.add(Line(x, 7, x + 18, 7, strokeColor=colors.black, strokeWidth=1.35,
                         strokeDashArray=dashes[key]))
        drawing.add(String(x + 23, 4, label, fontName="Text", fontSize=9.2, fillColor=INK))
    return drawing


def build(results, output):
    m = json.loads((results / "metrics.json").read_text(encoding="utf-8"))
    style = styles()
    story = []

    def p(value, name="body"):
        paragraph = Paragraph(value, style[name])
        if name == "caption" and story and isinstance(story[-1], (Table, Drawing)):
            block = [story.pop(), paragraph]
        elif name == "formula" and story and isinstance(story[-1], Paragraph):
            block = [story.pop(), paragraph]
        else:
            story.append(paragraph)
            return
        while story and isinstance(story[-1], Paragraph) and story[-1].style.name in {"Heading", "Subheading"}:
            block.insert(0, story.pop())
        story.append(KeepTogether(block))

    def h(value):
        p(value, "heading")

    def sub(value):
        p(value, "subhead")

    def table(rows, widths):
        cells = [[Paragraph(str(cell), style["cell"]) for cell in row] for row in rows]
        grid = Table(cells, colWidths=widths, hAlign="LEFT", repeatRows=1)
        grid.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EDF0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F6F8F9")]),
            ("LINEBELOW", (0, 0), (-1, 0), 0.6, RULE),
            ("LINEBELOW", (0, -1), (-1, -1), 0.6, RULE),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(grid)

    def link(url, label):
        return f'<link href="{url}" color="#146E72">{label}</link>'

    def pct(value):
        return f"{100 * value:.2f}%"

    original = m["test"]["original"]
    wrapped = m["test"]["wrapped"]
    original_gate = original["routing"]["learned_gate"]
    wrapped_gate = wrapped["routing"]["learned_gate"]
    without_neighbors = wrapped["routing"]["score_only_gate"]
    extended = json.loads((results / "extended_benchmark.json").read_text(encoding="utf-8"))
    routing = json.loads((results / "routing_analysis.json").read_text(encoding="utf-8"))
    selected = extended["selected_model"]
    selected_result = extended["official_test"][selected]
    ensemble = json.loads((results / "modern_ensemble.json").read_text(encoding="utf-8"))
    ensemble_result = ensemble["official_test"][ensemble["selected_model"]]
    uniform_result = ensemble["official_test"]["dense-average"]
    modern = []
    for key in ("qwen3", "embeddinggemma", "jina-v5"):
        directory = results / "modern" / key
        modern.append({
            "key": key,
            "encoder": json.loads((directory / "encoder.json").read_text(encoding="utf-8")),
            "metrics": json.loads((directory / "metrics.json").read_text(encoding="utf-8")),
            "analysis": json.loads((directory / "routing_analysis.json").read_text(encoding="utf-8")),
        })
    comparisons = [("BGE-small", "bge", m)] + [
        (item["encoder"]["name"], item["key"], item["metrics"]) for item in modern
    ]
    best_classifier = max([
        {"name": name, "method": label,
         "original": metrics["test"]["original"]["classification"][key]["accuracy"],
         "wrapped": metrics["test"]["wrapped"]["classification"][key]["accuracy"]}
        for name, semantic_key, metrics in comparisons
        for key, label in ((semantic_key, "linear head"), ("cosine_1nn", "cosine 1-NN"),
                           ("fusion", "probability fusion"))
    ], key=lambda item: item["wrapped"])
    policy_labels = {"max_prob": "maximum probability", "prob_margin": "probability margin",
                     "neighbors": "neighbor agreement", "score_only_gate": "gate without neighbors",
                     "learned_gate": "gate with neighbors"}
    best_routing = min([
        {"name": name, "policy": policy_labels[key], **values}
        for name, _, metrics in comparisons
        for key, values in metrics["test"]["wrapped"]["routing"].items()
    ], key=lambda item: item["aurc"])

    p("When Should a Banking Chatbot<br/>Defer to a Human?", "title")
    p("Karatorgayeva Dina", "author")
    p("NLP Course Final Project · Autumn 2026", "subtitle")
    sub("Abstract")
    p(f"We study selective intent routing when users add polite conversational text to banking "
      f"support requests. We compare TF-IDF and four frozen encoders: BGE-small, Qwen3-Embedding, "
      f"EmbeddingGemma and Jina v5 Nano, with linear, nearest-neighbor and probability-fusion classifiers. "
      f"On 3,080 paired BANKING77 requests, the highest observed wrapped accuracy is "
      f"{pct(best_classifier['wrapped'])} from {best_classifier['name']} with {best_classifier['method']} "
      f"({pct(best_classifier['original'])} on original inputs). Five acceptance scores per encoder "
      f"are calibrated independently with a nominal 5% error target. The lowest observed wrapped AURC "
      f"among these 20 combinations is {best_routing['aurc']:.5f} from {best_routing['name']} with "
      f"the {best_routing['policy']}; its frozen threshold yields {pct(best_routing['coverage'])} coverage "
      f"and {pct(best_routing['risk'])} accepted error. Paired gate ablations test the contribution of "
      f"neighbor agreement. A preset uniform average of six encoder heads reaches "
      f"{pct(uniform_result['accuracy'])} on the full-training official-test protocol; development "
      f"selection instead chooses a single head reaching {pct(ensemble_result['accuracy'])}. "
      f"Code and results: {link(REPO, REPO)}.")
    h("1 Introduction")
    p("Support systems use intent predictions to route questions to a handler. A card that has already "
      "been ordered and a question about expected delivery time can require different workflows. Every "
      "automatic routing error may add a correction step, and every deferred request uses operator time. "
      "We measure both costs through accepted error and coverage.")
    p("We construct a paired BANKING77 test with 3,080 messages supplemented by greetings and closings. "
      "The experiment asks whether evidence from nearby examples helps a fixed classifier decide "
      "when to defer. We extend the BGE experiment with Qwen3-Embedding, EmbeddingGemma and Jina v5 "
      "to test how the result depends on the encoder. The extension asks whether neighbor agreement "
      "lowers wrapped-test AURC relative to a gate using classifier scores alone for each encoder.")
    sub("1.1 Team")
    p("Karatorgayeva Dina approved the research question and is responsible for the course submission. "
      "OpenAI Codex generated and revised the experiment code, ran the analyses, and drafted the report.")
    h("2 Related Work")
    sub("2.1 Intent classification")
    p("Casanueva et al. [1] introduced BANKING77. Their frozen USE+ConveRT representations reach 85.19% "
      "accuracy with ten examples per intent and 93.36% with full training. Fine-tuned BERT-large reaches "
      "93.66% with full training. Qi et al. [2] compare commercial systems and fine-tuned encoders. "
      "RoBERTa-base reaches 94.10% on the full test and 83.8% on a TF-IDF-selected difficult subset. "
      "Song et al. [3] report 94.04% mean accuracy over three SAM continuation seeds and 94.12% for "
      "the best seed selected after evaluation. SPACE-3 [4] uses tree-induced contrastive "
      "pretraining on dialogue corpora and reaches 94.94% with full training.")
    sub("2.2 Selective classification")
    p("Geifman and El-Yaniv [5] choose a rejection threshold to trade off coverage and errors among "
      "accepted predictions. Liang et al. [6] study how the top-two logit gap and geometric margins "
      "rank errors as inputs shift. Cheng et al. [7] fit a correctness model using the semantic margin "
      "and a lexical model's signed support for the predicted class. They calibrate thresholds using "
      "simultaneous binomial bounds and evaluate an official BANKING77 test alongside two other corpora.")

    sub("2.3 Neighbor evidence and conversational wording")
    p("Deep k-Nearest Neighbors [8] uses label agreement among neighbors in several hidden layers to "
      "measure prediction support. Zhang et al. [9] use a pairwise NLI encoder to choose the best "
      "matching labeled utterance and detect out-of-scope requests in few-shot intent classification. "
      "Zhou et al. [10] use in-domain neighbors during contrastive training and then apply a density-based "
      "detector for out-of-domain intents. Here the encoder stays frozen, the neighbor labels supply "
      "one input to the confidence model, and all evaluated requests retain an in-scope label.")
    p("Bogachov et al. [11] compare Llama-3.2-1B-Instruct pooling methods on original and synthetically "
      "noised BANKING77 and CLINC150 utterances over ten seeds. On noisy BANKING77, attention pooling "
      "raises mean F1 from 87.67 to 90.31 versus the default head. Our experiment varies the confidence "
      "signals and measures how an original-text threshold transfers to texts with polite additions.")
    sub("2.4 More recent embedding models")
    p("Qwen3-Embedding [12], released in June 2025, adapts Qwen3 decoder models through contrastive "
      "training and supports task instructions. The 0.6B model returns 1,024-dimensional embeddings "
      "from the final non-padding token. EmbeddingGemma [13], released in September 2025, uses a "
      "308M-parameter bidirectional model with task-specific prefixes and a 768-dimensional output. "
      "Jina v5 Nano [14], released in February 2026, distills Qwen3-Embedding-4B into a EuroBERT-based "
      "239M-parameter model with task adapters. We evaluate the classification adapter merged into "
      "the backbone, using a 768-dimensional final-token vector. All three encoders use the same "
      "classification and routing heads as BGE.")

    h("3 Model Description")
    sub("3.1 Predicting an intent")
    p("The lexical head uses L2-normalized TF-IDF over word unigrams and bigrams (minimum document "
      "frequency 2) and within-word character 3-5-grams (minimum frequency 3). The concatenated features "
      "feed an L2-regularized multinomial logistic regression with C = 20.")
    p("The baseline semantic head encodes raw text with the frozen, int8 ONNX port of BGE-small-en-v1.5 [15], "
      "using dynamic padding and up to 128 tokens. The 384-dimensional final [CLS] vector is normalized "
      "to unit length and passed to multinomial logistic regression with C = 100. We combine the "
      "resulting class distributions and predict the largest fused probability.")
    p("<i>p</i>(<i>y</i> | <i>x</i>) = α <i>p</i><sub>lex</sub>(<i>y</i> | <i>x</i>) "
      "+ (1 − α) <i>p</i><sub>sem</sub>(<i>y</i> | <i>x</i>).", "formula")
    sub("3.2 Deciding whether to accept")
    p("Let <i>p</i><sub>(1)</sub> and <i>p</i><sub>(2)</sub> be the two highest fused probabilities. "
      "<i>N</i><sub>8</sub>(<i>x</i>) contains the eight fit rows with the highest cosine similarity "
      "in the normalized space of the current encoder. The BGE baseline uses its 384-dimensional [CLS] "
      "vectors. Every index uses only the 8,006 fit rows.")
    p("<i>z</i>(<i>x</i>) = "
      "[<i>p</i><sub>(1)</sub>, <i>p</i><sub>(1)</sub> − <i>p</i><sub>(2)</sub>, "
      "(1/8) Σ<sub>j in N8(x)</sub> 1[<i>y</i><sub>j</sub> = <i>f</i>(<i>x</i>)]].",
      "formula")
    p("Logistic regression (C = 1, L2 penalty) learns whether the fused label is correct on development "
      "data. It scores <i>z</i><sub>std</sub>(<i>x</i>), obtained by centering and scaling each signal "
      "using its development mean and standard deviation.")
    p("<i>s</i>(<i>x</i>) = 1 / [1 + exp(−(<i>b</i> + <i>w</i><sup>T</sup><i>z</i><sub>std</sub>(<i>x</i>)))].",
      "formula")
    story.append(pipeline_figure())
    p("Figure 1. The learned score controls acceptance of the fused intent. The threshold is fitted "
      "on the independent calibration split.", "caption")
    sub("3.3 Instructions for the newer encoders")
    p("Qwen3 receives “Instruct: Classify the intent of a banking customer support request.” followed "
      "by a newline and “Query: ” before the message. EmbeddingGemma receives “task: classification | "
      "query: ” and Jina receives “Document: ”. We use their pinned int8 ONNX exports, normalize each output vector, and cap the "
      "complete input at 128 tokens. Heads use C = 100 and each encoder supplies its own neighbor index. "
      "The manifests record model revisions, SHA-256 hashes and observed truncation counts. "
      "No original or wrapped test request exceeds the 128-token limit of any of the three newer "
      "encoders. Qwen's dynamic-int8 export is encoded one request at a time: a controlled probe found "
      "batch-composition dependence (cosine 0.914 for the same query alone versus in a mixed batch). "
      "EmbeddingGemma and Jina use batches of 16 and matched their single-query vectors in this probe. "
      "The inference audit is saved with the results. Model sizes differ and encoders remain frozen.")
    sub("3.4 Nearest-neighbor and nonlinear comparators")
    p("Cosine 1-NN assigns the label of the most similar fit vector without learning a linear head. "
      "The nonlinear comparator concatenates M normalized encoder vectors, scales each block by "
      "1/√M, and approximates the following kernel with 512 Nyström landmarks (seed 42).")
    p("<i>K</i>(<i>u</i>, <i>v</i>) = exp(−2 ||<i>u</i> − <i>v</i>||<sup>2</sup>).", "formula")
    p("A logistic classifier with C = 30 operates on the resulting features. The ensemble experiment "
      "combines class probabilities from the linear heads, this nonlinear head and TF-IDF. "
      "Its weights are chosen on grouped development data (§6.6).")

    h("4 Dataset")
    sub("4.1 Task definition")
    p("The input <i>x</i> is one English support message with a reference label <i>y</i> among 77 intents. "
      "The classifier <i>f</i>(<i>x</i>) predicts one label. The router accepts that label when "
      "<i>s</i>(<i>x</i>) ≥ τ and otherwise defers. The objective is to maximize coverage subject to "
      "observed calibration risk at most 5%.")
    sub("4.2 Data and test conditions")
    p(f"PolyAI released 13,083 English banking queries with a single labeled intent per query [1]. "
      f"We use the {link(DATA_URL, 'original CSVs and category list')} under "
      f"{link('https://creativecommons.org/licenses/by/4.0/', 'CC BY 4.0')}. The repository includes "
      f"these files. The preparation script verifies their SHA-256 hashes.")
    rows = [["<b>Split</b>", "<b>Rows</b>", "<b>Intents</b>", "<b>Rows per intent</b>",
             "<b>Median words</b>", "<b>90th pct. words</b>"]]
    for key, label in (("train", "Official train"), ("fit", "Fit"), ("development", "Development"),
                       ("calibration", "Calibration"), ("original", "Official test"),
                       ("wrapped", "Test with wrappers")):
        s = m["data"][key]
        rows.append([label, f"{s['rows']:,}", s["classes"],
                     f"{s['min_per_class']}-{s['max_per_class']}", int(s["median_words"]), int(s["p90_words"])])
    table(rows, [120, 50, 48, 91, 81, 90])
    p("Table 1. Dataset statistics. Fit, development and calibration partition the official train split. "
      "The two test conditions share the same labels and test row IDs.", "caption")
    p("We create one alternative text per test row by adding a greeting and closing from six fixed "
      "templates, using the zero-based row number modulo six. For example, the first prefix is "
      "“Hi there!” and its closing is “Thank you for your help.” The original text is preserved verbatim. "
      "The inherited label assumes the additions introduce no banking intent.")
    p("Case folding and replacement of non-word runs with spaces identify duplicate groups. We find "
      "31 repeated train rows beyond the first occurrence and 25 test rows whose normalized text "
      "appears in the official train set. The main scores retain the official test and we also "
      "measure accuracy after removing those 25 rows. Ying and Thomas [16] identify potential label noise.")

    h("5 Experiments")
    sub("5.1 Metrics")
    p("Let <i>a</i><sub>i</sub>(τ) = 1[<i>s</i>(<i>x</i><sub>i</sub>) ≥ τ] indicate acceptance and "
      "<i>e</i><sub>i</sub> = 1[<i>f</i>(<i>x</i><sub>i</sub>) ≠ <i>y</i><sub>i</sub>] indicate an error.")
    p("<i>C</i>(τ) = Σ<sub>i</sub> <i>a</i><sub>i</sub>(τ) / <i>N</i>"
      "&nbsp;&nbsp;&nbsp;&nbsp; "
      "<i>R</i>(τ) = Σ<sub>i</sub> <i>a</i><sub>i</sub>(τ)<i>e</i><sub>i</sub>"
      " / Σ<sub>i</sub> <i>a</i><sub>i</sub>(τ).", "formula")
    p("Risk is undefined when no rows are accepted. AURC averages cumulative risk over all nonempty "
      "prefixes sorted by score, with expected errors under random order within ties. Lower is better. "
      "We report accuracy, macro-F1 and exact paired McNemar tests for classification, and nominal "
      "95% Wilson intervals for accepted error.")

    sub("5.2 Experiment Setup")
    p("A stratified group split with seed 42 assigns a held-out fold from the original training "
      "file. A second group split divides it into development and calibration. The heads fit on "
      "8,006 examples. The initial grid {0, 0.1, …, 0.5} for α treats TF-IDF as a supplement "
      "to the semantic head. Development accuracy selects α = 0.2 (94.69%). A sensitivity check "
      "extends the same development grid to 1.0. The best accuracy at α &gt; 0.5 is 94.19% at 0.6, "
      "so the selected weight also wins on the full grid.")
    p("For calibration scores sorted in descending order, let <i>e</i><sub>(i)</sub> be the error "
      "at rank <i>i</i> and <i>K</i> the nonempty prefix sizes that include all rows tied at the boundary.")
    p("<i>k</i>* = max { <i>k</i> in <i>K</i> | "
      "(1/<i>k</i>) Σ<sub>i≤k</sub> <i>e</i><sub>(i)</sub> ≤ 0.05 },"
      "&nbsp;&nbsp; τ* = <i>s</i><sub>(k*)</sub>.", "formula")
    p("If the set is empty, the rule defers every request. Thresholds stay fixed across test variants. "
      "For comparison with prior work, we also refit classification heads on all 10,003 official train rows.")
    p("The three newer encoders use these same fit, development and calibration rows. Their fusion "
      "weights and acceptance thresholds are fitted independently on the corresponding development "
      "and calibration representations. This extension was added after feedback on the original study.")
    sub("5.3 Baselines")
    p("Classification baselines are the most frequent intent, TF-IDF, a linear head over frozen "
      "embeddings, and cosine 1-nearest-neighbor classification using the same fit index. The fused "
      "classifier combines lexical and semantic probabilities. Routing baselines score its predictions with maximum "
      "probability, the top-two probability gap, neighbor agreement alone, or a learned score using "
      "only the two classifier-derived signals. The last comparator isolates the contribution of neighbors.")
    h("6 Results")
    sub("6.1 Intent classification")
    rows = [["<b>Classifier</b>", "<b>Original accuracy</b>", "<b>Original macro-F1</b>",
             "<b>Wrapped accuracy</b>", "<b>Wrapped macro-F1</b>"]]
    names = [("majority", "Most frequent intent"), ("tfidf", "TF-IDF"),
             ("bge", "BGE-small + linear head"), ("cosine_1nn", "BGE-small cosine 1-NN"),
             ("fusion", "Probability fusion")]
    for key, label in names:
        rows.append([label] + [pct(group["classification"][key][metric])
                                for group in (original, wrapped) for metric in ("accuracy", "macro_f1")])
    table(rows, [128, 88, 88, 88, 88])
    p("Table 2. Test scores for models fitted on the 8,006-row split.", "caption")
    p(f"On the original test, fusion corrects 20 BGE errors and introduces four "
      f"({original['fusion_vs_bge_mcnemar']['delta'] * 100:+.2f} points, exact McNemar <i>p</i> = 0.00154). "
      f"Removing the 25 train-matching test rows changes fusion accuracy to "
      f"{pct(original['fusion_no_overlap_accuracy'])}. The wrapped test loses 215 net correct "
      f"fusion predictions (254 correct only before wrapping and 39 only after, <i>p</i> &lt; 0.001). "
      f"TF-IDF has the highest wrapped accuracy among these BGE baseline classifiers.")
    comparison_rows = [
        ["<b>Source</b>", "<b>Model</b>", "<b>Accuracy</b>"],
        ["This study", "TF-IDF / BGE-small / fusion", "91.14% / 93.08% / 93.60%"],
        ["This study", "Follow-up ensemble (§6.4)", pct(selected_result["accuracy"])],
        ["This study", "Development-selected candidate (§6.6)", pct(ensemble_result["accuracy"])],
        ["This study", "Preset uniform six-encoder average (§6.6)", pct(uniform_result["accuracy"])],
        ["This study", "RBF over six encoders (§6.6)", pct(ensemble["official_test"]["rbf"]["accuracy"])],
    ]
    for item in modern:
        full = item["metrics"]["full_train_original"]
        comparison_rows.append([
            "This study (§6.5)", item["encoder"]["name"] + " (linear / 1-NN / fusion)",
            " / ".join(pct(full[key]["accuracy"]) for key in (item["key"], "cosine_1nn", "fusion")),
        ])
    comparison_rows.extend([
        ["Casanueva et al. [1]", "USE+ConveRT / BERT-large", "93.36% / 93.66%"],
        ["Qi et al. [2]", "RoBERTa-base", "94.10%"],
        ["Song et al. [3]", "SAM continuation, mean of 3 seeds", "94.04%"],
        ["He et al. [4]", "SPACE-3", "94.94%"],
    ])
    table(comparison_rows, [124, 215, 141])
    p("Table 3. Full-training accuracy on the official BANKING77 test. Published rows are taken "
      "from the cited studies. Modern encoder rows show linear-head / cosine 1-NN / fusion accuracy. "
      "Training regimes and pretraining corpora differ across studies.", "caption")

    sub("6.2 Selective routing with BGE-small")
    rows = [["<b>Routing policy</b>", "<b>Orig. cov.</b>", "<b>Orig. risk</b>",
             "<b>Wrap. cov.</b>", "<b>Wrap. risk</b>", "<b>Wrap. AURC</b>"]]
    for key, label in (("max_prob", "Maximum probability"), ("prob_margin", "Probability margin"),
                       ("neighbors", "Neighbor agreement"), ("score_only_gate", "Gate without neighbors"),
                       ("learned_gate", "<b>Gate with neighbors</b>")):
        o, w = original["routing"][key], wrapped["routing"][key]
        rows.append([label, pct(o["coverage"]), pct(o["risk"]), pct(w["coverage"]),
                     pct(w["risk"]), f"{w['aurc']:.4f}"])
    table(rows, [147, 63, 62, 63, 64, 81])
    p("Table 4. BGE-small fusion: coverage and risk at thresholds chosen on original calibration data with a nominal "
      "5% target. AURC is computed over the entire ranking.", "caption")
    agreement = routing["test"]["wrapped"]["margin_vs_gate_without_neighbors"]
    p("Probability margin and the gate without neighbors each accept 2,480 wrapped requests and make "
      f"125 errors. Each accepts {agreement['margin_only_accepted']} additional request unique to that policy. "
      "Their score rankings produce different AURCs.")
    p(f"On the original BANKING77 test, Cheng et al. [7] report 89.63% coverage, 2.65% accepted error "
      f"and AURC 0.01044 (Tables II-III). Our gate obtains {pct(original_gate['coverage'])}, "
      f"{pct(original_gate['risk'])} and {original_gate['aurc']:.5f}, respectively. Their results "
      "average ten runs of a MiniLM classifier with simultaneous binomial calibration. Ours use "
      "one fit split, a fused classifier and an empirical calibration constraint. These protocol "
      "differences limit direct comparison.")

    story.append(risk_figure(results / "risk_coverage.csv"))
    p("Figure 2. BGE-small selective risk against coverage. The curves rank the "
      "same fused predictions by different scores. Table 4 reports the frozen operating points.", "caption")
    p(f"The combined gate accepts {wrapped_gate['accepted']:,} wrapped requests with "
      f"{wrapped_gate['errors']} errors. Its accepted error has a Wilson 95% interval of "
      f"{pct(wrapped_gate['risk_wilson_95'][0])} to {pct(wrapped_gate['risk_wilson_95'][1])}, which crosses 5%.")
    sub("Paired comparison of the two learned gates")
    rows = [["<b>Test</b>", "<b>Δ coverage in pp (95% CI)</b>", "<b>Δ AURC × 1,000 (95% CI)</b>"]]
    for key, name in (("original", "Original"), ("wrapped", "Wrapped")):
        values = []
        for metric, scale in (("coverage_delta", 100), ("aurc_delta", 1000)):
            stat = routing["test"][key][metric]
            lo, hi = stat["ci95"]
            values.append(f"{scale * stat['estimate']:+.2f} [{scale * lo:+.2f}, {scale * hi:+.2f}]")
        rows.append([name, *values])
    table(rows, [94, 193, 193])
    p("Table 5. BGE gate with neighbors minus gate without neighbors. Percentile intervals use 10,000 "
      "paired resamples of test rows, sharing row indices across both gates within each condition. "
      "Thresholds and fitted models stay fixed. AURC is recomputed per draw.", "caption")
    p("Both coverage and AURC intervals exclude zero for this split. The bootstrap measures test-row "
      "uncertainty conditional on the fitted gates and thresholds. Variation across training splits "
      "or calibration sets would require repeated fitting.")
    p("At 80% coverage on the wrapped test, error is 3.25% with neighbors and 4.95% without them. "
      "The neighbor-only score obtains lower error at its own operating point (4.20%), with 84 fewer "
      "accepted requests than the combined gate. Its coarse scores yield a higher overall AURC.")
    sub("6.3 Error analysis and limitations")
    p("The BGE baseline's largest wrapped confusion is unrecognized card payment classified as unrecognized direct debit "
      "(14 cases). One original request, “My card has not arrived yet,” is correctly classified as "
      "card arrival. After wrapping, the model predicts a delivery estimate and the gate defers it. "
      "Coverage for suspected card compromise falls from 36/40 to 12/40. Aggregate coverage can mask "
      "unequal operator load across intents.")
    p("This study uses one grouped split and six synthetic wrappers. Observed calibration error and test "
      "intervals describe this benchmark. Live traffic, novel intents, costs of particular misroutes, "
      "and the quality of human resolution require separate measurements.")
    sub("6.4 Follow-up with larger frozen encoders")
    p("The follow-up adds int8 BGE-base [15] with [CLS] pooling and E5-base-v2 [17] with a query prefix "
      "and masked mean pooling. It compares the linear heads, probability mixtures and a classifier "
      "with Nyström RBF features (512 landmarks, γ = 2, seed 42; logistic C = 30). We select by "
      "accuracy on 1,997 grouped development rows and refit on "
      "the complete training split. This classification experiment follows the initial test evaluation.")
    change = extended["paired_vs_original"]
    p(f"The chosen mixture gives TF-IDF 10% and each of the three encoders 30%. "
      f"Its {pct(selected_result['accuracy'])} test accuracy (Table 3) improves on the original fusion "
      f"by {change['delta'] * 100:.2f} points "
      f"(exact McNemar <i>p</i> = {change['p_value']:.3g}).")
    sub("6.5 Comparison with 2025-2026 encoders")
    p("Qwen3-Embedding-0.6B, EmbeddingGemma-300M and Jina v5 Nano use the same 8,006 fit rows, 999 development "
      "rows, 998 calibration rows and 3,080 paired test requests as the BGE routing experiment. "
      "For each encoder we fit the linear classifier, select its fusion weight and fit both gates "
      "independently using these same partitions.")
    rows = [["<b>Frozen encoder / classifier</b>", "<b>Original accuracy</b>", "<b>Wrapped accuracy</b>"]]
    for name, semantic_key, metrics in comparisons:
        for key, classifier in ((semantic_key, "linear head"), ("cosine_1nn", "cosine 1-NN")):
            rows.append([
                f"{name} / {classifier}",
                pct(metrics["test"]["original"]["classification"][key]["accuracy"]),
                pct(metrics["test"]["wrapped"]["classification"][key]["accuracy"]),
            ])
    table(rows, [268, 106, 106])
    p("Table 6. Frozen representations evaluated with a learned linear head and a cosine nearest-neighbor "
      "classifier. All rows use the same fit partition.", "caption")
    gemma = next(item["metrics"] for item in modern if item["key"] == "embeddinggemma")
    p(f"On wrapped requests, EmbeddingGemma's cosine 1-NN reaches "
      f"{pct(gemma['test']['wrapped']['classification']['cosine_1nn']['accuracy'])}, compared with "
      f"{pct(gemma['test']['wrapped']['classification']['fusion']['accuracy'])} for its probability fusion. "
      "The fused classifier is therefore not uniformly strongest as the representation changes.")
    rows = [["<b>Fusion encoder</b>", "<b>α</b>", "<b>Orig. acc.</b>", "<b>Wrap. acc.</b>",
             "<b>Wrap. cov.</b>", "<b>Wrap. risk</b>"]]
    for name, _, metrics in comparisons:
        original_metrics = metrics["test"]["original"]
        wrapped_metrics = metrics["test"]["wrapped"]
        gate_metrics = wrapped_metrics["routing"]["learned_gate"]
        rows.append([
            name, f"{metrics['fusion_weight']:.1f}",
            pct(original_metrics["classification"]["fusion"]["accuracy"]),
            pct(wrapped_metrics["classification"]["fusion"]["accuracy"]),
            pct(gate_metrics["coverage"]), pct(gate_metrics["risk"]),
        ])
    table(rows, [150, 28, 76, 76, 76, 74])
    p("Table 7. Fusion accuracy and the gate with neighbors at each encoder's independently calibrated "
      "threshold. α is the lexical weight. The nominal calibration error target is 5%.", "caption")
    qwen = next(item["metrics"] for item in modern if item["key"] == "qwen3")
    p(f"Qwen's α = 0.5 lies on the shared search boundary. The full development grid favors α = 0.6 "
      f"({pct(qwen['development_fusion_accuracy']['0.6'])}) over 0.5 "
      f"({pct(qwen['development_fusion_accuracy']['0.5'])}). Tables 6-8 retain the common 0-0.5 "
      "fusion protocol; the restricted search is a limitation for this configuration. These Qwen "
      "results concern the dynamic-int8 ONNX port, rather than a full-precision checkpoint.")
    sub("Routing ablations with the newer encoders")
    rows = [["<b>Encoder / gate</b>", "<b>Wrapped AURC</b>",
             "<b>Δ AURC × 1,000 (95% CI)</b>"]]
    for item in modern:
        metrics = item["metrics"]
        stat = item["analysis"]["test"]["wrapped"]["aurc_delta"]
        lo, hi = stat["ci95"]
        for key, label in (("max_prob", "maximum probability"), ("prob_margin", "probability margin"),
                           ("neighbors", "neighbor agreement"), ("score_only_gate", "gate without neighbors"),
                           ("learned_gate", "gate with neighbors")):
            interval = f"{stat['estimate'] * 1000:+.2f} [{lo * 1000:+.2f}, {hi * 1000:+.2f}]" if key == "learned_gate" else ""
            rows.append([
                f"{item['encoder']['name']} / {label}",
                f"{metrics['test']['wrapped']['routing'][key]['aurc']:.5f}", interval,
            ])
    table(rows, [255, 85, 140])
    p("Table 8. All five routing policies for each 2025-2026 encoder. Intervals use 10,000 test-row "
      "bootstrap draws with each fitted classifier and threshold held fixed. Δ is the gate with "
      "neighbors minus the gate without them.", "caption")
    gemma_gate = gemma["test"]["wrapped"]["routing"]["learned_gate"]
    p(f"EmbeddingGemma's gate accepts {pct(gemma_gate['coverage'])} of wrapped requests with "
      f"{pct(gemma_gate['risk'])} error. Accepted error is "
      f"{pct(gemma['test']['original']['routing']['learned_gate']['risk'])} on the original test and "
      f"{pct(gemma['calibration']['learned_gate']['risk'])} on calibration. The empirical calibration "
      "target does not guarantee 5% error on either test condition; higher coverage must be read "
      "together with risk.")
    p("Differences across encoders jointly reflect architecture, model size, task instruction and "
      "ONNX quantization. The constructed test probes six polite templates on one grouped split.")
    sub("6.6 Selection among individual and ensemble classifiers")
    p("A separate classification follow-up combines BGE-small, BGE-base, E5-base-v2, Qwen3, "
      "EmbeddingGemma and Jina v5 Nano. It fits a linear head per encoder and an RBF classifier "
      "over their concatenation, with each vector block scaled by 1/√6. The RBF approximation uses "
      "512 Nyström landmarks, γ = 2 and seed 42, followed by logistic regression with C = 30. "
      "The candidate grid includes uniform mixtures of nonempty subsets of the seven semantic "
      "scores, lexical weights {0, 0.1, ..., 0.5}, and the earlier baselines. Selection uses only "
      "the 1,997-row grouped development partition. Exact ties prefer fewer nonzero score components "
      "and then lower lexical weight. The chosen heads are refitted on all 10,003 training rows.")
    weights_text = ", ".join(f"{name}: {weight:.3f}" for name, weight in ensemble["selected_nonzero_weights"].items())
    if len(ensemble["selected_nonzero_weights"]) == 1:
        p("Development selection chooses a single classifier from the candidate grid. The search "
          "therefore supplies an individual-model comparator, despite including multi-model mixtures.")
    p(f"The selected weights (rounded to three decimals) are {weights_text}. Official-test accuracy is "
      f"{pct(ensemble_result['accuracy'])} ({ensemble_result['correct']:,}/3,080; Table 3). "
      f"Relative to the original full-training fusion, the change is "
      f"{ensemble['paired_vs_original']['delta'] * 100:+.2f} points "
      f"(exact McNemar <i>p</i> = {ensemble['paired_vs_original']['p_value']:.3g}). "
      f"The difference from SPACE-3's published 94.94% is "
      f"{(ensemble_result['accuracy'] - 0.9494) * 100:+.2f} percentage points. "
      "This experiment supplies a classification comparator; routing results above retain their "
      "independent fit/development/calibration protocol.")
    p(f"The preset uniform average of all six linear semantic heads obtains "
      f"{pct(uniform_result['accuracy'])} ({uniform_result['correct']:,}/3,080), "
      f"{(uniform_result['accuracy'] - 0.9494) * 100:+.2f} points relative to SPACE-3. "
      f"Its development accuracy is {pct(ensemble['development_accuracy']['dense-average'])}, "
      f"compared with {pct(ensemble['development_accuracy'][ensemble['selected_model']])} for the "
      f"selected individual head. Against the original full-training fusion, its paired test change "
      f"is {ensemble['paired_uniform_vs_original']['delta'] * 100:+.2f} points "
      f"(McNemar <i>p</i> = {ensemble['paired_uniform_vs_original']['p_value']:.3g}). "
      "This is a predefined baseline's test result. It is distinct from the development-selected "
      "winner, and the published comparison is a single-fit point comparison with the cited studies.")
    h("7 Conclusion")
    p(f"On the constructed polite-query test, the highest observed classifier accuracy is "
      f"{pct(best_classifier['wrapped'])} from {best_classifier['name']} with {best_classifier['method']}. "
      f"Across the 20 encoder-policy combinations, the lowest observed AURC is "
      f"{best_routing['aurc']:.5f} from {best_routing['name']} with the {best_routing['policy']}. "
      "These comparisons identify the best measured configurations within this constructed benchmark.")
    p(f"On the original official test with full training, the preset six-encoder average reaches "
      f"{pct(uniform_result['accuracy'])}, above the published full-training baselines cited here. "
      "Development selection favors a different candidate, illustrating that its ordering can "
      "differ from the official test ordering.")
    p(f"In the BGE-small baseline, neighbor agreement raises coverage by "
      f"{(wrapped_gate['coverage'] - without_neighbors['coverage']) * 100:.2f} points over the "
      f"matched gate without neighbors. The newer encoders' paired ablations quantify how this effect "
      f"changes with representation. Higher classifier accuracy and lower ranking loss must still be "
      f"distinguished from meeting a test-time error target. The empirical 5% calibration constraint "
      f"provides no population-risk guarantee; repeated splits and real conversational inputs would "
      f"test the stability of these single-split results.")

    h("References")
    references = [
        ("Casanueva, I., Temčinas, T., Gerz, D., Henderson, M., and Vulić, I. (2020). "
         "Efficient Intent Detection with Dual Sentence Encoders. NLP4ConvAI.",
         "https://aclanthology.org/2020.nlp4convai-1.5/", "ACL Anthology"),
        ("Qi, H., Pan, L., Sood, A., Shah, A., Kunc, L., Yu, M., and Potdar, S. (2021). "
         "Benchmarking Commercial Intent Detection Services with Practice-Driven Evaluations. "
         "NAACL Industry Papers.",
         "https://aclanthology.org/2021.naacl-industry.38/", "ACL Anthology"),
        ("Song, S., Naseem, M. L., and Yuan, Z. (2026). DeepOpen: Decision-Oriented Intent "
         "Classification on BANKING77 and CLINC150.",
         "https://arxiv.org/abs/2610.31104v2", "arXiv:2610.31104v2"),
        ("He, W., Dai, Y., Yang, M., Sun, J., Huang, F., Si, L., and Li, Y. (2022). SPACE-3: Unified "
         "Dialog Model Pre-training for Task-Oriented Dialog Understanding and Generation.",
         "https://arxiv.org/abs/2209.06664", "arXiv:2209.06664"),
        ("Geifman, Y. and El-Yaniv, R. (2017). Selective Classification for Deep Neural Networks. NeurIPS.",
         "https://arxiv.org/abs/1705.08500", "arXiv:1705.08500"),
        ("Liang, H., Peng, L., and Sun, J. (2024). Selective Classification Under Distribution Shifts. "
         "Transactions on Machine Learning Research.",
         "https://arxiv.org/abs/2405.05160", "arXiv:2405.05160"),
        ("Cheng, Y., Yang, Z., and Lin, B. (2026). Signed Lexical Confidence for Risk-Calibrated Intent Routing.",
         "https://arxiv.org/abs/2610.00262v1", "arXiv:2610.00262v1"),
        ("Papernot, N. and McDaniel, P. (2018). Deep k-Nearest Neighbors: Towards Confident, "
         "Interpretable and Robust Deep Learning.",
         "https://arxiv.org/abs/1803.04765", "arXiv:1803.04765"),
        ("Zhang, J. et al. (2020). Discriminative Nearest Neighbor Few-Shot Intent Detection by "
         "Transferring Natural Language Inference. EMNLP.",
         "https://aclanthology.org/2020.emnlp-main.411/", "ACL Anthology"),
        ("Zhou, Y., Liu, P., and Qiu, X. (2022). KNN-Contrastive Learning for Out-of-Domain Intent "
         "Classification. ACL.",
         "https://aclanthology.org/2022.acl-long.352/", "ACL Anthology"),
        ("Bogachov, B., Letov, N., and Zhao, Y. F. (2026). The Hidden States Cookbook: A Large-Scale "
         "Ablation Study for Noise-Robust Conversational Intent Classification in Industry.",
         "https://arxiv.org/abs/2610.05394v1", "arXiv:2610.05394v1"),
        ("Qwen Team (2025). Qwen3 Embedding: Advancing Text Embedding and Reranking Through "
         "Foundation Models. Released June 5, 2025.",
         "https://qwenlm.github.io/blog/qwen3-embedding/", "Official release"),
        ("Google DeepMind (2025). Introducing EmbeddingGemma. Released September 4, 2025. "
         "EmbeddingGemma-300M model card.",
         "https://huggingface.co/google/embeddinggemma-300m", "Model card"),
        ("Akram, M. K. et al. (2026). jina-embeddings-v5-text: Task-Targeted Embedding Distillation. "
         "Classification-specific Nano checkpoint.",
         "https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-classification", "Model card"),
        ("Beijing Academy of Artificial Intelligence. BGE English embedding model cards. "
         "ONNX export by Xenova.",
         "https://huggingface.co/BAAI/bge-small-en-v1.5", "Model card"),
        ("Ying, C. and Thomas, S. (2022). Label Errors in BANKING77. Insights Workshop, ACL.",
         "https://aclanthology.org/2022.insights-1.19/", "ACL Anthology"),
        ("Wang, L. et al. (2022). Text Embeddings by Weakly-Supervised Contrastive Pre-training. "
         "E5-base-v2 checkpoint.",
         "https://huggingface.co/intfloat/e5-base-v2", "Model card"),
    ]
    for index, (reference, url, label) in enumerate(references, 1):
        p(f"[{index}] {reference} {link(url, label)}.", "reference")

    def footer(canvas, doc):
        canvas.setStrokeColor(RULE)
        canvas.line(66, 43, 546, 43)
        canvas.setFont("Text", 9)
        canvas.setFillColor(MUTED)
        canvas.drawCentredString(letter[0] / 2, 28, str(doc.page))

    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output), pagesize=letter, topMargin=49, bottomMargin=54,
                            leftMargin=60, rightMargin=60, title=TITLE,
                            author="Karatorgayeva Dina", subject="NLP Course Final Project")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font-dir", type=Path)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "output/pdf/banking77-report.pdf")
    args = parser.parse_args()
    register_fonts(args.font_dir)
    build(args.results, args.output)


if __name__ == "__main__":
    main()
