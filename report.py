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
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle

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

    box(0, 46, 99, 59, "Classifier", ["Banking message", "TF-IDF + BGE", "Fused intent"])
    box(119, 46, 112, 59, "Three signals", ["Top probability", "Top-two gap", "Neighbor agreement"])
    box(251, 46, 99, 59, "Logistic gate", ["Standardize features", "Score correctness"])
    box(370, 46, 109, 59, "Threshold τ", ["Accept fused intent", "or defer to an operator"])
    box(119, 1, 112, 31, "Fit index", ["8,006 BGE vectors"])
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
        story.append(Paragraph(value, style[name]))

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

    p("When Should a Banking Chatbot<br/>Defer to a Human?", "title")
    p("Karatorgayeva Dina", "author")
    p("NLP Course Final Project · Autumn 2026", "subtitle")
    sub("Abstract")
    p(f"We study whether a banking intent classifier can identify requests that need human review "
      f"when users add polite conversational text. We evaluate probability fusion and a routing rule that "
      f"combines classifier scores with agreement among nearby labeled examples. On 3,080 paired BANKING77 "
      f"requests, adding greetings and closings reduces fusion accuracy from "
      f"{pct(original['classification']['fusion']['accuracy'])} to {pct(wrapped['classification']['fusion']['accuracy'])}. "
      f"A threshold chosen on original calibration data accepts {pct(wrapped_gate['coverage'])} of the "
      f"altered inputs with {pct(wrapped_gate['risk'])} error among accepted predictions. "
      f"The corresponding gate without neighbors accepts {pct(without_neighbors['coverage'])} with "
      f"{pct(without_neighbors['risk'])} error. Code and results are available at {link(REPO, REPO)}.")
    h("1 Introduction")
    p("Support systems use intent predictions to route questions to a handler. A card that has already "
      "been ordered and a question about expected delivery time can require different workflows. Every "
      "automatic routing error may add a correction step, and every deferred request uses operator time. "
      "We measure both costs through accepted error and coverage.")
    p("We construct a paired BANKING77 test with 3,080 messages supplemented by greetings and closings. "
      "The experiment asks whether evidence from nearby examples helps a fixed classifier decide "
      "when to defer. The gate with neighbors achieves the lowest AURC of the five routing methods "
      "compared on this constructed test.")
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

    story.append(PageBreak())
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
    h("3 Model Description")
    sub("3.1 Predicting an intent")
    p("The lexical head uses L2-normalized TF-IDF over word unigrams and bigrams (minimum document "
      "frequency 2) and within-word character 3–5-grams (minimum frequency 3). The concatenated features "
      "feed an L2-regularized multinomial logistic regression with C = 20.")
    p("The semantic head encodes raw text with the frozen, int8 ONNX port of BGE-small-en-v1.5 [12], "
      "using dynamic padding and up to 128 tokens. The 384-dimensional final [CLS] vector is normalized "
      "to unit length and passed to multinomial logistic regression with C = 100. We combine the "
      "resulting class distributions and predict the largest fused probability.")
    p("<i>p</i>(<i>y</i> | <i>x</i>) = α <i>p</i><sub>lex</sub>(<i>y</i> | <i>x</i>) "
      "+ (1 − α) <i>p</i><sub>sem</sub>(<i>y</i> | <i>x</i>).", "formula")
    sub("3.2 Deciding whether to accept")
    p("Let <i>p</i><sub>(1)</sub> and <i>p</i><sub>(2)</sub> be the two highest fused probabilities. "
      "<i>N</i><sub>8</sub>(<i>x</i>) contains the eight fit rows with the highest cosine similarity "
      "in the normalized 384-dimensional BGE-small [CLS] space. The index uses only the 8,006 fit rows.")
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

    story.append(PageBreak())
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
                     f"{s['min_per_class']}–{s['max_per_class']}", int(s["median_words"]), int(s["p90_words"])])
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
      "measure accuracy after removing those 25 rows. Ying and Thomas [13] identify potential label noise.")

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

    story.append(PageBreak())
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
    sub("5.3 Baselines")
    p("Classification baselines are the most frequent intent, the lexical head and the frozen "
      "semantic head. Routing baselines share the fused class predictions and score them with maximum "
      "probability, the top-two probability gap, neighbor agreement alone, or a learned score using "
      "only the two classifier-derived signals. The last comparator isolates the contribution of neighbors.")
    h("6 Results")
    sub("6.1 Intent classification")
    rows = [["<b>Classifier</b>", "<b>Original accuracy</b>", "<b>Original macro-F1</b>",
             "<b>Wrapped accuracy</b>", "<b>Wrapped macro-F1</b>"]]
    names = [("majority", "Most frequent intent"), ("tfidf", "TF-IDF"),
             ("bge", "BGE-small"), ("fusion", "Probability fusion")]
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
      f"TF-IDF has the highest accuracy after wrapping.")
    table([
        ["<b>Source</b>", "<b>Model</b>", "<b>Accuracy</b>"],
        ["This study", "TF-IDF / BGE-small / fusion", "91.14% / 93.08% / 93.60%"],
        ["This study", "Follow-up ensemble (§6.4)", pct(selected_result["accuracy"])],
        ["Casanueva et al. [1]", "USE+ConveRT / BERT-large", "93.36% / 93.66%"],
        ["Qi et al. [2]", "RoBERTa-base", "94.10%"],
        ["Song et al. [3]", "SAM continuation, mean of 3 seeds", "94.04%"],
        ["He et al. [4]", "SPACE-3", "94.94%"],
    ], [124, 215, 141])
    p("Table 3. Full-training accuracy on the official BANKING77 test. Published rows are taken "
      "from the cited studies. The follow-up ensemble exceeds Qi et al. and Song et al. and falls "
      "0.43 percentage points below SPACE-3.", "caption")

    story.append(PageBreak())
    sub("6.2 Selective routing")
    rows = [["<b>Routing policy</b>", "<b>Orig. cov.</b>", "<b>Orig. risk</b>",
             "<b>Wrap. cov.</b>", "<b>Wrap. risk</b>", "<b>Wrap. AURC</b>"]]
    for key, label in (("max_prob", "Maximum probability"), ("prob_margin", "Probability margin"),
                       ("neighbors", "Neighbor agreement"), ("score_only_gate", "Gate without neighbors"),
                       ("learned_gate", "<b>Gate with neighbors</b>")):
        o, w = original["routing"][key], wrapped["routing"][key]
        rows.append([label, pct(o["coverage"]), pct(o["risk"]), pct(w["coverage"]),
                     pct(w["risk"]), f"{w['aurc']:.4f}"])
    table(rows, [147, 63, 62, 63, 64, 81])
    p("Table 4. Coverage and risk at thresholds chosen on original calibration data with a nominal "
      "5% target. AURC is computed over the entire ranking.", "caption")
    agreement = routing["test"]["wrapped"]["margin_vs_gate_without_neighbors"]
    p("Probability margin and the gate without neighbors each accept 2,480 wrapped requests and make "
      f"125 errors. Each accepts {agreement['margin_only_accepted']} additional request unique to that policy. "
      "Their score rankings produce different AURCs.")
    p(f"On the original BANKING77 test, Cheng et al. [7] report 89.63% coverage, 2.65% accepted error "
      f"and AURC 0.01044 (Tables II–III). Our gate obtains {pct(original_gate['coverage'])}, "
      f"{pct(original_gate['risk'])} and {original_gate['aurc']:.5f}, respectively. Their results "
      "average ten runs of a MiniLM classifier with simultaneous binomial calibration. Ours use "
      "one fit split, a fused classifier and an empirical calibration constraint. These protocol "
      "differences limit direct comparison.")

    story.append(risk_figure(results / "risk_coverage.csv"))
    p("Figure 2. Selective risk against coverage. The curves rank the "
      "same fused predictions by different scores. Table 4 reports the frozen operating points.", "caption")
    p(f"The combined gate accepts {wrapped_gate['accepted']:,} wrapped requests with "
      f"{wrapped_gate['errors']} errors. Its accepted error has a Wilson 95% interval of "
      f"{pct(wrapped_gate['risk_wilson_95'][0])} to {pct(wrapped_gate['risk_wilson_95'][1])}, which crosses 5%.")
    story.append(PageBreak())
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
    p("Table 5. Gate with neighbors minus gate without neighbors. Percentile intervals use 10,000 "
      "paired resamples of test rows, sharing row indices across both gates within each condition. "
      "Thresholds and fitted models stay fixed. AURC is recomputed per draw.", "caption")
    p("Both coverage and AURC intervals exclude zero for this split. The bootstrap measures test-row "
      "uncertainty conditional on the fitted gates and thresholds. Variation across training splits "
      "or calibration sets would require repeated fitting.")
    p("At 80% coverage on the wrapped test, error is 3.25% with neighbors and 4.95% without them. "
      "The neighbor-only score obtains lower error at its own operating point (4.20%), with 84 fewer "
      "accepted requests than the combined gate. Its coarse scores yield a higher overall AURC.")
    sub("6.3 Error analysis and limitations")
    p("The largest wrapped confusion is unrecognized card payment classified as unrecognized direct debit "
      "(14 cases). One original request, “My card has not arrived yet,” is correctly classified as "
      "card arrival. After wrapping, the model predicts a delivery estimate and the gate defers it. "
      "Coverage for suspected card compromise falls from 36/40 to 12/40. Aggregate coverage can mask "
      "unequal operator load across intents.")
    p("This study uses one grouped split and six synthetic wrappers. Observed calibration error and test "
      "intervals describe this benchmark. Live traffic, novel intents, costs of particular misroutes, "
      "and the quality of human resolution require separate measurements.")
    sub("6.4 Follow-up with larger frozen encoders")
    p("The follow-up adds int8 BGE-base [12] with [CLS] pooling and E5-base-v2 [14] with a query prefix "
      "and masked mean pooling. It compares the linear heads, probability mixtures and a classifier "
      "with Nyström RBF features. We select by accuracy on 1,997 grouped development rows and refit on "
      "the complete training split. This classification experiment follows the initial test evaluation.")
    change = extended["paired_vs_original"]
    p(f"The chosen mixture gives TF-IDF 10% and each of the three encoders 30%. "
      f"Its {pct(selected_result['accuracy'])} test accuracy (Table 3) improves on the original fusion "
      f"by {change['delta'] * 100:.2f} points "
      f"(exact McNemar <i>p</i> = {change['p_value']:.3g}).")
    h("7 Conclusion")
    p(f"On our constructed BANKING77 test, the gate with neighbor agreement has the lowest AURC "
      f"({wrapped_gate['aurc']:.5f}) of the five evaluated policies. It raises coverage by "
      f"{(wrapped_gate['coverage'] - without_neighbors['coverage']) * 100:.2f} points over the "
      f"matched gate without neighbors. Polite additions still increase the share sent to an operator "
      f"from {pct(1 - original_gate['coverage'])} to {pct(1 - wrapped_gate['coverage'])} at the same threshold.")

    story.append(PageBreak())
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
