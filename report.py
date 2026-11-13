"""Build the course report from the saved experiment results."""

import argparse
import csv
import json
import os
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, PolyLine, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent
REPO = "https://github.com/amidachi/nlp-project_banking77"
DATA_URL = "https://github.com/PolyAI-LDN/task-specific-datasets/tree/9d081458ff52e53cf7e848f414e6e9344e4e6696/banking_data"
TITLE = "When Should a Banking Chatbot Defer to a Human?"
INK = colors.HexColor("#18212A")
MUTED = colors.HexColor("#505B65")
RULE = colors.HexColor("#CDD5DB")
TEAL = colors.HexColor("#167976")
NAVY = colors.HexColor("#2E4E83")


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


def risk_figure(path):
    policies = {"max_prob": "Maximum probability", "score_only_gate": "Gate without neighbors",
                "learned_gate": "Gate with neighbors"}
    palette = {"max_prob": MUTED, "score_only_gate": NAVY, "learned_gate": TEAL}
    series = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            coverage = float(row["coverage"])
            if row["policy"] in policies and coverage >= 0.50:
                series.setdefault((row["variant"], row["policy"]), []).append(
                    (coverage, float(row["risk"])),
                )
    drawing = Drawing(480, 165)
    bottom, height, width = 39, 100, 185
    ymax = 0.15
    for left, variant, title in ((43, "original", "Original test"), (279, "wrapped", "With polite wrappers")):
        drawing.add(String(left + width / 2, 147, title, fontName="TextBold",
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
        for policy in policies:
            values = series[(variant, policy)]
            points = values[::6] + [values[-1]]
            drawing.add(PolyLine(
                [coordinate for cx, ry in points for coordinate in
                 (left + width * (cx - 0.5) * 2, bottom + height * min(ry, ymax) / ymax)],
                strokeColor=palette[policy], strokeWidth=1.45,
                strokeDashArray=[3, 2] if policy == "score_only_gate" else None,
            ))
    for x, (key, label) in zip((42, 181, 325), policies.items()):
        drawing.add(Line(x, 7, x + 18, 7, strokeColor=palette[key], strokeWidth=1.6,
                         strokeDashArray=[3, 2] if key == "score_only_gate" else None))
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
      f"{pct(without_neighbors['risk'])} error. Code and results are available at {link(REPO, 'github.com/amidachi/nlp-project_banking77')}.")
    h("1 Introduction")
    p("Support systems use intent predictions to route questions to a handler. A card that has already "
      "been ordered and a question about expected delivery time can require different workflows. Every "
      "automatic routing error may add a correction step, and every deferred request uses operator time. "
      "We measure both costs through accepted error and coverage.")
    p("Given one English message and 77 supported intents, the classifier returns a candidate label. "
      "A separate rule accepts the candidate or defers the request. We ask whether eight nearby labeled "
      "examples improve the acceptance score and whether the same threshold transfers to a paired "
      "test with greetings and closings.")
    sub("1.1 Team")
    p("Karatorgayeva Dina defined the project direction and is responsible for the course submission. "
      "Codex assisted with implementation, experiments, and report preparation.")
    h("2 Related Work")
    p("Casanueva et al. [1] introduced BANKING77. Their frozen USE+ConveRT representations reach 85.19% "
      "accuracy with ten examples per intent and 93.36% with full training. Fine-tuned BERT-large reaches "
      "93.66% with full training. Qi et al. [2] compare commercial systems and fine-tuned encoders. "
      "RoBERTa-base reaches 94.10% on the full test and 83.8% on a TF-IDF-selected difficult subset. "
      "Song et al. [3] report 94.04% mean accuracy over three SAM continuation seeds and 94.12% for "
      "the best seed selected after evaluation. SPACE-3 [10] uses tree-induced contrastive "
      "pretraining on dialogue corpora and reaches 94.94% with full training.")
    p("Geifman and El-Yaniv [4] formalize selective classification as a trade-off between acceptance "
      "and errors among accepted predictions. Liang et al. [5] examine confidence margins under "
      "distribution shift. Cheng et al. [6] combine semantic confidence and signed lexical evidence. "
      "On BANKING77 they report 89.63% coverage, 2.65% accepted error and AURC 0.01044 across ten runs "
      "with binomial threshold calibration. Our study uses a different classifier and empirical calibration.")
    p("Bogachov et al. [7] report roughly 2.6–2.8 F1 points of improvement over default pooling with "
      "attention pooling under conversational noise. We study threshold transfer under a controlled "
      "wording change that preserves each message and its original label.")

    story.append(PageBreak())
    h("3 Model Description")
    sub("3.1 Predicting an intent")
    p("The lexical head uses L2-normalized TF-IDF over word unigrams and bigrams (minimum document "
      "frequency 2) and within-word character 3–5-grams (minimum frequency 3). The concatenated features "
      "feed an L2-regularized multinomial logistic regression with C = 20.")
    p("The semantic head encodes raw text with the frozen, int8 ONNX port of BGE-small-en-v1.5 [8], "
      "using dynamic padding and up to 128 tokens. The 384-dimensional final [CLS] vector is normalized "
      "to unit length and passed to multinomial logistic regression with C = 100. We combine the "
      "resulting class distributions and predict the largest fused probability.")
    p("<i>p</i>(<i>y</i> | <i>x</i>) = α <i>p</i><sub>lex</sub>(<i>y</i> | <i>x</i>) "
      "+ (1 − α) <i>p</i><sub>sem</sub>(<i>y</i> | <i>x</i>).", "formula")
    sub("3.2 Deciding whether to accept")
    p("For a fixed fused label, we extract three signals. The first is the largest fused probability. "
      "The second is its gap over the next highest probability. The third is the fraction of eight "
      "cosine-nearest fit examples whose labels agree with the fused prediction. We standardize these "
      "features on development data and train an L2 logistic regression (C = 1) on a binary target "
      "indicating whether the fused prediction is correct.")
    p("The resulting score ranks examples for acceptance. On separate calibration data, we sort scores "
      "in descending order and take the largest prefix whose observed accepted error is at most 5%. "
      "Complete score ties enter together. The last score in that prefix defines the threshold. "
      "We then apply the same score model and threshold to both test conditions.")
    h("4 Dataset")
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
      "measure accuracy after removing those 25 rows. Ying and Thomas [9] identify potential label noise.")

    story.append(PageBreak())
    h("5 Experiments")
    sub("5.1 Metrics")
    p("We report accuracy and macro-F1 for classification. For a threshold accepting a subset A of "
      "N messages, coverage is |A| / N and selective risk is the number of incorrect accepted "
      "predictions divided by |A|. For the ranking analysis, AURC is the mean of cumulative selective "
      "risk across all nonempty prefixes, averaging within score ties. Lower AURC is better. "
      "We report nominal 95% Wilson intervals for accepted error and exact McNemar tests for paired "
      "classification comparisons.")
    sub("5.2 Experiment Setup")
    p("A stratified group split with seed 42 assigns a held-out fold from the original training "
      "file. A second group split divides it into development and calibration. The heads fit on "
      "8,006 examples. The development set selects the lexical fusion weight from {0, 0.1, ..., 0.5} "
      f"by accuracy (selected α = {m['fusion_weight']:.1f}) and fits the correctness models. "
      "The calibration set fixes the routing thresholds. The test variants are evaluated with these "
      "frozen choices. We fit the classification heads once more on all 10,003 training rows for "
      "the published accuracy comparison. All runs use CPU. Data and encoder revisions are pinned.")
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
        ["Casanueva et al. [1]", "USE+ConveRT / BERT-large", "93.36% / 93.66%"],
        ["Qi et al. [2]", "RoBERTa-base", "94.10%"],
        ["Song et al. [3]", "SAM continuation, mean of 3 seeds", "94.04%"],
        ["He et al. [10]", "SPACE-3", "94.94%"],
    ], [124, 215, 141])
    p("Table 3. Full-training accuracy on the official BANKING77 test. Published rows are taken "
      "from the cited studies. The fusion result is below the strongest cited figures.", "caption")

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
    p("Table 4. Fixed thresholds from original calibration data at a nominal 5% target. "
      "Coverage is the share routed automatically. Risk is the error among accepted requests.", "caption")
    story.append(risk_figure(results / "risk_coverage.csv"))
    p("Figure 1. Selective risk (vertical) against coverage (horizontal). The curves rank the "
      "same fused predictions by different scores. Table 4 reports the frozen operating points.", "caption")
    p(f"On the wrapped test, the gate with neighbors accepts {wrapped_gate['accepted']:,} requests "
      f"and makes {wrapped_gate['errors']} errors. The ablation accepts {without_neighbors['accepted']:,} "
      f"and makes {without_neighbors['errors']} errors. On original text, the gate with neighbors "
      f"accepts {original_gate['accepted']:,} with {original_gate['errors']} errors. Its accepted error "
      f"interval rises from {pct(original_gate['risk_wilson_95'][0])}–{pct(original_gate['risk_wilson_95'][1])} "
      f"to {pct(wrapped_gate['risk_wilson_95'][0])}–{pct(wrapped_gate['risk_wilson_95'][1])} "
      "with wrappers. The latter interval crosses 5%.")
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
    story.append(PageBreak())
    sub("6.4 Follow-up with larger frozen encoders")
    p("We evaluated two more 768-dimensional int8 encoders on CPU, BGE-base [8] with [CLS] pooling "
      "and E5-base-v2 [11] with a query prefix and masked mean pooling. We use the same lexical and "
      "linear heads and train logistic regression (C = 30) on 512 Nyström RBF features (γ = 2) "
      "computed over the concatenated normalized embeddings. "
      "We compare the individual models with probability mixtures. Accuracy on 1,997 grouped "
      "development rows selects the mixture, then all heads are refit on all 10,003 training rows. "
      "This follow-up was designed after the first official test evaluation.")
    selected = extended["selected_model"]
    selected_result = extended["official_test"][selected]
    change = extended["paired_vs_original"]
    weights = ", ".join(f"{weight:.0%} {name}" for name, weight in extended["weights"].items() if weight)
    rows = [["<b>Model</b>", "<b>Official test accuracy</b>"]]
    for key, label in (("original-fusion", "Original TF-IDF + BGE-small"),
                       ("bge-base", "BGE-base"), ("e5-base", "E5-base-v2"),
                       ("dense-average", "Mean of three encoders"), ("rbf", "RBF feature classifier")):
        rows.append([label, pct(extended["official_test"][key]["accuracy"])])
    if selected not in ("original-fusion", "bge-base", "e5-base", "dense-average", "rbf"):
        rows.append(["Development-selected model", pct(selected_result["accuracy"])])
    table(rows, [330, 150])
    p(f"Table 5. Follow-up on the official test. Development selects {weights}. "
      f"Its test accuracy is {pct(selected_result['accuracy'])}, a change of "
      f"{change['delta'] * 100:+.2f} points from the original fusion "
      f"(exact McNemar <i>p</i> = {change['p_value']:.3g}).", "caption")
    h("7 Conclusion")
    p(f"Polite additions make intent routing harder even when the banking request is preserved. "
      f"On these paired test inputs, adding neighbor evidence improves the learned gate’s ranking "
      f"and raises accepted coverage by {(wrapped_gate['coverage'] - without_neighbors['coverage']) * 100:.2f} points "
      f"over the score-only gate. At the same threshold, the share deferred rises from "
      f"{pct(1 - original_gate['coverage'])} to {pct(1 - wrapped_gate['coverage'])} after wrapping.")

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
        ("Geifman, Y. and El-Yaniv, R. (2017). Selective Classification for Deep Neural Networks. NeurIPS.",
         "https://arxiv.org/abs/1705.08500", "arXiv:1705.08500"),
        ("Liang, H., Peng, L., and Sun, J. (2024). Selective Classification Under Distribution Shifts. "
         "Transactions on Machine Learning Research.",
         "https://arxiv.org/abs/2405.05160", "arXiv:2405.05160"),
        ("Cheng, Y., Yang, Z., and Lin, B. (2026). Signed Lexical Confidence for Risk-Calibrated Intent Routing.",
         "https://arxiv.org/abs/2610.00262", "arXiv:2610.00262"),
        ("Bogachov, B., Letov, N., and Zhao, Y. F. (2026). The Hidden States Cookbook: A Large-Scale "
         "Ablation Study for Noise-Robust Conversational Intent Classification in Industry.",
         "https://arxiv.org/abs/2610.05394", "arXiv:2610.05394"),
        ("Beijing Academy of Artificial Intelligence. BGE-small-en-v1.5 model card. "
         "ONNX export by Xenova.",
         "https://huggingface.co/BAAI/bge-small-en-v1.5", "Model card"),
        ("Ying, C. and Thomas, S. (2022). Label Errors in BANKING77. Insights Workshop, ACL.",
         "https://aclanthology.org/2022.insights-1.19/", "ACL Anthology"),
        ("He, W., Dai, Y., Yang, M., Sun, J., Huang, F., Si, L., and Li, Y. (2022). SPACE-3: Unified "
         "Dialog Model Pre-training for Task-Oriented Dialog Understanding and Generation.",
         "https://arxiv.org/abs/2209.06664", "arXiv:2209.06664"),
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
