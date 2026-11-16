"""PS06 proposal deck for PruneAgent. Usage: python build_ps06.py base_ps06.pptx proposal.pptx"""
import sys
from copy import deepcopy

from pptx import Presentation
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

from arch_diagram import (GOLD, GOLD_BG, GOLD_TX, GREEN, GREEN_BG, GREY_BG, INK, NAVY, RED, RED_BG, TEAL, TEAL_BG,
                          badge, line, rect, text)
from helpers import P, by_id, delete, notes, set_paras, txt

PURPLE, PURPLE_BG, BODY = "5B3F8C", "ECE6F5", "404040"

prs = Presentation(sys.argv[1])
s_title, s_prob, s_sol, s_risk, s_arch, s_plan, s_app, s_ref = prs.slides


def size(shape, pt):
    for para in shape.text_frame.paragraphs:
        for r in para.runs:
            r.font.size = Pt(pt)


def footnote(slide, src, paras, top=6.5):
    el = deepcopy(by_id(src, 9)._element)
    max_id = max(int(e.get("id")) for e in slide.shapes._spTree.iter(qn("p:cNvPr")))
    el.find(".//" + qn("p:cNvPr")).set("id", str(max_id + 1))
    el.find(".//" + qn("p:cNvPr")).set("name", "Footnotes")
    slide.shapes._spTree.append(el)
    f = [x for x in slide.shapes if x.name == "Footnotes"][0]
    f.left, f.top, f.width, f.height = Inches(0.75), Inches(top), Inches(11.4), Inches(0.28)
    set_paras(f.text_frame, paras)
    size(f, 11)
    return f


NOTES = {
    "title": "I’m Steph Xu, and today I’m proposing PruneAgent: an LLM agent that automates the layer-wise "
             "sensitivity analysis researchers do by hand when they structurally prune an open language model.",
    "prob": "The users are ML researchers and engineers compressing open LLMs for single-GPU deployment, where memory "
            "and speed are the binding constraints. Only structured pruning, which removes whole channels, attention "
            "heads or layers, makes a model smaller and faster on standard hardware. The tools exist: FLAP prunes "
            "without retraining, and ShortGPT removes whole layers. What remains manual is the sensitivity analysis "
            "around them: sweeping layers, reading the results, and deciding how much of each layer to remove. "
            "FLAP’s allocation is a fixed rule, RL search costs GPU-hours, and the one LLM-agent paper this year "
            "predicts unstructured levels, with no public code. That workflow is what I’m automating.",
    "today": "Here is how engineers prune a model today. The tools are mature: Torch-Pruning, LLM-Pruner and NVIDIA’s "
             "Model Optimizer all remove channels, heads and layers correctly. What they don’t do is decide how much to "
             "remove where. So the engineer runs a sensitivity sweep, cutting one layer at a time and measuring the "
             "quality loss, reads the results, sets an amount for each layer, prunes, evaluates, and usually retrains "
             "to win back quality. NVIDIA retrained its pruned Llama 3.1 4B on 94 billion tokens. When the result "
             "misses the target, the loop starts again at step 3. Each pass is a set of full-model evaluations and a "
             "human decision, which is why one model takes weeks to months. PruneAgent automates steps 2 to 4.",
    "sol": "The agent decides which experiment to run next, using existing pruning tools, instead of a researcher. "
           "Input: a Qwen3 model and a target, 20 or 30 percent of transformer-block parameters removed. Constraints: "
           "a 15 percent perplexity limit per step, a cumulative limit against the dense model, and five dollars per "
           "run. Output: the pruned model, its per-layer widths, a rationale for every decision, and a report against "
           "baselines at equal measured memory, with tokens per second. An optional last step quantizes every method "
           "to 8 or 4 bits the same way.",
    "arch": "One command, pruneagent run, takes the model, target and limits, and returns the pruned model, rationale "
            "log and report. FLAP scores every channel and attention group once. LangGraph fans out layer-local probes "
            "in parallel. The agent chooses the next experiment: probe further, apply a plan, start from FLAP, roll "
            "back or stop. Per layer, it narrows the MLP, removes attention groups, or drops the layer. The validator "
            "enforces fixed rules, the model is rebuilt from the dense weights, and the gate checks per-step and "
            "cumulative perplexity on the validation split. At the end: optional 8- or 4-bit quantization, then final "
            "evaluation on the test split, seven zero-shot tasks and MMLU, with memory and tokens per second on a T4.",
    "risk": "Scope: Qwen3-0.6B and 1.7B, structured pruning only, 20 to 30 percent, post-training, with optional "
            "uniform quantization on the T4. The main risk is that the agent doesn’t beat the baselines; it can start "
            "from FLAP’s allocation, and a well-supported negative result is still valid. Structured pruning loses "
            "quality faster than unstructured, so steps are small and the cumulative limit bounds drift. The FLAP code "
            "targets LLaMA, so week 1 adapts it to Qwen3’s grouped attention and builds a loader for per-layer widths.",
    "plan": "Week 1 is a feasibility spike: FLAP adapted to Qwen3, the loader, uniform Wanda-sp and FLAP baselines, a "
            "timed manual sweep, and one agent round end to end. Week 2 builds the agent graph; the midpoint is an "
            "end-to-end run on 0.6B. Week 3 tunes the instructions on 0.6B only and adds quantization. Week 4 runs the "
            "final evaluation on Qwen3-1.7B. Success: quality at least equal to uniform Wanda-sp and FLAP at equal "
            "measured memory, with fewer evaluations than the manual sweep and no manual decisions.",
}

# 1 title -----------------------------------------------------------------------
txt(s_title, 2, P("PruneAgent: Agentic Sensitivity Analysis for Structured LLM Pruning"))
txt(s_title, 3, P("Steph Xu"), P("1.125 Agentic Engineering & Architecture"), P("Term Project Proposal (PS06)"))
notes(s_title, NOTES["title"])

# 2 problem and audience ----------------------------------------------------------
txt(s_prob, 10, P("Layer-wise sensitivity analysis is a manual bottleneck in LLM pruning"))
txt(s_prob, 2, P("The Target Users"))
txt(s_prob, 4, P("The Bottleneck"))
txt(s_prob, 6, P("The Gap"))
txt(s_prob, 11,
    P(("ML researchers and engineers compressing open LLMs.", "b")),
    P("Target: single-GPU deployment, where memory and speed are the binding constraints."),
    P("Students and small teams without GPU clusters."))
txt(s_prob, 5,
    P(("Per-layer sensitivity analysis is manual and compute-intensive.", "b")),
    P("Only structured pruning (removing channels, heads or layers) shrinks and speeds up a model on a T4 or Mac."),
    P("Tools exist (FLAP", ("1", "s"), ", ShortGPT", ("2", "s"), "); choosing how much of each layer to remove still "
      "takes repeated sweeps and judgment."))
txt(s_prob, 7,
    P(("Automation is heuristic, costly or partial.", "b")),
    P("FLAP", ("1", "s"), ": fixed allocation rule."),
    P("RL search (AMC", ("3", "s"), ", FastForward", ("4", "s"), "): GPU-hours of policy training."),
    P("LLM agent (Kodathala & Vunnam", ("5", "s"), "): unstructured levels only, not the experiment workflow; no "
      "public code."))
for sid in (11, 5, 7):
    size(by_id(s_prob, sid), 16)
footnote(s_prob, s_sol, [P(("1", "s"), " An et al., 2024.  ", ("2", "s"), " Men et al., 2024.  ",
                           ("3", "s"), " He et al., 2018.  ", ("4", "s"), " Yuan et al., 2025.  ",
                           ("5", "s"), " Kodathala & Vunnam, 2026.  Full references on the last slide.")])
notes(s_prob, NOTES["prob"])

# 2b pruning today (new slide, drawn on the architecture slide's layout) ----------------
s_today = prs.slides.add_slide(s_arch.slide_layout)
s_today.shapes.title.text_frame.text = "Pruning today takes weeks to months"
for sh in list(s_today.placeholders):
    if sh.placeholder_format.type != 1:  # keep the title only
        sh._element.getparent().remove(sh._element)
t_title = s_today.shapes.title
t_title.left, t_title.top, t_title.width, t_title.height = Inches(0.75), Inches(0.33), Inches(11.8), Inches(1.24)
s_today.shapes._spTree.append(deepcopy(by_id(s_arch, 3)._element))  # slide number, as on the architecture slide

EY, EH = 1.32, 0.55
rect(s_today, "Tools label box", 0.75, EY, 2.4, EH, fill=NAVY)
text(s_today, "Tools label", 0.87, EY, 2.2, EH, [[("Mature Tools", {"b": True, "color": "FFFFFF", "size": 15})]],
     anchor=MSO_ANCHOR.MIDDLE)
rect(s_today, "Tools box", 3.2, EY, 9.4, EH, fill=GOLD_BG)
text(s_today, "Tools text", 3.35, EY, 9.15, EH,
     [[("Torch-Pruning", {"b": True, "color": INK, "size": 13}), ("6", {"color": INK, "size": 13, "s": True}),
       (" · ", {"color": BODY, "size": 13}), ("LLM-Pruner", {"b": True, "color": INK, "size": 13}),
       ("7", {"color": INK, "size": 13, "s": True}), (" · ", {"color": BODY, "size": 13}),
       ("NVIDIA Model Optimizer", {"b": True, "color": INK, "size": 13}),
       ("8", {"color": INK, "size": 13, "s": True}), (" · FLAP · ShortGPT: they cut correctly, ", {"color": BODY, "size": 13}),
       ("but don’t decide how much to cut where", {"b": True, "color": INK, "size": 13})]],
     anchor=MSO_ANCHOR.MIDDLE)

TW, TSTEP = 2.2, 2.41
txs = [0.75 + i * TSTEP for i in range(5)]
TY, TH = 2.75, 1.5
tmid = TY + TH / 2
steps = [
    ("1", "Choose a Method", "Pick a tool and a rule for scoring what to cut", "Engineer", GREY_BG, "5A6878", INK),
    ("2", "Sensitivity Sweep", "Cut one layer at a time; measure the quality loss", "One evaluation per layer and level",
     PURPLE_BG, PURPLE, PURPLE),
    ("3", "Set Per-Layer Cuts", "Read the results; decide how much of each layer to remove", "Judgment, rules of thumb",
     GOLD_BG, GOLD, GOLD_TX),
    ("4", "Prune and Evaluate", "Apply the plan; benchmark quality, memory and speed", "Full-model evaluation",
     TEAL_BG, TEAL, TEAL),
    ("5", "Retrain to Recover", "Retrain or distill to win back lost quality", "Billions of tokens on GPUs",
     GREY_BG, "5A6878", INK),
]
for i, (n, t, d, who, bg, acc, tc) in enumerate(steps):
    x = txs[i]
    rect(s_today, f"S{n} box", x, TY, TW, TH, fill=bg)
    badge(s_today, f"S{n} number", x + 0.12, TY + 0.12, n, acc)
    text(s_today, f"S{n} title", x + 0.12, TY + 0.5, TW - 0.2, 0.34, [[(t, {"b": True, "color": tc, "size": 14})]])
    text(s_today, f"S{n} text", x + 0.12, TY + 0.86, TW - 0.22, 0.62, [[(d, {"color": BODY, "size": 13})]])
    text(s_today, f"S{n} who", x, TY + TH + 0.08, TW, 0.3, [[(who, {"i": True, "color": "5A6878", "size": 12.5})]],
         align=PP_ALIGN.CENTER)
    if i < 4:
        line(s_today, f"Step arrow {i+1}", x + TW, tmid, txs[i + 1], tmid)

# bracket over steps 2-4: what PruneAgent automates
bx0, bx1, btop = txs[1], txs[3] + TW, 2.38
line(s_today, "Agent bracket left", bx0, btop, bx0, TY - 0.06, color=GREEN, arrow=False)
line(s_today, "Agent bracket top", bx0, btop, bx1, btop, color=GREEN, arrow=False)
line(s_today, "Agent bracket right", bx1, btop, bx1, TY - 0.06, color=GREEN, arrow=False)
text(s_today, "Agent bracket label", bx0, btop - 0.34, bx1 - bx0, 0.3,
     [[("PruneAgent automates steps 2–4", {"b": True, "color": GREEN, "size": 13})]], align=PP_ALIGN.CENTER)

# loop: missed target -> back to step 3
ly = TY + TH + 0.5
rx, sx3 = txs[4] + TW / 2, txs[2] + TW / 2
line(s_today, "Redo down", rx, TY + TH + 0.4, rx, ly, color=RED, arrow=False)
line(s_today, "Redo across", rx, ly, sx3, ly, color=RED, arrow=False)
line(s_today, "Redo up", sx3, ly, sx3, TY + TH + 0.4, color=RED)
text(s_today, "Redo label", sx3 + 0.2, ly + 0.04, rx - sx3 - 0.4, 0.3,
     [[("missed the quality or size target: back to step 3", {"i": True, "color": RED, "size": 13})]],
     align=PP_ALIGN.CENTER)

WY, WH = 5.2, 1.15
rect(s_today, "Why label box", 0.75, WY, 2.4, WH, fill=NAVY)
text(s_today, "Why label", 0.87, WY, 2.2, WH, [[("Weeks to Months", {"b": True, "color": "FFFFFF", "size": 15})]],
     anchor=MSO_ANCHOR.MIDDLE)
rect(s_today, "Why box", 3.2, WY, 9.4, WH, fill=RED_BG)
text(s_today, "Why text", 3.35, WY, 9.15, WH,
     [[("Every pass costs many full-model evaluations and a person’s judgment; ", {"color": BODY, "size": 14}),
       ("a bad per-layer choice shows up only after pruning, testing and often retraining.",
        {"b": True, "color": INK, "size": 14})],
      [("Example: NVIDIA retrained its pruned Llama 3.1 4B on 94 billion tokens", {"color": BODY, "size": 14}),
       ("8", {"color": BODY, "size": 14, "s": True}), (".", {"color": BODY, "size": 14})]],
     anchor=MSO_ANCHOR.MIDDLE)
footnote(s_today, s_sol, [P(("6", "s"), " Fang et al., 2023.  ", ("7", "s"), " Ma et al., 2023.  ",
                            ("8", "s"), " Sreenivas et al., 2024 (Minitron).  Timescale is the author’s estimate: "
                            "papers report the steps, not the person-time.")], top=6.6)
notes(s_today, NOTES["today"])

# 3 proposed solution -------------------------------------------------------------
txt(s_sol, 2, P("The Solution: Agentic Sensitivity Workflow"))
text(s_sol, "Value statement", 0.75, 1.2, 11.8, 0.62,
     [[("PruneAgent automates the researcher’s sensitivity-analysis workflow for structured pruning: an LLM agent "
        "decides which experiments to run with existing tools, producing a smaller, faster model at conventional-method "
        "quality with fewer evaluations and no manual decisions.", {"i": True, "color": NAVY, "size": 14})]])
txt(s_sol, 3, P("Inputs, Constraints and Outputs"))
txt(s_sol, 5,
    P(("Input: ", "b"), "Qwen3 model; target: 20% or 30% of transformer-block parameters removed."),
    P(("Constraints: ", "b"), "≤15% perplexity increase per step; cumulative limit vs. dense model; $5 per run."),
    P(("Output: ", "b"), "pruned model, per-layer widths, rationale log, report vs. baselines at equal measured "
      "memory, with tokens/s."),
    P(("Optional: ", "b"), "uniform 8- or 4-bit quantization of every method (bitsandbytes, T4)."),
    P(("Target not reached: ", "b"), "returns the last accepted plan and the reason."))
txt(s_sol, 6, P("Functional Requirements"))
txt(s_sol, 8,
    P(("1. ", "b"), "Probe each layer in parallel: narrower MLP, fewer attention groups, or dropped."),
    P(("2. ", "b"), "Agent selects the next experiment: probe, apply, start from FLAP, roll back or stop."),
    P(("3. ", "b"), "Validator enforces size, structure, quality and budget rules."),
    P(("4. ", "b"), "Full-model evaluation gates every applied step."),
    P(("5. ", "b"), "Save and reload models with per-layer widths; checkpoint and log every step."))
for sid in (5, 8):
    size(by_id(s_sol, sid), 16)
for sid in (3, 6):
    by_id(s_sol, sid).text_frame._txBody.find(qn("a:bodyPr")).set("anchor", "b")
txt(s_sol, 9, P("Source: author’s system architecture (ARCHITECTURE.md, ADR-001–007); FLAP reference implementation "
                "on GitHub."))
size(by_id(s_sol, 9), 11)
notes(s_sol, NOTES["sol"])

# 4 approach (architecture) ---------------------------------------------------------
txt(s_arch, 4, P("Technical Architecture"))
for r in by_id(s_arch, 4).text_frame.paragraphs[0].runs:
    r.font.bold = False
for sh in list(s_arch.shapes):
    if not sh.is_placeholder:
        sh._element.getparent().remove(sh._element)

EY, EH = 1.32, 0.55
rect(s_arch, "Entry label box", 0.75, EY, 2.4, EH, fill=NAVY)
text(s_arch, "Entry label", 0.87, EY, 2.2, EH, [[("Entry Point", {"b": True, "color": "FFFFFF", "size": 15})]],
     anchor=MSO_ANCHOR.MIDDLE)
rect(s_arch, "Entry box", 3.2, EY, 9.05, EH, fill=GOLD_BG)
text(s_arch, "Entry text", 3.35, EY, 8.8, EH,
     [[("Researcher → ", {"color": BODY, "size": 13}), ("pruneagent run", {"b": True, "color": INK, "size": 13}),
       (" (CLI / Colab notebook) · model, target, limits → ", {"color": BODY, "size": 13}),
       ("pruned model, rationale log, report", {"b": True, "color": INK, "size": 13}),
       ("; ", {"color": BODY, "size": 13}), ("resume", {"b": True, "color": INK, "size": 13}),
       (" after disconnect", {"color": BODY, "size": 13})]], anchor=MSO_ANCHOR.MIDDLE)
text(s_arch, "Tag", 0.75, 1.97, 2.8, 0.28,
     [[("LangGraph Agent Graph", {"b": True, "color": NAVY, "size": 13})]])
ROW_Y, ROW_H, W, STEP = 2.65, 1.6, 1.85, 2.0
mid = ROW_Y + ROW_H / 2
xs = [0.75 + i * STEP for i in range(6)]
# stacked cards behind node 1: parallel fan-out
for k, off in enumerate((0.16, 0.08)):
    rect(s_arch, f"Probe card {k+1}", xs[0] + off, ROW_Y - off, W, ROW_H, fill="F4F0FA", line=PURPLE, line_w=0.75)
nodes = [
    ("1", "Parallel Probes", "Width, attention-group and layer-drop error per layer", "FLAP scores, cached",
     PURPLE_BG, PURPLE, PURPLE),
    ("2", "LLM Planning", "Next experiment: probe, apply, roll back, stop", "LLM API", GOLD_BG, GOLD, GOLD_TX),
    ("3", "Plan Validation", "Size, structure and budget rules", "Python validator", TEAL_BG, TEAL, TEAL),
    ("4", "Structured Pruning", "Slice channels and groups; drop layers", "Rebuilt from dense", TEAL_BG, TEAL, TEAL),
    ("5", "Perplexity Gate", "Per-step and cumulative limits", "WikiText-2 validation", RED_BG, RED, RED),
    ("6", "Accept / Rollback", "Rationale logged per step", "LangGraph checkpoint", GREEN_BG, GREEN, GREEN),
]
for i, (n, t, d, tech, bg, acc, tc) in enumerate(nodes):
    x = xs[i]
    rect(s_arch, f"N{n} box", x, ROW_Y, W, ROW_H, fill=bg)
    badge(s_arch, f"N{n} number", x + 0.12, ROW_Y + 0.12, n, acc)
    text(s_arch, f"N{n} title", x + 0.12, ROW_Y + 0.5, W - 0.16, 0.34, [[(t, {"b": True, "color": tc, "size": 13})]])
    text(s_arch, f"N{n} text", x + 0.12, ROW_Y + 0.88, W - 0.2, 0.7, [[(d, {"color": BODY, "size": 12.5})]])
    text(s_arch, f"N{n} tech", x, ROW_Y + ROW_H + 0.08, W, 0.3,
         [[(tech, {"i": True, "color": "5A6878", "size": 12.5})]], align=PP_ALIGN.CENTER)
    if i < 5:
        line(s_arch, f"Arrow {i+1}", x + W, mid, xs[i + 1], mid)

top = 2.3
gx, ax = xs[5] + W / 2, xs[1] + W / 2
line(s_arch, "Loop up", gx, ROW_Y, gx, top, color=GREEN, arrow=False)
line(s_arch, "Loop across", gx, top, ax, top, color=GREEN, arrow=False)
line(s_arch, "Loop down", ax, top, ax, ROW_Y, color=GREEN)
text(s_arch, "Loop label", 5.2, top - 0.3, 5.0, 0.28,
     [[("next round: results + rationale feedback", {"i": True, "color": GREEN, "size": 13})]], align=PP_ALIGN.CENTER)

# final tests band
FY, FH = 4.72, 0.8
rect(s_arch, "Final label box", 0.75, FY, 2.4, FH, fill=NAVY)
text(s_arch, "Final label", 0.87, FY, 2.2, FH, [[("Final Evaluation", {"b": True, "color": "FFFFFF", "size": 15})]],
     anchor=MSO_ANCHOR.MIDDLE)
rect(s_arch, "Final box", 3.2, FY, 9.4, FH, fill=GREY_BG)
text(s_arch, "Final text", 3.35, FY, 9.15, FH,
     [[("Optional 8-/4-bit, then WikiText-2 test + 7 zero-shot tasks + MMLU + memory and tokens/s on T4 ",
        {"b": True, "color": INK, "size": 13}),
       ("vs. uniform Wanda-sp, FLAP, ShortGPT and a manual sweep at equal measured memory",
        {"color": BODY, "size": 13})]], anchor=MSO_ANCHOR.MIDDLE)

# runs on band
BY, BH = 5.62, 0.7
rect(s_arch, "Runs label box", 0.75, BY, 2.4, BH, fill=NAVY)
text(s_arch, "Runs label", 0.87, BY, 2.2, BH, [[("Infrastructure", {"b": True, "color": "FFFFFF", "size": 15})]],
     anchor=MSO_ANCHOR.MIDDLE)
outs = ["Mac · development", "Colab Pro · T4 / A100", "Drive or local · checkpoints", "Budget · $100 LLM, 200 CU"]
sx, sw, gap = 3.2, 2.3, 0.067
for i, label in enumerate(outs):
    x = sx + i * (sw + gap)
    rect(s_arch, f"Run {i+1}", x, BY, sw, BH, fill=GREEN_BG)
    text(s_arch, f"Run {i+1} text", x + 0.12, BY, sw - 0.2, BH, [[(label, {"b": True, "color": GREEN, "size": 14})]],
         anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
notes(s_arch, NOTES["arch"])

# 5 scope and risks -----------------------------------------------------------------
txt(s_risk, 4, P("Main risk: the agent doesn’t beat the benchmarks"))
for r in by_id(s_risk, 4).text_frame.paragraphs[0].runs:
    r.font.bold = False
text(s_risk, "Scope", 0.75, 1.72, 11.8, 0.62,
     [[("In scope: ", {"b": True, "color": GREEN, "size": 15}),
       ("Qwen3-0.6B / 1.7B · MLP channels, attention groups, whole layers · 20–30% · optional 8-/4-bit on T4",
        {"color": BODY, "size": 15})],
      [("Out of scope: ", {"b": True, "color": RED, "size": 15}),
       ("unstructured pruning · retraining · Mac or agent-chosen quantization · models >2B",
        {"color": BODY, "size": 15})]])
tshape = [s for s in s_risk.shapes if s.has_table][0]
tshape.top = Inches(2.42)
tbl = tshape.table
for col, w in zip(tbl.columns, (3.9, 2.8, 5.13)):
    col.width = Inches(w)
cells = [
    [("Risk", "b")], [("Impact", "b")], [("Mitigation", "b")],
    [("Agent doesn’t beat baselines", "b")], ["Below baseline quality"],
    [("Start from FLAP allocation; report negative result")],
    [("Structured pruning loses quality fast", "b")], [">2× perplexity at 20–30%"],
    [("Small steps; cumulative limit; lower target if FLAP fails")],
    [("Layer-local probes mispredict", "b")], [("Poor experiment choices")],
    [("Gate on full-model perplexity; validate probes in week 1")],
    [("FLAP code assumes LLaMA", "b")], [("Week-1 slip")],
    [("Prune attention by key/value group; custom loader")],
    [("Colab compute units exhausted", "b")], [("No GPU for final runs")],
    [("T4 for development; A100 (bf16) for final search; 200-CU cap")],
    [("Run time or cost overrun", "b")], [(">2 h or >$5 per run")],
    [("Qwen3-0.6B only")],
]
while len(tbl.rows) < len(cells) // 3:
    tbl._tbl.append(deepcopy(tbl._tbl.tr_lst[-1]))
for i, (c, runs) in enumerate(zip([c for r in tbl.rows for c in r.cells], cells)):
    set_paras(c.text_frame, [P(*runs)])
    size(c, 14 if i < 3 else 13)
for r in tbl.rows:
    r.height = Inches(0.5)
notes(s_risk, NOTES["risk"])

# 6 plan and success ------------------------------------------------------------------
txt(s_plan, 2, P("Project Plan and Success Criteria"))
for sid, t in [(4, "Week 1: Feasibility"), (6, "Week 2: Agent Graph"), (8, "Week 3: Tuning"), (10, "Week 4: Evaluation")]:
    txt(s_plan, sid, P((t, "b")))
for t, d, a, b in [
    (14, 15, "Feasibility Spike", "FLAP adapted to Qwen3; loader; one agent round end to end"),
    (20, 21, "Baseline Implementation", "Uniform Wanda-sp, FLAP, timed manual sweep"),
    (26, 27, "Agent Graph Build", "Parallel probes, planner, validator, gate; ShortGPT"),
    (32, 33, "Accept / Rollback Loop", "Midpoint: end-to-end run on Qwen3-0.6B"),
    (38, 39, "Prompt Tuning", "Tune agent instructions on 0.6B; freeze; add 8-/4-bit step"),
    (44, 45, "Final Evaluation & Demo", "Qwen3-1.7B vs. 4 baselines; memory and tokens/s on T4"),
]:
    txt(s_plan, t, P((a, "b")))
    txt(s_plan, d, P(b))
    size(by_id(s_plan, t), 14)
    size(by_id(s_plan, d), 12.5)
    by_id(s_plan, d).height = Inches(1.0)
for sid in (4, 6, 8, 10):
    size(by_id(s_plan, sid), 13)
delete(s_plan, 46, 47, 48, 49, 50, 51, 52, 53, 54)
SY, SH = 5.12, 1.55
rect(s_plan, "Success label box", 0.75, SY, 2.4, SH, fill=NAVY)
text(s_plan, "Success label", 0.87, SY, 2.2, SH,
     [[("Success Criteria", {"b": True, "color": "FFFFFF", "size": 15})]], anchor=MSO_ANCHOR.MIDDLE)
rect(s_plan, "Success box", 3.2, SY, 9.4, SH, fill=GREEN_BG)
text(s_plan, "Success text", 3.35, SY, 9.15, SH,
     [[("1. ", {"b": True, "color": GREEN, "size": 14}),
       ("Quality ≥ uniform Wanda-sp and FLAP at equal measured memory (floor)", {"color": INK, "size": 14})],
      [("2. ", {"b": True, "color": GREEN, "size": 14}),
       ("Fewer evaluations and GPU time than the manual sweep; zero manual decisions", {"color": INK, "size": 14})],
      [("3. ", {"b": True, "color": GREEN, "size": 14}),
       ("≤2 h on a T4 and ≤$5 per run; memory and tokens/s measured for every method", {"color": INK, "size": 14})],
      [("4. ", {"b": True, "color": GREEN, "size": 14}),
       ("A well-supported negative quality result is a valid outcome", {"color": INK, "size": 14})]],
     anchor=MSO_ANCHOR.MIDDLE)
txt(s_plan, 55, P("Effort target ratio set after week-1 timing. Descoping order: quantization step, coarser manual "
                  "sweep, 30% target, then Qwen3-0.6B only. Source: author’s system architecture."))
by_id(s_plan, 55).top = Inches(6.72)
size(by_id(s_plan, 55), 11)
notes(s_plan, NOTES["plan"])

# 7 appendix: speaker notes ------------------------------------------------------------
txt(s_app, 5, P("Appendix: Speaker Notes"))
order = [("2 · Problem and Audience. ", "prob"), ("3 · Pruning Today. ", "today"), ("4 · Proposed Solution. ", "sol"),
         ("5 · Technical Architecture. ", "arch"), ("6 · Scope and Risks. ", "risk"),
         ("7 · Plan and Success Criteria. ", "plan")]
txt(s_app, 2, *[P((h, "b"), NOTES[k]) for h, k in order])
body = by_id(s_app, 2)
body.left, body.top, body.width, body.height = Inches(0.75), Inches(1.5), Inches(11.83), Inches(5.2)
body.text_frame._txBody.find(qn("a:bodyPr")).clear()
size(body, 12.5)
for para in body.text_frame.paragraphs:
    para.space_after = Pt(6)
notes(s_app, "Appendix; not presented.")

# 8 references -------------------------------------------------------------------------
txt(s_ref, 5, P("References"))
refs = [
    [("An, Yongqi, et al.", "b"), " “Fluctuation-Based Adaptive Structured Pruning for Large Language Models.” ",
     ("AAAI Conference on Artificial Intelligence", "i"), ", 2024. Code: github.com/CASIA-IVA-Lab/FLAP."],
    [("Dettmers, Tim, et al.", "b"), " “LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.” ",
     ("Advances in Neural Information Processing Systems", "i"), ", 2022."],
    [("Fang, Gongfan, et al.", "b"), " “DepGraph: Towards Any Structural Pruning.” ",
     ("IEEE/CVF Conference on Computer Vision and Pattern Recognition", "i"), ", 2023. Code: Torch-Pruning."],
    [("He, Yihui, et al.", "b"), " “AMC: AutoML for Model Compression and Acceleration on Mobile Devices.” ",
     ("European Conference on Computer Vision", "i"), ", 2018."],
    [("Kodathala, Sai Varun, and Rakesh Vunnam.", "b"), " “LLMs can Compress LLMs: Adaptive Pruning by Agents.” ",
     ("arXiv", "i"), ":2601.09694, 2026."],
    [("Ma, Xinyin, Gongfan Fang, and Xinchao Wang.", "b"), " “LLM-Pruner: On the Structural Pruning of Large "
     "Language Models.” ", ("Advances in Neural Information Processing Systems", "i"), ", 2023."],
    [("Men, Xin, et al.", "b"), " “ShortGPT: Layers in Large Language Models are More Redundant Than You Expect.” ",
     ("arXiv", "i"), ":2403.03853, 2024."],
    [("Sreenivas, Sharath Turuvekere, et al.", "b"), " “LLM Pruning and Distillation in Practice: The Minitron "
     "Approach.” ", ("arXiv", "i"), ":2408.11796, 2024. Tool: NVIDIA Model Optimizer."],
    [("Sun, Mingjie, et al.", "b"), " “A Simple and Effective Pruning Approach for Large Language Models.” ",
     ("International Conference on Learning Representations", "i"), ", 2024."],
    [("Yuan, Xin, et al.", "b"), " “FastForward Pruning: Efficient LLM Pruning via Single-Step Reinforcement "
     "Learning.” ", ("arXiv", "i"), ":2511.18977, 2025."],
]
txt(s_ref, 2, *[P(*r) for r in refs])
notes(s_ref, "Reference slide; not presented.")

# order: approach before scope and risks
ids = prs.slides._sldIdLst
lst = list(ids)
ids.remove(lst[4])
ids.insert(3, lst[4])
# pruning-today slide (added last) goes right after the problem slide
lst = list(ids)
ids.remove(lst[-1])
ids.insert(2, lst[-1])

prs.save(sys.argv[2])
print("saved", sys.argv[2])
