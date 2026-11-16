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
             "sensitivity analysis researchers do by hand when they prune an open language model.",
    "prob": "The users are ML researchers and engineers compressing open LLMs for single-GPU deployment. The tools to "
            "prune already exist: Wanda prunes a model in minutes, and OWL sets non-uniform levels with a fixed "
            "heuristic. What remains manual is the sensitivity analysis around them: sweeping layers, reading the "
            "results, deciding the next experiment, and choosing per-layer levels. RL search like AMC and FastForward "
            "automates the choice but costs GPU-hours, and the one LLM-agent paper this year predicts levels without "
            "automating the experiment workflow, and released no code. That workflow is what I’m automating.",
    "sol": "The value of the agentic architecture is that the agent decides which experiment to run next, using "
           "existing pruning tools, instead of a researcher. Input: a Qwen3 model and a target size, measured as "
           "memory. Constraints: a 15 percent perplexity limit per step against the previous step, a cumulative limit "
           "against the dense model, and five dollars per run. Output: the pruned model, per-layer sparsity, a "
           "rationale for every decision, and a report against baselines at equal measured size.",
    "arch": "Here is the technical architecture. The entry point is a single command, pruneagent run, from the CLI "
            "or a Colab notebook: the researcher gives the model, target memory and limits, and gets back the pruned "
            "model, the rationale log and the report; resume restarts after a disconnect. LangGraph fans out layer-local sensitivity probes in parallel, each "
            "on one layer with cached activations, and a reducer merges them into one sensitivity table. The agent "
            "then chooses the next experiment: probe further, apply a plan, start from OWL, roll back or stop. The "
            "validator enforces size, quality and budget rules; Wanda pruning runs sequentially from the dense weights; "
            "and the perplexity gate checks both the per-step and the cumulative limit on the WikiText-2 validation "
            "split. Final evaluation uses the test split, Wanda’s seven zero-shot tasks and MMLU.",
    "risk": "Scope is narrow on purpose: Qwen3-0.6B and 1.7B, Wanda unstructured pruning only, post-training. "
            "Quantization is out. The main risk is that the agent doesn’t beat the benchmarks; in that case it starts "
            "from the OWL allocation, and a well-supported negative result is still a valid outcome. Two risks are new: "
            "layer-local probes may not predict whole-model sensitivity, so every applied step is gated on full-model "
            "perplexity; and sparse weights only save memory in a sparse format, so I report measured file size and "
            "peak GPU memory.",
    "plan": "Week 1 is a feasibility spike: Wanda on Qwen3-0.6B, the baselines including a timed manual sweep, and one "
            "agent round end to end. Week 2 builds the agent graph; the midpoint is an end-to-end run on 0.6B. Week 3 "
            "tunes the instructions on 0.6B only. Week 4 runs final evaluation on Qwen3-1.7B. Success: quality at least "
            "equal to uniform Wanda and OWL at equal measured memory, with fewer evaluations than the manual sweep and "
            "no manual decisions.",
}

# 1 title -----------------------------------------------------------------------
txt(s_title, 2, P("PruneAgent: Agentic Sensitivity Analysis for LLM Pruning"))
txt(s_title, 3, P("Steph Xu"), P("1.125 Agentic Engineering & Architecture"), P("Term Project Proposal (PS06)"))
notes(s_title, NOTES["title"])

# 2 problem and audience ----------------------------------------------------------
txt(s_prob, 10, P("Layer-wise sensitivity analysis is a manual bottleneck in LLM pruning"))
txt(s_prob, 2, P("The Target Users"))
txt(s_prob, 4, P("The Bottleneck"))
txt(s_prob, 6, P("The Gap"))
txt(s_prob, 11,
    P(("ML researchers and engineers compressing open LLMs.", "b")),
    P("Target: single-GPU deployment, where memory is the binding constraint."),
    P("Students and small teams without GPU clusters."))
txt(s_prob, 5,
    P(("Per-layer sensitivity analysis is manual and compute-intensive.", "b")),
    P("Existing tools (Wanda", ("1", "s"), ", OWL", ("2", "s"), ") prune well."),
    P("Choosing per-layer levels still takes repeated sweeps, interpretation and judgment by a researcher."))
txt(s_prob, 7,
    P(("Automation is heuristic, costly or partial.", "b")),
    P("OWL", ("2", "s"), ": fixed outlier-based heuristic."),
    P("RL search (AMC", ("3", "s"), ", FastForward", ("4", "s"), "): GPU-hours of policy training."),
    P("LLM agent (Kodathala & Vunnam", ("5", "s"), "): predicts levels, not the experiment workflow; no public code."))
for sid in (11, 5, 7):
    size(by_id(s_prob, sid), 16)
footnote(s_prob, s_sol, [P(("1", "s"), " Sun et al., 2024.  ", ("2", "s"), " Yin et al., 2024.  ",
                           ("3", "s"), " He et al., 2018.  ", ("4", "s"), " Yuan et al., 2025.  ",
                           ("5", "s"), " Kodathala & Vunnam, 2026.  Full references on the last slide.")])
notes(s_prob, NOTES["prob"])

# 3 proposed solution -------------------------------------------------------------
txt(s_sol, 2, P("The Solution: Agentic Sensitivity Workflow"))
text(s_sol, "Value statement", 0.75, 1.2, 11.8, 0.62,
     [[("PruneAgent automates the researcher’s sensitivity-analysis workflow: an LLM agent decides which experiments "
        "to run with existing pruning tools, reaching conventional-method quality with fewer evaluations and no "
        "manual decisions.", {"i": True, "color": NAVY, "size": 14})]])
txt(s_sol, 3, P("Inputs, Constraints and Outputs"))
txt(s_sol, 5,
    P(("Input: ", "b"), "Qwen3 model; target size as measured memory."),
    P(("Constraints: ", "b"), "≤15% perplexity increase per step (vs. previous step); cumulative limit vs. dense "
      "model; $5 per run."),
    P(("Output: ", "b"), "pruned model, per-layer sparsity, decision rationale log, report vs. baselines at equal "
      "measured size."),
    P(("Target not reached: ", "b"), "returns the last accepted plan with the stop reason."))
txt(s_sol, 6, P("Functional Requirements"))
txt(s_sol, 8,
    P(("1. ", "b"), "Run layer-local sensitivity probes in parallel."),
    P(("2. ", "b"), "Agent selects the next experiment: probe, apply, start from OWL, roll back or stop."),
    P(("3. ", "b"), "Validator enforces size, quality and budget rules."),
    P(("4. ", "b"), "Full-model evaluation gates every applied step."),
    P(("5. ", "b"), "Checkpoint each step to Google Drive or local drive; log its rationale."))
for sid in (5, 8):
    size(by_id(s_sol, sid), 16)
for sid in (3, 6):
    by_id(s_sol, sid).text_frame._txBody.find(qn("a:bodyPr")).set("anchor", "b")
txt(s_sol, 9, P("Source: author’s system architecture (ARCHITECTURE.md, ADR-001–006); Wanda and OWL reference "
                "implementations on GitHub."))
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
       (" (CLI / Colab notebook) · model, target memory, limits → ", {"color": BODY, "size": 13}),
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
    ("1", "Parallel Probes", "Layer-local error per layer; reducer merges", "Cached activations",
     PURPLE_BG, PURPLE, PURPLE),
    ("2", "LLM Planning", "Next experiment: probe, apply, roll back, stop", "LLM API", GOLD_BG, GOLD, GOLD_TX),
    ("3", "Plan Validation", "Size, quality and budget rules", "Python validator", TEAL_BG, TEAL, TEAL),
    ("4", "Wanda Pruning", "Sequential, from dense weights", "Wanda reference code", TEAL_BG, TEAL, TEAL),
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
     [[("WikiText-2 test + 7 zero-shot tasks + MMLU ", {"b": True, "color": INK, "size": 14}),
       ("(lm-eval harness) vs. uniform Wanda, OWL and a manual sensitivity sweep at equal measured memory",
        {"color": BODY, "size": 14})]], anchor=MSO_ANCHOR.MIDDLE)

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
       ("Qwen3-0.6B / 1.7B · Wanda unstructured pruning · post-training only", {"color": BODY, "size": 15})],
      [("Out of scope: ", {"b": True, "color": RED, "size": 15}),
       ("quantization · other model families · inference speed-up · models >2B", {"color": BODY, "size": 15})]])
tshape = [s for s in s_risk.shapes if s.has_table][0]
tshape.top = Inches(2.42)
tbl = tshape.table
for col, w in zip(tbl.columns, (3.9, 2.8, 5.13)):
    col.width = Inches(w)
cells = [
    [("Risk", "b")], [("Impact", "b")], [("Mitigation", "b")],
    [("Agent doesn’t beat benchmarks", "b")], ["Below baseline quality"],
    ["Start from OWL allocation; report negative result"],
    [("Layer-local probes mispredict", "b")], ["Poor experiment choices"],
    ["Gate on full-model perplexity; validate probes in week 1"],
    [("Sparse weights don’t reduce memory", "b")], ["Size claim unsupported"],
    ["Report sparse-format file size and peak GPU memory"],
    [("Colab compute units exhausted", "b")], ["No GPU for final runs"],
    ["T4 for development; A100 for final runs only; 200-CU cap"],
    [("No bf16 support on T4", "b")], ["Numerical errors in some layers"],
    ["Final runs on A100 (bf16); fp32 fallback on T4"],
    [("Run time or cost overrun", "b")], [">2 h or >$5 per run"],
    ["Qwen3-0.6B only"],
]
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
    (14, 15, "Feasibility Spike", "Wanda on Qwen3-0.6B; one agent round end to end"),
    (20, 21, "Baseline Implementation", "Uniform Wanda, OWL, timed manual sweep"),
    (26, 27, "Agent Graph Build", "Parallel probes, planner, validator, gate"),
    (32, 33, "Accept / Rollback Loop", "Midpoint: end-to-end run on Qwen3-0.6B"),
    (38, 39, "Prompt Tuning", "Tune agent instructions on 0.6B; freeze"),
    (44, 45, "Final Evaluation & Demo", "Qwen3-1.7B vs. 3 baselines"),
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
       ("Quality ≥ uniform Wanda and OWL at equal measured memory (floor)", {"color": INK, "size": 14})],
      [("2. ", {"b": True, "color": GREEN, "size": 14}),
       ("Fewer evaluations and GPU time than the manual sweep; zero manual decisions", {"color": INK, "size": 14})],
      [("3. ", {"b": True, "color": GREEN, "size": 14}),
       ("≤2 h on a T4 and ≤$5 per run; resumes after interruption", {"color": INK, "size": 14})],
      [("4. ", {"b": True, "color": GREEN, "size": 14}),
       ("A well-supported negative quality result is a valid outcome", {"color": INK, "size": 14})]],
     anchor=MSO_ANCHOR.MIDDLE)
txt(s_plan, 55, P("Effort target ratio set after week-1 timing. Descoping order: coarser manual sweep (layer groups), "
                  "then Qwen3-0.6B only. Source: author’s system architecture."))
by_id(s_plan, 55).top = Inches(6.72)
size(by_id(s_plan, 55), 11)
notes(s_plan, NOTES["plan"])

# 7 appendix: speaker notes ------------------------------------------------------------
txt(s_app, 5, P("Appendix: Speaker Notes"))
order = [("2 · Problem and Audience. ", "prob"), ("3 · Proposed Solution. ", "sol"), ("4 · Technical Architecture. ", "arch"),
         ("5 · Scope and Risks. ", "risk"), ("6 · Plan and Success Criteria. ", "plan")]
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
    [("He, Yihui, et al.", "b"), " “AMC: AutoML for Model Compression and Acceleration on Mobile Devices.” ",
     ("European Conference on Computer Vision", "i"), ", 2018."],
    [("Jafari, Sadegh, et al.", "b"), " “ProfilingAgent: Profiling-Guided Agentic Reasoning for Adaptive Model "
     "Optimization.” ", ("arXiv", "i"), ":2509.05584, 2025."],
    [("Kodathala, Sai Varun, and Rakesh Vunnam.", "b"), " “LLMs can Compress LLMs: Adaptive Pruning by Agents.” ",
     ("arXiv", "i"), ":2601.09694, 2026."],
    [("Sun, Mingjie, et al.", "b"), " “A Simple and Effective Pruning Approach for Large Language Models.” ",
     ("International Conference on Learning Representations", "i"), ", 2024. Code: github.com/locuslab/wanda."],
    [("Yin, Lu, et al.", "b"), " “Outlier Weighed Layerwise Sparsity (OWL): A Missing Secret Sauce for Pruning LLMs "
     "to High Sparsity.” ", ("International Conference on Machine Learning", "i"), ", 2024. Code: github.com/luuyin/OWL."],
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

prs.save(sys.argv[2])
print("saved", sys.argv[2])
