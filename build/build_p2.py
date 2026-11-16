"""PruneAgent proposal deck. Usage: python build_p2.py base.pptx ../PruneAgent_Proposal.pptx"""
import sys
from copy import deepcopy

from pptx import Presentation
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

from helpers import A, P, by_id, delete, notes, set_paras, txt
from slides_p2 import add_who_slide, draw_arch

prs = Presentation(sys.argv[1])
s1, s2, s3, s4, s5, s6, s7, s8 = prs.slides

# 1 title ---------------------------------------------------------------------
txt(s1, 2, P("An AI agent that prunes language models, layer by layer"))
txt(s1, 3, P("Steph Xu"), P("1.125 Agentic Engineering & Architecture"), P("Course project, 4 weeks"))
notes(s1, "I’m building a system where an AI agent decides how much to cut from each layer of a language model, "
          "checks the damage after every step, and undoes bad steps.")

# 2 context -------------------------------------------------------------------
txt(s2, 10, P("Pruning shrinks language models, but every layer gets the same cut"))
txt(s2, 2, P("The Landscape"))
txt(s2, 4, P("The Gap"))
txt(s2, 6, P("The Proposal"))
txt(s2, 11,
    P(("Pruning removes weights to make a model smaller.", "b")),
    P("Wanda", ("1", "s"), " prunes a language model in minutes, without retraining, but removes the same share "
      "from every layer."))
txt(s2, 5,
    P(("Layers differ, so equal cuts waste quality.", "b")),
    P("OWL", ("2", "s"), " sets each layer’s cut with a fixed rule."),
    P("Learning-based search", ("3,4", "s"), " adapts, but needs many GPU-hours."),
    P("An AI agent did it in 2026,", ("5", "s"), " with no code and weak comparisons."))
txt(s2, 7,
    P(("An open system where an AI agent picks each layer’s cut, tests the damage, and undoes bad steps.", "b")),
    P("It runs on a free Colab GPU and compares every result with standard methods at the same size."))
fn = deepcopy(by_id(s3, 9)._element)
max_id = max(int(e.get("id")) for e in s2.shapes._spTree.iter(qn("p:cNvPr")))
fn.find(".//" + qn("p:cNvPr")).set("id", str(max_id + 1))
fn.find(".//" + qn("p:cNvPr")).set("name", "Footnotes")
s2.shapes._spTree.append(fn)
f = [x for x in s2.shapes if x.name == "Footnotes"][0]
f.left, f.top, f.width, f.height = Inches(0.75), Inches(6.52), Inches(11.0), Inches(0.28)
set_paras(f.text_frame, [P(("1", "s"), " Sun et al., 2024 (Wanda).  ", ("2", "s"), " Yin et al., 2024 (OWL).  ",
                           ("3", "s"), " He et al., 2018 (AMC).  ", ("4", "s"), " Yuan et al., 2025 (FastForward).  ",
                           ("5", "s"), " Kodathala & Vunnam, 2026. Full references at the end.")])
notes(s2, "Pruning removes weights so a model takes less memory. Wanda, the standard quick method, cuts the same "
          "share from every layer. But layers differ. OWL sets each layer’s cut with a fixed rule; learning-based "
          "search adapts, but takes many GPU-hours. In January, a paper had an AI agent choose the cuts, but it "
          "released no code and skipped the obvious comparisons. I’ll build it openly, on free hardware, and compare "
          "every result with the standard methods.")

# 3 what the system must do ----------------------------------------------------
txt(s3, 2, P("What the system must do"))
txt(s3, 3, P(("In and out", "b")))
txt(s3, 5,
    P(("You give: ", "b"), "an open model, such as Qwen3-1.7B, and a target: “remove 50% of the weights.”"),
    P(("You set: ", "b"), "how much quality may drop in one step (default 15%) and a cost cap."),
    P(("You get: ", "b"), "the pruned model, the cut chosen for each layer, and a report comparing it with three "
      "standard methods at the same size."),
    P(("If the target can’t be reached: ", "b"), "it stops at the last good step and says why."))
txt(s3, 6, P(("Requirements", "b")))
txt(s3, 8,
    P(("1. ", "b"), "Measure each layer once before pruning."),
    P(("2. ", "b"), "Let an AI agent propose each layer’s next cut, or stop."),
    P(("3. ", "b"), "Check every proposal against fixed rules."),
    P(("4. ", "b"), "Test after every step; undo any step that loses too much quality."),
    P(("5. ", "b"), "Compare with uniform Wanda, OWL and magnitude pruning at the same size."),
    P(("6. ", "b"), "Pick up again after a Colab disconnect; log every step."),
    P(("Target: ", "b"), "beat uniform Wanda on both final models at 50% of weights removed."))
txt(s3, 9, P("Scope: open models up to about 2 billion parameters; 40–60% of weights removed; no retraining. "
             "Not in scope: making models faster, models above 4 billion parameters."))
for sid in (3, 6):
    by_id(s3, sid).text_frame._txBody.find(A("bodyPr")).set("anchor", "b")
for sid, size in ((5, 16), (8, 14)):
    for para in by_id(s3, sid).text_frame.paragraphs:
        for r in para.runs:
            r.font.size = Pt(size)
notes(s3, "You give it a model and a target, like removing half the weights. You set how much quality may drop in one "
          "step, and a cost cap. You get back the pruned model, the cut for each layer, and a report that compares it "
          "with three standard methods at the same size. The target: beat uniform Wanda on both final models.")

# 4 data and tests ------------------------------------------------------------
txt(s4, 4, P("Data and tests: all public, all standard"))
tbl = [s for s in s4.shapes if s.has_table][0].table
rows = [
    [("Source", "b")], [("What it is", "b")], [("How I use it", "b")],
    [("Qwen3-0.6B, 1.7B", "b")], ["Open language models (Apache 2.0)"],
    ["0.6B to build and tune the agent; 1.7B for the final runs"],
    [("SmolLM2-1.7B", "b")], ["Open model from a different family (Apache 2.0)"],
    ["Final runs: checks the agent isn’t tuned to one family"],
    [("C4", "b")], ["Web text"], ["128 samples to measure each layer before pruning"],
    [("WikiText-2", "b")], ["Wikipedia text"],
    ["How well the model still predicts text: a quick test every step, the full test at the end"],
    [("Benchmarks", "b")], ["5 reasoning and common-sense sets, plus MMLU knowledge questions"],
    ["Final tests for the agent and the three standard methods"],
]
for c, runs in zip([c for r in tbl.rows for c in r.cells], rows):
    set_paras(c.text_frame, [P(*runs)])
notes(s4, "All data is public and standard. I build and tune on the smallest Qwen3 model, then run the final tests on "
          "Qwen3-1.7B and on SmolLM2, a different family, so the agent isn’t tuned to one model. A little web text "
          "measures each layer. Wikipedia text gives the quick test after every step. Standard benchmarks, including "
          "knowledge questions, give the final verdict, because a model can predict text well and still forget facts.")

# 5 architecture ----------------------------------------------------------------
txt(s5, 4, P("Technical architecture"))
for sh in list(s5.shapes):
    if not sh.is_placeholder:
        sh._element.getparent().remove(sh._element)
draw_arch(s5)
notes(s5, "One run. First the system measures every layer. Then each round, the AI agent proposes how much more to cut "
          "from each layer. Fixed rules check the proposal. The model is pruned from its original weights, so undo is "
          "exact, and a quick test checks quality. If quality drops more than 15 percent, the step is undone and the "
          "agent is told why. At the end come the full tests, next to three standard methods at the same size.")

# 6 timeline ------------------------------------------------------------------
txt(s6, 2, P("Proposed timeline"))
for sid, t in [(4, "Week 1: Baselines"), (6, "Week 2: Agent loop"), (8, "Week 3: Tune"), (10, "Week 4: Finish")]:
    txt(s6, sid, P((t, "b")))
for t, d, a, b in [
    (14, 15, "Set up", "Mac and Colab; load the models; adapt the Wanda code"),
    (20, 21, "Standard methods", "Uniform Wanda, OWL, magnitude; quick and final tests"),
    (26, 27, "Agent and checks", "Layer profile, agent instructions, fixed rules"),
    (32, 33, "Keep or undo", "Test after each step; undo; save to Drive"),
    (38, 39, "Tune on the small model", "Adjust instructions on Qwen3-0.6B, then freeze them"),
    (44, 45, "Final runs and demo", "Qwen3-1.7B and SmolLM2 vs. 3 methods; report; demo"),
]:
    txt(s6, t, P((a, "b")))
    txt(s6, d, P(b))
txt(s6, 50, P(("Test results feed fixes back into steps 3 and 4", "i")))
delete(s6, 51, 52, 53)
b = by_id(s6, 54)
b.left, b.width = Inches(0.75), Inches(11.8)
set_paras(b.text_frame, [P(("Budget: ", "bi"), ("AI agent well under $1 per run; spending stops at $100. "
                                               "GPU: free Colab T4; the Mac for development.", "i"))])
txt(s6, 55, P("Source: my architecture plan. If I fall behind, I cut SmolLM2 first, then the knowledge questions, "
              "then magnitude pruning."))
notes(s6, "Week one builds the standard methods first, so I know the tests work. Week two adds the agent and the "
          "keep-or-undo loop. Week three tunes the agent’s instructions on the smallest model only. Week four runs the "
          "final tests on two models and prepares the demo. The AI costs well under a dollar per run, and the GPU is "
          "free. Thank you.")

# 7 references ------------------------------------------------------------------
txt(s7, 5, P("References"))
refs = [
    [("He, Yihui, et al.", "b"), " “AMC: AutoML for Model Compression and Acceleration on Mobile Devices.” ",
     ("European Conference on Computer Vision", "i"), ", 2018."],
    [("Jafari, Sadegh, et al.", "b"), " “ProfilingAgent: Profiling-Guided Agentic Reasoning for Adaptive Model "
     "Optimization.” ", ("arXiv", "i"), ":2509.05584, 2025."],
    [("Kodathala, Sai Varun, and Rakesh Vunnam.", "b"), " “LLMs can Compress LLMs: Adaptive Pruning by Agents.” ",
     ("arXiv", "i"), ":2601.09694, 2026."],
    [("Sun, Mingjie, et al.", "b"), " “A Simple and Effective Pruning Approach for Large Language Models.” ",
     ("International Conference on Learning Representations", "i"), ", 2024."],
    [("Yin, Lu, et al.", "b"), " “Outlier Weighed Layerwise Sparsity (OWL): A Missing Secret Sauce for Pruning LLMs "
     "to High Sparsity.” ", ("International Conference on Machine Learning", "i"), ", 2024."],
    [("Yuan, Xin, et al.", "b"), " “FastForward Pruning: Efficient LLM Pruning via Single-Step Reinforcement "
     "Learning.” ", ("arXiv", "i"), ":2511.18977, 2025."],
]
txt(s7, 2, *[P(*r) for r in refs])
notes(s7, "Reference slide; not presented.")
notes(s8, "Happy to take questions.")
add_who_slide(prs, s5, notes)

prs.save(sys.argv[2])
print("saved", sys.argv[2])
