"""PruneAgent proposal: architecture diagram and 'who builds what' slide, native shapes in the deck palette."""
from copy import deepcopy

from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn

from arch_diagram import (GOLD, GOLD_BG, GOLD_TX, GREEN, GREEN_BG, GREY_BG, INK, NAVY, RED, RED_BG, TEAL, TEAL_BG,
                          line, node, rect, text)

PURPLE, PURPLE_BG = "5B3F8C", "ECE6F5"


def draw_arch(slide):
    text(slide, "Tag", 0.75, 1.62, 8.0, 0.3,
         [[("One pruning run, built with LangGraph (a tool for AI workflows)", {"b": True, "color": NAVY, "size": 13})]])
    ROW_Y, ROW_H, W, STEP = 2.75, 1.3, 1.6, 1.95
    mid = ROW_Y + ROW_H / 2
    xs = [0.75 + i * STEP for i in range(6)]
    nodes = [
        ("Profile", "Measure each layer once: how sensitive, how many outliers", PURPLE_BG, PURPLE, None),
        ("Agent proposes", "AI picks how much more to cut from each layer, or stops", GOLD_BG, GOLD, GOLD_TX),
        ("Check plan", "Fixed rules; a broken rule goes back to the agent", TEAL_BG, TEAL, None),
        ("Prune", "Wanda, from the original weights, at the planned amounts", TEAL_BG, TEAL, None),
        ("Quick test", "How well the model still predicts text (perplexity)", RED_BG, RED, None),
        ("Keep or undo", "Quality down more than 15 %? Undo and tell the agent", GREEN_BG, GREEN, None),
    ]
    for i, (x, (t, d, bg, acc, tc)) in enumerate(zip(xs, nodes)):
        node(slide, f"N{i+1}", x, ROW_Y, W, ROW_H, i + 1, t, d, bg, acc, tc)
        if i < 5:
            line(slide, f"Arrow {i+1}", x + W, mid, xs[i + 1], mid)

    # return loop: keep/undo -> agent
    top = 2.15
    gx, ax = xs[5] + W / 2, xs[1] + W / 2
    line(slide, "Loop up", gx, ROW_Y, gx, top, color=GREEN, arrow=False)
    line(slide, "Loop across", gx, top, ax, top, color=GREEN, arrow=False)
    line(slide, "Loop down", ax, top, ax, ROW_Y, color=GREEN)
    text(slide, "Loop label", 5.0, top + 0.05, 4.0, 0.28,
         [[("next round, with the result of this one", {"i": True, "color": GREEN, "size": 10})]], align=PP_ALIGN.CENTER)

    # final tests
    FX, FY, FW, FH = xs[4], 4.6, xs[5] + W - xs[4], 0.9
    rect(slide, "Final box", FX, FY, FW, FH, fill=GREY_BG)
    from arch_diagram import badge
    badge(slide, "Final number", FX + 0.1, FY + 0.1, 7, "595959")
    text(slide, "Final title", FX + 0.5, FY + 0.08, FW - 0.6, 0.32,
         [[("Final tests and report", {"b": True, "color": INK, "size": 12})]], anchor=MSO_ANCHOR.MIDDLE)
    text(slide, "Final text", FX + 0.12, FY + 0.46, FW - 0.22, 0.4,
         [[("Full text test, 5 benchmarks, knowledge questions", {"color": "404040", "size": 10.5})]])
    line(slide, "Down to final", gx, ROW_Y + ROW_H, gx, FY)
    text(slide, "Done label", xs[3] + 0.1, ROW_Y + ROW_H + 0.05, gx - xs[3] - 0.2, 0.28,
         [[("target reached, agent stops, or out of rounds", {"i": True, "color": "5A6878", "size": 9})]], align=PP_ALIGN.RIGHT)

    # baselines feed the final tests
    BX, BW = xs[1], xs[4] - xs[1] - 0.35
    rect(slide, "Base box", BX, FY + 0.08, BW, FH - 0.16, fill="FFFFFF", line=PURPLE, dash=True, line_w=1.0)
    text(slide, "Base text", BX + 0.12, FY + 0.08, BW - 0.24, FH - 0.16,
         [[("Same size, standard methods: ", {"b": True, "color": PURPLE, "size": 10.5}),
           ("uniform Wanda, OWL, magnitude pruning", {"color": "404040", "size": 10.5})]], anchor=MSO_ANCHOR.MIDDLE)
    line(slide, "Base to final", BX + BW, FY + FH / 2, FX, FY + FH / 2, color=PURPLE, w=1.0, dash=True)

    # what you get
    BY, BH = 5.85, 0.68
    rect(slide, "Band label box", 0.75, BY, 2.75, BH, fill=NAVY)
    text(slide, "Band label", 0.87, BY + 0.04, 2.55, BH - 0.08,
         [[("What you get", {"b": True, "color": "FFFFFF", "size": 12})],
          [("for every run", {"color": "FFFFFF", "size": 10})]], anchor=MSO_ANCHOR.MIDDLE)
    outs = ["Pruned model files", "Per-layer plan and chart", "Scores vs. 3 standard methods", "Rounds, undos and AI cost"]
    sx, sw, gap = 3.6, 2.2, 0.06
    for i, label in enumerate(outs):
        x = sx + i * (sw + gap)
        rect(slide, f"Out {i+1}", x, BY, sw, BH, fill=GREEN_BG)
        text(slide, f"Out {i+1} text", x + 0.15, BY, sw - 0.25, BH,
             [[(label, {"b": True, "color": GREEN, "size": 11})]], anchor=MSO_ANCHOR.MIDDLE)


def add_who_slide(prs, after_slide, notes_fn):
    s = prs.slides.add_slide(after_slide.slide_layout)
    have = {sh.placeholder_format.type for sh in s.placeholders}
    for sh in after_slide.placeholders:
        if sh.placeholder_format.type not in have:
            s.shapes._spTree.append(deepcopy(sh._element))
    ids = prs.slides._sldIdLst
    new = ids[-1]
    rid = prs.part.relate_to(after_slide.part,
                             "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide")
    idx = list(ids).index([e for e in ids if e.get(qn("r:id")) == rid][0])
    ids.remove(new)
    ids.insert(idx + 1, new)

    title = [sh for sh in s.placeholders if sh.placeholder_format.type == 1][0]
    title.text_frame.text = "Who builds what, and with which tools"
    for r in title.text_frame.paragraphs[0].runs:
        r.font.bold = False

    text(s, "Who lane", 0.75, 1.62, 8, 0.3,
         [[("Who does what", {"b": True, "color": NAVY, "size": 13}),
           ("  no hand-made data; all models and tests are public", {"color": "5A6878", "size": 11})]])
    who = [
        ("Me", "Write the agent’s instructions. Tune them on the smallest model only. Make the design decisions.",
         GOLD_BG, GOLD_TX),
        ("Coding agent", "Writes the code: profiler, plan checker, pruner (from the Wanda code), tests, loop and report.",
         TEAL_BG, TEAL),
        ("The system, automatically", "Profiles, prunes, tests, keeps or undoes, runs the standard methods and final "
                                      "tests, and logs every step.", GREEN_BG, GREEN),
    ]
    x0, w, gap, y, h = 0.75, 3.85, 0.14, 2.02, 1.55
    for i, (t, d, bg, tc) in enumerate(who):
        x = x0 + i * (w + gap)
        rect(s, f"Who {i+1} box", x, y, w, h, fill=bg)
        text(s, f"Who {i+1} title", x + 0.15, y + 0.12, w - 0.3, 0.32, [[(t, {"b": True, "color": tc, "size": 13})]])
        text(s, f"Who {i+1} text", x + 0.15, y + 0.5, w - 0.3, h - 0.55, [[(d, {"color": "404040", "size": 12.5})]])

    text(s, "Stack lane", 0.75, 4.45, 8, 0.3,
         [[("Tools I build with", {"b": True, "color": NAVY, "size": 13}),
           ("  each result records which versions made it", {"color": "5A6878", "size": 11})]])
    tools = [
        ("Git + GitHub", "Saves every change; each result notes which version made it"),
        ("uv", "Fixes the exact version of every Python package"),
        ("PyTorch", "Loads and runs the open models (with Transformers)"),
        ("lm-eval harness", "Runs the standard benchmark tests"),
        ("Colab T4 + Drive", "Free GPU; saves after every step, so a disconnect loses one step"),
        ("Mac + cost cap", "Develop on the small model; AI spending stops at $100"),
    ]
    x0, w, gap, y, h = 0.75, 1.9, 0.086, 4.85, 1.4
    for i, (t, d) in enumerate(tools):
        x = x0 + i * (w + gap)
        rect(s, f"Tool {i+1} box", x, y, w, h, fill=GREEN_BG)
        text(s, f"Tool {i+1} title", x + 0.12, y + 0.12, w - 0.22, 0.3, [[(t, {"b": True, "color": GREEN, "size": 11})]])
        text(s, f"Tool {i+1} text", x + 0.12, y + 0.46, w - 0.22, h - 0.5, [[(d, {"color": "404040", "size": 10})]])

    notes_fn(s, "My own work is writing the agent’s instructions and tuning them on the smallest model. A coding agent "
                "writes the code. The system does the rest: profiling, pruning, testing, undoing, and running the "
                "standard methods for comparison. Underneath: version control, fixed package versions, PyTorch, the "
                "standard benchmark tool, a free Colab GPU that saves after every step, and a spending cap.")
    return s
