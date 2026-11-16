"""Fill the 8 kept template slides with DimCAD content. Input: base.pptx  Output: DimCAD_Proposal.pptx"""
import sys
from copy import deepcopy
from lxml import etree
from pptx import Presentation
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

A = lambda tag: qn("a:" + tag)


def set_paras(tf, paras):
    """paras: list of paragraphs; each a list of (text, fmt) with fmt containing 'b' and/or 'i'.
    Reuses the template paragraph at the same index (or the last one) for pPr and rPr."""
    body = tf._txBody
    old = body.findall(A("p"))
    tmpl = [deepcopy(p) for p in old]

    def rpr_of(p):
        r = p.find(A("r"))
        return deepcopy(r.find(A("rPr"))) if r is not None and r.find(A("rPr")) is not None else None

    base = next((rpr_of(p) for p in tmpl if rpr_of(p) is not None), None)
    for p in old:
        body.remove(p)
    for i, runs in enumerate(paras):
        tp = deepcopy(tmpl[min(i, len(tmpl) - 1)])
        rp = rpr_of(tp)
        if rp is None:
            rp = deepcopy(base) if base is not None else etree.Element(A("rPr"))
        for child in list(tp):
            if child.tag != A("pPr"):
                tp.remove(child)
        for text, fmt in runs:
            r = etree.SubElement(tp, A("r"))
            rr = deepcopy(rp)
            for h in rr.findall(A("hlinkClick")):
                rr.remove(h)
            rr.set("b", "1" if "b" in fmt else "0")
            rr.set("i", "1" if "i" in fmt else "0")
            if "s" in fmt:
                rr.set("baseline", "30000")
            r.append(rr)
            t = etree.SubElement(r, A("t"))
            t.text = text
            if text != text.strip():
                t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        body.append(tp)


def P(*runs):
    """P(('bold', 'b'), ' plain') -> paragraph."""
    return [(r, "") if isinstance(r, str) else r for r in runs]


def by_id(slide, sid):
    for s in slide.shapes:
        if s.shape_id == sid:
            return s
    raise KeyError(sid)


def txt(slide, sid, *paras):
    set_paras(by_id(slide, sid).text_frame, list(paras))


def delete(slide, *sids):
    for sid in sids:
        el = by_id(slide, sid)._element
        el.getparent().remove(el)


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


