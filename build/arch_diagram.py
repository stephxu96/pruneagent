"""Architecture diagram slide, drawn with native, editable PowerPoint shapes in the deck's palette."""
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt
from lxml import etree

NAVY, INK, MUTED, RULE, ARROW = "1E4D8C", "0A1628", "5A6878", "CBD5E0", "7A7A7A"
GOLD, GOLD_BG, GOLD_TX = "E3A21A", "FBF0D2", "9A6A00"
TEAL, TEAL_BG = "1B6B7A", "DCE9F7"
RED, RED_BG = "A31F34", "F6E3E6"
GREEN, GREEN_BG = "0B6E5E", "DDEFEA"
GREY_BG = "F2F2F2"

rgb = RGBColor.from_string


def _name(shape, name):
    shape.name = name
    return shape


def rect(slide, name, x, y, w, h, fill=None, line=None, dash=False, line_w=1.0):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    _name(s, name)
    s.shadow.inherit = False
    if fill:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill)
    else:
        s.fill.background()
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(line_w)
        if dash:
            ln = s.line._get_or_add_ln()
            pd = etree.SubElement(ln, qn("a:prstDash"))
            pd.set("val", "dash")
    else:
        s.line.fill.background()
    s.text_frame.text = ""
    return s


def text(slide, name, x, y, w, h, paras, size=11, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    """paras: list of paragraphs; each a list of (text, {'b':bool,'i':bool,'s':superscript,'color':hex,'size':pt})."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    _name(tb, name)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    for i, runs in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if i:
            p.space_before = Pt(2)
        for t, o in runs:
            r = p.add_run()
            r.text = t
            f = r.font
            f.size = Pt(o.get("size", size))
            f.bold = o.get("b", False)
            f.italic = o.get("i", False)
            f.color.rgb = rgb(o.get("color", color))
            if o.get("s"):
                f._rPr.set("baseline", "30000")
    return tb


def badge(slide, name, x, y, n, fill, d=0.32):
    c = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    _name(c, name)
    c.shadow.inherit = False
    c.fill.solid()
    c.fill.fore_color.rgb = rgb(fill)
    c.line.fill.background()
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = str(n)
    r.font.size, r.font.bold = Pt(12), True
    r.font.color.rgb = rgb("FFFFFF")
    return c


def line(slide, name, x1, y1, x2, y2, color=ARROW, w=1.5, arrow=True, dash=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    _name(c, name)
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(w)
    ln = c.line._get_or_add_ln()
    if dash:
        etree.SubElement(ln, qn("a:prstDash")).set("val", "dash")
    if arrow:
        te = etree.SubElement(ln, qn("a:tailEnd"))
        te.set("type", "triangle")
        te.set("w", "med")
        te.set("len", "med")
    return c


def node(slide, key, x, y, w, h, n, title, desc, bg, accent, title_color=None):
    rect(slide, f"{key} box", x, y, w, h, fill=bg)
    if n is not None:
        badge(slide, f"{key} number", x + 0.1, y + 0.1, n, accent)
        tx = x + 0.5
    else:
        tx = x + 0.12
    text(slide, f"{key} title", tx, y + 0.06, x + w - tx - 0.08, 0.42,
         [[(title, {"b": True, "color": title_color or accent, "size": 12})]], anchor=MSO_ANCHOR.MIDDLE)
    text(slide, f"{key} text", x + 0.12, y + 0.56, w - 0.22, h - 0.6, [[(desc, {"color": "404040", "size": 10.5})]])


