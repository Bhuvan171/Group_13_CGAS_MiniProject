"""Small reportlab layer for the Sustainome research report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

PAGE_W, PAGE_H = A4
MARGIN = 1.8 * cm
TEXT_W = PAGE_W - 2 * MARGIN

_ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("title", parent=_ss["Title"], fontName="Helvetica-Bold", fontSize=17, leading=21, spaceAfter=4),
    "subtitle": ParagraphStyle("subtitle", parent=_ss["Normal"], fontName="Helvetica", fontSize=10.5, leading=13,
                               textColor=colors.HexColor("#444444"), alignment=1, spaceAfter=10),
    "h1": ParagraphStyle("h1", parent=_ss["Heading1"], fontName="Helvetica-Bold", fontSize=13, leading=16, spaceBefore=10, spaceAfter=5,
                         textColor=colors.HexColor("#0B3D63")),
    "h2": ParagraphStyle("h2", parent=_ss["Heading2"], fontName="Helvetica-Bold", fontSize=10.5, leading=13, spaceBefore=7, spaceAfter=3),
    "body": ParagraphStyle("body", parent=_ss["BodyText"], fontName="Helvetica", fontSize=9.3, leading=12.6, alignment=TA_JUSTIFY, spaceAfter=5),
    "bullet": ParagraphStyle("bullet", parent=_ss["BodyText"], fontName="Helvetica", fontSize=9.3, leading=12.4, leftIndent=12,
                             bulletIndent=3, spaceAfter=2.5, alignment=TA_LEFT),
    "caption": ParagraphStyle("caption", parent=_ss["BodyText"], fontName="Helvetica-Oblique", fontSize=8.2, leading=10.4,
                              textColor=colors.HexColor("#333333"), spaceAfter=8, alignment=TA_JUSTIFY),
    "abstract": ParagraphStyle("abstract", parent=_ss["BodyText"], fontName="Helvetica", fontSize=9.2, leading=12.4,
                               alignment=TA_JUSTIFY, leftIndent=8, rightIndent=8),
    "cell": ParagraphStyle("cell", parent=_ss["BodyText"], fontName="Helvetica", fontSize=7.4, leading=9, alignment=TA_LEFT),
    "cellb": ParagraphStyle("cellb", parent=_ss["BodyText"], fontName="Helvetica-Bold", fontSize=7.4, leading=9, alignment=TA_LEFT),
    "small": ParagraphStyle("small", parent=_ss["BodyText"], fontName="Helvetica", fontSize=8, leading=10),
}


class Report:
    def __init__(self, figures: Path):
        self.story: list = []
        self.figures = figures
        self.n_fig = 0
        self.n_tab = 0

    # text -----------------------------------------------------------------
    def h1(self, t):
        self.story.append(Paragraph(t, S["h1"]))

    def h2(self, t):
        self.story.append(Paragraph(t, S["h2"]))

    def p(self, t):
        self.story.append(Paragraph(t, S["body"]))

    def bullets(self, items):
        for it in items:
            self.story.append(Paragraph(it, S["bullet"], bulletText="•"))
        self.story.append(Spacer(1, 3))

    def space(self, h=6):
        self.story.append(Spacer(1, h))

    def pagebreak(self):
        self.story.append(PageBreak())

    def boxed(self, title, text):
        t = Table([[Paragraph(f"<b>{title}</b>", S["abstract"])], [Paragraph(text, S["abstract"])]], colWidths=[TEXT_W])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F2F6FA")),
                               ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#9DB4C8")),
                               ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        self.story.append(t)
        self.space(8)

    # figures / tables -----------------------------------------------------------
    def figure(self, name: str, caption: str, width: float = 1.0, max_h: float = 0.55):
        path = self.figures / name
        if not path.exists():
            raise FileNotFoundError(path)
        with PILImage.open(path) as im:
            w, h = im.size
        dw = TEXT_W * width
        dh = dw * h / w
        if dh > PAGE_H * max_h:
            dh = PAGE_H * max_h
            dw = dh * w / h
        self.n_fig += 1
        self.story.append(KeepTogether([Image(str(path), width=dw, height=dh),
                                        Paragraph(f"<b>Figure {self.n_fig}.</b> {caption}", S["caption"])]))
        return self.n_fig

    def table(self, df: pd.DataFrame, caption: str, col_widths: list[float] | None = None, fmt: dict | None = None):
        fmt = fmt or {}
        head = [Paragraph(str(c), S["cellb"]) for c in df.columns]
        rows = [head]
        for _, r in df.iterrows():
            cells = []
            for c in df.columns:
                v = r[c]
                if c in fmt and pd.notna(v):
                    v = fmt[c](v)
                elif isinstance(v, float):
                    v = f"{v:.3g}"
                cells.append(Paragraph(str(v), S["cell"]))
            rows.append(cells)
        if col_widths is None:
            col_widths = [TEXT_W / len(df.columns)] * len(df.columns)
        else:
            tot = sum(col_widths)
            col_widths = [TEXT_W * w / tot for w in col_widths]
        t = Table(rows, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F0")),
            ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#6F8CA8")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F7F7")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ]))
        self.n_tab += 1
        self.story.append(KeepTogether([Paragraph(f"<b>Table {self.n_tab}.</b> {caption}", S["caption"]), t]))
        self.space(8)
        return self.n_tab

    # build -----------------------------------------------------------------------
    def build(self, out: Path, title: str):
        def deco(canvas, doc):
            canvas.saveState()
            canvas.setFont("Helvetica", 7.5)
            canvas.setFillColor(colors.HexColor("#666666"))
            canvas.drawString(MARGIN, 1.0 * cm, title)
            canvas.drawRightString(PAGE_W - MARGIN, 1.0 * cm, f"page {doc.page}")
            canvas.restoreState()

        doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=1.6 * cm,
                                bottomMargin=1.6 * cm, title=title, author="BIO544 Group 13", subject="Sustainome mini-project report")
        doc.build(self.story, onFirstPage=deco, onLaterPages=deco)


def check(condition: bool, statement: str) -> str:
    """Guard for narrative text: the report build fails if a sentence's premise is false."""
    if not condition:
        raise AssertionError(f"report statement no longer supported by the data: {statement}")
    return statement
