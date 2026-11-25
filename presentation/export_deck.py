"""Render the Sustainome slide deck (presentation/deck) to PNG, PDF and PPTX.

    python presentation/export_deck.py

The deck's source is HTML slides (Claude Slides format) whose images point to
uploaded assets (/_blob/<id>); here each id is mapped to the local copy in
presentation/assets/. Output: presentation/export/Sustainome_Presentation.{pdf,pptx}
plus one PNG per slide. The PPTX holds one full-slide picture per slide (not
editable text); an editable PPTX can be exported from the online deck page.
"""

import json
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright
from pptx import Presentation
from pptx.util import Emu

HERE = Path(__file__).resolve().parent
DECK = HERE / "deck"
ASSETS = HERE / "assets"
OUT = HERE / "export"
OUT.mkdir(exist_ok=True)

BLOBS = {
    "6670e332926ed88319c8f1c1e4697a86": "d1_join.png",
    "32bf08356eb58d3bd00e926dd80473c0": "d1_missing.png",
    "e5ee142af4043b22cfc5a1f75d0e90e0": "d2_biplot.png",
    "2ab71043e2fd14ae22d9e2b4b6b058a8": "d2_fronts.png",
    "197106c8786108765a8fdb4e521d5812": "d2_protein_views.png",
    "41656908dbfe00234897cad591015766": "d3_edgemap.png",
    "efb6bdc9bdb56af6fa4720b0c10fec88": "d3_network.png",
    "b0a93927d2679fdd77b93cf3d96abfca": "d3_tau.png",
    "6512fa6b41115ab3d6ff3e82d967bd99": "d4_beforeafter.png",
    "0625776be2052a6c47485186274958fa": "d5_grid.png",
    "bcdeaa68a9c439fba29a393631f9e080": "d5_opt_robust.png",
    "8a9194c42e763eac571bf1c6a38e3298": "d5_reversal.png",
    "149748b7564dd3f98f7673486d7c5f04": "p_conflict_counts.png",
    "5631e0fa7089591c4a352f919a613e15": "p_d4_fronts.png",
    "65946d3a8ebef139e392a81dc78d73a9": "p_d4_summary.png",
    "ed40539d7593e54000a0efdc037d2fc9": "p_d4_swaps.png",
    "cacdb3d3764363cad49d7217f7625ef3": "p_protein_dumbbell.png",
}

index = json.loads((DECK / "project" / "deck.json").read_text(encoding="utf-8"))
fonts = "".join(f"<link rel='stylesheet' href='{f['href']}'>" for f in index["faces"].values() if "href" in f)
base_css = """
* { margin: 0 }
body { font: 32px/1.2 sans-serif }
body > section { width: 1920px !important; height: 1080px !important; overflow: hidden !important; box-sizing: border-box !important; position: relative; isolation: isolate }
aside { display: none !important }
h1 { font-size: 96px; font-weight: 600 } h2 { font-size: 64px; font-weight: 600 } h3 { font-size: 44px; font-weight: 600 }
table { width: 100%; border-collapse: separate; border-spacing: 0; border: solid #0003; border-width: 2px 0 0 2px }
th, td { padding: .35em .6em; text-align: left; border: solid #0003; border-width: 0 2px 2px 0 }
x-shape, x-icon { display: block }
x-shape[kind="arrow-right"], x-shape[kind="arrow-left"] { clip-path: polygon(0 30%, 60% 30%, 60% 0, 100% 50%, 60% 100%, 60% 70%, 0 70%) }
x-shape[kind="arrow-left"] { transform: scaleX(-1) }
section .grow { min-height: 0 }
"""
slides = "".join((DECK / "project" / "slides" / f"{sid}.html").read_text(encoding="utf-8") for sid in index["order"])
html = ("<!doctype html><html><head><meta charset='utf-8'>" + fonts + "<style>" + base_css + "</style>"
        "<link rel='stylesheet' href='/project/deck.css'></head><body>" + slides + "</body></html>")


def handle(route):
    url = route.request.url
    if url.startswith(("https://fonts.googleapis.com/", "https://fonts.gstatic.com/")):
        return route.continue_()
    path = url.split("://", 1)[1].split("/", 1)[1] if "/" in url.split("://", 1)[1] else ""
    if path == "":
        return route.fulfill(status=200, content_type="text/html", body=html)
    if path.startswith("_blob/") and path[6:] in BLOBS:
        return route.fulfill(status=200, content_type="image/png", body=(ASSETS / BLOBS[path[6:]]).read_bytes())
    if path.startswith("project/") and (DECK / path).is_file():
        return route.fulfill(status=200, content_type="text/css", body=(DECK / path).read_bytes())
    return route.abort()


pngs = []
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_context(java_script_enabled=False, viewport={"width": 1920, "height": 1080}).new_page()
    page.route("**/*", handle)
    page.goto("http://deck.local/", wait_until="networkidle")
    sections = page.locator("body > section")
    for i in range(sections.count()):
        sections.nth(i).scroll_into_view_if_needed()
        page.wait_for_timeout(150)
        f = OUT / f"slide_{i + 1:02d}.png"
        sections.nth(i).screenshot(path=str(f), animations="disabled")
        pngs.append(f)
    browser.close()

ims = [Image.open(f).convert("RGB") for f in pngs]
ims[0].save(OUT / "Sustainome_Presentation.pdf", save_all=True, append_images=ims[1:], resolution=144)

prs = Presentation()
prs.slide_width, prs.slide_height = Emu(12192000), Emu(6858000)  # 16:9
for f in pngs:
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.shapes.add_picture(str(f), 0, 0, prs.slide_width, prs.slide_height)
prs.save(OUT / "Sustainome_Presentation.pptx")
print(f"{len(pngs)} slides -> {OUT}")
