"""Render docs/APPROACH.md to a PDF with a headless Chromium browser.

    python tools/build_pdf.py out/Yamal_Fan_Boys.pdf

Needs the `markdown` package (requirements-dev.txt) and Chrome, Chromium or Brave.
"""
import os
import shutil
import subprocess
import sys

import markdown

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(HERE, "docs", "APPROACH.md")
BROWSERS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome", "chromium", "chromium-browser",
]

CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; font-size: 10pt;
       line-height: 1.42; color: #0b0b0b; }
h1 { font-size: 17pt; margin: 0 0 4pt 0; }
h2 { font-size: 12.5pt; margin: 14pt 0 4pt 0; border-bottom: 1px solid #e1e0d9; padding-bottom: 2pt; }
h3 { font-size: 10.5pt; margin: 10pt 0 2pt 0; }
p, li { margin: 3pt 0; }
ul, ol { padding-left: 18pt; }
table { border-collapse: collapse; margin: 6pt 0; font-size: 9pt; width: 100%; }
th, td { border-bottom: 1px solid #e1e0d9; padding: 3pt 6pt; text-align: left; vertical-align: top; }
th { color: #52514e; font-weight: 600; }
code { font-family: Menlo, Consolas, monospace; font-size: 8.6pt; background: #f4f4f1; padding: 0 2pt; }
pre { background: #f4f4f1; padding: 6pt 8pt; font-size: 8.4pt; overflow-x: auto; }
pre code { background: none; padding: 0; }
img { max-width: 100%; margin: 4pt 0; }
h2, h3 { page-break-after: avoid; }
table, img, pre { page-break-inside: avoid; }
"""


def browser():
    for b in BROWSERS:
        if os.path.isfile(b) or shutil.which(b):
            return b
    raise SystemExit("no Chrome / Chromium / Brave found")


def main(out_pdf):
    with open(DOC, encoding="utf-8") as handle:
        body = markdown.markdown(handle.read(), extensions=["tables", "fenced_code"])
    figures = "file://" + os.path.join(HERE, "docs", "figures") + "/"
    body = body.replace('src="figures/', 'src="' + figures)
    html = ("<!doctype html><html><head><meta charset='utf-8'><title>Yamal Fan Boys</title>"
            "<style>%s</style></head><body>%s</body></html>" % (CSS, body))
    out_pdf = os.path.abspath(out_pdf)
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    html_path = os.path.splitext(out_pdf)[0] + ".html"
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(html)
    subprocess.run([browser(), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    "--print-to-pdf-no-header", "--print-to-pdf=" + out_pdf,
                    "file://" + html_path], check=True, capture_output=True)
    os.remove(html_path)
    print("wrote", out_pdf)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "out", "Yamal_Fan_Boys.pdf"))
