"""Print a self-contained HTML report to PDF with headless Chrome.

    python3 -m src.analysis.html_pdf results/reports/crypto/<study>/<name>.html [...]

Writes <name>.pdf next to each file. A print stylesheet is injected into a temporary copy: A4
landscape, light theme, no navigation, tables and charts kept whole across page breaks. Chrome is
not installed on every box, so when it is missing here the print runs over ssh on --host
(home is shared NFS, so the paths are the same there).
"""
import argparse
import os
import shutil
import subprocess

PRINT_CSS = """<style media="print">
@page{size:A4 landscape;margin:10mm}
:root{color-scheme:light!important}
html,body{background:#fff!important}
nav{display:none!important}
main{max-width:none!important;padding:0!important}
table{font-size:9px!important;page-break-inside:auto}
tr,svg,figure,.verdict{break-inside:avoid;page-break-inside:avoid}
h2,h3{break-after:avoid;page-break-after:avoid}
</style>"""


def chrome():
    return next((c for c in ("google-chrome", "chromium", "chromium-browser") if shutil.which(c)), None)


def to_pdf(html, host):
    html = os.path.abspath(html)
    pdf = os.path.splitext(html)[0] + ".pdf"
    tmp = os.path.splitext(html)[0] + ".print.html"
    doc = open(html).read()
    # force the light palette whatever the viewer's scheme, then add the print rules
    doc = doc.replace("<html", '<html data-theme="light"', 1).replace("</head>", PRINT_CSS + "</head>", 1)
    open(tmp, "w").write(doc)
    cmd = ["--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=5000",
           "--print-to-pdf=" + pdf, "file://" + tmp]
    exe = chrome()
    try:
        if exe:
            subprocess.run([exe] + cmd, check=True, capture_output=True, timeout=300)
        else:
            subprocess.run(["ssh", "-o", "BatchMode=yes", host, " ".join(["google-chrome"] + ["'%s'" % c for c in cmd])],
                           check=True, capture_output=True, timeout=300)
    finally:
        os.remove(tmp)
    if not os.path.exists(pdf):
        raise SystemExit("no PDF written for " + html)
    print(pdf, "%.0f KB" % (os.path.getsize(pdf) / 1e3))
    return pdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("html", nargs="+")
    ap.add_argument("--host", default="lovelace", help="box with google-chrome, used when this one has none")
    a = ap.parse_args()
    for h in a.html:
        to_pdf(h, a.host)


if __name__ == "__main__":
    main()
