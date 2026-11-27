"""Apply the final E229 image sizing without changing frozen scientific outputs."""
import json
import subprocess
from experiments import e212_volatility99_tsfm as S
ROOT=S.REPO/'results/reports/crypto/e229_momentum_volatility_ensembles'
page=ROOT/'E229_momentum_volatility_ensembles.html'
old=page.read_text()
new=old.replace('max-width:100%;max-height:160mm;object-fit:contain','width:100%;height:auto;max-width:100%;object-fit:contain')
assert new!=old or 'width:100%;height:auto;max-width:100%' in old
page.write_text(new)
subprocess.run([S.PYTHON,'-m','src.analysis.html_pdf',str(page)],check=True,timeout=300)
complete=json.loads((ROOT/'COMPLETE.json').read_text());complete['pdf_sha256']=S.sha(page.with_suffix('.pdf'));complete['report_layout_source']=str(__file__);complete['report_layout_sha256']=S.sha(__file__);S.save(ROOT/'COMPLETE.json',complete)
