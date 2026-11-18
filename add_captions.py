"""
Add a bold centred figure-caption paragraph immediately after each <w:drawing> block
in HIQ_Clinical_Risk_Article_v6.docx, producing v7.docx.

Drawing order in the document (confirmed from embed_figs.py + PDF page check):
  Drawing 1 (rId8 / docPr id=101) = Figure 1 — HIQ* bar chart           (p26)
  Drawing 2 (rId9 / docPr id=103) = Figure 2 — forest plot              (p27; PNG: fig3_forest.png)
  Drawing 3 (rId10/ docPr id=102) = Figure 3 — rank reversal            (p28; PNG: fig2_rank_reversal.png)
  Drawing 4 (rId11/ docPr id=104) = Figure 4 — scorecard                (p53; PNG: figA1_scorecard.png)

Note: Document-order numbering (v8) differs from PNG filename order:
  fig3_forest.png     → Figure 2 in document
  fig2_rank_reversal.png → Figure 3 in document
  figA1_scorecard.png → Figure 4 in document
"""
import zipfile, shutil, re
from pathlib import Path

SCRATCHPAD = Path('/tmp/claude-0/-home-claude/240e3105-e79d-52b9-becb-b336b6aa69c5/scratchpad')
SRC  = SCRATCHPAD / 'HIQ_Clinical_Risk_Article_v6.docx'
DST  = SCRATCHPAD / 'HIQ_Clinical_Risk_Article_v7.docx'
WORK = SCRATCHPAD / 'v7_unpack'

# ── Unpack ──────────────────────────────────────────────────────────────────
if WORK.exists():
    shutil.rmtree(WORK)
with zipfile.ZipFile(SRC) as z:
    z.extractall(WORK)

doc_path = WORK / 'word' / 'document.xml'
doc = doc_path.read_text(encoding='utf-8')

# ── Caption paragraph builder ────────────────────────────────────────────────
def caption_para(text):
    """
    Returns a centred paragraph with bold italic 'Figure N.' prefix + normal text.
    Style mirrors common journal figure-caption formatting.
    """
    # Split "Figure N." from the rest
    m = re.match(r'(Figure [^.]+\.)(.*)', text, re.DOTALL)
    if m:
        label_part = m.group(1)   # e.g. "Figure 1."
        rest_part  = m.group(2).strip()
    else:
        label_part = ''
        rest_part  = text

    def run(txt, bold=False, italic=False, size_half_pts=18):
        b   = '<w:b/><w:bCs/>'     if bold   else ''
        i   = '<w:i/><w:iCs/>'     if italic else ''
        sz  = f'<w:sz w:val="{size_half_pts}"/><w:szCs w:val="{size_half_pts}"/>'
        col = '<w:color w:val="444444"/>'
        return (
            f'<w:r><w:rPr>{b}{i}{sz}{col}</w:rPr>'
            f'<w:t xml:space="preserve">{txt}</w:t></w:r>'
        )

    runs = ''
    if label_part:
        runs += run(label_part + ' ', bold=True, italic=True)
    if rest_part:
        runs += run(rest_part, bold=False, italic=False)

    return (
        '<w:p>'
        '<w:pPr>'
        '<w:jc w:val="center"/>'
        '<w:spacing w:before="60" w:after="160"/>'
        '</w:pPr>'
        + runs +
        '</w:p>'
    )

# ── Captions for each drawing ────────────────────────────────────────────────
CAPTIONS = [
    # Drawing 1 — Figure 1 (document order p26; PNG: fig1_hiq_bar.png)
    ("Figure 1. All-hospital HIQ* with 95% cluster-robust confidence intervals. "
     "LightGBM (HIQ* = 113.2) is the only model whose confidence interval lies "
     "entirely above 100 (the panel mean). Neural Net (HIQ* = 10.1) has a wide "
     "interval spanning zero, flagging instability. Synthetic data; illustrative only."),

    # Drawing 2 — Figure 2 (document order p27; PNG: fig3_forest.png; previously Figure 3)
    ("Figure 2. Forest plot of HIQ* with 95% cluster-robust confidence intervals. "
     "Models are sorted best to worst by point estimate. Only LightGBM's confidence "
     "interval lies entirely above 100 (panel mean, dashed line). "
     "Synthetic data; illustrative only."),

    # Drawing 3 — Figure 3 (document order p28; PNG: fig2_rank_reversal.png; previously Figure 2)
    ("Figure 3. Naive BSS vs corrected BSS* per hospital — LightGBM. "
     "Arrows show the direction and magnitude of the IPCW-and-transport correction. "
     "Regional Hospital reverses from naive rank 4 to corrected rank 1 (+0.032); "
     "Academic Hospital drops from rank 1 to rank 4 (−0.017). "
     "Synthetic data; illustrative only."),

    # Drawing 4 — Figure 4 (document order p53; PNG: figA1_scorecard.png; previously Figure A1)
    ("Figure 4. Performance rating: HIQ* vs naive BSS on five evaluation dimensions "
     "(1 = inadequate, 5 = excellent). HIQ* scores 4.4/5 overall; naive BSS scores "
     "1.6/5 in the multi-centre, right-censored evaluation setting for which it was "
     "not designed."),
]

# ── Insert caption after each </w:drawing>  block ───────────────────────────
# Each drawing is wrapped in a <w:p>…<w:drawing>…</w:drawing>…</w:r></w:p>
# We insert the caption paragraph immediately after the closing </w:p> of that drawing.

draws = list(re.finditer(r'<w:drawing>.*?</w:drawing>', doc, re.DOTALL))
print(f'Found {len(draws)} drawings')
assert len(draws) == 4, f'Expected 4 drawings, got {len(draws)}'

# Work from last to first so positions don't shift
for i in reversed(range(4)):
    m = draws[i]
    # Find the </w:p> that closes this drawing's paragraph
    para_end_idx = doc.find('</w:p>', m.end())
    assert para_end_idx != -1, f'No </w:p> after drawing {i+1}'
    insert_pos = para_end_idx + len('</w:p>')
    cap = '\n' + caption_para(CAPTIONS[i])
    doc = doc[:insert_pos] + cap + doc[insert_pos:]
    print(f'Drawing {i+1}: caption inserted at pos {insert_pos}')

doc_path.write_text(doc, encoding='utf-8')
print('document.xml updated')

# ── Repack ──────────────────────────────────────────────────────────────────
if DST.exists():
    DST.unlink()
with zipfile.ZipFile(DST, 'w', zipfile.ZIP_DEFLATED) as zout:
    for fpath in sorted(WORK.rglob('*')):
        if fpath.is_file():
            zout.write(fpath, fpath.relative_to(WORK).as_posix())

print(f'\nWritten: {DST}  ({DST.stat().st_size/1024:.0f} KB)')
