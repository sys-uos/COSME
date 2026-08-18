#!/usr/bin/env bash
# Build the SIGCOMM'26 poster and render a PNG preview for inspection.
#
# Deliberately plain `pdflatex` (twice, for tcolorbox/tcbposter's saved box positions) rather than
# latexmk: the poster must compile with the same minimal, stock TeX Live that the self-hosted
# Overleaf instance provides. Nothing here needs installing beyond texlive-latex-extra +
# texlive-pictures, both already present.
set -euo pipefail

cd "$(dirname "$0")/../poster"

DPI="${1:-60}"   # 60dpi over A0 gives a ~2000x2800 preview: enough to judge layout and type

# pdflatex -> bibtex -> pdflatex x2 is the classic four-pass dance: the first pass records the
# \citation entries, bibtex resolves them into poster.bbl, and the last two settle the numbering
# and tcolorbox/tcbposter's saved box positions.
echo "==> pdflatex pass 1/3"
pdflatex -interaction=nonstopmode -halt-on-error poster.tex > /dev/null
echo "==> bibtex"
bibtex poster > /dev/null || { echo "    bibtex FAILED:"; bibtex poster | tail -20; exit 1; }
grep -E "Warning--" poster.blg && echo "    ^ bibtex warnings" || echo "    no bibtex warnings"
echo "==> pdflatex pass 2/3"
pdflatex -interaction=nonstopmode -halt-on-error poster.tex > /dev/null
echo "==> pdflatex pass 3/3"
pdflatex -interaction=nonstopmode -halt-on-error poster.tex > /dev/null

echo "==> unresolved citations"
grep -c "Citation.*undefined" poster.log && echo "    ^ UNDEFINED CITATIONS" || echo "    none"

echo "==> overfull/underfull boxes"
grep -E "^(Overfull|Underfull)" poster.log || echo "    none"

echo "==> preview at ${DPI}dpi"
pdftoppm -r "$DPI" -png -singlefile poster.pdf preview

echo "==> 10-metre legibility check (10% scale)"
pdftoppm -r "$(echo "$DPI" | awk '{print $1/6}')" -png -singlefile poster.pdf preview-far

ls -la poster.pdf preview.png preview-far.png
