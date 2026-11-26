# Build the paper. `make` regenerates every number/figure from result files, compiles with tectonic, checks pages.
#   make fetch    # Stage 2 cell outputs (manifests, lm-eval; no generations) from the private store
#   make assets   # paper/numbers.tex, tables/, figures/ (needs results/ and the private judged labels)
#   make overleaf # aftermarket_overleaf.zip: self-contained sources for Overleaf (pdfLaTeX + BibTeX)
#   make arxiv    # aftermarket_arxiv.zip: Overleaf files + compiled main.bbl, \pdfoutput=1 (arXiv runs no BibTeX)
#   make pdf      # main.pdf; fails on a TODO or an eaten escape in any .tex (check_tex.py) or > 6 main pages
all: assets pdf

fetch:
	python fetch_stage2.py

assets:
	cd .. && PYTHONPATH=src:paper python paper/make_assets.py

pdf:
	python check_tex.py
	tectonic -X compile main.tex --keep-logs -Z search-path=template
	python check_pages.py main.pdf

overleaf:
	python make_overleaf.py

arxiv: overleaf
	rm -rf build_arxiv && mkdir build_arxiv && cd build_arxiv && unzip -q ../aftermarket_overleaf.zip && tectonic -X compile main.tex --keep-intermediates
	python make_arxiv.py build_arxiv/main.bbl
	rm -rf build_arxiv

.PHONY: all fetch assets pdf overleaf arxiv
