# Build the paper. `make` regenerates every number/figure from result files, compiles with tectonic, checks pages.
#   make assets   # paper/numbers.tex, tables/, figures/ (needs results/ and the private judged labels)
#   make pdf      # main.pdf + page-budget check (fails if the main text exceeds 6 pages)
all: assets pdf

assets:
	cd .. && PYTHONPATH=src:paper python paper/make_assets.py

pdf:
	tectonic -X compile main.tex --keep-logs -Z search-path=template
	python check_pages.py main.pdf

.PHONY: all assets pdf
