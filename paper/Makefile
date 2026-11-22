# Build the paper. `make` regenerates every number/figure from result files, then compiles with tectonic.
#   make assets   # paper/numbers.tex, tables/, figures/ (needs results/ and the private judged labels)
#   make pdf      # main.pdf
all: assets pdf

assets:
	cd .. && PYTHONPATH=src:paper python paper/make_assets.py

pdf:
	tectonic -X compile main.tex --keep-logs -Z search-path=template

.PHONY: all assets pdf
