# make check      tests + a small run of every experiment except MovieLens (what CI runs)
# make reproduce  all experiments (writes results/*.csv), then the figures; about 40 minutes
# make figures    figures/*.png and *-dark.png from the committed results
# make data       download MovieLens-100K into data/ (also done by the MovieLens experiment)
# make notes      rebuild notes/pmf.pdf (needs latexmk)

PYTHON ?= python
WORKERS ?= 3
EXPERIMENTS := landscape inpainting saddle_escape missing_data convergence movielens
SMOKE := landscape inpainting saddle_escape missing_data convergence
PLOTS := hero movielens convergence_rates saddle_escape landscape missing_data inpainting

.PHONY: check test smoke reproduce experiments figures data notes notebooks

check: test smoke

test:
	$(PYTHON) -m pytest -q

smoke:
	@for e in $(SMOKE); do \
		echo "== $$e --quick"; \
		$(PYTHON) experiments/$$e.py --quick --workers 1 > /dev/null || exit 1; \
	done

experiments:
	@for e in $(EXPERIMENTS); do \
		echo "== $$e"; \
		$(PYTHON) experiments/$$e.py --workers $(WORKERS) || exit 1; \
	done

figures:
	@for f in $(PLOTS); do $(PYTHON) plotting/$$f.py || exit 1; done

reproduce: experiments figures

data:
	$(PYTHON) experiments/movielens.py --download-only

notes:
	cd notes && latexmk -pdf -interaction=nonstopmode pmf.tex && latexmk -c

notebooks:
	@for nb in experiments/*.ipynb; do \
		jupyter nbconvert --to notebook --execute --inplace "$$nb" || exit 1; \
	done
