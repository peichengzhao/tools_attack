PYTHON ?= python3

.PHONY: ci check-repository test compile dry-run baseline-check benchmark-check

ci: check-repository test compile

check-repository:
	$(PYTHON) scripts/check_repository.py

test:
	cd experiments/aaai27_supplement && $(PYTHON) -m unittest discover -s tests -v

compile:
	$(PYTHON) -m compileall -q InjecAgent/src exp7 experiments/aaai27_supplement scripts

dry-run:
	cd experiments/aaai27_supplement && \
		$(PYTHON) scripts/run_pipeline.py --stage all --dry-run --budget-calls 20000

baseline-check:
	$(PYTHON) experiments/aaai27_supplement/scripts/check_baselines.py

benchmark-check:
	$(PYTHON) experiments/aaai27_supplement/scripts/check_external_benchmarks.py
