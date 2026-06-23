PYTHON ?= python3
PROJECT_PYTHON ?= /home/zpc/anaconda3/envs/tools_attack/bin/python
MODEL_CONFIG ?= /home/zpc/contract_heldout_ipi/model_api_ablation.py

.PHONY: test check check-baselines check-benchmarks interactive-plan interactive-table dry-run verify

test:
	$(PYTHON) -m unittest discover -s tests -v

check:
	$(PYTHON) scripts/check_environment.py \
		--project-python "$(PROJECT_PYTHON)" \
		--model-config "$(MODEL_CONFIG)"

check-baselines:
	$(PYTHON) scripts/check_baselines.py

check-benchmarks:
	$(PYTHON) scripts/check_external_benchmarks.py

interactive-plan:
	$(PYTHON) scripts/run_interactive_baselines.py \
		--python "$(PROJECT_PYTHON)" \
		--model-config "$(MODEL_CONFIG)"

interactive-table:
	$(PYTHON) scripts/build_interactive_table.py

dry-run:
	$(PYTHON) scripts/run_pipeline.py \
		--stage all \
		--project-python "$(PROJECT_PYTHON)" \
		--model-config "$(MODEL_CONFIG)" \
		--dry-run \
		--budget-calls 20000

verify:
	$(PYTHON) scripts/verify_artifacts.py
