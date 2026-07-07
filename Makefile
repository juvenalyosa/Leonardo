.PHONY: installer-check package manuscript-assets manuscript-tests reviewer-check

CONDA_ENV ?= protliginteract-full
PYTHON_RUN ?= conda run -n $(CONDA_ENV) python
PYTEST_RUN ?= conda run -n $(CONDA_ENV) env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest

installer-check:
	./install_leonardo.sh --check-only --skip-zip --no-pymol-loader

package:
	./build_leonardo_zip.sh

manuscript-assets:
	$(PYTHON_RUN) draft/scripts/run_interaction_external_benchmark.py
	$(PYTHON_RUN) draft/scripts/generate_bmc_validation_assets.py

manuscript-tests:
	$(PYTEST_RUN) -q --junitxml=draft/results/pytest_validation.xml

reviewer-check: installer-check package manuscript-tests manuscript-assets
