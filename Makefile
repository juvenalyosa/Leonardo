.PHONY: installer-check manuscript-assets manuscript-tests reviewer-check

installer-check:
	./install_leonardo.sh --check-only --skip-zip --no-pymol-loader

manuscript-assets:
	python draft/scripts/run_interaction_external_benchmark.py
	python draft/scripts/generate_bmc_validation_assets.py

manuscript-tests:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q --junitxml=draft/results/pytest_validation.xml

reviewer-check: installer-check manuscript-tests manuscript-assets
