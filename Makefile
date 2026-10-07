# Standard commands. `make help` lists targets. Uses the project venv created by `make install`.
PY      ?= .venv/bin/python
UV      ?= uv
CONFIG  ?= experiments/toy.yaml
export TWIST_NUM_THREADS ?= 1

.PHONY: help install lint format typecheck test test-fast smoke toy controls sanity audit suite-dev suite-pilot pilot ladder robustness validity study study-report discovery compat check clean-artifacts container

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

install:  ## create .venv (Python 3.11) and install the package with dev extras
	$(UV) venv --python 3.11 .venv
	$(UV) pip install --python $(PY) -e ".[dev,baselines,discovery]"

lint:  ## ruff lint + format check
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

format:  ## apply ruff fixes and formatting
	$(PY) -m ruff check --fix src tests scripts
	$(PY) -m ruff format src tests scripts

typecheck:  ## mypy on the package
	$(PY) -m mypy

test:  ## full test suite
	$(PY) -m pytest

test-fast:  ## tests without the end-to-end smoke test
	$(PY) -m pytest -m "not smoke and not slow"

smoke:  ## end-to-end smoke test only
	$(PY) -m pytest -m smoke

toy:  ## Week-1 toy run: world → classifier → explainers → A/B/C → certificates
	$(PY) -m twist.cli run --config $(CONFIG)

controls:  ## positive/negative structural control suite on the linear toy world
	$(PY) -m twist.cli run --config experiments/controls.yaml

sanity:  ## metric sanity report: control suite × {linear, nonlinear, spring} + audits
	$(PY) -m twist.cli sanity --config experiments/sanity.yaml

audit:  ## traceability audit of a run: make audit RUN=artifacts/evaluations/<run_id>
	$(PY) -m twist.cli audit $(RUN)

suite-dev:  ## build and re-check the development suite (40 worlds)
	$(PY) -m twist.cli build-suite --config suites/dev.yaml
	$(PY) -m twist.cli check-worlds artifacts/suites/twist_dev_v1

pilot:  ## Week-5 pilot: 3 dev worlds × 2 classifiers × baselines/controls × 3 seeds (+ dashboard)
	$(PY) -m twist.cli pilot --config experiments/pilot.yaml

suite-pilot:  ## build the Week-8 pilot-study suite (25 dev worlds, fresh seed)
	$(PY) -m twist.cli build-suite --config suites/pilot.yaml

ladder:  ## Week-6 mechanism-access ladder + misspecification sweep (SLURM recommended)
	$(PY) -m twist.cli ladder --config experiments/access_ladder.yaml

robustness:  ## Week-7 shifts, stability, uncertainty, tolerance (SLURM recommended)
	$(PY) -m twist.cli robustness --config experiments/robustness.yaml

validity:  ## Week-8 construct/discriminant validity report (5 exact dev worlds, ~7 min on 1 core)
	$(PY) -m twist.cli validity --config experiments/validity.yaml

study:  ## Week-8 pilot multi-world study (50 tasks; use SLURM: see scripts/slurm/multiworld.sbatch)
	$(PY) -m twist.cli study --config experiments/study.yaml

study-report:  ## rebuild the study report from finished runs
	$(PY) -m twist.cli study --config experiments/study.yaml --report-only

discovery:  ## causal-discovery study: discovered graphs as mechanism source (SLURM recommended)
	$(PY) -m twist.cli discovery --config experiments/discovery.yaml

compat:  ## regenerate docs/compatibility_matrix.md from explainer metadata
	$(PY) -m twist.cli compat

check: lint typecheck test  ## everything CI runs

clean-artifacts:  ## remove generated artifacts (keeps directory skeleton)
	find artifacts -mindepth 2 ! -name .gitkeep -delete 2>/dev/null || true
	rm -f artifacts/manifest.jsonl

container:  ## build the Apptainer image (HPC); use `docker build .` elsewhere
	apptainer build twist.sif apptainer.def
