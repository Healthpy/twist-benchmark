# Reproducibility guide

## Environment

```bash
make install            # uv venv (Python 3.11) + editable install with dev extras
make check              # ruff, mypy, pytest (incl. deterministic-seed and artifact-schema tests)
make toy                # end-to-end toy run
```

Containers: `docker build -t twist .` (CPU) or `make container` (Apptainer, for Snellius/HPC).
On shared nodes set `TWIST_NUM_THREADS` (default 1); under SLURM `SLURM_CPUS_PER_TASK` is used.

## External data (energy bridge)

The semi-synthetic energy worlds read `src/twist/data/energy/opsd_nl_2016_2019.npz`. It is
shipped in the repository (270 kB, CC-BY 4.0) and verified by SHA-256 on every load. To rebuild it
from the original Open Power System Data packages (≈ 360 MB download), run
`python scripts/fetch_energy_profiles.py`. The rebuild must reproduce the recorded SHA-256 in
`opsd_nl_2016_2019.json`. Energy world cards record the profile hash
(`profiles_sha256_16`).

## What every run records

`artifacts/manifest.jsonl` (append-only) gets one `RunManifest` per explainer × reading with:
`world_id, world_family, world_seed, classifier_architecture, classifier_seed, explainer,
explainer_seed, action_semantics, mechanism_access_regime, evaluation_seed, dataset_hash,
config_hash, git_commit, timestamp_utc, hardware, runtime_seconds, package_version`.

Each run directory `artifacts/evaluations/<name>__<UTC>__<config_hash8>/` contains the fully
resolved config, per-explanation records with provenance, the summary table, explanation
certificates and case-study figures.

`git_commit` is suffixed `-dirty` when the working tree has uncommitted changes; results that
enter the paper must come from clean commits.

`run_info.json` also records a `code_fingerprint`: a hash of every `.py` file of the package, taken
once per process when its first run starts. Multi-world commands reuse a finished task only when
both the config hash and the fingerprint match. `twist study --report-only` rebuilds a report from
runs with the same config even when the code has changed since, and logs a warning for each one.

## Determinism

- World generation and data simulation are deterministic given the world seed
  (`tests/test_worlds`, `tests/test_reproducibility`).
- Classifier training is deterministic given the classifier seed on CPU
  (`torch.use_deterministic_algorithms(True)`), checked by a test.
- GPU runs may differ at floating-point level; record `hardware` and compare within tolerance.

## Clean-room checklist (Week 12)

1. install package; 2. generate a development world; 3. train one classifier; 4. run two
explainers; 5. compute all metrics; 6. reproduce one figure; 7. export one action certificate.
