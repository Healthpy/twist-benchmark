# Benchmark card — TWIST (draft, completed in Week 11)

| Field | Content |
|---|---|
| Name | TWIST: Temporal World-relative Intervention, Simulation, and Traceability benchmark |
| Task | Evaluate counterfactual explanations of time-series classifiers against known temporal SCMs |
| Evaluated object | Explanation `e = (𝒜, x̂, y*, 𝒞, r, u)` |
| Axes | model-relative, mechanism consistency, world-relative, actionability, robustness, traceability |
| Primary endpoints | `A/B/C`, `V_world`, `V_safe-world`, `E_replay` (see `evaluation_protocol.md`) |
| Worlds | five families (linear VAR, nonlinear SCM, spring, regime-switching, partially observed state-space) from a 12-factor generator; dev/val/test/challenge splits, test/challenge hidden. Domain bridge: semi-synthetic `energy` family (real NL weather/demand covariates, simulated transformer/battery/DR dynamics, native controls, domain constraints) |
| Protocol | v1.2: paired design, four-level hierarchical bootstrap, ITT comparisons, no scalar leaderboard (v1.0, Week 8); causal discovery (v1.1); energy bridge (v1.2) |
| Validity evidence | construct (5 controlled pairs) and discriminant (4 tests) validity reports, `twist validity` |
| Mechanism access | R0–R6 |
| Intended use | Research on action-grounded explanation methods; auditing explanation pipelines |
| Out-of-scope use | Certifying real-world deployment; claims about human utility; claims about real grids from the energy bridge (its dynamics are a simulator) |
| Known failure modes | `D` is representation- and tolerance-dependent; exact abduction only in additive-noise observed worlds (partially observed worlds use oracle replay from stored exogenous variables; abduction metrics are undefined there) |
| Maintenance | TODO (Week 12) |
