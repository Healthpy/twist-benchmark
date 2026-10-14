#!/usr/bin/env python
"""Exploratory feasible-dispatch ceiling for the Week-9 bridge (decision log Week 9, item 14):
exhaustive single-window constant-power battery dispatch on the bridge units, with and without the
SoC / C-rate constraints. Run from the repository root after `twist bridge`."""

import glob
import json

import numpy as np

from twist.actions import HighLevelControl
from twist.actions.feasibility import build_constraints, check_feasibility
from twist.replay.pearl_replay import to_dense
from twist.worlds.base_scm import TemporalSCM
from twist.worlds.io import load_world_data

cons = build_constraints(
    [
        {"name": "state_bounds", "id": "soc_capacity", "channels": [4], "lower": 0, "upper": 1},
        {"name": "max_rate_of_change", "id": "battery_c_rate", "channels": [4], "max_rate": 0.55},
    ]
)
out = {}
for rd in sorted(glob.glob("artifacts/bridge/bridge_week9/evaluations/*/")):
    cfg = json.load(open(rd + "config.resolved.json"))
    wid = cfg["world"]["world_id"]
    w = TemporalSCM.load(f"artifacts/bridge/bridge_week9/worlds/{wid}")
    d = load_world_data(f"artifacts/bridge/bridge_week9/worlds/{wid}")
    recs = [
        r
        for r in json.load(open(rd + "records.json"))
        if r["explainer"] == "Dispatch[R6 constraint-aware]"
    ]
    ids = [d["n_train"] + int(r["factual_id"].rsplit("/", 1)[1]) for r in recs]
    exo = {k: v[ids] for k, v in d["exo"].items()}
    tg = np.array([r["target"] for r in recs])
    P = w.controls["u_dispatch"].upper
    space = w.action_space()
    ok_any = np.zeros(len(ids), bool)
    succ_any = np.zeros(len(ids), bool)
    for start in range(25, 47):
        for dur in (1, 2, 3, 4):
            for frac in (-1, -0.5, 0.25, 0.5, 0.75, 1.0):
                acts = [
                    HighLevelControl(
                        "u_dispatch",
                        start,
                        min(start + dur, 50),
                        [frac * P] * (min(start + dur, 50) - start),
                    )
                    for _ in ids
                ]
                _, s = w.replay_units(exo, acts)
                succ = w.outcome(s) == tg
                feas = np.array(
                    [
                        check_feasibility(
                            cons, exo["state"][i], to_dense(a, 50, 8, space), s[i]
                        ).passed
                        for i, a in enumerate(acts)
                    ]
                )
                succ_any |= succ
                ok_any |= succ & feas
    out[wid] = (
        succ_any.mean(),
        ok_any.mean(),
        np.mean([r["world_replay"] == r["target"] and r["feasible"] for r in recs]),
    )
    print(
        wid,
        "any-success {:.2f}  feasible-success ceiling {:.2f}  R6 acceptable {:.2f}".format(
            *out[wid]
        ),
        flush=True,
    )
print(
    "mean",
    np.mean([v[0] for v in out.values()]).round(3),
    np.mean([v[1] for v in out.values()]).round(3),
    np.mean([v[2] for v in out.values()]).round(3),
)
