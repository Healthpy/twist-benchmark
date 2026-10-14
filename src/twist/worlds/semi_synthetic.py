"""Semi-synthetic energy worlds: a distribution transformer with PV, a battery and demand response
(plan §Week 9, the domain bridge).

The worlds keep TWIST's contract: an additive-noise SCM with lagged parents, exact abduction and
unit-level replay. Real covariates drive them, and the dynamics on top are an explicit simulator.
Exogenous channels (weather and demand forecasts) are roots whose innovations *are* real hourly
profiles from Open Power System Data (Netherlands, 2016–2019; see
``scripts/fetch_energy_profiles.py``). Abduction therefore recovers the real covariates, and replay
holds them fixed: an action never changes the weather. Units are 00:00-UTC-aligned windows of
consecutive real hours.

State ``s_{t+1} = F(s_t, u_t, w_t)``, hourly steps, power in per-unit of the transformer rating
(1 p.u. = 10 MW), with channels

| # | name | meaning | equation (all inputs at ``t−1``) |
|---|---|---|---|
| 0 | ``temp_fcst`` | ambient temperature forecast for the next hour, ``(°C − 10)/10`` | root (real) |
| 1 | ``ghi_fcst`` | irradiance forecast for the next hour, kW/m² | root (real) |
| 2 | ``load_fcst`` | demand forecast for the next hour, relative to the mean | root (real) |
| 3 | ``grid`` | transformer loading ``K``, p.u. (negative = reverse flow) | ``nd − p + r·backlog + η`` |
| 4 | ``soc`` | battery state of charge, fraction of capacity | ``(1−δ)soc − (dis/η_b − chg·η_b)/E + η`` |
| 5 | ``backlog`` | demand deferred by demand response, p.u.·h | ``(1−r)·backlog + η`` |
| 6 | ``hot_spot`` | winding hot-spot temperature, ``(°C − 80)/20`` | ``a·hs + (1−a)·θ_ss(temp, K) + η`` |
| 7 | ``stress`` | insulation ageing stress (EWMA of the ageing rate) | ``b·stress + (1−b)·softplus((θ_hs − 98)/6) + η`` |

with forecast net demand ``nd = c_load·load − c_pv·ghi``, a default peak-shaving battery policy
``p = P·tanh(g(nd − τ)/P)`` split into discharge ``dis`` and charge ``chg``, each capped by the
energy available inside the policy's operating band ``[0.1, 0.95]``, and the steady-state
hot-spot temperature of a simplified IEC 60076-7 thermal model
``θ_ss = θ_amb + Δθ_r·(1 + R·K²)/(1 + R)``. ``stress`` is a downstream symptom of
overheating (a non-causal distractor that is predictive of the label).

Native actions are two high-level controls entering through the input map (soft interventions):

* ``u_dispatch`` — battery redispatch on top of the policy (+ discharge / − charge), p.u.;
  ``grid −= u``, ``soc −= u/E``; delay 0.
* ``u_dr`` — demand-response curtailment, p.u.; ``grid −= u``, ``backlog += u``, after one hour of
  notice. Deferred demand returns at rate ``r`` (rebound).

The input map is linear and the policy does not see the control, so a control can drain the
battery below empty or exceed the inverter rating. Those are the domain feasibility constraints
(``configs/experiments/bridge.yaml``) checked on the replayed trajectory, not built into ``F``.

Claim boundary: the simulator is a controlled oracle inspired by distribution-grid physics. It is
not a validated model of any real feeder, and its causal truth is true by construction only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from twist.actions.schemas import ControlChannel
from twist.utils import backend as bk
from twist.utils.backend import Array
from twist.worlds.base_scm import InnovationProcess, TemporalSCM
from twist.worlds.labels import LabelFunction, OnsetLabel

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "energy"
CHANNELS = ("temp_fcst", "ghi_fcst", "load_fcst", "grid", "soc", "backlog", "hot_spot", "stress")
TEMP, GHI, LOAD, GRID, SOC, BACKLOG, HOT_SPOT, STRESS = range(8)
EXOGENOUS = (TEMP, GHI, LOAD)
SEASONS = {"winter": (11, 12, 1, 2), "summer": (5, 6, 7, 8), "all": tuple(range(1, 13))}

# fixed constants of the simulator (not world-specific, not perturbed by misspecification)
EFF = 0.95  # one-way battery efficiency
SELF_DISCHARGE = 5e-4  # per hour
SOC_LO, SOC_HI = 0.1, 0.95  # operating band of the default policy
STRESS_KEEP = 0.9  # EWMA persistence of the ageing stress
HS_REF, HS_SCALE = 80.0, 20.0  # hot spot normalisation (°C)
AGEING_REF, AGEING_STEP = 98.0, 6.0  # IEC: ageing rate doubles every 6 K above 98 °C
P_BASE_MW = 10.0  # 1 p.u.
BATTERY_HOURS = 2.0  # E = 2·P (fixed C-rate 0.5): the C-rate constraint is world-independent

# per channel: (physical meaning, unit, why it is not directly actionable — "" if actionable)
CHANNEL_INFO = {
    "temp_fcst": ("ambient temperature forecast, next hour", "(°C − 10)/10", "exogenous weather"),
    "ghi_fcst": ("global horizontal irradiance forecast, next hour", "kW/m²", "exogenous weather"),
    "load_fcst": ("demand forecast, next hour", "× mean NL demand", "exogenous demand"),
    "grid": (
        "transformer loading K (abstract set-point when intervened on: load shedding or "
        "upstream redispatch by unspecified means)",
        "p.u. of 10 MVA",
        "",
    ),
    "soc": ("battery state of charge", "fraction of capacity", "state: moved only by power flows"),
    "backlog": ("demand deferred by demand response", "p.u.·h", "state: moved only by DR"),
    "hot_spot": ("winding hot-spot temperature", "(°C − 80)/20", "outcome: thermal state"),
    "stress": ("insulation ageing stress (EWMA ageing rate)", "relative", "symptom of overheating"),
}

# channel innovation scales relative to the world's noise ``scale`` (exogenous: real profiles)
CHANNEL_SCALE = (0.0, 0.0, 0.0, 1.0, 0.01, 0.01, 1.0, 0.33)

PARAM_NAMES = (
    "c_load",  # p.u. at mean demand
    "c_pv",  # p.u. per kW/m²
    "p_max",  # battery inverter rating, p.u.
    "e_cap",  # battery energy, p.u.·h
    "policy_gain",
    "policy_target",  # net demand above which the policy discharges, p.u.
    "rebound",  # share of deferred demand returning per hour
    "thermal_keep",  # a = exp(−1/τ_th)
    "rise_rated",  # hot-spot rise over ambient at K = 1, K
    "loss_ratio",  # R: load losses / no-load losses
)


# ----------------------------------------------------------------------------------- profiles
def _sha256(arrays: dict[str, np.ndarray]) -> str:
    return hashlib.sha256(b"".join(arrays[k].tobytes() for k in sorted(arrays))).hexdigest()


def synthetic_profiles(years: int = 2, seed: int = 0) -> dict[str, np.ndarray]:
    """Self-contained stand-in with the same layout as the real profiles (tests, no data file):
    seasonal and diurnal temperature, clear-sky irradiance with cloud cover, a two-peak demand."""
    rng = np.random.default_rng(seed)
    n = years * 8760
    t = np.arange(n)
    hour, day = t % 24, t / 24.0
    season = np.cos(2 * np.pi * (day - 15) / 365.25)  # +1 mid-January
    month = (np.floor(day % 365.25 / 30.44).astype(int) % 12 + 1).astype(np.int8)
    ar = np.zeros(n)
    e = rng.normal(0, 0.6, n)
    for i in range(1, n):
        ar[i] = 0.97 * ar[i - 1] + e[i]
    temp = 10 - 8 * season + 4 * np.sin(2 * np.pi * (hour - 9) / 24) + ar
    elev = np.clip(np.sin(np.pi * (hour - 5) / 15), 0, None) * (hour >= 5) * (hour <= 20)
    cloud = np.clip(0.6 + 0.3 * np.repeat(rng.random(n // 24 + 1), 24)[:n], 0, 1)
    ghi = (0.85 - 0.45 * (season + 1) / 2) * elev * cloud
    diurnal = 0.12 * np.exp(-((hour - 8) ** 2) / 8) + 0.2 * np.exp(-((hour - 18) ** 2) / 6)
    load = 0.85 + 0.12 * season + diurnal + 0.03 * rng.normal(size=n)
    return {
        "load": (load / load.mean()).astype(np.float32),
        "temp": temp.astype(np.float32),
        "ghi": ghi.astype(np.float32),
        "hour_utc": hour.astype(np.int8),
        "month": month,
    }


@lru_cache(maxsize=4)
def load_profiles(name: str) -> dict[str, np.ndarray]:
    """Hourly covariate profiles by name: ``synthetic`` or a file in ``twist/data/energy``.

    Real profiles are checked against the SHA-256 recorded by the fetch script."""
    if name == "synthetic":
        return synthetic_profiles()
    path = DATA_DIR / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/fetch_energy_profiles.py")
    arrays = dict(np.load(path))
    meta = json.loads(path.with_suffix(".json").read_text())
    if _sha256(arrays) != meta["arrays_sha256"]:
        raise ValueError(f"{path} does not match its recorded SHA-256")
    return arrays


def profile_hash(name: str) -> str:
    return _sha256(load_profiles(name))[:16]


@dataclass
class ProfileInnovations(InnovationProcess):
    """Innovations of an energy world: real profile windows on the exogenous channels, i.i.d.
    ``kind`` noise of scale ``scale × channel_scale[i]`` elsewhere.

    A window covers consecutive real hours and starts so that index ``day_offset`` falls on
    00:00 UTC of a day in ``months`` (for a simulated population, ``day_offset`` = burn-in).
    Exogenous channels hold *next-hour* values (forecasts for hour ``t+1``), so the lagged
    mechanism uses the right hour's weather and demand."""

    channel_scale: list[float] = field(default_factory=lambda: list(CHANNEL_SCALE))
    profiles: str = "opsd_nl_2016_2019"
    months: list[int] = field(default_factory=lambda: list(SEASONS["all"]))
    day_offset: int = 48

    def exogenous_windows(self, rng: np.random.Generator, n: int, length: int) -> np.ndarray:
        """``(n, length, 3)`` windows of (temp, ghi, load) in channel units."""
        p = load_profiles(self.profiles)
        N = len(p["load"])
        days = np.flatnonzero((p["hour_utc"] == 0) & np.isin(p["month"], self.months))
        starts = days - self.day_offset
        starts = starts[(starts >= 0) & (starts + length + 1 <= N)]
        if starts.size == 0:
            raise ValueError(f"no {length}-hour windows in months {self.months}")
        s = rng.choice(starts, n)
        idx = s[:, None] + 1 + np.arange(length)[None, :]  # next-hour values
        return np.stack(
            [(p["temp"][idx] - 10.0) / 10.0, p["ghi"][idx], p["load"][idx]], axis=-1
        ).astype(float)

    def sample(self, rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
        eta = super().sample(rng, shape) * np.asarray(self.channel_scale, float)
        n, length, _ = shape
        eta[..., list(EXOGENOUS)] = self.exogenous_windows(rng, n, length)
        return eta

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "channel_scale": [float(v) for v in self.channel_scale],
            "profiles": self.profiles,
            "months": [int(m) for m in self.months],
            "day_offset": int(self.day_offset),
        }


# ------------------------------------------------------------------------------------- world
def _min(a: Array, b: Array) -> Array:
    return 0.5 * (a + b - abs(a - b))


@dataclass
class EnergyGridSCM(TemporalSCM):
    family: ClassVar[str] = "energy"
    phys: dict[str, float] = field(default_factory=dict)

    channel_info: ClassVar[dict[str, tuple[str, str, str]]] = CHANNEL_INFO

    def _p(self, name: str) -> float:
        return float(self.phys[name])

    def battery_policy(self, past: Array) -> tuple[Array, Array, Array]:
        """(net demand forecast, discharge, charge) of the default policy for the next hour."""
        x = past[:, 0]
        nd = self._p("c_load") * x[:, LOAD] - self._p("c_pv") * x[:, GHI]
        P = self._p("p_max")
        req = P * bk.tanh(self._p("policy_gain") * (nd - self._p("policy_target")) / P)
        soc, E = x[:, SOC], self._p("e_cap")
        # the policy never plans beyond its operating band: at most the energy above SOC_LO
        # (discharge) or the headroom below SOC_HI (charge) within the hour
        dis = _min(bk.relu(req), bk.relu(soc - SOC_LO) * E * EFF)
        chg = _min(bk.relu(-req), bk.relu(SOC_HI - soc) * E / EFF)
        return nd, dis, chg

    def structural_mean(self, past: Array) -> Array:
        x = past[:, 0]
        nd, dis, chg = self.battery_policy(past)
        r = self._p("rebound")
        grid = nd - (dis - chg) + r * x[:, BACKLOG]
        soc = (1 - SELF_DISCHARGE) * x[:, SOC] - (dis / EFF - chg * EFF) / self._p("e_cap")
        backlog = (1 - r) * x[:, BACKLOG]
        R, a = self._p("loss_ratio"), self._p("thermal_keep")
        ambient = 10.0 + 10.0 * x[:, TEMP]
        steady = ambient + self._p("rise_rated") * (1 + R * x[:, GRID] ** 2) / (1 + R)
        hot = a * x[:, HOT_SPOT] + (1 - a) * (steady - HS_REF) / HS_SCALE
        theta = HS_SCALE * x[:, HOT_SPOT] + HS_REF
        stress = STRESS_KEEP * x[:, STRESS] + (1 - STRESS_KEEP) * bk.softplus(
            (theta - AGEING_REF) / AGEING_STEP
        )
        zero = bk.zeros_like(grid)
        return bk.stack([zero, zero, zero, grid, soc, backlog, hot, stress], axis=1)

    def params(self) -> dict[str, np.ndarray]:
        # one array per parameter, so misspecification perturbs each relative to its own size
        return {n: np.atleast_1d(np.asarray(self.phys[n], float)) for n in PARAM_NAMES}

    def card(self) -> dict[str, Any]:
        return {
            **super().card(),
            "phys": {n: float(self.phys[n]) for n in PARAM_NAMES},
            "profiles_sha256_16": profile_hash(self.noise.profiles)
            if isinstance(self.noise, ProfileInnovations)
            else None,
        }

    @classmethod
    def from_saved(cls, card: dict[str, Any], arrays: dict[str, np.ndarray]) -> EnergyGridSCM:
        common = cls._common_from_card({**card, "noise": {"kind": "laplace"}}, arrays)
        common["noise"] = ProfileInnovations(**card["noise"])
        phys = {n: float(np.asarray(arrays[n]).reshape(-1)[0]) for n in PARAM_NAMES}
        return cls(phys=phys, **common)


def energy_graph() -> np.ndarray:
    """``G[i, j, 0]``: ``j_{t−1} → i_t`` for the energy simulator (L = 1)."""
    g = np.zeros((8, 8, 1), bool)
    for child, parents in {
        GRID: (LOAD, GHI, SOC, BACKLOG),
        SOC: (LOAD, GHI, SOC),
        BACKLOG: (BACKLOG,),
        HOT_SPOT: (TEMP, GRID, HOT_SPOT),
        STRESS: (HOT_SPOT, STRESS),
    }.items():
        g[child, list(parents), 0] = True
    return g


def energy_controls(phys: dict[str, float], dr_max: float) -> dict[str, ControlChannel]:
    """The two native controls and their action-card fields."""
    load_d = [0.0] * 8
    load_d[GRID], load_d[SOC] = -1.0, -1.0 / phys["e_cap"]
    load_r = [0.0] * 8
    load_r[GRID], load_r[BACKLOG] = -1.0, 1.0
    mw = P_BASE_MW
    return {
        "u_dispatch": ControlChannel(
            name="u_dispatch",
            loadings=load_d,
            delay=0,
            lower=-phys["p_max"],
            upper=phys["p_max"],
            unit=f"p.u. (1 p.u. = {mw:g} MW), + discharge / − charge",
            description="battery redispatch on top of the default peak-shaving policy",
            cost=0.05 * mw,
            cost_unit="k€ per p.u.·h (battery degradation, ≈ 50 €/MWh)",
            side_effects="drains (or fills) the battery: less peak-shaving capacity later; "
            "combined with the policy it can exceed the inverter rating",
        ),
        "u_dr": ControlChannel(
            name="u_dr",
            loadings=load_r,
            delay=1,
            lower=0.0,
            upper=dr_max,
            unit=f"p.u. (1 p.u. = {mw:g} MW) of curtailed demand",
            description="demand-response curtailment, activated with one hour of notice",
            cost=0.15 * mw,
            cost_unit="k€ per p.u.·h (customer compensation, ≈ 150 €/MWh)",
            side_effects=f"deferred demand returns at rate r={phys['rebound']:.2f}/h (rebound "
            "peak after the event)",
        ),
    }


PV_LEVELS = {"low": (0.2, 0.4), "high": (0.7, 1.0)}
BATTERY_LEVELS = {"small": (0.1, 0.2), "large": (0.25, 0.4)}


def make_energy_grid(
    world_id: str,
    seed: int,
    T: int = 50,
    pv_penetration: str = "low",
    battery: str = "small",
    season: str = "all",
    profiles: str = "opsd_nl_2016_2019",
    burn_in: int = 48,
    noise: InnovationProcess | None = None,
    label: LabelFunction | None = None,
) -> EnergyGridSCM:
    """One energy world: physical parameters drawn from ``seed`` within the factor levels."""
    rng = np.random.default_rng(seed)
    c_load = rng.uniform(0.8, 1.0)
    p_max = rng.uniform(*BATTERY_LEVELS[battery])
    phys = {
        "c_load": c_load,
        "c_pv": rng.uniform(*PV_LEVELS[pv_penetration]),
        "p_max": p_max,
        "e_cap": BATTERY_HOURS * p_max,
        "policy_gain": rng.uniform(1.0, 3.0),
        "policy_target": c_load * rng.uniform(1.0, 1.2),
        "rebound": rng.uniform(0.1, 0.3),
        "thermal_keep": float(np.exp(-1.0 / rng.uniform(2.0, 4.0))),
        "rise_rated": rng.uniform(70.0, 85.0),
        "loss_ratio": rng.uniform(4.0, 8.0),
    }
    base = noise or InnovationProcess("laplace", 0.03)
    innov = ProfileInnovations(
        kind=base.kind,
        scale=base.scale,
        correlation=base.correlation,
        heteroscedastic=base.heteroscedastic,
        profiles=profiles,
        months=list(SEASONS[season]),
        day_offset=burn_in,
    )
    world = EnergyGridSCM(
        world_id=world_id,
        seed=seed,
        k=8,
        L=1,
        T=T,
        graph=energy_graph(),
        noise=innov,
        label=label or OnsetLabel(channel=HOT_SPOT, t_start=T // 2, t_end=-1),
        burn_in=burn_in,
        variable_names=list(CHANNELS),
        phys=phys,
    )
    world.actionable = [GRID]  # abstract loading set-point; the native actions are the controls
    world.controls = energy_controls(phys, dr_max=float(rng.uniform(0.15, 0.3)))
    world.roles = {
        "label_channel": HOT_SPOT,
        "route_source": GRID,
        "distractor": STRESS,
        "exogenous": list(EXOGENOUS),
        "battery_soc": SOC,
    }
    world.assumptions = {**world.assumptions, "real_covariates": True}
    return world
