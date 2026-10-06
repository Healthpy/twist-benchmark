"""Figures. Week 1: trajectory case study (factual / proposal / replay with the action marked).
Planned (Week 11): A→B→C flow, validity–cost Pareto, access curve, robustness heatmap,
tolerance sensitivity."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_case_study(
    factual: np.ndarray,
    proposal: np.ndarray | None,
    replayed: np.ndarray,
    channels: list[int],
    action_times: list[int],
    t_label: int,
    path: str | Path,
    title: str = "",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(channels), 1, figsize=(7, 1.8 * len(channels)), sharex=True, squeeze=False
    )
    for ax, c in zip(axes[:, 0], channels, strict=True):
        ax.plot(factual[:, c], color="0.5", lw=1.2, label="factual")
        if proposal is not None:
            ax.plot(proposal[:, c], color="C0", lw=1.2, ls="--", label="proposal")
        ax.plot(replayed[:, c], color="C3", lw=1.2, label="replayed action")
        for t in action_times:
            ax.axvline(t, color="C2", lw=0.8, alpha=0.6)
        ax.axvline(t_label, color="k", lw=0.8, ls=":")
        ax.set_ylabel(f"x{c}")
    axes[0, 0].legend(loc="upper left", fontsize=7, ncol=3, frameon=False)
    axes[-1, 0].set_xlabel("t")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
