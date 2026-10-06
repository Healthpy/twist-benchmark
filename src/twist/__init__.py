"""TWIST: Temporal World-relative Intervention, Simulation, and Traceability benchmark.

Conventions used throughout the package (see ``docs/evaluation_protocol.md``):

* Trajectories are arrays of shape ``(T, k)`` for one unit and ``(B, T, k)`` for a batch
  (batch x time x channel). Time is zero-indexed, ``t = 0, ..., T-1``.
* An *action* is a declared intervention plan; a *proposal* is an explainer-proposed
  trajectory; a *replay* is the trajectory obtained by propagating the action through the
  world mechanism with the factual innovations retained (Pearl semantics) or zeroed after
  the first intervened timestep (noiseless semantics).
"""

__version__ = "0.1.0"
