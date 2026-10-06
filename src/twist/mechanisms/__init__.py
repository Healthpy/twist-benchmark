"""Mechanism estimation, quality assessment and controlled misspecification (Week 6).

Estimated and misspecified mechanisms are ordinary ``TemporalSCM`` objects, so abduction, replay
and the recourse optimizers run on them unchanged. Only the benchmark compares them with the true
world; explainers see the estimate alone.
"""
