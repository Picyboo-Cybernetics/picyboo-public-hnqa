"""Contextual reasoning demo built atop :mod:`superposition_memory`.

The function ``reason_about_signal`` shows how an application could inject
sensor features and scenario priors into the ``SuperpositionMemory`` class.
It keeps the example deterministic and numerically stable while still
illustrating the qualitative behaviour described in the whitepaper.
"""

from __future__ import annotations

from typing import Dict, Iterable

from .superposition_memory import Hypothesis, SuperpositionMemory


def reason_about_signal(
    priors: Dict[str, float],
    sensor_features: Iterable[str],
) -> Hypothesis:
    """Fuse priors and feature cues to pick the most plausible hypothesis.

    Parameters
    ----------
    priors:
        Mapping from hypothesis label to prior probability mass.  Values do not
        need to sum to one; they are normalised automatically.
    sensor_features:
        Iterable of feature names extracted from a sensor snapshot.  The
        function maps these to contextual weights that bias the superposition.

    Returns
    -------
    Hypothesis
        The hypothesis favoured by the contextual collapse procedure.
    """

    memory = SuperpositionMemory()
    for label, prior in priors.items():
        memory.add(label, prior ** 0.5)

    memory.normalise()

    context_weights = _feature_to_context(sensor_features)
    winner = memory.collapse(context=context_weights.items())
    return winner


def _feature_to_context(features: Iterable[str]) -> Dict[str, float]:
    """Translate simple feature flags into contextual weight adjustments."""

    weights: Dict[str, float] = {
        "thermal anomaly": 0.0,
        "sensor fault": 0.0,
        "benign fluctuation": 0.0,
    }

    for feature in features:
        if feature == "redundant_sensor_agreement":
            # Reliability increases, penalise the sensor fault interpretation.
            weights["sensor fault"] -= 0.25
        elif feature == "external_heat_source":
            weights["thermal anomaly"] += 0.35
        elif feature == "maintenance_recently_completed":
            weights["benign fluctuation"] += 0.20

    return weights


if __name__ == "__main__":
    result = reason_about_signal(
        priors={
            "thermal anomaly": 0.55,
            "sensor fault": 0.30,
            "benign fluctuation": 0.15,
        },
        sensor_features=["redundant_sensor_agreement", "external_heat_source"],
    )
    print(f"Selected hypothesis: {result.label} (p={result.probability:.3f})")
