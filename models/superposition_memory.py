"""Minimal reference implementation of HNQA-inspired superposition memory.

The Hybrid Neural Quantum Architecture whitepaper describes storage units that
retain multiple incompatible hypotheses simultaneously.  The implementation
below models that idea with a tiny, dependency-free class that keeps a weighted
superposition of labelled hypotheses.  The class exposes utilities to:

* register hypotheses with complex amplitudes,
* normalise the internal state so probabilities sum to one, and
* "collapse" the superposition in a context-sensitive fashion.

The behaviour is intentionally simple and deterministic so the code can serve as
an executable companion to the whitepaper without introducing correctness
risks.  All math fits in standard Python and NumPy is not required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class Hypothesis:
    """Container that stores a hypothesis label and its complex amplitude."""

    label: str
    amplitude: complex

    @property
    def probability(self) -> float:
        """Return the Born-rule style probability mass of the hypothesis."""

        return (self.amplitude.conjugate() * self.amplitude).real


class SuperpositionMemory:
    """Store and collapse hypotheses as described in the HNQA whitepaper.

    Attributes
    ----------
    hypotheses:
        Mapping from hypothesis label to complex amplitude.
    """

    def __init__(self) -> None:
        self.hypotheses: Dict[str, complex] = {}

    # ---------------------------------------------------------------------
    # State management
    # ---------------------------------------------------------------------
    def add(self, label: str, amplitude: complex) -> None:
        """Add a hypothesis with the given amplitude.

        If the label already exists the amplitudes are summed, mirroring the
        constructive interference behaviour discussed in the whitepaper.
        """

        if label in self.hypotheses:
            self.hypotheses[label] += amplitude
        else:
            self.hypotheses[label] = amplitude

    def normalise(self) -> None:
        """Scale amplitudes such that their probability mass sums to one."""

        total_probability = sum(
            (amp.conjugate() * amp).real for amp in self.hypotheses.values()
        )
        if total_probability == 0:
            raise ValueError("Cannot normalise an empty or zero-amplitude state")

        scale = total_probability ** 0.5
        for label, amplitude in list(self.hypotheses.items()):
            self.hypotheses[label] = amplitude / scale

    # ---------------------------------------------------------------------
    # Collapse heuristics
    # ---------------------------------------------------------------------
    def collapse(self, context: Iterable[Tuple[str, float]]) -> Hypothesis:
        """Collapse the superposition using a contextual weighting scheme.

        Parameters
        ----------
        context:
            Iterable of ``(label, weight)`` pairs.  The weight represents the
            strength of contextual support for the corresponding hypothesis.  A
            weight of zero expresses neutrality.  Hypotheses missing from the
            context default to zero weight.

        Returns
        -------
        Hypothesis
            The hypothesis with the highest context-weighted score.  Scores are
            computed as ``probability * (1 + weight)`` so that neutral contexts
            fall back to probability ranking, while positive weights reinforce
            and negative weights penalise hypotheses.
        """

        if not self.hypotheses:
            raise ValueError("Cannot collapse an empty superposition state")

        weight_map = {label: weight for label, weight in context}
        scored: List[Tuple[float, Hypothesis]] = []

        for label, amplitude in self.hypotheses.items():
            probability = (amplitude.conjugate() * amplitude).real
            weight = weight_map.get(label, 0.0)
            score = probability * (1.0 + weight)
            scored.append((score, Hypothesis(label=label, amplitude=amplitude)))

        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    # ------------------------------------------------------------------
    # Convenience utilities
    # ------------------------------------------------------------------
    def as_ranked_list(self) -> List[Hypothesis]:
        """Return hypotheses sorted by probability mass."""

        ranked = [Hypothesis(label, amplitude) for label, amplitude in self.hypotheses.items()]
        ranked.sort(key=lambda hyp: hyp.probability, reverse=True)
        return ranked

    def reset(self) -> None:
        """Clear the internal state.  Useful in small experiments."""

        self.hypotheses.clear()


def run_demo() -> List[Hypothesis]:
    """Run a deterministic toy scenario used in documentation examples.

    The scenario introduces three competing interpretations for an ambiguous
    sensor signal and applies a mild positive context weight to one of them.
    The function returns the ranked hypothesis list after normalisation so the
    behaviour can be asserted in tests or experiments.
    """

    memory = SuperpositionMemory()
    memory.add("thermal anomaly", 0.6 + 0.1j)
    memory.add("sensor fault", 0.3)
    memory.add("benign fluctuation", 0.1)
    memory.normalise()
    memory.collapse(context=[("thermal anomaly", 0.2)])
    return memory.as_ranked_list()


if __name__ == "__main__":
    for hypothesis in run_demo():
        print(f"{hypothesis.label}: amplitude={hypothesis.amplitude:.3f}, probability={hypothesis.probability:.3f}")
