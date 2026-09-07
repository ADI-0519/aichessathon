"""Generalized SPRT for colour-swapped chess-game pairs.

The five observations are the candidate's normalized score over a pair:
LL, LD/DL, LW/DD/WL, DW/WD, and WW.  The likelihood calculation follows
the constrained multinomial maximum-likelihood construction used by fishtest,
implemented here with a dependency-free bisection solver.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

Decision = Literal["accept_h0", "accept_h1", "continue"]

PAIR_SUPPORT = (0.0, 0.25, 0.5, 0.75, 1.0)
REGULARIZATION = 1e-3
ROOT_EPSILON = 1e-12
ROOT_ITERATIONS = 160


def elo_to_score(elo: float) -> float:
    """Convert a logistic Elo difference to expected game score."""
    return float(1.0 / (1.0 + 10.0 ** (-elo / 400.0)))


def score_to_elo(score: float) -> float:
    """Convert expected game score to logistic Elo."""
    clamped = min(max(score, 1e-9), 1.0 - 1e-9)
    return 400.0 * math.log10(clamped / (1.0 - clamped))


@dataclass(frozen=True, slots=True)
class Verdict:
    """One sequential-test snapshot."""

    decision: Decision
    llr: float
    pairs: int
    score: float
    elo: float
    lower_bound: float
    upper_bound: float
    elo0: float
    elo1: float
    alpha: float
    beta: float
    min_pairs: int

    @property
    def summary(self) -> str:
        label = {
            "accept_h0": "FAIL",
            "accept_h1": "PASS",
            "continue": "....",
        }[self.decision]
        return (
            f"{label} LLR {self.llr:+.3f} "
            f"[{self.lower_bound:+.3f}, {self.upper_bound:+.3f}] "
            f"after {self.pairs} pairs, score {self.score:.2%}, Elo {self.elo:+.1f}"
        )


def _validate_counts(counts: Sequence[int]) -> tuple[int, int, int, int, int]:
    if len(counts) != len(PAIR_SUPPORT):
        raise ValueError("pentanomial counts must contain exactly five bins")
    try:
        normalized = tuple(operator.index(value) for value in counts)
    except TypeError as error:
        raise ValueError("pentanomial counts must be integers") from error
    if any(value < 0 for value in normalized):
        raise ValueError("pentanomial counts cannot be negative")
    return normalized[0], normalized[1], normalized[2], normalized[3], normalized[4]


def _constrained_mle(empirical: Sequence[float], target: float) -> tuple[float, ...]:
    """Fit probabilities with maximum likelihood at a fixed expectation."""
    if len(empirical) != len(PAIR_SUPPORT):
        raise ValueError("empirical distribution must contain five bins")
    if not 0.0 < target < 1.0:
        raise ValueError("target score must lie strictly between zero and one")

    offsets = tuple(outcome - target for outcome in PAIR_SUPPORT)
    lower = -1.0 / max(offsets) + ROOT_EPSILON
    upper = -1.0 / min(offsets) - ROOT_EPSILON

    def secular(multiplier: float) -> float:
        return sum(
            probability * offset / (1.0 + multiplier * offset)
            for probability, offset in zip(empirical, offsets, strict=True)
        )

    if secular(lower) <= 0.0 or secular(upper) >= 0.0:
        raise ArithmeticError("failed to bracket constrained-MLE root")
    for _ in range(ROOT_ITERATIONS):
        midpoint = (lower + upper) / 2.0
        if secular(midpoint) > 0.0:
            lower = midpoint
        else:
            upper = midpoint

    multiplier = (lower + upper) / 2.0
    fitted = tuple(
        probability / (1.0 + multiplier * offset)
        for probability, offset in zip(empirical, offsets, strict=True)
    )
    total = sum(fitted)
    expectation = sum(
        outcome * probability
        for outcome, probability in zip(PAIR_SUPPORT, fitted, strict=True)
    )
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ArithmeticError(f"constrained MLE is not normalized: {total}")
    if not math.isclose(expectation, target, rel_tol=0.0, abs_tol=1e-9):
        raise ArithmeticError(
            f"constrained MLE expectation mismatch: {expectation} != {target}"
        )
    if any(probability <= 0.0 for probability in fitted):
        raise ArithmeticError("constrained MLE produced a non-positive probability")
    return fitted


def log_likelihood_ratio(counts: Sequence[int], elo0: float, elo1: float) -> float:
    """Return the generalized log likelihood ratio ``log(L(H1) / L(H0))``."""
    if not elo0 < elo1:
        raise ValueError("elo0 must be smaller than elo1")
    validated = _validate_counts(counts)
    observed = tuple(
        float(value) if value else REGULARIZATION for value in validated
    )
    total = sum(observed)
    empirical = tuple(value / total for value in observed)
    fitted_h0 = _constrained_mle(empirical, elo_to_score(elo0))
    fitted_h1 = _constrained_mle(empirical, elo_to_score(elo1))
    return sum(
        count * math.log(probability_h1 / probability_h0)
        for count, probability_h0, probability_h1 in zip(
            observed, fitted_h0, fitted_h1, strict=True
        )
    )


def evaluate(
    counts: Sequence[int],
    *,
    elo0: float = 0.0,
    elo1: float = 20.0,
    alpha: float = 0.05,
    beta: float = 0.05,
    min_pairs: int = 25,
) -> Verdict:
    """Evaluate a pentanomial logistic-Elo GSPRT snapshot."""
    validated = _validate_counts(counts)
    if not elo0 < elo1:
        raise ValueError("elo0 must be smaller than elo1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must lie strictly between zero and one")
    if min_pairs < 0:
        raise ValueError("min_pairs cannot be negative")

    pairs = sum(validated)
    points = sum(
        count * outcome
        for count, outcome in zip(validated, PAIR_SUPPORT, strict=True)
    )
    score = points / pairs if pairs else 0.5
    lower_bound = math.log(beta / (1.0 - alpha))
    upper_bound = math.log((1.0 - beta) / alpha)
    llr = log_likelihood_ratio(validated, elo0, elo1) if pairs else 0.0

    decision: Decision = "continue"
    if pairs >= min_pairs:
        if llr <= lower_bound:
            decision = "accept_h0"
        elif llr >= upper_bound:
            decision = "accept_h1"

    return Verdict(
        decision=decision,
        llr=llr,
        pairs=pairs,
        score=score,
        elo=score_to_elo(score),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        elo0=elo0,
        elo1=elo1,
        alpha=alpha,
        beta=beta,
        min_pairs=min_pairs,
    )
