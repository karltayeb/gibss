from dataclasses import dataclass
import numpy as np


@dataclass
class CS:
    cs: np.ndarray
    alpha: np.ndarray
    size: int
    coverage: float
    target_coverage: float


def compute_cs(alpha, target_coverage=0.95):
    idx = np.argsort(-alpha)
    mass = np.cumsum(alpha[idx])
    size = np.argmax(mass > 0.95) + 1
    return CS(
        cs=idx[:size],
        alpha=alpha[idx[:size]],
        size=int(size),
        coverage=float(mass[size - 1]),
        target_coverage=target_coverage,
    )
