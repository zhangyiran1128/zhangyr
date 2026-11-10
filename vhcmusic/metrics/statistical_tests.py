"""Statistical tests and confidence intervals."""

from __future__ import annotations

import numpy as np


def bootstrap_ci(values, statistic=np.mean, n_boot: int = 1000, ci: float = 0.95, seed: int = 16) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        stats.append(float(statistic(sample)))
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 1.0 - alpha))


def paired_ttest(a, b) -> dict[str, float | None]:
    try:
        from scipy.stats import ttest_rel
    except ImportError:
        return {"statistic": None, "pvalue": None}
    stat, p = ttest_rel(a, b)
    return {"statistic": float(stat), "pvalue": float(p)}


def wilcoxon_test(a, b) -> dict[str, float | None]:
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return {"statistic": None, "pvalue": None}
    stat, p = wilcoxon(a, b)
    return {"statistic": float(stat), "pvalue": float(p)}
