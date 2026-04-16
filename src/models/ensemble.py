"""Regime-based ensemble weight optimization (any number of models)."""

import numpy as np

from .metrics import compute_rmse

REGIMES = {
    "night":   [0, 1, 2, 3, 4, 5],
    "morning": [6, 7, 8, 9],
    "day":     [10, 11, 12, 13, 14, 15, 16],
    "peak":    [17, 18, 19, 20, 21],
    "late":    [22, 23],
}

HOUR_TO_REGIME = {}
for _rname, _hours in REGIMES.items():
    for _h in _hours:
        HOUR_TO_REGIME[_h] = _rname


def _weight_combos(n_models, step=0.1):
    """Generate all non-negative weight tuples of length n_models summing to 1.0."""
    levels = int(round(1.0 / step))

    def _recurse(remaining, n_left):
        if n_left == 1:
            yield (round(remaining * step, 2),)
            return
        for k in range(remaining + 1):
            for rest in _recurse(remaining - k, n_left - 1):
                yield (round(k * step, 2),) + rest

    yield from _recurse(levels, n_models)


def optimize_regime_weights(models_dict, actual, hours, label, step=0.1):
    """Per-regime weight optimization for any number of models.

    Uses grid search over all weight combinations summing to 1.0.
    For n=6 step=0.1: 3003 combos/regime. For n=9: 43758 combos/regime.
    """
    names = list(models_dict.keys())
    n = len(names)
    regime_weights = {}
    ens_preds = np.zeros(len(actual))

    for rname, rhours in REGIMES.items():
        rmask = np.isin(hours, rhours)
        if rmask.sum() == 0:
            continue

        best_rmse = 999.0
        best_w = {names[0]: 1.0}
        a = actual[rmask]
        P = np.stack([models_dict[nm][rmask] for nm in names])

        for combo in _weight_combos(n, step):
            w = np.array(combo)
            e = w @ P
            r = float(np.sqrt(np.mean((a - e) ** 2)))
            if r < best_rmse:
                best_rmse = r
                best_w = {names[i]: round(combo[i], 2) for i in range(n)}

        regime_weights[rname] = best_w
        ens_preds[rmask] = sum(best_w.get(nm, 0) * models_dict[nm][rmask] for nm in names)
        w_str = " / ".join(f"{nm}={best_w.get(nm, 0):.1f}" for nm in names)
        print(f"    {rname:8s} (h={REGIMES[rname]}): {w_str}  RMSE={best_rmse:.2f}  n={rmask.sum()}")

    total_rmse = compute_rmse(actual, ens_preds)
    print(f"  {label} regime ensemble: RMSE={total_rmse:.2f}")
    return regime_weights, ens_preds


def apply_regime_weights(models_dict, hours, regime_weights):
    """Apply pre-computed regime weights to test predictions."""
    names = list(models_dict.keys())
    n = len(list(models_dict.values())[0])
    ens = np.zeros(n)
    for i in range(n):
        h = hours[i]
        rname = HOUR_TO_REGIME.get(h, "day")
        w = regime_weights.get(rname, {names[0]: 1.0})
        ens[i] = sum(w.get(nm, 0) * models_dict[nm][i] for nm in names)
    return ens
