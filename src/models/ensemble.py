"""Regime-based ensemble weight optimization."""

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


def optimize_regime_weights(models_dict, actual, hours, label):
    """Per-regime weight optimization for 2-5 models. Returns dict of {regime: weights}."""
    names = list(models_dict.keys())
    n = len(names)
    regime_weights = {}
    ens_preds = np.zeros(len(actual))

    for rname, rhours in REGIMES.items():
        rmask = np.isin(hours, rhours)
        if rmask.sum() == 0:
            continue

        best = {"rmse": 999, "w": {names[0]: 1.0}}
        a = actual[rmask]
        p = {nm: models_dict[nm][rmask] for nm in names}

        if n == 1:
            best = {"rmse": compute_rmse(a, p[names[0]]), "w": {names[0]: 1.0}}
        elif n == 2:
            for w1 in np.arange(0.0, 1.05, 0.1):
                w2 = round(1.0 - w1, 1)
                e = w1 * p[names[0]] + w2 * p[names[1]]
                r = compute_rmse(a, e)
                if r < best["rmse"]:
                    best = {"rmse": r, "w": {names[0]: round(w1, 1), names[1]: w2}}
        elif n == 3:
            for w1 in np.arange(0.0, 1.05, 0.1):
                for w2 in np.arange(0.0, 1.05 - w1, 0.1):
                    w3 = round(1.0 - w1 - w2, 1)
                    if w3 < -0.01:
                        continue
                    e = w1 * p[names[0]] + w2 * p[names[1]] + w3 * p[names[2]]
                    r = compute_rmse(a, e)
                    if r < best["rmse"]:
                        best = {"rmse": r,
                                "w": {names[0]: round(w1, 1), names[1]: round(w2, 1), names[2]: w3}}
        elif n == 4:
            for w1 in np.arange(0.0, 1.05, 0.1):
                for w2 in np.arange(0.0, 1.05 - w1, 0.1):
                    for w3 in np.arange(0.0, 1.05 - w1 - w2, 0.1):
                        w4 = round(1.0 - w1 - w2 - w3, 1)
                        if w4 < -0.01:
                            continue
                        e = (w1 * p[names[0]] + w2 * p[names[1]] +
                             w3 * p[names[2]] + w4 * p[names[3]])
                        r = compute_rmse(a, e)
                        if r < best["rmse"]:
                            best = {"rmse": r,
                                    "w": {names[0]: round(w1, 1), names[1]: round(w2, 1),
                                          names[2]: round(w3, 1), names[3]: w4}}
        elif n == 5:
            for w1 in np.arange(0.0, 1.05, 0.1):
                for w2 in np.arange(0.0, 1.05 - w1, 0.1):
                    for w3 in np.arange(0.0, 1.05 - w1 - w2, 0.1):
                        for w4 in np.arange(0.0, 1.05 - w1 - w2 - w3, 0.1):
                            w5 = round(1.0 - w1 - w2 - w3 - w4, 1)
                            if w5 < -0.01:
                                continue
                            e = (w1 * p[names[0]] + w2 * p[names[1]] +
                                 w3 * p[names[2]] + w4 * p[names[3]] +
                                 w5 * p[names[4]])
                            r = compute_rmse(a, e)
                            if r < best["rmse"]:
                                best = {"rmse": r,
                                        "w": {names[0]: round(w1, 1), names[1]: round(w2, 1),
                                              names[2]: round(w3, 1), names[3]: round(w4, 1),
                                              names[4]: w5}}

        regime_weights[rname] = best["w"]
        ens_preds[rmask] = sum(best["w"].get(nm, 0) * p[nm] for nm in names)
        w_str = " / ".join(f"{nm}={best['w'].get(nm, 0):.1f}" for nm in names)
        print(f"    {rname:8s} (h={REGIMES[rname]}): {w_str}  RMSE={best['rmse']:.2f}  n={rmask.sum()}")

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
