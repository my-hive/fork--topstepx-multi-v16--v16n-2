"""Corn/soy ratio reversion: fade 40-session causal ZC/ZS close-ratio z-score in next RTH, hedged legs.
Hypothesis: supply/demand cross-substitution makes extreme daily ZC/ZS departures mean-revert."""
import numpy as np
import pandas as pd

FAMILY = "grain-corn-soy-ratio-reversion"
PARAMS = {"lookback": 40, "z_entry": 1.5, "entry_sec": 30600, "exit_sec": 54000, "risk": 280}


def _positions(panel):
    out = {sym: np.zeros(len(df), dtype=np.int64) for sym, df in panel.items()}
    if "ZC" not in panel or "ZS" not in panel:
        return out, None
    zc, zs = panel["ZC"], panel["ZS"]
    c1 = zc.groupby("session", sort=False).close.last()
    c2 = zs.groupby("session", sort=False).close.last()
    common = c1.index.intersection(c2.index)
    if len(common) < PARAMS["lookback"] + 2:
        return out, None
    ratio = np.log(c1.loc[common] / c2.loc[common])
    z = (ratio - ratio.rolling(PARAMS["lookback"]).mean()) / ratio.rolling(PARAMS["lookback"]).std()
    side = pd.Series(0.0, index=common)
    side[z > PARAMS["z_entry"]] = -1.0
    side[z < -PARAMS["z_entry"]] = 1.0
    return out, side.shift(1).fillna(0.0)


def _apply(panel, out, side, key, sign):
    df = panel[key]
    sec = df.ct_sec.to_numpy()
    sess = df.session.to_numpy()
    for session, pos in side.items():
        if not pos:
            continue
        ix = np.flatnonzero(sess == session)
        if not len(ix):
            continue
        ix = ix[(sec[ix] >= PARAMS["entry_sec"]) & (sec[ix] < PARAMS["exit_sec"])]
        out[key][ix] = int(sign * pos)


def signals(panel):
    out, side = _positions(panel)
    if side is None:
        return out
    _apply(panel, out, side, "ZC", 1)
    _apply(panel, out, side, "ZS", -1)
    return out


def entry_risk(panel):
    return {s: np.full(len(df), PARAMS["risk"], dtype=float) for s, df in panel.items()}
