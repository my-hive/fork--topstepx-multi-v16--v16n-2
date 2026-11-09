"""RTY/YM relative-value reversion: 60-session causal RTH ratio z-score, hedged pair next RTH.
Hypothesis: small-vs-bluechip relative mispricing mean-reverts after extreme daily ratio departures."""
import numpy as np
import pandas as pd

FAMILY = "index-small-large-relative-reversion"
PARAMS = {"lookback": 60, "z_entry": 2.0, "entry_sec": 30600, "exit_sec": 54000, "risk": 280}


def _positions(panel):
    out = {}
    for sym, df in panel.items():
        out[sym] = np.zeros(len(df), dtype=np.int64)
    if "RTY" not in panel or "YM" not in panel:
        return out, None
    rty, ym = panel["RTY"], panel["YM"]
    rty_last = rty.groupby("session", sort=False).close.last()
    ym_last = ym.groupby("session", sort=False).close.last()
    common = rty_last.index.intersection(ym_last.index)
    if len(common) < PARAMS["lookback"] + 2:
        return out, None
    ratio = np.log(rty_last.loc[common] / ym_last.loc[common])
    z = (ratio - ratio.rolling(PARAMS["lookback"]).mean()) / ratio.rolling(PARAMS["lookback"]).std()
    side = pd.Series(0.0, index=common)
    side[z > PARAMS["z_entry"]] = -1.0
    side[z < -PARAMS["z_entry"]] = 1.0
    side = side.shift(1).fillna(0.0)
    return out, side


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
    _apply(panel, out, side, "RTY", 1)
    _apply(panel, out, side, "YM", -1)
    return out


def entry_risk(panel):
    return {s: np.full(len(df), PARAMS["risk"], dtype=float) for s, df in panel.items()}
