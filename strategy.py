"""Gold weekly momentum reversal: fade the trailing 5-session RTH return sign, hold RTH 08:30-14:30 CT.
Hypothesis: gold intraday flows overextend over a week and partially unwind in the next session."""
import numpy as np

FAMILY = "metals-gold-weekly-momentum-reversal"
PARAMS = {"lookback_sessions": 5, "entry_sec": 30600, "exit_sec": 52200, "risk": 280}


def signals(panel):
    df = panel["GC"]
    sec = df.ct_sec.to_numpy()
    close = df.close.to_numpy()
    sess = df.session.to_numpy()
    starts = np.r_[0, np.flatnonzero(sess[1:] != sess[:-1]) + 1, len(df)]
    closes = close[starts[1:] - 1]
    n = len(closes)
    k = PARAMS["lookback_sessions"]
    side = np.zeros(n, dtype=np.int64)
    side[k:] = -np.sign(closes[k - 1:-1] - closes[:-k])
    target = np.zeros(len(df), dtype=np.int64)
    for i, (a, b) in enumerate(zip(starts[:-1], starts[1:])):
        if side[i]:
            ix = np.arange(a, b)
            target[ix[(sec[ix] >= PARAMS["entry_sec"]) & (sec[ix] < PARAMS["exit_sec"])]] = side[i]
    return {"GC": target}


def entry_risk(panel):
    return {s: np.full(len(df), PARAMS["risk"], dtype=float) for s, df in panel.items()}
