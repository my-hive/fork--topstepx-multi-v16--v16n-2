"""CAD futures weekly reversal: fade trailing five-session causal return during RTH.
Hypothesis: commodity-FX weekly positioning mean-reverts in liquid US session."""
import numpy as np

FAMILY = "fx-cad-weekly-momentum-reversal"
PARAMS = {"lookback_sessions": 5, "entry_sec": 30600, "exit_sec": 54000, "risk": 280}


def signals(panel):
    df = panel["6M"]
    sec, close, sess = df.ct_sec.to_numpy(), df.close.to_numpy(), df.session.to_numpy()
    starts = np.r_[0, np.flatnonzero(sess[1:] != sess[:-1]) + 1, len(df)]
    ends = starts[1:] - 1
    k = PARAMS["lookback_sessions"]
    side = np.zeros(len(ends), dtype=np.int64)
    side[k:] = -np.sign(close[ends[k - 1:-1]] - close[ends[:-k]])
    target = np.zeros(len(df), dtype=np.int64)
    for i, (a, b) in enumerate(zip(starts[:-1], starts[1:])):
        ix = np.arange(a, b)
        target[ix[(sec[ix] >= PARAMS["entry_sec"]) & (sec[ix] < PARAMS["exit_sec"])]] = side[i]
    return {"6M": target}


def entry_risk(panel):
    return {s: np.full(len(df), PARAMS["risk"], dtype=float) for s, df in panel.items()}
