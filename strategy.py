"""Gold daily trend regime: causal 20-session return selects RTH direction.
Hypothesis: slow macro trend persists through liquid COMEX US session, avoiding overnight noise."""
import numpy as np

FAMILY = "metals-daily-trend-rth"
PARAMS = {"lookback_sessions": 20, "entry_sec": 30600, "exit_sec": 54000, "risk": 280}


def signals(panel):
    result = {}
    for sym, df in panel.items():
        sec = df.ct_sec.to_numpy()
        close = df.close.to_numpy()
        session = df.session.to_numpy()
        target = np.zeros(len(df), dtype=np.int64)
        starts = np.r_[0, np.flatnonzero(session[1:] != session[:-1]) + 1, len(df)]
        closes = close[starts[1:] - 1]
        direction = np.zeros(len(closes), dtype=np.int64)
        direction[PARAMS["lookback_sessions"]:] = np.sign(closes[PARAMS["lookback_sessions"] - 1:-1] - closes[:-PARAMS["lookback_sessions"]])
        for k, (a, b) in enumerate(zip(starts[:-1], starts[1:])):
            ix = np.arange(a, b)
            target[ix[(sec[ix] >= PARAMS["entry_sec"]) & (sec[ix] < PARAMS["exit_sec"])]] = direction[k]
        result[sym] = target
    return result


def entry_risk(panel):
    return {sym: np.full(len(df), PARAMS["risk"], dtype=float) for sym, df in panel.items()}
