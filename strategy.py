"""6E First-Friday NFP release momentum drift: enter 07:45 CT if 06:45-07:45 move > 0, exit 14:45 CT."""
import numpy as np
import pandas as pd

FAMILY = "fx-nfp-session-momentum"
PARAMS = {"entry_sec": 27900, "exit_sec": 53100, "risk": 280}


def signals(panel):
    df = panel["6E"]
    sec = df.ct_sec.to_numpy()
    close = df.close.to_numpy()
    sess = df.session.to_numpy()
    starts = np.r_[0, np.flatnonzero(sess[1:] != sess[:-1]) + 1, len(df)]
    target = np.zeros(len(df), dtype=np.int64)
    for a, b in zip(starts[:-1], starts[1:]):
        s_date = pd.Timestamp(sess[a])
        if s_date.dayofweek != 4 or s_date.day > 7:
            continue
        s_sec = sec[a:b]
        s_c = close[a:b]
        pre = s_c[s_sec < 26100]   # before 07:15 CT
        post = s_c[s_sec < 27900]  # before 07:45 CT
        if not len(pre) or not len(post):
            continue
        side = int(np.sign(post[-1] - pre[-1]))
        if not side:
            continue
        ix = np.arange(a, b)
        target[ix[(s_sec >= PARAMS["entry_sec"]) & (s_sec < PARAMS["exit_sec"])]] = side
    return {"6E": target}


def entry_risk(panel):
    return {s: np.full(len(df), PARAMS["risk"], dtype=float) for s, df in panel.items()}
