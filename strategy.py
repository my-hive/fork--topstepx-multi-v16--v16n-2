"""PL overnight-gap continuation: one trade per session at the 08:20 CT metals open, exit 13:20 CT."""
import numpy as np
FAMILY = "metals-pl-gap-follow"
PARAMS = {"ref_sec": 48000, "entry_sec": 30000, "exit_sec": 48000, "min_gap_ticks": 20}

def signals(panel):
    res = {}
    tick = {"PL": 0.1}
    for sym, df in panel.items():
        c, s = df.close.to_numpy(), df.ct_sec.to_numpy()
        out = np.zeros(len(df), dtype=np.int64)
        prev_ref = None
        for sess, ix in df.groupby('session', sort=False).indices.items():
            ix = np.asarray(ix)
            en = ix[s[ix] >= PARAMS["entry_sec"]]
            if prev_ref is not None and len(en):
                gap = c[en[0]] - prev_ref
                if abs(gap) >= PARAMS["min_gap_ticks"] * tick.get(sym, 0.1):
                    win = ix[(s[ix] > PARAMS["entry_sec"]) & (s[ix] < PARAMS["exit_sec"])]
                    out[win] = int(np.sign(gap))
            ref = ix[s[ix] <= PARAMS["ref_sec"]]
            if len(ref): prev_ref = c[ref[-1]]
        res[sym] = out
    return res

def entry_risk(panel):
    return {sym: np.zeros(len(df)) for sym, df in panel.items()}
