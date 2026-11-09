"""SI cash-session gap reversion: fade 09:00 CT gap versus prior session close."""
import numpy as np

FAMILY = "metals-gap-close-reversion"
PARAMS = {"entry_sec": 32400, "exit_sec": 54540}

def signals(panel):
    result = {}
    for sym, df in panel.items():
        close = df.close.to_numpy()
        sec = df.ct_sec.to_numpy()
        target = np.zeros(len(df), dtype=np.int64)
        prior_close = None
        for _, indices in df.groupby("session", sort=False).indices.items():
            ix = np.asarray(indices)
            at_entry = ix[sec[ix] >= PARAMS["entry_sec"]]
            if prior_close is not None and len(at_entry):
                gap = close[at_entry[0]] - prior_close
                window = ix[(sec[ix] >= PARAMS["entry_sec"]) & (sec[ix] < PARAMS["exit_sec"])]
                target[window] = -int(np.sign(gap))
            closes = ix[sec[ix] <= PARAMS["exit_sec"]]
            if len(closes):
                prior_close = close[closes[-1]]
        result[sym] = target
    return result

def entry_risk(panel):
    return {sym: np.zeros(len(df)) for sym, df in panel.items()}
