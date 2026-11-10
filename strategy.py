"""MNQ Monday post-10:00 CT long drift."""
import numpy as np
import pandas as pd

FAMILY = "equity-dow-seasonality"
PARAMS = {"entry_sec": 36000, "exit_sec": 54540, "target_dow": 0}

def signals(panel):
    result = {}
    for sym, df in panel.items():
        sec = df.ct_sec.to_numpy()
        target = np.zeros(len(df), dtype=np.int64)
        for _, indices in df.groupby("session", sort=False).indices.items():
            ix = np.asarray(indices)
            session_date = pd.to_datetime(df.session.iloc[ix[0]])
            if session_date.dayofweek == PARAMS["target_dow"]:
                window = ix[(sec[ix] >= PARAMS["entry_sec"]) & (sec[ix] < PARAMS["exit_sec"])]
                target[window] = 1
        result[sym] = target
    return result

def entry_risk(panel):
    return {sym: np.full(len(df), 200.0) for sym, df in panel.items()}
