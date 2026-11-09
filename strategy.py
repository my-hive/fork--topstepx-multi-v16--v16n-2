"""Gold Friday RTH long drift: enter 07:30 CT Friday, exit at daily flat (15:09 CT).
Hypothesis: pre-weekend risk premium in gold; positions held only within the CME US session."""
import numpy as np
import pandas as pd

FAMILY = "metals-gold-friday-rth-drift"
PARAMS = {"target_dow": 4, "entry_sec": 30600, "exit_sec": 54000, "point_value": 10}


def signals(panel):
    result = {}
    for sym, df in panel.items():
        sec = df.ct_sec.to_numpy()
        dow = pd.to_datetime(df.session).dt.day_of_week.to_numpy()
        target = ((dow == PARAMS["target_dow"]) & (sec >= PARAMS["entry_sec"]) & (sec < PARAMS["exit_sec"])).astype(np.int64)
        result[sym] = target
    return result


def entry_risk(panel):
    return {s: np.full(len(df), 280.0) for s, df in panel.items()}
