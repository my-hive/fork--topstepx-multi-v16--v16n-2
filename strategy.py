"""Equity index closing-ramp: long ES from 14:00 CT to the 15:09 flat.
Hypothesis: passive/index MOC imbalance creates a measurable upward ramp into the close."""
import numpy as np

FAMILY = "equity-index-close-ramp"
PARAMS = {"entry_sec": 50400, "exit_sec": 54000, "risk": 280}


def signals(panel):
    df = panel["ES"]
    sec = df.ct_sec.to_numpy()
    return {"ES": ((sec >= PARAMS["entry_sec"]) & (sec < PARAMS["exit_sec"])).astype(np.int64)}


def entry_risk(panel):
    return {s: np.full(len(df), PARAMS["risk"], dtype=float) for s, df in panel.items()}
