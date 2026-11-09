"""Metals cash-open (08:20 ET / 07:20 CT) breakout, NY session only, vol cap, stop at range opposite edge."""
import numpy as np
FAMILY = "metals-comex-orb-2"
PARAMS = {"or_start": 26400, "or_end": 28200, "exit_sec": 50400, "risk_budget": 250, "max_qty": 20, "fade": True}
PV = {"GC":100,"MGC":10,"SI":5000,"SIL":1000,"HG":25000,"MHG":2500,"PL":50}

def _levels(df):
    p=PARAMS; hi=np.full(len(df),np.nan); lo=np.full(len(df),np.nan)
    h,l,s=df.high.to_numpy(),df.low.to_numpy(),df.ct_sec.to_numpy()
    for ix in df.groupby('session',sort=False).indices.values():
        m=(s[ix]>=p['or_start'])&(s[ix]<p['or_end'])
        if m.sum()<5: continue
        rh,rl=h[ix][m].max(),l[ix][m].min()
        after=s[ix]>=p['or_end']
        hi[ix[after]]=rh;lo[ix[after]]=rl
    return hi,lo

def entry_risk(panel):
    out={}
    for sym,df in panel.items():
        hi,lo=_levels(df)
        r=(hi-lo)*PV[sym]
        out[sym]=np.nan_to_num(r,nan=0.0)
    return out

def signals(panel):
    p=PARAMS;res={}
    for sym,df in panel.items():
        hi,lo=_levels(df);c,s=df.close.to_numpy(),df.ct_sec.to_numpy();out=np.zeros(len(df),dtype=np.int64)
        for ix in df.groupby('session',sort=False).indices.values():
            pos=0;done=False;q=0
            for j in ix:
                if np.isnan(hi[j]) or s[j]>=p['exit_sec']: pos=0
                elif pos==0 and not done:
                    rk=(hi[j]-lo[j])*PV[sym]
                    if 0<rk<=p['risk_budget'] and (c[j]>hi[j] or c[j]<lo[j]):
                        q=int(min(p['max_qty'],max(1,p['risk_budget']//rk)))
                        pos=q if c[j]>hi[j] else -q
                        if p.get('fade'): pos=-pos
                elif pos>0 and c[j]<lo[j]: pos=0;done=True
                elif pos<0 and c[j]>hi[j]: pos=0;done=True
                out[j]=pos
        res[sym]=out
    return res
