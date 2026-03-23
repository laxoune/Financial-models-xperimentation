
import numpy as np, pandas as pd

def gex_by_strike(df_calls, df_puts, S0, T, r=0.0):
    c = df_calls.groupby("strike")["open_interest"].sum().rename("call_oi")
    p = df_puts.groupby("strike")["open_interest"].sum().rename("put_oi")
    g = pd.concat([c,p], axis=1).fillna(0.0)
    g["gex"] = (g["call_oi"] - g["put_oi"]) * (g.index.values.astype(float)**2)
    g = g.reset_index().rename(columns={"index":"strike"})
    return g

def cluster_peaks(gex_df, top_n=3, smooth=3):
    x = gex_df["strike"].values.astype(float)
    y = gex_df["gex"].values.astype(float)
    if smooth>1:
        y = np.convolve(y, np.ones(smooth)/smooth, mode="same")
    peaks = []
    for i in range(1, len(y)-1):
        if y[i] >= y[i-1] and y[i] >= y[i+1]:
            peaks.append((x[i], y[i]))
    peaks = sorted(peaks, key=lambda t: abs(t[1]), reverse=True)[:top_n]
    return peaks

def pinning_uplift(grid, peaks, cap=1.5, width_frac=0.01, strength=0.25):
    grid = np.asarray(grid, dtype=float)
    uplift = np.ones_like(grid)
    for Kp, amp in peaks:
        scale = max(1e-6, Kp*width_frac)
        bump = 1.0 + strength * np.exp(-0.5*((grid - Kp)/scale)**2)
        uplift = np.minimum(uplift * bump, cap)
    return uplift
