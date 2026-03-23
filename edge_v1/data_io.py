
import numpy as np, pandas as pd

def clean_chain(df: pd.DataFrame):
    df = df.copy()
    req = ["type","strike","bid","ask"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Missing column {c}")
    df["type"] = df["type"].str.upper().str.strip()
    for c in ["strike","bid","ask","iv","open_interest"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[(df["bid"]>=0) & (df["ask"]>=0) & (df["ask"]>=df["bid"])]
    mid = (df["bid"]+df["ask"])/2.0
    spread = df["ask"]-df["bid"]
    df = df[spread <= 4*mid.replace(0,np.nan).fillna(spread)]
    return df

def mid_at(df: pd.DataFrame, opt_type: str, K: float):
    row = df[(df["type"]==opt_type.upper()) & (np.isclose(df["strike"],K))]
    if len(row)==0: return None
    b,a = float(row["bid"].iloc[0]), float(row["ask"].iloc[0])
    return 0.5*(b+a)

def nbbo_mid(df: pd.DataFrame):
    df = df.copy()
    df["mid"] = (df["bid"] + df["ask"])/2.0
    return df
