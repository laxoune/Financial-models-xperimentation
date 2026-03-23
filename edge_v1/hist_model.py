
import numpy as np, pandas as pd

class RollingBandModel:
    def __init__(self, band_edges, window=260, kernel_bw=None):
        self.band_edges = np.asarray(band_edges, dtype=float)
        self.window = int(window)
        self.kernel_bw = kernel_bw

    def _band_index(self, ret):
        absret = np.abs(ret)
        idx = np.searchsorted(self.band_edges, absret, side="right") - 1
        idx = np.clip(idx, 0, len(self.band_edges)-2)
        return idx

    def fit(self, df, feature_cols, ret_col, date_col):
        self.feature_cols = feature_cols
        self.df = df[[date_col, ret_col] + feature_cols].copy().reset_index(drop=True)
        self.df["band_idx"] = self.df[ret_col].apply(self._band_index)
        self.date_col = date_col; self.ret_col = ret_col
        return self

    def predict_proba_at(self, i):
        if i <= 0: 
            n_bands = len(self.band_edges)-1
            return np.ones(n_bands)/n_bands
        start = max(0, i - self.window)
        hist = self.df.iloc[start:i]
        n_bands = len(self.band_edges)-1

        xq = self.df.iloc[i][self.feature_cols].values.astype(float)
        X = hist[self.feature_cols].values.astype(float)
        if self.kernel_bw is None:
            bw = np.std(X, axis=0) + 1e-9
        else:
            bw = np.asarray(self.kernel_bw, dtype=float)
            if bw.size==1: bw = np.ones(X.shape[1])*bw

        d2 = ((X - xq)**2 / (bw**2)).sum(axis=1)
        w = np.exp(-0.5*d2)
        if w.sum()==0: w = np.ones_like(w)

        probs = np.zeros(n_bands)
        for j in range(start, i):
            idx = int(self.df.iloc[j]["band_idx"])
            probs[idx] += w[j-start]
        if probs.sum()==0: probs += 1.0
        probs /= probs.sum()
        return probs
