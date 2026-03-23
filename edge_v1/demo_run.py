
import numpy as np, pandas as pd, os, matplotlib.pyplot as plt, json
from calibration import pava, brier_score, reliability_curve
from payoff import payoff_from_legs, butterfly_legs
from data_io import clean_chain, nbbo_mid, mid_at
from iv_surface import fit_svi_slice
from rnd import call_curve_from_iv, density_from_calls
from hist_model import RollingBandModel
from pinning import gex_by_strike, cluster_peaks, pinning_uplift

out_dir = "/Users/alexanderdagher/Documents/edge_v1/outputs"
os.makedirs(out_dir, exist_ok=True)

np.random.seed(7)
S0 = 100.0
r = 0.02
T_years = 5/252
dates = pd.date_range("2024-01-05", periods=30, freq="W-FRI")
S_path = S0 * np.cumprod(1 + np.random.normal(0, 0.02, size=len(dates)))

strikes = np.arange(70, 131, 1.0)
F = S_path[-1] * np.exp(r*T_years)
iv = 0.24 + 0.0007*(strikes - S_path[-1])**2/1000.0
bid = 0.01 + 0.5*np.maximum(0, (S_path[-1]-strikes))
ask = bid + 0.05 + 0.02*np.abs(strikes - S_path[-1])/S_path[-1]
call_df = pd.DataFrame({"type":"C","strike":strikes,"bid":bid,"ask":ask,"iv":iv,"open_interest":(800*np.exp(-((strikes - S_path[-1])/5)**2)).astype(int)})
put_df  = pd.DataFrame({"type":"P","strike":strikes,"bid":bid/2,"ask":ask/2,"iv":iv,"open_interest":(400*np.exp(-((strikes - (S_path[-1]-2))/6)**2)).astype(int)})
chain = clean_chain(pd.concat([call_df, put_df], ignore_index=True))

iv_fun = fit_svi_slice(strikes, iv, F)
K_grid = np.linspace(strikes.min(), strikes.max(), 800)
call_curve = call_curve_from_iv(K_grid, iv_fun, S_path[-1], r, T_years)
p_rnd = density_from_calls(K_grid, call_curve, r, T_years)

rets = pd.Series(np.diff(S_path) / S_path[:-1]).reindex(range(len(S_path)), fill_value=0)
df_hist = pd.DataFrame({
    "date": dates,
    "ret_w": rets.values,
    "rv": pd.Series(rets).rolling(5).std().bfill().values,
    "vov": pd.Series(rets).rolling(5).std().diff().abs().bfill().values,
    "slope": pd.Series(rets).rolling(5).mean().bfill().values,
    "gaps": np.abs(np.random.normal(0,0.005,size=len(dates)))
})
band_edges = np.array([0.0, 0.005, 0.01, 0.015, 0.03])
rbm = RollingBandModel(band_edges=band_edges, window=20, kernel_bw=0.01)
rbm.fit(df_hist, feature_cols=["rv","vov","slope","gaps"], ret_col="ret_w", date_col="date")
P_hist = rbm.predict_proba_at(len(df_hist)-1)

def hist_pdf_over_K(K_grid, S_ref, band_edges, probs):
    pdf = np.zeros_like(K_grid, dtype=float)
    centers = (band_edges[:-1]+band_edges[1:])/2.0
    for c, p in zip(centers, probs):
        mean = S_ref*(1+c)
        sd = S_ref*0.005 + S_ref*c/2
        pdf += p * np.exp(-0.5*((K_grid-mean)/sd)**2) / (sd*np.sqrt(2*np.pi))
    if pdf.sum()>0:
        pdf /= np.trapz(pdf, K_grid)
    return pdf
p_hist = hist_pdf_over_K(K_grid, S_path[-1], band_edges, P_hist)

gex = gex_by_strike(call_df, put_df, S_path[-1], T_years, r)
peaks = cluster_peaks(gex, top_n=2, smooth=5)
uplift = pinning_uplift(K_grid, peaks, cap=1.4, width_frac=0.012, strength=0.3)

w1, w2 = 0.55, 0.45
p_ens = w1*p_rnd + w2*p_hist
p_ens = p_ens * uplift
p_ens = p_ens / np.trapz(p_ens, K_grid)

y_prob = np.clip(np.random.beta(2,5, size=100), 0, 1)
y_true = (np.random.rand(100) < y_prob*0.9).astype(int)
thr, vals, isotonic_predict = pava(y_prob, y_true)
brier_raw = brier_score(y_true, y_prob)
brier_cal = brier_score(y_true, isotonic_predict(y_prob))

K_body, w = float(np.median(strikes)), 2.0
legs = butterfly_legs(K_body, w)
payoff = payoff_from_legs(K_grid, legs)
debit = (mid_at(chain,"C",K_body-w) - 2*mid_at(chain,"C",K_body) + mid_at(chain,"C",K_body+w))
debit = float(debit if debit is not None else 0.10)
EV = float(np.trapz(payoff * p_ens, K_grid) - debit)

plt.figure()
plt.plot(K_grid, p_rnd, label="P_RND")
plt.plot(K_grid, p_hist, label="P_HIST")
plt.plot(K_grid, p_ens, label="P_ENS*")
plt.legend(); plt.title("Distributions (same expiry)"); plt.xlabel("S_T"); plt.ylabel("Density")
plt.savefig(os.path.join(out_dir,"curves.png"), dpi=160, bbox_inches="tight"); plt.close()

plt.figure()
plt.plot(K_grid, payoff, label="Butterfly Payoff")
plt.plot(K_grid, p_ens, label="P_ENS*")
plt.title(f"Example Fly K={K_body}, w={w} | EV≈{EV:.3f}")
plt.legend(); plt.xlabel("S_T"); plt.ylabel("Value / Density")
plt.savefig(os.path.join(out_dir,"payoff_vs_pdf.png"), dpi=160, bbox_inches="tight"); plt.close()

exp, obs, cnt = reliability_curve(y_true, isotonic_predict(y_prob))
plt.figure()
plt.plot(exp, obs, marker="o"); plt.plot([0,1],[0,1],"--")
plt.title(f"Reliability (Brier raw={brier_raw:.3f} → cal={brier_cal:.3f})")
plt.xlabel("Predicted"); plt.ylabel("Observed")
plt.savefig(os.path.join(out_dir,"reliability.png"), dpi=160, bbox_inches="tight"); plt.close()

summary = {
  "example_EV": EV,
  "brier_raw": brier_raw,
  "brier_calibrated": brier_cal,
  "pinning_peaks": [(float(k), float(a)) for (k,a) in peaks]
}
with open(os.path.join(out_dir,"summary.json"),"w") as f:
    json.dump(summary, f, indent=2)

print("Demo complete.")
