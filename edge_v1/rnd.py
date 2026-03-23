
import numpy as np

def bs_call_price(S, K, r, sigma, T):
    if T<=0: return max(S-K,0.0)
    from math import log, sqrt, exp
    try:
        from math import erf
        def cdf(x): return 0.5*(1+erf(x/np.sqrt(2)))
    except Exception:
        import scipy.stats as st
        def cdf(x): return st.norm.cdf(x)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T) + 1e-12)
    d2 = d1 - sigma*np.sqrt(T)
    return S*cdf(d1) - K*np.exp(-r*T)*cdf(d2)

def call_curve_from_iv(K_grid, iv_fun, S0, r, T):
    K_grid = np.asarray(K_grid, dtype=float)
    C = []
    for K in K_grid:
        sigma = float(iv_fun(K))
        C.append(bs_call_price(S0, K, r, sigma, T))
    return np.asarray(C)

def density_from_calls(K_grid, call_prices, r, T):
    K = np.asarray(K_grid, dtype=float)
    C = np.asarray(call_prices, dtype=float)
    n = len(K)
    f = np.zeros_like(C)
    win = max(5, int(0.03*n))
    if win % 2 == 0: win += 1
    half = win//2
    for i in range(n):
        lo = max(0, i-half)
        hi = min(n, i+half+1)
        Ki = K[lo:hi]; Ci = C[lo:hi]
        if len(Ki)>=3:
            A = np.vstack([Ki**2, Ki, np.ones_like(Ki)]).T
            coef, *_ = np.linalg.lstsq(A, Ci, rcond=None)
            a = coef[0]
            f[i] = np.exp(r*T) * 2*a
        else:
            f[i] = 0.0
    f = np.maximum(f, 0.0)
    area = np.trapz(f, K)
    if area>0:
        f /= area
    return f
