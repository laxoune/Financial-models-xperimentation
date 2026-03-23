
import numpy as np
try:
    import scipy.optimize as opt
    SCIPY=True
except Exception:
    SCIPY=False

def fit_svi_slice(strikes, ivs, F):
    strikes = np.asarray(strikes, dtype=float)
    ivs = np.asarray(ivs, dtype=float)
    k = np.log(strikes / F + 1e-12)
    var = ivs**2

    if SCIPY and len(k)>=5:
        def obj(theta):
            a,b,rho,m,sigma = theta
            if b<=0 or abs(rho)>=1 or sigma<=0: return 1e9
            w = a + b*(rho*(k-m) + np.sqrt((k-m)**2 + sigma**2))
            return np.mean((w - var)**2)
        x0 = np.array([var.mean(), 0.2, 0.0, 0.0, 0.5])
        res = opt.minimize(obj, x0, method="Nelder-Mead")
        theta = res.x
        def iv_fun(K):
            kk = np.log(np.asarray(K)/F + 1e-12)
            w = theta[0] + theta[1]*(theta[2]*(kk-theta[3]) + np.sqrt((kk-theta[3])**2 + theta[4]**2))
            return np.sqrt(np.maximum(w,1e-9))
        return iv_fun
    else:
        coeffs = np.polyfit(k, var, deg=2)
        def iv_fun(K):
            kk = np.log(np.asarray(K)/F + 1e-12)
            w = coeffs[0]*kk**2 + coeffs[1]*kk + coeffs[2]
            return np.sqrt(np.maximum(w,1e-9))
        return iv_fun
