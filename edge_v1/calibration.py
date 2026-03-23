
import numpy as np

def pava(y_pred, y_true, sample_weight=None):
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    if sample_weight is None:
        sample_weight = np.ones_like(y_true, dtype=float)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)

    order = np.argsort(y_pred)
    x = y_pred[order]
    y = y_true[order]
    w = sample_weight[order]

    avg = y.copy()
    weight = w.copy()
    i = 0
    while i < len(avg) - 1:
        if avg[i] > avg[i+1]:
            new_avg = (avg[i]*weight[i] + avg[i+1]*weight[i+1]) / (weight[i]+weight[i+1])
            new_w   = weight[i] + weight[i+1]
            avg[i] = new_avg
            weight[i] = new_w
            avg = np.delete(avg, i+1)
            weight = np.delete(weight, i+1)
            x = np.delete(x, i+1)
            if i > 0: i -= 1
        else:
            i += 1

    thresholds = np.concatenate([[-np.inf], x, [np.inf]])
    values = np.concatenate([[avg[0]], avg, [avg[-1]]])

    def predict(p):
        p = np.asarray(p, dtype=float)
        idx = np.searchsorted(thresholds, p, side='right') - 1
        return values[idx]

    return (thresholds, values, predict)

def brier_score(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return np.mean((y_prob - y_true)**2)

def reliability_curve(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0,1,n_bins+1)
    idx = np.digitize(y_prob, bins) - 1
    obs, exp, count = [], [], []
    for b in range(n_bins):
        mask = idx==b
        if mask.sum()>0:
            obs.append(y_true[mask].mean())
            exp.append(y_prob[mask].mean())
            count.append(mask.sum())
        else:
            obs.append(np.nan); exp.append((bins[b]+bins[b+1])/2); count.append(0)
    return np.array(exp), np.array(obs), np.array(count)
