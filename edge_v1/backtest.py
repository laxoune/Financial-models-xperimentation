
import numpy as np, pandas as pd

def clipped_kelly(ev, var, cap=0.02):
    if var<=0: return 0.0
    f = ev/var
    return float(np.clip(f, 0.0, cap))

class WalkForwardEV:
    def __init__(self, ev_threshold=0.10, fees=0.00, slippage=0.0):
        self.ev_threshold = ev_threshold
        self.fees = fees
        self.slippage = slippage

    def run(self, dates, choose_trade_fn, realize_fn, bankroll=1.0):
        equity = bankroll
        history = []
        for d in dates:
            tr = choose_trade_fn(d)
            if tr is None or tr.get("EV",0.0) < self.ev_threshold:
                history.append({"date": d, "equity": equity, "trade": None, "pnl": 0.0})
                continue
            f = clipped_kelly(tr["EV"], tr.get("var", (tr["max_pay"]/2)**2), cap=0.05)
            units = max(0, int(equity * f / max(tr["debit"], 1e-6)))
            if units<=0:
                history.append({"date": d, "equity": equity, "trade": None, "pnl": 0.0})
                continue
            pnl_per = realize_fn(d, tr) - self.fees - self.slippage
            pnl = units * pnl_per
            equity += pnl
            history.append({"date": d, "equity": equity, "trade": tr, "pnl": pnl, "units": units})
        return pd.DataFrame(history)
