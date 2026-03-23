
import numpy as np

def payoff_from_legs(St, legs):
    St = np.asarray(St, dtype=float)
    total = np.zeros_like(St)
    for leg in legs:
        K = float(leg["K"]); q = float(leg["qty"])
        t = leg["type"].upper()
        if t == "C":
            total += q * np.maximum(St - K, 0.0)
        elif t == "P":
            total += q * np.maximum(K - St, 0.0)
        else:
            raise ValueError("Unknown leg type: "+str(leg["type"]))
    return total

def butterfly_legs(K, w):
    return [
        {"type":"C","K":K-w,"qty":+1.0},
        {"type":"C","K":K,"qty":-2.0},
        {"type":"C","K":K+w,"qty":+1.0},
    ]
