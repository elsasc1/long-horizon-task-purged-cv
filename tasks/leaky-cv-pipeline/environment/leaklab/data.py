"""Synthetic customer-level dataset used by the examples and the tests."""
import numpy as np
import pandas as pd

N_FEATURES = 30
FEATURE_COLS = [f"x{j}" for j in range(N_FEATURES)]


def make_dataset(n_customers=400, rows_per_customer=5, n_segments=120, seed=0, null_labels=False):
    rng = np.random.default_rng(seed)
    seg_effect = rng.normal(size=n_segments)
    cust_seg = rng.integers(0, n_segments, size=n_customers)
    cust_latent = seg_effect[cust_seg] + 0.5 * rng.normal(size=n_customers)

    cust = np.repeat(np.arange(n_customers), rows_per_customer)
    n = len(cust)
    X = rng.normal(size=(n, N_FEATURES)) * (1.0 + np.arange(N_FEATURES))
    X[rng.random(size=X.shape) < 0.1] = np.nan
    base = np.nan_to_num(X[:, 0] / 1.0) * 0.9 - np.nan_to_num(X[:, 1] / 2.0) * 1.4
    logit = base + 1.2 * cust_latent[cust] + 0.5 * rng.normal(size=n)
    y = (rng.random(size=n) < 1.0 / (1.0 + np.exp(-logit))).astype(float)
    if null_labels:  # labels carry no information about X, but are shared inside a customer
        y = np.repeat(rng.integers(0, 2, size=n_customers), rows_per_customer).astype(float)

    df = pd.DataFrame(X, columns=FEATURE_COLS)
    df.insert(0, "segment", [f"seg_{s:03d}" for s in cust_seg[cust]])
    df.insert(0, "customer_id", cust)
    df["y"] = y
    perm = rng.permutation(n)
    return df.iloc[perm].reset_index(drop=True)
