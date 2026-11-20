import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _loss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


class LogisticModel:
    """Full-batch gradient-descent logistic regression with optional early stopping."""

    def __init__(self, lr=0.1, max_epochs=200, patience=10):
        self.lr, self.max_epochs, self.patience = lr, max_epochs, patience

    def fit(self, X, y, eval_set=None):
        n, d = X.shape
        w, b = np.zeros(d), 0.0
        best = (np.inf, w.copy(), b, 0)
        stale = 0
        for epoch in range(self.max_epochs):
            p = _sigmoid(X @ w + b)
            w = w - self.lr * (X.T @ (p - y) / n)
            b = b - self.lr * float((p - y).mean())
            if eval_set is not None:
                Xe, ye = eval_set
                cur = _loss(ye, _sigmoid(Xe @ w + b))
                if cur < best[0] - 1e-9:
                    best, stale = (cur, w.copy(), b, epoch + 1), 0
                else:
                    stale += 1
                    if stale >= self.patience:
                        break
        if eval_set is not None:
            _, self.w_, self.b_, self.n_epochs_ = best
        else:
            self.w_, self.b_, self.n_epochs_ = w, b, self.max_epochs
        return self

    def predict_proba(self, X):
        return _sigmoid(X @ self.w_ + self.b_)
