import numpy as np


class LinearSVM:
    def __init__(
        self,
        lr=0.01,
        C=1.0,
        epochs=100,
        batch_size=32,
        seed=None,
        verbose=False,
        callback=None,
    ):
        self.lr = lr
        self.C = C
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose
        self.callback = callback
        self.w = None
        self.b = 0.0

    def _init_params(self, d):
        rng = np.random.RandomState(self.seed)
        self.w = rng.normal(scale=0.01, size=d)
        self.b = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n, d = X.shape
        if self.w is None:
            self._init_params(d)

        for ep in range(self.epochs):
            if self.seed is not None:
                rng = np.random.RandomState(self.seed + ep)
                perm = rng.permutation(n)
            else:
                perm = np.random.permutation(n)

            for start in range(0, n, self.batch_size):
                xb = X[perm[start : start + self.batch_size]]
                yb = y[perm[start : start + self.batch_size]]

                if xb.shape[0] == 0:
                    continue
                grad_w, grad_b = hinge_grad_step(self.w, self.b, xb, yb, self.C)
                self.w -= grad_w * self.lr
                self.b -= grad_b * self.lr

            if self.callback is not None:
                self.callback(self, ep, X, y)

            if self.verbose and (ep % max(1, self.epochs // 5) == 0):
                print(f"epoch {ep}/{self.epochs}  obj={self._objective(X,y):.4f}")

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        return X.dot(self.w) + self.b

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)

    def _objective(self, X, y):
        scores = y * (X.dot(self.w) + self.b)
        hinge = np.maximum(0.0, 1.0 - scores).mean()
        obj = 0.5 * np.dot(self.w, self.w) + self.C * hinge
        return obj


class DAGSVM:
    def __init__(
        self,
        lr=0.001,
        C=10.0,
        epochs=300,
        batch_size=64,
        seed=None,
        verbose=False,
        callback=None,
    ):
        self.lr = lr
        self.C = C
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose
        self.callback = callback
        self.classes = None
        self.pair_clfs = {}

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes = np.unique(y)
        self.pair_clfs = {}

        for i in range(len(self.classes)):
            for j in range(i + 1, len(self.classes)):
                c1, c2 = self.classes[i], self.classes[j]

                mask = (y == c1) | (y == c2)
                X_pair = X[mask]
                y_pair = y[mask]

                y_binary = np.where(y_pair == c1, -1, 1)

                clf = LinearSVM(
                    lr=self.lr,
                    C=self.C,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    seed=self.seed,
                    verbose=self.verbose,
                    callback=self.callback,
                )
                clf.fit(X_pair, y_binary)

                self.pair_clfs[(i, j)] = clf

                if self.verbose:
                    y_pred = clf.predict(X_pair)
                    acc = np.mean(y_pred == y_binary)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        preds = []

        k = len(self.classes)
        for idx in range(n):
            x = X[idx : idx + 1]
            winner_idx = 0
            for challenger_idx in range(1, k):
                if winner_idx < challenger_idx:
                    i, j = winner_idx, challenger_idx

                    clf = self.pair_clfs[(i, j)]
                    score = clf.decision_function(x)[0]
                    if score > 0:
                        winner_idx = challenger_idx
                else:
                    i, j = challenger_idx, winner_idx

                    clf = self.pair_clfs[(i, j)]
                    score = clf.decision_function(x)[0]
                    if score < 0:
                        winner_idx = challenger_idx
            preds.append(self.classes[winner_idx])
        return np.array(preds, dtype=self.classes.dtype)


def hinge_grad_step(w, b, X_batch, y_batch, C):
    X_batch = np.asarray(X_batch, dtype=float)
    y_batch = np.asarray(y_batch, dtype=float)
    scores = X_batch.dot(w) + b
    margins = y_batch * scores
    mask = margins < 1.0
    if mask.any():
        grad_w_hinge = -np.mean((y_batch[mask][:, None] * X_batch[mask]), axis=0)
        grad_b_hinge = -np.mean(y_batch[mask])
    else:
        grad_w_hinge = np.zeros_like(w)
        grad_b_hinge = 0.0

    grad_w = w + C * grad_w_hinge
    grad_b = C * grad_b_hinge
    return grad_w, grad_b
