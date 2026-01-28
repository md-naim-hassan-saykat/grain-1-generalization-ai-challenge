import numpy as np

class Model:
    def __init__(self):
        self.mean = None
        self.out_shape = None  # shape of one target (excluding batch)

    def train(self, X, y):
        y = np.asarray(y)
        self.mean = float(np.mean(y))

        # If y is (N, ...) store per-sample shape (...)
        if y.ndim >= 1:
            self.out_shape = y.shape[1:]  # could be () if y is 1D

    def predict(self, X):
        # batch size
        if hasattr(X, "shape") and X is not None and len(X.shape) >= 1:
            n = int(X.shape[0])
        else:
            n = len(X)

        # Build prediction with correct shape
        if self.out_shape is None or self.out_shape == ():
            y_pred = np.full((n,), self.mean, dtype=np.float32)
        else:
            y_pred = np.full((n,) + tuple(self.out_shape), self.mean, dtype=np.float32)

        return y_pred