import numpy as np
from math import log10

class LogMinScaler:
    def __init__(self):
        self.scale_ = None
        self.min_ = None

    def fit(self, X):
        X = np.asarray(X)
        if np.any(X <= 10):
            raise ValueError("All values must be strictly positive to compute log10 and > 10 for scaling")
        self.min_ = np.min(X)
        self.scale_ = int(log10(self.min_))
        return self

    def transform(self, X):
        if self.scale_ is None:
            raise RuntimeError("You must fit the scaler before transforming data.")
        X = np.asarray(X)
        return X / (10 ** self.scale_)

    def inverse_transform(self, X_transformed):
        if self.scale_ is None:
            raise RuntimeError("You must fit the scaler before inverse transforming data.")
        X_transformed = np.asarray(X_transformed)
        return X_transformed * (10 ** self.scale_)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

if __name__ == "__main__":
    # Example usage
    data = np.array([1200, 100456700, 12000, 14356, 1987])
    print("Original Data:", data)
    scaler = LogMinScaler()
    
    # Fit the scaler
    scaler.fit(data)
    print("Scale Factor:", scaler.scale_)
    print("Minimum Value:", scaler.min_)
    
    # Transform the data
    transformed_data = scaler.transform(data)
    print("Transformed Data:", transformed_data)
    # Inverse transform the data
    inverse_data = scaler.inverse_transform(transformed_data)
    print("Inverse Transformed Data:", inverse_data)
    # Check if the inverse transformation is correct
    assert np.allclose(data, inverse_data), "Inverse transformation did not recover the original data."
