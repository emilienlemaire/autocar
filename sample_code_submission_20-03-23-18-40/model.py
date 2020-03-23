import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator

class preprocessing ():
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.fit = None
    
    def fit_transform(self, X, y=None):
        self.fit = self.scaler.fit(X, y=y)
        X_scaled = self.scaler.fit_transform(X, y=y)

        return X_scaled
    
    def transform(self, X, y=None):
        if self.fit == None:
            self.fit = self.scaler.fit(X, y=y)
        X_scaled = self.scaler.transform(X)

        return X_scaled

class model (BaseEstimator):
    def __init__(self):
        self.num_train_samples = 38563
        self.num_feat = 59
        self.num_labels = 1
        self.is_trained = False
        self.preprocess = preprocessing()
        self.mod = RandomForestRegressor(n_estimators=60)

    def fit(self, X, y):
        X_preprocess = self.preprocess.fit_transform(X, y)
        self.mod.fit(X_preprocess, y)
        self.is_trained = True

    def predict(self, X):
        num_test_samples = X.shape[0]
        if X.ndim>1: self.num_feat = X.shape[1]
        y = np.zeros([num_test_samples, self.num_labels])
        
        X_preprocess = self.preprocess.transform(X)
        y = self.mod.predict(X_preprocess)
        return y

    def save(self, path="./"):
        pass

    def load(self, path="./"):
        pass

def test():
    mod = model()


if __name__ == "__main__":
    test()
