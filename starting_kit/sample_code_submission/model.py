from os.path import isfile
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator

class preprocessing ():
    def __init__(self, n_comp):
        self.scaler = MinMaxScaler()
        self.fit = None

    def fit_transform(self, X, y=None):
        self.fit = self.scaler.fit(X, y=y)
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def transform(self, X, y=None):
        self.fit = self.scaler.fit(X, y=y)
        X_scaled = self.scaler.transform(X)

        return X_scaled

class model(BaseEstimator):
    def __init__(self):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation. 
        '''
        self.num_train_samples = 38563
        self.num_feat = 59
        self.num_labels = 1
        self.is_trained = False
        self.preprocess = preprocessing(59)
        self.mod = RandomForestRegressor(n_estimators=60)

    def fit(self, X, y):
        '''
        This function should train the model parameters.
        Here we do nothing in this example...
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        For classification, labels could be either numbers 0, 1, ... c-1 for c
        classe or one-hot encoded vector of zeros, with a 1 at the kth position for
        class k.
        The AutoML format support on-hot encoding, which also works for
        multi-labels problems.
        Use data_converter.convert_to_num() to convert to the category number
        format.
        For regression, labels are continuous values.
        '''

        self.num_train_samples = X.shape[0]
        if X.ndim > 1: self.num_feat= X.shape[1]
        print("FIT: dim(X)= [{:d}, {:d}]".format(self.num_train_samples, self.num_feat))
        num_train_samples = y.shape[0]
        if y.ndim > 1: self.num_labels = y.shape[1]
        print("FIT: dim(y)= [{:d}, {:d}]".format(num_train_samples, self.num_labels))
        if (self.num_train_samples != num_train_samples):
            print("ARRGH: number of samples in X and y do not match!")
        X_preprocess = self.preprocess.fit_transform(X, y)
        self.mod.fit(X_preprocess, y)
        self.is_trained = True

    def predict(self, X):
        '''
        This function should provide predictions of labels on (test) data.
        Here we just return zeros...
        Make sure that the predicted values are in the correct format for the scoring
        metric. For example, binary classification problems often expect predictions
        in the form of a discriminant value (if the area under the ROC curve it the metric)
        rather that predictions of the class labels themselves. For multi-class or multi-labels
        problems, class probabilities are often expected if the metric is cross-entropy.
        Scikit-learn also has a function predict-proba, we do not require it.
        The function predict eventually can return probabilities.
        '''
        num_test_samples = X.shape[0]
        if X.ndim > 1: num_feat = X.shape[1]
        print("PREDICT: dim(X)= [{:d}, {:d}]".format(num_test_samples, num_feat))
        if (self.num_feat != num_feat):
            print("ARRGH: number of features in X does not match training data!")
        print("PREDICT: dim(y)= [{:d}, {:d}]".format(num_test_samples, self.num_labels))
        y = np.zeros([num_test_samples, self.num_labels])
        X_preprocess = self.preprocess.transform(X)
        y = self.mod.predict(X_preprocess)
        return y

    def save(self, path="./"):
        pickle.dump(self, open(path + "_model.pickle", "wb"))

    def load(self, path="./"):
        modelfile = path + "model.pickle"
        if isfile(modelfile):
            with open(modelfile, "rb") as f:
                self = pickle.load(f)
                print("Model reloaded from: " + modelfile)
        return self

def test():
    mod = model()


if __name__ == "__main__":
    test()
