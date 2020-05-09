from os.path import isfile
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator


class preprocessing ():
    '''
    This class is the preprocessing class of our model.
    It only scales the data (from 0. to 1.) as in our tests
    we found out that outlier detection and dimension reduction
    reduced our scores on the test data.
    '''
    def __init__(self):
        '''
        This constructor initialize the scaler and set the
        fit function to none
        '''
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.fitted = False

    def fit(self, X, y=None):
        '''
        This function fit the scaler model to the X data of the arguments.
        Args:
            X: The dat amatrix on which to train the preprocessing.
        The input should be numpy array.
        '''
        self.fitted = True
        self.scaler.fit(X, y=y)

    def fit_transform(self, X, y=None):
        '''
        This function train the scaler and returns the scaled data.
        Args:
            X: The data matrix to fit the preprocessing on and to be transformed.
            y: The label matrix to help fit the preprocessing
        Both inputs should be numpy arrays.
        '''
        if self.fitted == False:
            self.scaler.fit(X, y=y)
            self.fitted = True
        X_scaled = self.scaler.transform(X)

        return X_scaled
    
    def transform(self, X):
        '''
        This functions returns the scaled version of the X data
        Args:
            X: The data matrix to be preprocessed.
        '''
        return self.scaler.transform(X)


class model (BaseEstimator):
    '''
    This class is our model. We choose the RandomForestRegressor model
    as it gave us the best scores overhall during our tests. The arguments of
    the model are as follow:
        n_estimators=60
    We found out that these arguments gave us the best score for our
    RandomForestRegressor.
    '''
    def __init__(self):
        '''
        This constructor is supposed to initialize data members.
        '''
        self.num_train_samples = 38563
        self.num_feat = 59
        self.num_labels = 1
        self.is_trained = False
        self.preprocess = preprocessing()
        self.mod = RandomForestRegressor(n_estimators=60)

    def fit(self, X, y):
        '''
        This function should train the model parameters.
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        '''
        self.num_train_samples = X.shape[0]
        if X.ndim > 1: self.num_feat= X.shape[1]
        print("FIT: dim(X)= [{:d}, {:d}]".format(self.num_train_samples, self.num_feat))
        num_train_samples = y.shape[0]
        if y.ndim > 1: self.num_labels = y.shape[1]
        print("FIT: dim(y)= [{:d}, {:d}]".format(num_train_samples, self.num_labels))
        if (self.num_train_samples != num_train_samples):
            print("ARRGH: number of samples in X and y do not match!")
        # We preprocess the data.
        X_preprocess = self.preprocess.fit_transform(X, y)
        # We train the model
        self.mod.fit(X_preprocess, y)
        self.is_trained = True

    def predict(self, X):
        '''
        This method predicts the value of our label from the X data.
        Args:
            X: Test data matrix of dim num_test_samples * num_feat.
        The inout should be a numpy array.
        '''
        num_test_samples = X.shape[0]
        if X.ndim > 1: num_feat = X.shape[1]
        print("PREDICT: dim(X)= [{:d}, {:d}]".format(num_test_samples, num_feat))
        if (self.num_feat != num_feat):
            print("ARRGH: number of features in X does not match training data!")
        print("PREDICT: dim(y)= [{:d}, {:d}]".format(num_test_samples, self.num_labels))
        # We preprocess the data.
        X_preprocess = self.preprocess.transform(X) 
        # We return the predicted labels.
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


import numpy as np
from sklearn.datasets import make_regression

def test():
    mod = model()
    preproc = preprocessing()
    # Test du preprocessing
    X_train = np.array([[ 1., -1.,  2.],
                        [ 2.,  0.,  0.],
                        [ 0.,  1., -1.]])
    # Test preprocess.fit()
    X_preproc = preproc.fit(X_train)
    assert X_preproc is None, "Fit is returning a modified version"
    # Test preprocess.transform()
    X_preproc = preproc.transform(X_train)
    assert np.array_equal(X_preproc, np.array([[0.5       , 0.        , 1.        ],
                                               [1.        , 0.5       , 1./3.],
                                               [0.        , 1.        , 0.        ]])), "Transform not going right, result: {}".format(X_preproc)
    # Test preprocessing.fit_transform()
    preproc = preprocessing()
    X_preproc = preproc.fit_transform(X_train)
    assert np.array_equal(X_preproc, np.array([[0.5       , 0.        , 1.        ],
                                               [1.        , 0.5       , 1./3.],
                                               [0.        , 1.        , 0.        ]])), "Fit transform not going right, result: {}".format(X_preproc)
    X, y = make_regression(n_features=4, n_informative=2,
                       random_state=0, shuffle=False)
    # Test de model.fit()
    mod.fit(X,y)
    
    # Test de model.predict
    print("Prediction:", mod.predict(np.array([[0, 0, 0, 0]])))

if __name__ == "__main__":
    test()
    print("All test passed successfully")
