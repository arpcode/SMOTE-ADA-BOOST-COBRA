import numpy as np
import pandas as pd

import sklearn
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

class AdaBoost(BaseEstimator):
    def __init__(self, weak_learner = DecisionTreeClassifier(max_depth = 1)):
        self.alphas = []
        self.weaks = []
        self.M = None
        self.training_errors = []
        self.prediction_errors = []
        self.weak_learner = weak_learner
    
    def fit(self, X, y, M = 40):
        self.M = M
        self.alphas = []
        self.training_errors = []
        
        for m in range(M):
            if m==0:
                w_i = np.ones(len(y)) * 1/len(y)
            else:
                w_i = self.update_weights(w_i, alpha_m, y, y_pred)
                
            
            weak = sklearn.base.clone(self.weak_learner)
            weak.fit(X, y, sample_weight = w_i)
            
            y_pred = weak.predict(X)
            weak.predict(X)
            
            self.weaks.append(weak)
            
            error_m = self.compute_error(y, y_pred, w_i)
            self.training_errors.append(error_m)
            
            alpha_m = self.alpha(error_m)
            self.alphas.append(alpha_m)
            
    def predict(self, X):
#         X = np.array(X)
        
        weak_preds = pd.DataFrame(index = range(len(X)), columns = range(self.M))
        
        for m in range(self.M):
            y_pred_m = self.weaks[m].predict(X) * self.alphas[m]
            weak_preds.iloc[:, m] = y_pred_m
            
        y_pred = (1*np.sign(weak_preds.T.sum())).astype(int)
        return y_pred
            
    def compute_error(self, y, y_pred, w_i):
        return (sum(w_i * (np.not_equal(y, y_pred)).astype(int)))/sum(w_i)

    def alpha(self, error):
        return np.log((1 - error) / error)

    def update_weights(self, w_i, alpha, y, y_pred):
        return w_i * np.exp(alpha * (np.not_equal(y, y_pred)).astype(int))



