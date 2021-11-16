from collections import Counter

import numpy as np
import pandas as pd

import sklearn
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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

class SMOTE(object):
    def __init__(self, k_neighbours=5, random_state = None):
        self.k = k_neighbours
        self.random_state = random_state
        
    def fit(self, X):
        self.X = X
        self.n_minority_samples, self.n_features = self.X.shape
        
        self.neigh = NearestNeighbors(n_neighbors = self.k + 1)
        self.neigh.fit(self.X)
        
        return self
        
    def sample(self, n_samples):
        np.random.seed(seed=self.random_state)
        
        S = np.zeros(shape = (n_samples, self.n_features))
        for i in range(n_samples):
            j = np.random.randint(0, self.X.shape[0])
            nn = self.neigh.kneighbors(self.X[j].reshape(1, -1), 
                                        return_distance=False)[:, 1:]
            nn_index = np.random.choice(nn[0])
            
            dif = self.X[nn_index] - self.X[j]
            gap = np.random.random()
            
            S[i, :] = self.X[j, :] + gap * dif[:]
            
        return S

class SMOTEBoost(AdaBoost):
    def __init__(self, n_samples = 20, k_neighbours = 5, 
                weak_learner = DecisionTreeClassifier(max_depth=1), random_state = None):
        self.n_samples = n_samples
        self.smote = SMOTE(k_neighbours = k_neighbours, random_state=random_state)
        
        self.alphas = []
        self.weaks = []
        self.M = None
        self.training_errors = []
        self.preecition_errors = []
        
        super().__init__(weak_learner = weak_learner)
        
    def fit(self, X, y, M=20):
        self.M = M
        self.alphas = []
        self.training_errors = []
        
        stats_c = Counter(y)
        maj_c = max(stats_c, key = stats_c.get)
        min_c = min(stats_c, key = stats_c.get)
        self.minority_target = min_c
        
        X, y = np.array(X), np.array(y)
        
        for m in range(M):
            if m==0:
                w_i = np.ones(len(y)) * 1/len(y)
            else:
                w_i = self.update_weights(w_i, alpha_m, y, y_pred)
        
                
            X_min = X[np.where(y==self.minority_target)]
            
            if len(X_min) >= self.smote.k:
                self.smote.fit(X_min)
                X_syn = self.smote.sample(self.n_samples)
                y_syn = np.full(X_syn.shape[0], fill_value = self.minority_target, dtype = np.int64)
                
                w_i_syn = np.empty(X_syn.shape[0],dtype = np.float64)
                w_i_syn[:] = 1.0/X.shape[0]
                
                X = np.vstack((X, X_syn))
                y = np.append(y, y_syn)
                
                w_i = np.append(w_i, w_i_syn).reshape(-1, 1)
                w_i = np.squeeze(normalize(w_i, axis=0, norm = 'l1'))
#                 w_i = normalize(w_i, norm = 'l1')

            weak = sklearn.base.clone(self.weak_learner)
            weak.fit(X, y, sample_weight = w_i)
            
            y_pred = weak.predict(X)
            
            self.weaks.append(weak)
            
            error_m = self.compute_error(y, y_pred, w_i)
            self.training_errors.append(error_m)
            
            alpha_m = self.alpha(error_m)
            self.alphas.append(alpha_m)