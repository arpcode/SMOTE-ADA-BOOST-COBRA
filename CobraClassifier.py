import numpy as np
import pandas as pd

import sklearn
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils.validation import check_array
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

class CobraClassifier(BaseEstimator):
    def __init__(self, estimators = [DecisionTreeClassifier(max_depth = 3)]*4, random_state = 0):
        self.random_state = random_state
        self.estimators = estimators
        
        self.n_machines = len(estimators)
        
    def fit(self, X, y, sample_weight = None, split = 0.5):
        l = len(X)
        k = split*l if split<=1 else split
        k = int(k)
        
        self.X_k, self.y_k = X[:k], y[:k]
        self.X_l, self.y_l = X[k:l].reset_index(drop=True), y[k:l].reset_index(drop=True)
        self.sample_weights = None
        if sample_weight is not None:
            self.sample_weights = sample_weight[:k]
        
        
        self.train()
        self.fit_cobra()
        
        return self
    
    def predict(self, X):
        X = np.array(X)
        M = len(self.estimators)
        results = np.zeros(len(X))
        avg_points = 0
        index = 0
        
        for sample in X:
            results[index] = self.pred(sample.reshape(1, -1))
            index+=1
        
        return results
        
    def train(self):
        for estimator in self.estimators:
            estimator.fit(self.X_k, self.y_k, sample_weight = self.sample_weights)
            
        return self
    
    def fit_cobra(self):
        self.machine_predictions = [None]*self.n_machines
        for i in range(self.n_machines):
            self.machine_predictions[i] = self.estimators[i].predict(self.X_l)
            
        return self

    def pred(self, X):
        n_machines = self.n_machines
        M = n_machines
        
        select = [set()]*n_machines
        for i in range(n_machines):
            label = self.estimators[i].predict(X)
            for point in range(len(self.X_l)):
                if self.machine_predictions[i][point] == label:
                    select[i].add(point)
                    
        points = []
        for sample in range(len(self.X_l)):
            row_check = 0
            for i in range(n_machines):
                if sample in select[i]:
                    row_check+=1

            if row_check == M:
                points.append(sample)


        if len(points) == 0:
            print('No Points found')
            return 2


        classes = {}
        for label in np.unique(self.y_l):
            classes[label] = 0

        for point in points:
            classes[self.y_l[point]] += 1

        result = int(max(classes, key = classes.get))
        return result