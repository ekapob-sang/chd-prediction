# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 09:57:51 2020

@author: ratch
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class ai:
    
    def __init__(self, data, features, target, test_size):
        self.X = data.loc[:, features].values
        self.y = data[target].values
        self.model = self.learn(self.X, self.y, test_size)
        
    def learn(self, X, y, test_size):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
        clf = LogisticRegression(max_iter=10000, penalty="l2")
        clf.fit(X_train, y_train)
        return clf