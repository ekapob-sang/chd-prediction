# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class ai:
    
    def __init__(self, data, features, target, test_size):
        self.X = data.loc[:, features].values
        self.y = data[target].values
        self.model = self.learn(self.X, self.y, test_size)
        
    def learn(self, X, y, test_size):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y,random_state=12345)
        clf = LogisticRegression(C=10,solver='liblinear',max_iter=10000, penalty="l2",class_weight='balanced')
        clf.fit(X_train, y_train)
        return clf