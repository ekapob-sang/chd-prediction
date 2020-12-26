# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 13:29:50 2020

@author: admin-1309
"""

import pandas as  pd
import numpy as np
import medml as ml
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

df1 = pd.read_csv(r"E:\dataset\framingham.csv")
df1.isnull().sum()  
#df1.info()
df=df1.fillna(df1.median())
num_factor=['age','cigsPerDay','sysBP','diaBP']
heart = df.loc[:, num_factor].values
scaler = MinMaxScaler()
scaler.fit(heart)
heart1=scaler.transform(heart) 
#pd dummy 
male = pd.get_dummies(df1["male"])
male.columns=["F","M"]
dm = pd.get_dummies(df1["diabetes"])
dm.columns=["nodm","dm"]
cat_factor=[male,dm]
heart2 = pd.concat(cat_factor,axis=1)

#concat data
chd= pd.DataFrame(data=heart1,columns=['age','cigsPerDay','sysBP','diaBP'])
chd=pd.concat([chd,heart2], axis=1)
features = chd.columns.tolist()
chd=pd.concat([chd,df['TenYearCHD']], axis=1)
chd.rename(columns={"TenYearCHD": "labels"},inplace = True)
X=chd.drop(['labels'],axis=1)
y=chd['labels']





#from sklearn.cross_validation import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=5)
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, confusion_matrix
# import sklearn
# logreg=LogisticRegression()
# logreg.fit(x_train,y_train)
# y_pred=logreg.predict(x_test)
# sklearn.metrics.accuracy_score(y_test,y_pred)
# X_train,X_test , y_train , y_test = train_test_split (X, y,test_size=0.20,random_state=56294,stratify=y)
# hcv_logis = LogisticRegression(penalty='l2',C = 200,solver='liblinear',class_weight='balanced')
# hcv_logis.fit(X_train, y_train)
# y_pred = hcv_logis.predict(X_test)
# from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))


# solvers = ['newton-cg', 'lbfgs', 'liblinear']
# penalty = ['l2','L1']
# c_values = [1000,100, 10, 1.0, 0.1, 0.01,0.001]
# rand = dict(solver=solvers,penalty=penalty,C=c_values)
# # logreg = LogisticRegression(class_weight='balanced') 
# logreg = LogisticRegression() 
# clf = RandomizedSearchCV(logreg,rand, n_jobs=-1, cv=10,random_state=1)
# clf.fit(X_train, y_train)
# print("Best parameters set found on development set from randomsearch:")
# print()
# print(clf.best_params_)
# print()
# print("Best: %f using %s" % ( clf.best_score_,clf.best_params_))
# print()
# means = clf.cv_results_['mean_test_score']
# stds = clf.cv_results_['std_test_score']
# params = clf.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
# # confusion matrix for best param
# hcv_logis = LogisticRegression(penalty='l2',C = 1,solver='lbfgs')
# hcv_logis.fit(X_train, y_train)
# y_pred = hcv_logis.predict(X_test)
# from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# solvers = ['lbfgs']
# penalty = ['l2']
# c_values = [0.1,0.3,0.5,0.8,1,2,3,4,5,6,7,8,9,10,20,30]
# grid = dict(solver=solvers,penalty=penalty,C=c_values)
# clf = GridSearchCV(logreg,grid,cv=10)
# clf.fit(X_train, y_train)
# print("Best parameters set found on development set for Grid search:")
# print()
# print(clf.best_params_)
# print()
# # confusion matrix for best param
# hcv_logis = LogisticRegression(penalty='l2',C =5,solver='lbfgs')
# hcv_logis.fit(X_train, y_train)
# y_pred = hcv_logis.predict(X_test)
# from sklearn.metrics import  classification_report, confusion_matrix
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))


chdmodel = ml.ai(data=chd, 
              features=features, 
              target="labels", 
              test_size=0.2)


# x_new=np.array([20, 2, 120, 80]).reshape(1, -1)
# logreg.predict_proba(x_new)
# import pickle
# pickle.dump(chdmodel.model,open("logisticchd2.pkl","wb"))
# pickle.dump(scaler, open('scaler.pkl', 'wb'))


