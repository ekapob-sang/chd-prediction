# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 13:29:50 2020

@author: admin-1309
"""

import pandas as  pd
import numpy as np
import medml as ml

df1 = pd.read_csv(r"E:\dataset\framingham.csv")
df1.isnull().sum()  
df=df1.fillna(df1.median())
num_factor=['age','cigsPerDay','sysBP','diaBP']
# heart = df.loc[:, num_factor].values
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(heart)
# heart1=scaler.transform(heart)  
# cat_factor=['male','diabetes']
# heart2 = df.loc[:, cat_factor].values
#concat data
# heart3= np.concatenate((heart1,heart2),axis=1)
chd= pd.DataFrame(data=df,columns=['age','cigsPerDay','sysBP','diaBP','male','diabetes'])

features = chd.columns.tolist()
chd=pd.concat([chd,df['TenYearCHD']], axis=1)
chd.rename(columns={"TenYearCHD": "labels"},inplace = True)
chdmodel = ml.ai(data=chd, 
              features=features, 
              target="labels", 
              test_size=0.2)




# import pickle
# pickle.dump(chdmodel.model,open("logisticchd.pkl","wb"))