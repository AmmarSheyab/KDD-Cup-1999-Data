# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd

df=pd.read_csv('data.csv',header=None,chunksize=(10000),on_bad_lines=('skip'))

data= pd.concat(df)
data.drop_duplicates(subset=(None),inplace=True)

data.info()
'''
#i have in this data mltydatatype
#int =(23)
#float=(15)
#object=(4)
'''
a=data.isnull().sum()
'''
#no missing values in data

'''
z=data.nunique()
#data=pd.get_dummies(data,columns=[1],drop_first=True)
#data=pd.get_dummies(data,columns=[2],drop_first=True)
#data=pd.get_dummies(data,columns=[3],drop_first=True)


X = data.iloc[:,:-1].values

y = data.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

X[:,1] = labelencoder.fit_transform(X[:,1])

X[:,2] = labelencoder.fit_transform(X[:,2])

X[:,3] = labelencoder.fit_transform(X[:,3])

y = labelencoder.fit_transform(y)

from category_encoders import CatBoostEncoder

encoder = CatBoostEncoder()

X[:,1:2] = encoder.fit_transform(X[:,1:2],y)

X[:,2:3] = encoder.fit_transform(X[:,2:3],y)

X[:,3:4] = encoder.fit_transform(X[:,3:4],y)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler

standardscaler= StandardScaler()
X_train = standardscaler.fit_transform(X_train)
X_test = standardscaler.transform(X_test)


from sklearn.linear_model import LogisticRegression

regressor = LogisticRegression()

regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print('Sklearn Accuracy Score:',accuracy)














