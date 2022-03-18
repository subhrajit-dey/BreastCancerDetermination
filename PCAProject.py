# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 22:20:02 2022

@author: SUBHRAJIT_DEY
"""

from sklearn.datasets import load_breast_cancer
import pandas as pd

lbc = load_breast_cancer()


x = pd.DataFrame(lbc['data'], columns = lbc['feature_names'])
y = pd.DataFrame(lbc['target'],columns = ['type'])

#Prediction without PCA

from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, random_state=1234, stratify=y)

from sklearn.ensemble import RandomForestClassifier

rfc1 = RandomForestClassifier(random_state=1234)
rfc1.fit(x_train,y_train)
y_predict1 = rfc1.predict(x_test)
score1 = rfc1.score(x_test, y_test)

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test,y_predict1)



# =============================================================================
# Implementation of PCA
# =============================================================================

#Center the data

from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()
#Mean 0 and sd = 1;
x_scale1 = scalar.fit_transform(x)
#Scaled values around 0

print(x_scale1[:,0].mean())


from sklearn.decomposition import PCA

pca = PCA(n_components=5)

x_pca = pca.fit_transform(x_scale1)



from sklearn.model_selection import train_test_split

x_train1, x_test1,y_train1,y_test1 = train_test_split(x_pca,y,test_size=0.3, random_state=1234, stratify=y)

from sklearn.ensemble import RandomForestClassifier

rfc2 = RandomForestClassifier(random_state=1234)
rfc2.fit(x_train1,y_train1)
y_predict2 = rfc2.predict(x_test1)
score2 = rfc2.score(x_test1, y_test1)

from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test1,y_predict2)
























