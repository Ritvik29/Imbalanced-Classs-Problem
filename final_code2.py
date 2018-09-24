# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 15:22:09 2018

@author: ritvik
"""

import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from os import chdir, getcwd
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import LinearSVC
from imblearn.pipeline import make_pipeline
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
wd=getcwd()
abs_path="C:\\Users\\ritvik\\EnsembleEnergyInterview\\Data_for_classification.xlsx"

#chdir(wd)

df1=pd.read_excel(abs_path)
df1.fillna(0,inplace=True)
df1.head()

#---------------------------------------------------------------------------------------------------------
"""Data Engineering and Visualisation. The Number of 0's and 1's are counted.
Target-> 1657 values=1, 13413 values=0

feature_3 -> 114 values=1, 14956 values=0
feature_4-> 6519 values=1, 8551 values=0
feature_5-> 6507 values=1, 8563 values=0
feature_6-> 6417 values=1, 8653 values=0
feature_7-> 1687 values=1, 13383 values=0
feature_8-> 2657 values=1, 10785 values=0, 995 values=2, 367 values=3, 130 values=4, 70 values=5, 28 values=6, 9 values=7, 6 values=8, 3 values=9, 3 values=10, 3 values=11
feature_9-> 15070 values=0

feature_10-> 9574 values=1, 152 values=0, 4326 values=2, 900 values=3, 113 values=4, 5 values=5, 
feature_13-> 1999 values=1, 13071 values=0, 
feature_14-> 807 values=1, 14263 values=0
feature_15-> 242 values=1, 14828 values=0
feature_16-> 91 values=1, 14979 values=0
feature_17-> 7047 values=1, 8023 values=0
feature_18-> 1609 values=1, 13461 values=0
feature_19-> 1288 values=1, 13782 values=0

"""
df1.drop(['ID'],axis=1,inplace=True)
#-------------------------------------------------------------------------------------------------------------
df1['feature_4'][df1['feature_4']==1].shape
df1['feature_5'][df1['feature_5']==1].shape
df1['feature_6'][df1['feature_6']==1].shape
df1['feature_7'][df1['feature_7']==0].shape
df1['feature_8'][df1['feature_8']==11].shape
df1['feature_9'][df1['feature_9']==2].shape
df1['feature_10'][df1['feature_10']==0].shape
df1['feature_13'][df1['feature_13']==0].shape
df1['feature_14'][df1['feature_14']==0].shape
df1['feature_15'][df1['feature_15']==1].shape
df1['feature_16'][df1['feature_16']==0].shape
df1['feature_17'][df1['feature_17']==0].shape
df1['feature_18'][df1['feature_18']==0].shape
df1['feature_19'][df1['feature_19']==0].shape
df1['Target'][df1['Target']==1].shape


#Exploratory Data Analysis----------------------------------------------------------------------------------------
#It is decided that feature 8 will be plotted as a categorical Variable 
plt.figure()
sns.distplot(df1['Target'],kde=False)
plt.figure()
sns.distplot(df1['feature_1'],kde=False)
plt.figure()
sns.distplot(df1['feature_2'],kde=False)
plt.figure()
sns.distplot(df1['feature_3'],kde=False)
plt.figure()
sns.distplot(df1['feature_4'],kde=False)
plt.figure()
sns.distplot(df1['feature_5'],kde=False)
plt.figure()
sns.distplot(df1['feature_6'],kde=False)
plt.figure()
sns.distplot(df1['feature_7'],kde=False)
plt.figure()
sns.distplot(df1['feature_8'],kde=False)
plt.figure()
sns.distplot(df1['feature_9'],kde=False)
plt.figure()
sns.distplot(df1['feature_10'],kde=False)
plt.figure()
sns.distplot(df1['feature_11'],kde=False)
plt.figure()
sns.distplot(df1['feature_12'],kde=False)

#--------------------------------------------------------------------------------
"""
Categorical predictors are- feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9
feature_10, 
Continous predictors are:- feature_1, feature_2, feature_11, feature_12
"""
#------------------------------------------------------------------------------
#Pre-processing Categorical predictors- One Hot Encoding of Categorical Predictors
pd.get_dummies(df1['feature_4'])
pd.get_dummies(df1['feature_5'])
pd.get_dummies(df1['feature_6'])
pd.get_dummies(df1['feature_7'])
pd.get_dummies(df1['feature_9'])
pd.get_dummies(df1['feature_10'])
pd.get_dummies(df1['feature_13'])
pd.get_dummies(df1['feature_14'])
pd.get_dummies(df1['feature_15'])
pd.get_dummies(df1['feature_16'])
pd.get_dummies(df1['feature_17'])
pd.get_dummies(df1['feature_18'])
pd.get_dummies(df1['feature_19'])
#-------------------------------------------------------------------------------------------
# The highest correlation between features is between feature 11 and feature 12
df2=df1[['Target','feature_1','feature_2','feature_11','feature_12']]
sns.pairplot(df2)
print(df2.corr())

# The numerical variables are standardized

df1['feature_1']=(df1['feature_1']-df1['feature_1'].mean())/df1['feature_1'].std()
df1['feature_2']=(df1['feature_2']-df1['feature_2'].mean())/df1['feature_2'].std()
df1['feature_11']=(df1['feature_11']-df1['feature_11'].mean())/df1['feature_11'].std()
df1['feature_12']=(df1['feature_12']-df1['feature_12'].mean())/df1['feature_12'].std()

#------------------------------------------------------------------------------------------
#Use Unbalanced Sampling to give better results in classification

def plot_pie(y):
    target_stats = Counter(y)
    labels = list(target_stats.keys())
    sizes = list(target_stats.values())
    explode = tuple([0.1] * len(target_stats))

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, shadow=True,
           autopct='%1.1f%%')
    ax.axis('equal')
def plot_auc(d_y_val,y2hat):   #Plots roc curve
    fpr, tpr, threshold = roc_curve(d_y_val, y2hat)
    roc_auc = auc(fpr, tpr)
    # method I: plt
    plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
y=df1['Target']
X1=df1.drop(['Target'],axis=1)

d_x_train,d_x_val,d_y_train,d_y_val= train_test_split(X1,y,test_size=0.2,random_state=42)
a1=5*sum(d_y_train==1)
a0=sum(d_y_train==0)
ratio = {1:a1,0:a0}
sm = SMOTE(ratio=ratio,random_state=42)
X_res1, y_res1 = sm.fit_sample(d_x_train, d_y_train)
sme = SMOTEENN(ratio=ratio,random_state=42)
X_res2, y_res2 = sme.fit_sample(d_x_train, d_y_train)
plot_pie(y_res2)


# Random Forest Fit on SMOTE---------------------------------------------------------------------------------
w1={0: 1, 1: 8}   # The ratio of 0's to 1's in the original dataset is roughly 8:1
rf=RandomForestClassifier(class_weight=w1)
rf.fit(X_res1,y_res1)
yhat=rf.predict(d_x_val)
d_y_val1=pd.DataFrame(rf.predict_proba(d_x_val))
d_y_val1['y2hat']=d_y_val1[1].apply(lambda x:1 if x>0.11 else 0)
y2hat=np.array(d_y_val1['y2hat'])
print(confusion_matrix(d_y_val,yhat))
print('RF1',f1_score( d_y_val,y2hat))
plot_auc(d_y_val,y2hat)


#Random Forest fit on SMOTE +EEN
rf1=RandomForestClassifier(class_weight=w1)
rf1.fit(X_res1,y_res1)
yhat=rf.predict(d_x_val)
d_y_val1=pd.DataFrame(rf1.predict_proba(d_x_val))
d_y_val1['y2hat']=d_y_val1[1].apply(lambda x:1 if x>0.11 else 0)
y2hat=np.array(d_y_val1['y2hat'])
print(confusion_matrix(d_y_val,yhat))
print('RF1',f1_score( d_y_val,y2hat))
plot_auc(d_y_val,y2hat)

#The results for SMOTE and SMOTE +EEN are almost the same
#--------------------------------------------------------------------------------------------------
""" KFOLD cross validation is implemented with  Oversamoling

"""
#Implement K fold Cross Validation

from sklearn.model_selection import KFold # import KFold
X_t,X_test,y_t,y_test= train_test_split(X1,y,test_size=0.2,random_state=42,shuffle=True)

sme2 = SMOTEENN(ratio=ratio,random_state=42)
X_res3, y_res3 = sme2.fit_sample(X_t, y_t)
plot_pie(y_res3)

w1={0: 1, 1: 15}

kf = KFold(n_splits=5,random_state=42,shuffle=True) # Define the split - into 2 folds 
kf.get_n_splits(X_res3) # returns the number of splitting iterations in the cross-validator
for train_index, val_index in kf.split(X_res3):
    print('TRAIN:', train_index, 'TEST:', val_index)
    X_train, X_val = X_res3[train_index], X_res3[val_index]
    y_train, y_val = y_res3[train_index], y_res3[val_index]
    rf1=RandomForestClassifier(class_weight=w1)
    rf1.fit(X_train,y_train)
    d_y1=pd.DataFrame(rf1.predict_proba(X_val))
    d_y2=rf1.predict(X_val)
    yhat_test=rf1.predict(X_test)
    print('f1_score', 'RF2',f1_score( y_val,d_y2))
    print('f1_score', 'RF ON TEST SET (NO OVERSAMPLING)',f1_score( y_test,yhat_test))
    print('Confusion Matrix on Validation Data with oversampling',confusion_matrix(y_val,d_y2))
    print('Confusion Matrix on Validation Data(NO OVERSAMPLING)' ,confusion_matrix(y_test,yhat_test))
    
    print('ROC on Validation data, With Oversampling')
    plot_auc(y_val,d_y2)
    print('ROC on Validation data, NO Oversampling')
    plot_auc(y_test,yhat_test)

    #----------------------------------------------------------------------------------
    #Gradient boosting
    GB=GradientBoostingClassifier(learning_rate=0.3)
    GB.fit(X_train,y_train)
    y_gb=GB.predict(X_val)
    yhat_gb_test=GB.predict(X_test)
    print('f1_score on Validation Set, With Oversampling', 'GB',f1_score( y_val,y_gb))
    print('F1 ON TEST SET  (NO Oversampling)', 'GB',f1_score( y_test,yhat_gb_test))
    print('Confusion Matrix on Validation Data with oversampling',confusion_matrix(y_val,y_gb))
    print('Confusion Matrix on Validation Data(NO OVERSAMPLING)' ,confusion_matrix(y_test,yhat_gb_test))
    print('ROC on Validation data, With Oversampling')
    plot_auc(y_val,y_gb)
    print('ROC on TEST data, NO Oversampling')
    plot_auc(y_test,yhat_gb_test)
    
    
    #print(GB.score(d_x_train,d_y_train))
    #------------------------------------------------------------------------------------------
    #SVM   
    clfn = svm.SVC(probability=True,kernel='rbf')
    clfn.fit(X_train,y_train)
    yhat_svm=clfn.predict(X_val)
    yhat_svm_test=clfn.predict(X_test)
    print('f1_score on validatiob set with oversampling', 'SVM',f1_score( y_val,yhat_svm))
    print('F1 ON TEST SET  (NO DATA SYNTHESIS)', 'SVM',f1_score( y_test,yhat_svm_test))
    print('Confusion Matrix on Validation Data with oversampling',confusion_matrix(y_val,yhat_svm))
    print('Confusion Matrix on Validation Data(NO OVERSAMPLING)' ,confusion_matrix(y_test,yhat_svm_test))
    print('ROC on Validation data, With Oversampling')
    plot_auc(y_val,yhat_svm)
    print('ROC on Test data, With NO Oversampling')
    plot_auc(y_test,yhat_svm_test)
