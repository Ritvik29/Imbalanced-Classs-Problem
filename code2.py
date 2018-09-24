# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 20:33:32 2018

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
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

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

#-------------------------------------------------------------------------------------------------------------


abs_path="C:\\Users\\ritvik\\EnsembleEnergyInterview\\Data_for_classification.xlsx"


df1=pd.read_excel(abs_path)

df1.head()
df1.fillna(0,inplace=True)
df1.head()
#Categorical Variables - get dummy variables
df1.drop(['ID'],axis=1,inplace=True)
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
 #The numerical variables are standardized

df1['feature_1']=(df1['feature_1']-df1['feature_1'].mean())/df1['feature_1'].std()
df1['feature_2']=(df1['feature_2']-df1['feature_2'].mean())/df1['feature_2'].std()
df1['feature_11']=(df1['feature_11']-df1['feature_11'].mean())/df1['feature_11'].std()
df1['feature_12']=(df1['feature_12']-df1['feature_12'].mean())/df1['feature_12'].std()

#--------------------------------------------------------------------

#----------------------
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

for i in range(5,0,-1):
    print('the ratio is ',str(i)+"00"+'%','for the minority class','-----------------------------------------')
    a1=i*sum(y==1)
    a0=sum(y==0)
    ratio = {1:a1,0:a0}
    sme2 = SMOTEENN(ratio=ratio,random_state=42)


    #X_t,X_test,y_t,y_test= train_test_split(X1,y,test_size=0.2,random_state=42,shuffle=True)
    kf = KFold(n_splits=5,random_state=42,shuffle=True) # Define the split - into 2 folds 
    kf.get_n_splits(X1) # returns the number of splitting iterations in the cross-validator
    X1=np.array(X1)
    y=np.array(y)
    counter=0
    for train_index, val_index in kf.split(X1):
        #print('TRAIN:', train_index, 'TEST:', val_index)
        counter=counter+1
        svc = svm.SVC(probability=True,gamma='auto' )
        X_train, X_val = X1[train_index], X1[val_index]
        y_train, y_val = y[train_index], y[val_index]
        x_resn,y_resn=sme2.fit_sample(X_train, y_train)
        tuned_parameters = [{'kernel': ['rbf'], 'class_weight': [{0: 1, 1: 1}  , {0: 1, 1: 2}  , {0: 1, 1: 4}  , {0: 1, 1: 6}],
                     'C': [0.001, 0.10, 0.1, 10]},
                    {'kernel': ['sigmoid'], 'class_weight': [{0: 1, 1: 1}  , {0: 1, 1: 2}  , {0: 1, 1: 4}  ,{0: 1, 1: 6}],
                     'C': [0.001, 0.10, 0.1, 10]},
                    {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10],
                   'class_weight': [{0: 1, 1: 1}  , {0: 1, 1: 2}  , {0: 1, 1: 4}  , {0: 1, 1: 6}]}
                    ]
        scores = ['f1']
        print("# Tuning hyper-parameters for %s" % scores)
        clf = GridSearchCV(svc, tuned_parameters,cv=None)
        clf.fit(x_resn,y_resn)
        yhat1=clf.predict(X_val)
        print('Confusion Matrix', confusion_matrix(y_val,yhat1))
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()
        plot_auc(y_val,yhat1)
        print('f1_score of final model on test/validation data(NO OVERSAMPLING)',f1_score(y_val,yhat1))

"""



"""