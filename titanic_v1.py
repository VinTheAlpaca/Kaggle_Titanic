### What this work is?
# This work is a practice for prediction models
# The data was from a Kaggle Competition which
# was aimming to predict the survial of passagers
# on Titanic

### What does it mainly included?
# In this work, the core part in this word is
# the usage of 'Decision Tree', 'Random Forest', 'XGboost'
# as well as the cross validation and parameter tuning.

### How about the result?
# Since it was a pure practice, I did not force myself to
# get a great mark on the competition. The accuracy
# for the finial prediction was around 78% which was tested by
# kaggle. The highest mark in this competition was 100%
# (I have no idea how did they achieved that).

###
# The result will be put on github for readers to test,
# although it was not a very good result. Readers can search
# competition 'Titanic' on Kaggle to reach the website and
# upload my result to have a test in order to inspire themselves
# or get familiar with CV, DT, RF, XGboost and etc.



import pandas as pd
import numpy as np
import os

### data dictionary
# survival	Survival	0 = No, 1 = Yes
# pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
# sex	    Sex
# Age	    Age in years
# sibsp     # of siblings / spouses aboard the Titanic
# parch 	# of parents / children aboard the Titanic
# ticket	Ticket number
# fare	    Passenger fare
# cabin	    Cabin number
# embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

### functions
def naPercent(df):
    return round(sum(df.isnull())/len(df),3)*100

def getTitle(s):
    return s.split(', ')[1].split('. ')[0]

def titleGroup(s):
    if s in ['Master', 'Don', 'Jonkheer', 'Sir', 'the Countess']:
        return 'Noble'
    elif s in ['Col', 'Major', 'Capt']:
        return 'Soldier'
    else:
        return 'General'

def parTuning(x, y, model, parameter, kf, scorer):
    for train_index, test_index in kf.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        gs_model = GridSearchCV(model, parameter, scoring=scorer, cv=10)
        gs_model.fit(x_train, y_train)
        best_index = gs_model.cv_results_['rank_test_score'].argmin()
        best_param = gs_model.cv_results_['params'][best_index]

    return best_param


def xgbParmTuning(features, label, parameters, learning_rate=0.1, n_estimators=100, max_depth=6, min_child_weight=3,
                  gamma=0, subsample=0.5, colsample_bytree=0.6, reg_alpha=0.01):
    gsearch_step = GridSearchCV(estimator=XGBClassifier(learning_rate=learning_rate,
                                                        n_estimators=n_estimators,
                                                        max_depth=max_depth,
                                                        min_child_weight=min_child_weight,
                                                        gamma=gamma,
                                                        subsample=subsample,
                                                        colsample_bytree=colsample_bytree,
                                                        reg_alpha=reg_alpha,
                                                        objective='binary:logistic',
                                                        nthread=4,
                                                        scale_pos_weight=1,
                                                        seed=123456),
                                param_grid=parameters,
                                scoring='roc_auc', n_jobs=4, cv=5)

    gsearch_step.fit(features, label)
    return gsearch_step.best_params_, gsearch_step.best_score_

### loading data
data0 = pd.read_csv(r'C:\Users\44757\Desktop\Python_Projects\titanic\data\train.csv')
data = data0.copy()

### transform sex from string to binary
data['Sex'] = data['Sex'].replace({"female":0,"male":1})

### cheak missing data
data.apply(naPercent, axis=0)


## according to the missing data,
# 1. column 'Cabin' contains
# more than 77% of NA value
# (imputation may cause large bias)
# thus, we decide to remove this column
# 2. column 'Age' contains less than 20%,
# and this feature means a lot for
# predicting the survival possibility,
# thus, we decide to keep it
# and use MICE algorithm to impute it
# 3. column 'Embarked' has small number
# of missing value, and it is category value,
# thus, we'd use the value occured most to impute it

## choose features
data = data.filter(['Name', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked'])

data['Title'] = data['Name'].map(getTitle)

## impute Age
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
imp = IterativeImputer(estimator=lr,
                       missing_values=np.nan,
                       max_iter=10,
                       verbose=2,
                       imputation_order='roman',
                       random_state=1,
                       min_value = 0)

# temp = data_train.filter(['Survived', 'Pclass', 'Sex', 'Age', 'Fare'])
# temp = imp.fit_transform(data_train[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]).T[3]

data['Age'] = imp.fit_transform(data[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]).T[3]

#data_train.groupby('Embarked').count()
data['Embarked'].fillna('S', inplace=True)

## title grouping
data['title_group'] = data['Title'].map(titleGroup)

### sampling
# does it need sampling?
data.groupby('Survived').count()

data = pd.concat([data, pd.get_dummies(data[['Embarked', 'title_group']])], axis=1).\
    drop(['Name', 'Ticket', 'Title', 'Embarked', 'title_group'], axis=1)

# data = data.iloc[:,0:7]

from sklearn.model_selection import train_test_split
data_train, data_test, label_train, label_test = train_test_split(data.drop('Survived',axis=1),
                                                                  data['Survived'],
                                                                  test_size=0.25,
                                                                  random_state=123456,
                                                                  stratify=data['Survived'])

### Decision Tree and Random Forest

import sklearn.metrics as mt
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn import metrics

scorer = {
    'f1': mt.make_scorer(mt.f1_score, pos_label=1),
    # Define different scoring metric to be used
    # Define “positive” label for F-measure
    'accuracy': 'accuracy'
}

dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()

scores = cross_validate(dtc, data_train, label_train, cv=10, scoring=scorer)
scores1 = cross_validate(dtc, data_train.iloc[:,0:6], label_train, cv=10, scoring=scorer)

scores['test_f1']
scores1['test_f1']

### Random Forest
### Cross Validation + parameter Tuning

parameters_dtc = {'min_impurity_decrease': [0.05*i for i in range(3)],
                    'criterion': ["gini", "entropy"],
                    'max_depth': [i for i in range(3,11)]}

kf = KFold(n_splits=10, shuffle=True)

best_param_dtc = parTuning(data_train, label_train, dtc, parameters_dtc, kf, mt.make_scorer(mt.f1_score, pos_label=1))


dtc_cv = DecisionTreeClassifier(**best_param_dtc)
dtc_cv.fit(data_train, label_train)
dtc_pred = dtc_cv.predict(data_test)
confusion_matrix(label_test, dtc_pred)

print("Accuracy on test dataset:", round(metrics.accuracy_score(label_test, dtc_pred),4)*100,"%")
print("ROC_AUC on test dataset:", round(metrics.roc_auc_score(label_test, dtc_cv.predict_proba(data_test)[:,1]),4)*100,"%")



### random forest
parameters_rfc = {'min_impurity_decrease': [0.05*i for i in range(3)],
              'criterion': ["gini", "entropy"]}


kf = KFold(n_splits=10, shuffle=True)

best_param_rfc = parTuning(data_train, label_train, rfc, parameters_rfc, kf, mt.make_scorer(mt.f1_score, pos_label=1))

rfc_cv = RandomForestClassifier(**best_param_rfc)
rfc_cv.fit(data_train, label_train)
rfc_pred = rfc_cv.predict(data_test)
confusion_matrix(label_test, rfc_pred)

print("Accuracy on test dataset:", round(metrics.accuracy_score(label_test, rfc_pred),4)*100,"%")
print("ROC_AUC on test dataset:", round(metrics.roc_auc_score(label_test, rfc_cv.predict_proba(data_test)[:,1]),4)*100,"%")


### xgboost
import xgboost as xgb
from xgboost import XGBClassifier

### Parameter Tuning
# step1: the optimal number for n_estimators
# (the number of tree in the model)
#
# set a group of initial value and find the
# optimal n_estimators by using cross validation
xgc = XGBClassifier(
    learning_rate =0.1,
    n_estimators=500,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.75,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=123456
)


xgtrain = xgb.DMatrix(data_train.values, label=label_train.values)
cv_result = xgb.cv(xgc.get_xgb_params(),
       xgtrain,
       num_boost_round=xgc.get_params()['n_estimators'],
       nfold=10,
       metrics='auc',
       early_stopping_rounds=50)

# the optimal n_estimators
cv_result.shape[0]




### step2: tuning for max_depth and min_child_weight
# ({'max_depth': 6, 'min_child_weight': 3}, 0.9036199230390201)
parameters_step2 = {'max_depth': [i for i in range(3,11)],
                    'min_child_weight':range(0,6,1)
                   }

step2_result=xgbParmTuning(data_train, label_train, parameters_step2, n_estimators=cv_result.shape[0])
step2_result


### step3: tuning for gamma
# ({'gamma': 0.0}, 0.9036199230390201)
parameters_step3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
step3_result = xgbParmTuning(data_train, label_train, parameters_step3,
                             n_estimators=cv_result.shape[0],
                             max_depth=step2_result[0]['max_depth'],
                             min_child_weight=step2_result[0]['min_child_weight'])
step3_result

### step4: tuning for subsample and colsample_bytree
# ({'colsample_bytree': 0.5, 'subsample': 0.4}, 0.904593496498574)
parameters_step4 = {
    # 'subsample':[i/10.0 for i in range(6,10)],
    # 'colsample_bytree':[i/10.0 for i in range(6,10)]
    'subsample':[i/100.0 for i in range(40,70,5)],
    'colsample_bytree':[i/100.0 for i in range(50,70,5)]
}
step4_result = xgbParmTuning(data_train, label_train, parameters_step4,
                             n_estimators=cv_result.shape[0],
                             max_depth=step2_result[0]['max_depth'],
                             min_child_weight=step2_result[0]['min_child_weight'],
                             gamma=step3_result[0]['gamma'])
step4_result

### step5: tuning for reg_alpha
# ({'reg_alpha': 0.01}, 0.904593496498574)
parameters_step5 = {
  'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}

step5_result = xgbParmTuning(data_train, label_train, parameters_step5,
                             n_estimators=cv_result.shape[0],
                             max_depth=step2_result[0]['max_depth'],
                             min_child_weight=step2_result[0]['min_child_weight'],
                             gamma=step3_result[0]['gamma'],
                             colsample_bytree=step4_result[0]['colsample_bytree'],
                             subsample=step4_result[0]['subsample']
                              )
step5_result

### step6: increase n_estimators (adding trees)
xgbParmTuning(data_train, label_train, parameters_step5,
              n_estimators=5000, max_depth=6, min_child_weight=3,
              gamma=0, subsample=0.4, colsample_bytree=0.5)

### step7: reduce learning rate
xgc = XGBClassifier(learning_rate =0.1,
                   n_estimators=cv_result.shape[0],
                   max_depth=step2_result[0]['max_depth'],
                   min_child_weight=step2_result[0]['min_child_weight'],
                   gamma=step3_result[0]['gamma'],
                   subsample=step4_result[0]['subsample'],
                   colsample_bytree=step4_result[0]['colsample_bytree'],
                   objective= 'binary:logistic',
                   nthread=4,
                   reg_alpha=0.01,
                   scale_pos_weight=1,
                   seed=123456)

xgc.fit(data_train, label_train)
xgc_pred = xgc.predict(data_test)
confusion_matrix(label_test, xgc_pred)

print("Accuracy on test dataset:", round(metrics.accuracy_score(label_test, xgc_pred),4)*100,"%")
print("ROC_AUC on test dataset:", round(metrics.roc_auc_score(label_test, xgc.predict_proba(data_test)[:,1]),4)*100,"%")


########################### Make prediction on unlabelled data ####################


### make prediction on unlabelled data
data1 = pd.read_csv(r'C:\Users\44757\Desktop\Python_Projects\titanic\data\test.csv')

### transform sex from string to binary
data1['Sex'] = data1['Sex'].replace({"female":0,"male":1})

### cheak missing data
data1.apply(naPercent, axis=0)

## choose features
result = pd.DataFrame(columns=['PassengerId', 'Survived'])
result['PassengerId'] = data1['PassengerId']

data1 = data1.filter(['Name', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked'])

data1['Title'] = data1['Name'].map(getTitle)

## impute Age
lr = LinearRegression()
imp = IterativeImputer(estimator=lr,
                       missing_values=np.nan,
                       max_iter=10,
                       verbose=2,
                       imputation_order='roman',
                       random_state=1,
                       min_value = 0)

# temp = data_train.filter(['Survived', 'Pclass', 'Sex', 'Age', 'Fare'])
# temp = imp.fit_transform(data_train[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]).T[3]

data1['Age'] = imp.fit_transform(data1[['Pclass', 'Sex', 'Age', 'Fare']]).T[2]
data1['Fare'] = imp.fit_transform(data1[['Pclass', 'Sex', 'Age', 'Fare']]).T[3]

#data_train.groupby('Embarked').count()
data1['Embarked'].fillna('S', inplace=True)

## title grouping
data1['title_group'] = data1['Title'].map(titleGroup)


data1 = pd.concat([data1, pd.get_dummies(data1[['Embarked', 'title_group']])], axis=1).\
    drop(['Name', 'Ticket', 'Title', 'Embarked', 'title_group'], axis=1)



# data_target = data1[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
data_target = data1

result['Survived']=xgc.predict(data_target)
# result.to_csv('xgboost_prediction.csv')
result.to_csv('xgboost_prediction2.csv')


result['Survived']=rfc_cv.predict(data_target)
result.to_csv('randomforest_prediction.csv')

result['Survived']=dtc_cv.predict(data_target)
result.set_index('PassengerId', inplace=True)
result.to_csv('decisiontree_prediction.csv')
