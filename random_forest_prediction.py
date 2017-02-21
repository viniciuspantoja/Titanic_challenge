#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 10:24:01 2017

@author: ViniciusPantoja
"""
#%%
# Use first the file data_adjsutments.py to work with the variables.

# Now lets optimize the model
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.metrics import roc_auc_score as ras

#First, number of estimators
results = []

n_estimator_option = [100, 250, 300, 700]

for trees in n_estimator_option:
    model = rfr(trees, oob_score = True, random_state = 42)
    model.fit(X_train,Y_train)
    print trees, 'trees'
    roc = ras(Y_train, model.oob_prediction_)
    print 'c-stat:', roc
    results.append(roc)
    print ""

pd.Series(results,n_estimator_option).plot()

# n_estimator_option = 300 is the highest. Let's use it
#%%

results = []

max_feature_option = ['auto', None, "sqrt", "log2", 0.9, 0.2]

for max_feature in max_feature_option:
    model = rfr(n_estimators = 300, oob_score = True, random_state = 42, max_features = max_feature)
    model.fit(X_train,Y_train)
    print max_feature, 'option'
    roc = ras(Y_train, model.oob_prediction_)
    print 'c-stat:', roc
    results.append(roc)
    print ""

pd.Series(results,max_feature_option).plot()


# Lets choose the Auto option
#%%
# Now, the number of leaves
results = []

min_samples_leaf_option = [1,2,3,4,5,6,7,8,9,10]

for min_samples  in min_samples_leaf_option:
    model = rfr(n_estimators = 300, oob_score = True, random_state = 42, 
                max_features = 'auto', min_samples_leaf= min_samples)
    model.fit(X_train,Y_train)
    print min_samples, 'min_samples'
    roc = ras(Y_train, model.oob_prediction_)
    print 'c-stat:', roc
    results.append(roc)
    print ""

pd.Series(results,min_samples_leaf_option).plot()


# The best choice is 6. We need high number of leaves. Therefore I've chose 
# the local optimum.
#%%
# Final model

model = rfr(n_estimators = 300, oob_score = True, random_state = 42, 
                max_features = 'auto', min_samples_leaf= 6)

model.fit(X_train,Y_train)

y_oob = model.oob_prediction_

print "c-stat:", ras(Y_train,y_oob)


# Ok, our prediction has exploded to 0.85

final_prediction = model.predict(Y)




