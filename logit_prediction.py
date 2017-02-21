#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 15:43:55 2017

@author: ViniciusPantoja
"""

#%%

# After runnig the data_adjustments file, this code will deliver the final 
# predictions

from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import roc_auc_score as roc

model = lr(penalty = 'l2', fit_intercept = False, n_jobs = -1)

resultado = model.fit(X_train,Y_train)

prediction = resultado.predict(X_test)

roc(Y_test,prediction)

final_prediction = resultado.predict(Y)
