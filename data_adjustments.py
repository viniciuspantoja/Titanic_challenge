#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:42:22 2017

@author: ViniciusPantoja
"""
#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re





X = pd.read_csv('path/train.csv')
Y = pd.read_csv('path/test.csv')



Z = X.append(Y,ignore_index = True )

# Very good video on regular expression
# https://www.youtube.com/watch?v=ZdDOauFIDkw&t=7s 

for i in range (0, len(Z["Name"])):
    Z['Name'][i] = re.split(r'[,.]', Z['Name'][i])[1]

Title_Dictionary = {
                        " Capt":       "Officer",
                        " Col":        "Army",
                        " Major":      "Army",
                        " Jonkheer":   "Civ",
                        " Don":        "Civ",
                        " Sir" :       "Sir",
                        " Dr":         "Dr",
                        " Rev":        "Rev",
                        " the Countess":"Civ_female",
                        " Dona":       "Dona",
                        " Mme":        "Mrs",
                        " Mlle":       "Miss",
                        " Ms":         "Mrs",
                        " Mr" :        "Mr",
                        " Mrs" :       "Mrs",
                        " Miss" :      "Miss",
                        " Master" :    "Master",
                        " Lady" :      "Civ_female"

                        }
    
    # we map each title
Z_name = Z.Name.map(Title_Dictionary)    


Z["Cabin"] = Z["Cabin"].isnull()*1



# Temos que idade tem muitas linhas sem informação. Vamos fazer preencher de 
# acordo com a média

Z_Age_Missing = Z[Z["Age"].isnull()]
                  
for i in range (0,len(Z_Age_Missing)):
    if Z_Age_Missing["Sex"].iloc[i] == 'female' and  Z_Age_Missing["Pclass"].iloc[i] ==1:
        if Z_Age_Missing['Name'].iloc[i] == ' Miss':
            a = 30
        elif Z_Age_Missing['Name'].iloc[i] == ' Mrs':
            a = 45
        elif Z_Age_Missing['Name'].iloc[i] == ' Officer':
            a = 49
        elif Z_Age_Missing['Name'].iloc[i] == ' Royalty':
            a = 39

    elif Z_Age_Missing["Sex"].iloc[i] == 'female' and  Z_Age_Missing["Pclass"].iloc[i] ==2:
        if Z_Age_Missing['Name'].iloc[i] == ' Miss':
            a = 20
        elif Z_Age_Missing['Name'].iloc[i] == ' Mrs':
            a = 35
            
    elif Z_Age_Missing["Sex"].iloc[i] == 'female' and  Z_Age_Missing["Pclass"].iloc[i] ==3:
        if Z_Age_Missing['Name'].iloc[i] == ' Miss':
            a = 18
        elif Z_Age_Missing['Name'].iloc[i] == ' Mrs':
            a = 31
                
    
    elif Z_Age_Missing["Sex"].iloc[i] == 'male' and  Z_Age_Missing["Pclass"].iloc[i] ==1:
        if Z_Age_Missing['Name'].iloc[i] == ' Master':
            a = 6
        elif Z_Age_Missing['Name'].iloc[i] == ' Mr':
            a = 41.5
        elif Z_Age_Missing['Name'].iloc[i] == ' Officer':
            a = 52
        elif Z_Age_Missing['Name'].iloc[i] == ' Royalty':
            a = 40
            
    elif Z_Age_Missing["Sex"].iloc[i] == 'male' and  Z_Age_Missing["Pclass"].iloc[i] ==2:
        if Z_Age_Missing['Name'].iloc[i] == ' Master':
            a = 2
        elif Z_Age_Missing['Name'].iloc[i] == ' Mr':
            a = 30
        elif Z_Age_Missing['Name'].iloc[i] == ' Officer':
            a = 41
                
                
    elif Z_Age_Missing["Sex"].iloc[i] == 'male' and  Z_Age_Missing["Pclass"].iloc[i] ==3:
        if Z_Age_Missing['Name'].iloc[i] == ' Master':
            a = 6
        elif Z_Age_Missing['Name'].iloc[i] == ' Mr':
            a = 26
                
    Z_Age_Missing["Age"].iloc[i] = a
                
    
Z_Age_Missing = Z_Age_Missing["Age"]



for i in range (0,len(Z['Age'])):
    if pd.isnull(Z["Age"].iloc[i]):
        a = Z_Age_Missing[Z_Age_Missing.index == i]
        Z["Age"].iloc[i] = a.iloc[0]

Z["Fare"] = Z['Fare'].fillna(Z["Fare"].mean() )

Z = pd.get_dummies(Z)


# Ok. We have deleted only the rows that the column age does not have info. 
# Now lets get the numeric part of our database, and make our first prediction 
# machine learning on that.

X = Z[0:891]
Y = Z[891:1310]


from sklearn.cross_validation import train_test_split

#Let's divide our database to check our performance. 
X_train, X_test = train_test_split(X, test_size = 0.2, random_state = 42)
 
Y_train = X_train.pop("Survived")
Y_test = X_test.pop("Survived")
Y = Y.drop('Survived', axis = 1)












