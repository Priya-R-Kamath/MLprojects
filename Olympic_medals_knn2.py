# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 13:17:49 2023

@author: priya
"""
#from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors
import pandas as pd
from sklearn.metrics import mean_absolute_error

teams= pd.read_csv("teams.csv")

teams.corr()["medals"]
teams =  teams[["team", "country","year", "events", "athletes", "age", "prev_medals", "prev_3_medals", "medals"]]

teams[teams.isnull().any(axis =1)]
teams.dropna(how='any', inplace=True)

teams.shape

train = teams[teams["year"]<2012].copy()
test = teams[teams["year"]>=2012].copy()

predictor = ["athletes", "prev_medals", "events","prev_3_medals" ]
target = "medals"


err_val = [] #to store rmse values for different k
for K in range(40):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(train[predictor], train[target])  #fit the model
    pred=model.predict(test[predictor]).round() #make prediction on test set
    error = mean_absolute_error(test[target],pred) #calculate abs error
    err_val.append(error) #store rmse values
    print('error value for k= ' , K , 'is:', error)

k = err_val.index(min(err_val))+1
print("Ideal value of K is  ", k)   
model = neighbors.KNeighborsRegressor(n_neighbors = k)
model.fit(train[predictor], train[target])
pred=model.predict(test[predictor]).round()
test["prediction"] = pred
error_knn = mean_absolute_error(test["medals"], test["prediction"])

print(error_knn)