# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 10:02:05 2023

@author: priya
"""

import pandas as pd
import seaborn as sns
from sklearn import neighbors
from sklearn.metrics import mean_absolute_error 



teams  = pd.read_csv("teams.csv")
teams = teams[["team", "country","year", "athletes","age","prev_medals", "medals"]]
teams.shape
teams.corr()["medals"]

sns.lmplot(x="age", y ="medals", data=teams, fit_reg= True, ci=None)
teams.hist(column="medals", bins = 25)
teams[teams.isnull().any(axis =1)]
teams.dropna(how='any', inplace=True)

teams.shape
teams["year"].max()

#divide the data to training and testing 
train = teams[teams["year"]<2012].copy()
test = teams[teams["year"]>=2012].copy()
#test_data.shape
#train_data.shape
predictor = ["athletes", "prev_medals"]
target = "medals"


err_val = [] #to store rmse values for different k
for K in range(20):
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
