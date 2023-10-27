# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 13:35:07 2023

@author: priya
"""

from sklearn.ensemble import RandomForestRegressor
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

regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(train[predictor], train[target])
pred=regressor.predict(test[predictor]).round()
test["prediction"] = pred
error_RF = mean_absolute_error(test["medals"], test["prediction"])

print("abs error using  all coefs with high corr", error_RF)

predictor = ["athletes", "prev_medals" ]
target = "medals"

regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(train[predictor], train[target])
pred=regressor.predict(test[predictor]).round()
test["prediction"] = pred
error_RF = mean_absolute_error(test["medals"], test["prediction"])

print("abs error using few coefs with high-corr  considered", error_RF)