# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 22:58:21 2023

@author: priya
"""

import pandas as pd
import seaborn as sns
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
train_data = teams[teams["year"]<2012].copy()
test_data = teams[teams["year"]>=2012].copy()
#test_data.shape
#train_data.shape
predictor = ["athletes", "prev_medals"]
target = "medals"
from sklearn.linear_model import LinearRegression
reg = LinearRegression()


reg.fit(train_data[predictor], train_data["medals"])
predictions = reg.predict(test_data[predictor])
test_data["prediction"] = predictions
#test_data
test_data.loc[test_data["prediction"]<0,"prediction"] = 0
test_data["prediction"] = test_data["prediction"].round()


#MEASURE hte ERROR
from sklearn.metrics import mean_absolute_error
error_linearReg = mean_absolute_error(test_data["medals"], test_data["prediction"])

print(error_linearReg)

