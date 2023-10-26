# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 22:58:21 2023

@author: priya
"""

import pandas as pd
import seaborn as sns
teams  = pd.read_csv("teams.csv")
teams = teams[["team", "country","year", "athletes","age","prev_medals", "medals"]]
teams
teams.corr()["medals"]

sns.lmplot(x="age", y ="medals", data=teams, fit_reg= True, ci=None)
teams.hist(column="medals", bins = 25)
teams[teams.isnull().any(axis = 1)]
teams.dropna()
teams
teams["year"].max()

#divide the data to training and testing 
train_data = teams[teams["year"]<2012].copy()
test_data = teams[teams["year"]>=2012].copy()


from sklearn.linear_model import LinearRegression
reg = LinearRegression()

predictor = ["athletes", "prev_medals"]
target = "medals"
reg.fit(train_data[predictor], train_data["medals"])