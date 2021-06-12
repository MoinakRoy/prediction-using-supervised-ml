import pandas as pd
import numpy as np  

import lux # new data visualization library 

import matplotlib.pyplot as plt  
#%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

path = r"C:\Users\Moinak\OneDrive\Documents\sparks\student_scores.csv.txt"
s_data = pd.read_csv(path)

s_data.head()

s_data # press toggle button to view the graph

s_data.columns
s_data.shape
train,test = train_test_split(s_data,test_size=0.25,random_state=123)
train.shape
test.shape
train_x=train.drop("Scores",axis=1)
train_y=train["Scores"]
test_x=test.drop("Scores",axis=1)
test_y=test["Scores"]
lr=LinearRegression()
lr.fit(train_x,train_y)
lr.coef_

lr.intercept_

# Plotting the regression line # formula for line is y=m*x + c
line = lr.coef_*train_x+lr.intercept_

# Plotting for the test data
plt.scatter(train_x,train_y)
plt.plot(train_x, line);
plt.show()
pr=lr.predict(test_x)
list(zip(test_y,pr))

from sklearn.metrics import mean_squared_error

mean_squared_error(test_y,pr,squared=False)
hour =[9.25]
own_pr=lr.predict([hour])
print("No of Hours = {}".format([hour]))
print("Predicted Score = {}".format(own_pr[0]))
