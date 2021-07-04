import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
df=pd.read_csv("internship_train.csv")
df['squ6']=np.square(df['6'])
x,y=df.drop(['target','8'],axis=1),df['target']
linm=LinearRegression()
linm.fit(x,y)