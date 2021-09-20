# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 22:56:18 2021

TODO: how many models to we want? import all data across all seasons

I am currently working on aggregating the 2013 and 2014 data together

add what you think we need. This is just the base framework. 

@author: Brian
"""
import pandas as pd
import numpy as np
pd.options.plotting.backend = 'holoviews'
import hvplot
#from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import time

class nflCombine:

    def __init__(self):
        snaps_cum_2013 = pd.Series(dtype = float)
        snaps_cum_2014 = pd.Series(dtype = float)

    def read_in(self,path):
        self.pd_2013 = pd.read_excel('C:/Users/Bharath/Desktop/nfl_combine/NFL 2013_edit.xlsx')
        self.pd_2014 = pd.read_excel('C:/Users/Bharath/Desktop/nfl_combine/NFL 2014_edit.xlsx')
        self.snaps_2013 = pd.read_excel("C:/Users/Bharath/Desktop/nfl_combine/NFL 2013_edit.xlsx",
                                       sheet_name="Snaps")
        self.snaps_2014 = pd.read_excel("C:/Users/Bharath/Desktop/nfl_combine/NFL 2014_edit.xlsx",
                                       sheet_name="Snaps")

    def cumulative_snaps(self):
        self.snaps_cum_2013 = self.snaps_2013.sum(axis = 1,skipna = True)
        self.snaps_cum_2014 = self.snaps_2014.sum(axis = 1,skipna = True)
    
    def split_test(self):
        index_nonzero = self.snaps_cum_2013[self.snaps_cum_2013 !=0 ].index.tolist()
        
        snaps_parse = self.snaps_cum_2013.iloc[index_nonzero]
        pd_2013_nozero = self.pd_2013.iloc[index_nonzero,:]
        
        x_data = pd_2013_nozero[['40yd','Vertical','BP','Broad Jump',
                          'Shuttle','3Cone']]
        index_nan = x_data.dropna().index.tolist()
        
        x = x_data.loc[index_nan]
        y = snaps_parse.loc[index_nan]
        
        self.x_train,self.X_test,self.y_train,self.y_test = train_test_split(x,y)
     
    def model_test(self):
        model = GradientBoostingRegressor(n_estimators=100,max_depth=5)
        model2 = RandomForestRegressor()
        model3 = LinearRegression()
        model4 = DecisionTreeRegressor(max_depth=5)
        model5 = svm.SVC()
        
        model.fit(self.x_train,self.y_train)
        model2.fit(self.x_train,self.y_train)
        model3.fit(self.x_train,self.y_train)
        model4.fit(self.x_train,self.y_train)
        model5.fit(self.x_train,self.y_train)
        
        y_pred = model.predict(self.X_test)
        y_pred1 = model2.predict(self.X_test)
        y_pred2 = model3.predict(self.X_test)
        y_pred3 = model4.predict(self.X_test)
        y_pred4 = model5.predict(self.X_test)

        mse = mean_squared_error(self.y_test,y_pred)
        mse1 = mean_squared_error(self.y_test,y_pred1)
        mse2 = mean_squared_error(self.y_test,y_pred2)
        mse3 = mean_squared_error(self.y_test,y_pred3)
        mse4 = mean_squared_error(self.y_test,y_pred4)
        
        return np.sqrt(mse),np.sqrt(mse1),np.sqrt(mse2),np.sqrt(mse3),np.sqrt(mse4)
        
if __name__ == '__main__':
    start_time = time.time()
    nfl = nflCombine()
    nfl.read_in("")
    nfl.cumulative_snaps()
    nfl.split_test()
    nfl.model_test()
    cols = ['Gradient_error', 'RFR_error', 'Linear_error','DT_error',
            'SVM_error']
    lst = []
    for i in range(0,20):
       save_list =  nfl.model_test()
       lst.append(save_list)
    error = pd.DataFrame(lst,columns=cols)
    hvplot.show(error.plot())
    print("--- %s seconds ---" % (time.time() - start_time))


