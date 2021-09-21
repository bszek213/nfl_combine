# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 22:56:18 2021

TODO: how many models to we want? import all data across all seasons

TODO: make a separate script that assesses whether the combine metrics
predicts on whether a player will play within the next 4 seasons - a classification
question

I am currently working on aggregating the 2013 and 2014 data together

add what you think we need. This is just the base framework. 

@author: Brian
"""
import pandas as pd
import numpy as np
pd.options.plotting.backend = 'holoviews'
import hvplot
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import time

class nflCombine:

    def __init__(self):
        snaps_cum_2013 = pd.Series(dtype = float)
        snaps_cum_2014 = pd.Series(dtype = float)
        snaps_cum_2015 = pd.Series(dtype = float)

    def read_in(self,path):
        self.pd_2013 = pd.read_excel('/home/bszekely/Desktop/ProjectsResearch/nfl_combine/NFL 2013_edit.xlsx')
        self.pd_2014 = pd.read_excel('/home/bszekely/Desktop/ProjectsResearch/nfl_combine/NFL 2014_edit.xlsx')
        self.pd_2015 = pd.read_excel('/home/bszekely/Desktop/ProjectsResearch/nfl_combine/NFL 2015_edit.xlsx')
        self.snaps_2013 = pd.read_excel("/home/bszekely/Desktop/ProjectsResearch/nfl_combine/NFL 2013_edit.xlsx",
                                       sheet_name="Snaps")
        self.snaps_2014 = pd.read_excel("/home/bszekely/Desktop/ProjectsResearch/nfl_combine/NFL 2014_edit.xlsx",
                                       sheet_name="Snaps")
        self.snaps_2015 = pd.read_excel("/home/bszekely/Desktop/ProjectsResearch/nfl_combine/NFL 2015_edit.xlsx",
                                       sheet_name="Snaps")

    def cumulative_snaps(self):
        """
        Sums all data across seasons,defense,offense, and special teams.
        This will change the the NANs to zeros
        """
        self.snaps_cum_2013 = self.snaps_2013.sum(axis = 1)
        self.snaps_cum_2014 = self.snaps_2014.sum(axis = 1)
        print(len(self.snaps_cum_2013), "Samples started with - 2013")
        print(len(self.snaps_cum_2014), "Samples started with - 2014")
    
    def split_test(self):
        index_nonzero_13 = self.snaps_cum_2013[self.snaps_cum_2013 !=0 ].index.tolist()
        index_nonzero_14 = self.snaps_cum_2014[self.snaps_cum_2014 !=0 ].index.tolist()
        
        snaps_parse_13 = self.snaps_cum_2013.iloc[index_nonzero_13]
        snaps_parse_14 = self.snaps_cum_2014.iloc[index_nonzero_14]
        pd_2013_nozero = self.pd_2013.iloc[index_nonzero_13,:]
        pd_2014_nozero = self.pd_2014.iloc[index_nonzero_14,:]
        
        cols =['40yd','Vertical','BP','Broad Jump',
                          'Shuttle','3Cone']
        
        x_data_13_ = pd_2013_nozero[['40yd','Vertical','BP','Broad Jump',
                          'Shuttle','3Cone']]
        x_data_14_ = pd_2014_nozero[['40yd','Vertical','BP','Broad Jump',
                          'Shuttle','3Cone']]
    
    
        index_nan_13 = x_data_13_.dropna().index.tolist()
        index_nan_14 = x_data_14_.dropna().index.tolist()
    
        x_data_13_nonan = x_data_13_.loc[index_nan_13]
        x_data_14_nonan = x_data_14_.loc[index_nan_14]
        y_data_13_nonan = snaps_parse_13.loc[index_nan_13]
        y_data_14_nonan = snaps_parse_14.loc[index_nan_14]


        scaler = StandardScaler()
        x_data_13 = scaler.fit_transform(x_data_13_nonan)
        x_data_14 = scaler.fit_transform(x_data_14_nonan)        
        
        df_13 = pd.DataFrame(x_data_13, columns = cols)
        df_14 = pd.DataFrame(x_data_14, columns = cols)
        
        x = df_13.append(df_14)  
        y = y_data_13_nonan.append(y_data_14_nonan)     
        
        #print(np.median(y),"median snaps", np.std(y), "STD snaps")
        print(len(x_data_13_nonan), "Samples started with - 2013")
        print(len(x_data_14_nonan), "Samples started with - 2014")
        #hvplot.show(y.hist())
        self.x_train,self.X_test,self.y_train,self.y_test = train_test_split(x,y)
     
    def model_test(self):
        model = GradientBoostingRegressor(max_depth=3)
        model2 = RandomForestRegressor()
        model3 = LinearRegression()
        model4 = DecisionTreeRegressor(max_depth=3)
        model5 = svm.SVC(kernel = 'poly', gamma = "auto")
        
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

        # mse = mean_squared_error(self.y_test,y_pred,squared = False)
        # mse1 = mean_squared_error(self.y_test,y_pred1,squared = False)
        # mse2 = mean_squared_error(self.y_test,y_pred2,squared = False)
        # mse3 = mean_squared_error(self.y_test,y_pred3,squared = False)
        # mse4 = mean_squared_error(self.y_test,y_pred4,squared = False)
        mse = r2_score(self.y_test, y_pred)
        mse1 = r2_score(self.y_test, y_pred1)
        mse2 = r2_score(self.y_test, y_pred2)
        mse3 = r2_score(self.y_test, y_pred3)
        mse4 = r2_score(self.y_test, y_pred4)
        
        return mse, mse1, mse2, mse3, mse4
        
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
    print("--- %s seconds ---" % (time.time() - start_time))
    hvplot.show(error.plot())
    


