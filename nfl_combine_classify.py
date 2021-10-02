#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sept  28 11:16:50 2021
TO DO: remove data that does not have all combine features,
convert all snap data to binary
run classifiers:
1. RandomForestClassifier instead of Logistic Regression
2. Naive Bayes
3. K-Nearest Neighbors
4. Decision Tree
5. Support Vector Machines

@author: bszekely
"""

from nfl_combine import nflCombine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


class nflCombineClassify(nflCombine):
    
    def __init__(self,path):
        super().__init__()
        super().read_in(path) #change this to be relative via argparse()
        super().cumulative_snaps()
        
    def snaps_to_binary(self):
        cols =['40yd','Vertical','BP','Broad Jump',
                          'Shuttle','3Cone']
        
        x_data_13_ = self.pd_2013[['40yd','Vertical','BP','Broad Jump',
                          'Shuttle','3Cone']]
        x_data_14_ = self.pd_2014[['40yd','Vertical','BP','Broad Jump',
                          'Shuttle','3Cone']]
        x_data_15_ = self.pd_2015[['40yd','Vertical','BP','Broad Jump',
                          'Shuttle','3Cone']]

        index_nan_13 = x_data_13_.dropna().index.tolist()
        index_nan_14 = x_data_14_.dropna().index.tolist()
        index_nan_15 = x_data_15_.dropna().index.tolist()

        y_data_13_nonan = self.snaps_cum_2013.loc[index_nan_13]
        y_data_14_nonan = self.snaps_cum_2014.loc[index_nan_14]
        y_data_15_nonan = self.snaps_cum_2015.loc[index_nan_15]
        
        x_data_13_nonan = x_data_13_.loc[index_nan_13]
        x_data_14_nonan = x_data_14_.loc[index_nan_14]
        x_data_15_nonan = x_data_15_.loc[index_nan_15]
        
        print(len(y_data_13_nonan), "Samples ended with - 2013")
        print(len(y_data_14_nonan), "Samples ended with - 2014")
        print(len(y_data_15_nonan), "Samples ended with - 2015")
        
        #convert to binary
        y_data_13_nonan[y_data_13_nonan > 0] = 1
        y_data_14_nonan[y_data_14_nonan > 0] = 1
        y_data_15_nonan[y_data_15_nonan > 0] = 1
        
        y = pd.concat([y_data_13_nonan, y_data_14_nonan, y_data_15_nonan]).astype(int)
        x = pd.concat([x_data_13_nonan, x_data_14_nonan, x_data_15_nonan])
        
        self.x_train_classify,self.X_test_classify,self.y_train_classify,self.y_test_classify = train_test_split(x,y)
        
    def model_test_classify(self):
        model1 = DecisionTreeClassifier()
        model2 = KNeighborsClassifier()
        model3 = SVC()
        model4 = GaussianNB()
        model5 = RandomForestClassifier()
        
        model1.fit(self.x_train_classify,self.y_train_classify)
        model2.fit(self.x_train_classify,self.y_train_classify)
        model3.fit(self.x_train_classify,self.y_train_classify)
        model4.fit(self.x_train_classify,self.y_train_classify)
        model5.fit(self.x_train_classify,self.y_train_classify)
        
        y_pred1 = model1.predict(self.X_test_classify)
        y_pred2 = model2.predict(self.X_test_classify)
        y_pred3 = model3.predict(self.X_test_classify)
        y_pred4 = model4.predict(self.X_test_classify)
        y_pred5 = model5.predict(self.X_test_classify)
        
        print("Accuracy:",metrics.accuracy_score(self.y_test_classify, y_pred1))
        print("Accuracy:",metrics.accuracy_score(self.y_test_classify, y_pred2))
        print("Accuracy:",metrics.accuracy_score(self.y_test_classify, y_pred3))
        print("Accuracy:",metrics.accuracy_score(self.y_test_classify, y_pred4))
        print("Accuracy:",metrics.accuracy_score(self.y_test_classify, y_pred5))
            
if __name__ == '__main__':
    classify = nflCombineClassify('')
    classify.snaps_to_binary()
    classify.model_test_classify()