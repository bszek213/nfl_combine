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
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import hvplot


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
        x_data_17_ = self.pd_2017[['40yd','Vertical','BP','Broad Jump',
                          'Shuttle','3Cone']]

        index_nan_13 = x_data_13_.dropna().index.tolist()
        index_nan_14 = x_data_14_.dropna().index.tolist()
        index_nan_15 = x_data_15_.dropna().index.tolist()
        index_nan_17 = x_data_17_.dropna().index.tolist()

        y_data_13_nonan = self.snaps_cum_2013.loc[index_nan_13]
        y_data_14_nonan = self.snaps_cum_2014.loc[index_nan_14]
        y_data_15_nonan = self.snaps_cum_2015.loc[index_nan_15]
        y_data_17_nonan = self.snaps_cum_2017.loc[index_nan_17]
        
        x_data_13_nonan = x_data_13_.loc[index_nan_13]
        x_data_14_nonan = x_data_14_.loc[index_nan_14]
        x_data_15_nonan = x_data_15_.loc[index_nan_15]
        x_data_17_nonan = x_data_17_.loc[index_nan_17]
        
        print(len(y_data_13_nonan), "Samples ended with - 2013")
        print(len(y_data_14_nonan), "Samples ended with - 2014")
        print(len(y_data_15_nonan), "Samples ended with - 2015")
        print(len(y_data_17_nonan), "Samples ended with - 2017")
        
        #convert to binary
        y_data_13_nonan[y_data_13_nonan > 0] = 1
        y_data_14_nonan[y_data_14_nonan > 0] = 1
        y_data_15_nonan[y_data_15_nonan > 0] = 1
        y_data_17_nonan[y_data_17_nonan > 0] = 1
        
        y = pd.concat([y_data_13_nonan, y_data_14_nonan, y_data_15_nonan, y_data_17_nonan]).astype(int)
        x = pd.concat([x_data_13_nonan, x_data_14_nonan, x_data_15_nonan, x_data_17_nonan])
        
        self.x_train_classify,self.X_test_classify,self.y_train_classify,self.y_test_classify = train_test_split(x,y)
        
    def model_test_classify(self):
        
        self.model1_classify = DecisionTreeClassifier(criterion='entropy')
        self.model2_classify = GradientBoostingClassifier(n_estimators=105,max_depth=4,tol=0.001)
        self.model3_classify = SVC(kernel='linear')
        self.model4_classify = GaussianNB()
        self.model5_classify = RandomForestClassifier(n_estimators=105,criterion='entropy',min_samples_leaf=4)
        self.model6_classify = LogisticRegression(max_iter=105)
        
        self.model1_classify.fit(self.x_train_classify,self.y_train_classify)
        self.model2_classify.fit(self.x_train_classify,self.y_train_classify)
        self.model3_classify.fit(self.x_train_classify,self.y_train_classify)
        self.model4_classify.fit(self.x_train_classify,self.y_train_classify)
        self.model5_classify.fit(self.x_train_classify,self.y_train_classify)
        self.model6_classify.fit(self.x_train_classify,self.y_train_classify)

        
        y_pred1 = self.model1_classify.predict(self.X_test_classify)
        y_pred2 = self.model2_classify.predict(self.X_test_classify)
        y_pred3 = self.model3_classify.predict(self.X_test_classify)
        y_pred4 = self.model4_classify.predict(self.X_test_classify)
        y_pred5 = self.model5_classify.predict(self.X_test_classify)
        y_pred6 = self.model6_classify.predict(self.X_test_classify)

        
        print("DecisionTreeClassifier Accuracy:",metrics.accuracy_score(self.y_test_classify, y_pred1))
        print("GradientBoostingClassifier Accuracy:",metrics.accuracy_score(self.y_test_classify, y_pred2))
        print("SVC Accuracy:",metrics.accuracy_score(self.y_test_classify, y_pred3))
        print("GaussianNB Accuracy:",metrics.accuracy_score(self.y_test_classify, y_pred4))
        print("RandomForestClassifier Accuracy:",metrics.accuracy_score(self.y_test_classify, y_pred5))
        print("LogisticRegression Accuracy:",metrics.accuracy_score(self.y_test_classify, y_pred6))
        

    def plot_feature_importance_classify(self):
        imps = permutation_importance(self.model4_classify, self.X_test_classify, self.y_test_classify)
        #Calculate feature importance 
        feature_imp1 = pd.Series(self.model1_classify.feature_importances_,index=self.X_test_classify.columns).sort_values(ascending=False)
        feature_imp2 = pd.Series(self.model2_classify.feature_importances_,index=self.X_test_classify.columns).sort_values(ascending=False)
        feature_imp4 = pd.Series(imps.importances_mean,index=self.X_test_classify.columns).sort_values(ascending=False)
        feature_imp3 = pd.Series(self.model3_classify.coef_[0],index=self.X_test_classify.columns).sort_values(ascending=False)
        feature_imp5 = pd.Series(self.model5_classify.feature_importances_,index=self.X_test_classify.columns).sort_values(ascending=False)
        feature_imp6 = pd.Series(self.model6_classify.coef_[0],index=self.X_test_classify.columns).sort_values(ascending=False)
        fig, axs = plt.subplots(2, 3)
        axs = axs.flatten()
    
        sns.barplot(ax=axs[0],x=feature_imp1,y=feature_imp1.index)
        sns.barplot(ax=axs[1],x=feature_imp2,y=feature_imp2.index)
        sns.barplot(ax=axs[2],x=feature_imp3,y=feature_imp3.index)
        sns.barplot(ax=axs[3],x=feature_imp4,y=feature_imp4.index)
        sns.barplot(ax=axs[4],x=feature_imp5,y=feature_imp5.index)
        sns.barplot(ax=axs[5],x=feature_imp6,y=feature_imp6.index)
        plt.xlabel('Feature Importance')
        axs[0].set_title('DecisionTreeClassifier')
        axs[1].set_title('GradientBoostingClassifier')
        axs[2].set_title('SVC')
        axs[3].set_title('GaussianNB')
        axs[4].set_title('RandomForestClassifier')
        axs[5].set_title('LogisticRegression')
        plt.draw()
        plt.show()
            
if __name__ == '__main__':
    classify = nflCombineClassify('')
    classify.snaps_to_binary()
    classify.model_test_classify()
    lst = []
    cols = ['acc']
    # h_para = 100
    # for i in range(0,20):
    #     save_list =  classify.model_test_classify(h_para)
    #     lst.append(save_list)
    #     h_para =+ 5 
    # acc = pd.DataFrame(lst,columns=cols)
    # hvplot.show(acc.plot())
    classify.plot_feature_importance_classify()