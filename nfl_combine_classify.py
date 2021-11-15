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

from nfl_combine_regressor import nflCombineRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


class nflCombineClassify(nflCombineRegressor):
    
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
        
        scaler = StandardScaler()
        x_data_13 = scaler.fit_transform(x_data_13_nonan)
        x_data_14 = scaler.fit_transform(x_data_14_nonan) 
        x_data_15 = scaler.fit_transform(x_data_15_nonan)
        #x_data_16 = scaler.fit_transform(x_data_16_nonan)
        x_data_17 = scaler.fit_transform(x_data_17_nonan)
        
        df_13 = pd.DataFrame(x_data_13, columns = cols)
        df_14 = pd.DataFrame(x_data_14, columns = cols)
        df_15 = pd.DataFrame(x_data_15, columns = cols)
        #df_16 = pd.DataFrame(x_data_16, columns = cols)
        df_17 = pd.DataFrame(x_data_17, columns = cols)
        
        # y = pd.concat([y_data_13_nonan, y_data_14_nonan, y_data_15_nonan, y_data_17_nonan]).astype(int)
        # x = pd.concat([x_data_13_nonan, x_data_14_nonan, x_data_15_nonan, x_data_17_nonan])
        
        y = pd.concat([y_data_13_nonan, y_data_14_nonan, y_data_15_nonan, y_data_17_nonan]).astype(int)
        x = pd.concat([df_13, df_14, df_15, df_17])
        
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x,y, train_size=0.8) 
        # self.x_train, self.x_rem, self.y_train, self.y_rem = train_test_split(x,y, train_size=0.5)
        # self.x_valid, self.x_test, self.y_valid, self.y_test = train_test_split(self.x_rem,self.y_rem, test_size=0.2)
        
    def model_test_classify(self):
            # self.model1_classify = DecisionTreeClassifier()
            # self.model2_classify = GradientBoostingClassifier() 
            # self.model3_classify = SVC(kernel='linear')
            # self.model4_classify = GaussianNB()
            # self.model5_classify = RandomForestClassifier()
            # self.model6_classify = LogisticRegression()
            
            # Determine which models performs best
            DT = cross_validate(DecisionTreeClassifier(),
                                            self.x_train, self.y_train, cv=10,
                                            scoring=['accuracy'],return_train_score=True)
            GB = cross_validate(GradientBoostingClassifier(),
                                            self.x_train, self.y_train, cv=10,
                                            scoring=['accuracy'],return_train_score=True)
            SV_C = cross_validate(SVC(kernel='rbf'),
                                            self.x_train, self.y_train, cv=10,
                                            scoring=['accuracy'],return_train_score=True)
            RF = cross_validate(RandomForestClassifier(),
                                            self.x_train, self.y_train, cv=10,
                                            scoring=['accuracy'],return_train_score=True)
            LR = cross_validate(LogisticRegression(),
                                            self.x_train, self.y_train, cv=10,
                                            scoring=['accuracy'],return_train_score=True)

            print('results of DT: ',np.mean(DT['test_accuracy']))
            print('results of GB: ',np.mean(GB['test_accuracy']))
            print('results of SV_C: ',np.mean(SV_C['test_accuracy'])) #WINNER
            print('results of RF: ',np.mean(RF['test_accuracy']))
            print('results of LF: ',np.mean(LR['test_accuracy']))  
            
            #Tune the winner
            Cs = [2,5,10,15]
            toler = [0.003, 0.0003]
            SV_C1 = cross_validate(SVC(kernel='linear',C=Cs[0],tol=toler[0]),
                                            self.x_train, self.y_train, cv=10,
                                            scoring=['accuracy'],return_train_score=True)
            SV_C2 = cross_validate(SVC(kernel='linear',C=Cs[0],tol=toler[1]),
                                            self.x_train, self.y_train, cv=10,
                                            scoring=['accuracy'],return_train_score=True)
            
            SV_C3 = cross_validate(SVC(kernel='linear',C=Cs[1],tol=toler[0]),
                                            self.x_train, self.y_train, cv=10,
                                            scoring=['accuracy'],return_train_score=True)
            SV_C4 = cross_validate(SVC(kernel='linear',C=Cs[1],tol=toler[1]),
                                            self.x_train, self.y_train, cv=10,
                                            scoring=['accuracy'],return_train_score=True)
            
            SV_C5 = cross_validate(SVC(kernel='linear',C=Cs[2],tol=toler[0]),
                                            self.x_train, self.y_train, cv=10,
                                            scoring=['accuracy'],return_train_score=True)
            SV_C6 = cross_validate(SVC(kernel='linear',C=Cs[2],tol=toler[1]),
                                            self.x_train, self.y_train, cv=10,
                                            scoring=['accuracy'],return_train_score=True)
            
            SV_C7 = cross_validate(SVC(kernel='linear',C=Cs[3],tol=toler[0]),
                                            self.x_train, self.y_train, cv=10,
                                            scoring=['accuracy'],return_train_score=True)
            SV_C8= cross_validate(SVC(kernel='linear',C=Cs[3],tol=toler[1]),
                                            self.x_train, self.y_train, cv=10,
                                            scoring=['accuracy'],return_train_score=True)
            SV_C9= cross_validate(SVC(kernel='linear',C=Cs[3],tol=toler[1],gamma='auto'),
                                            self.x_train, self.y_train, cv=10,
                                            scoring=['accuracy'],return_train_score=True)
            
            print('Print all cross outcomes: ',SV_C1['test_accuracy'])
            
            print('results of SV_C1: ',np.mean(SV_C1['test_accuracy']), Cs[0],toler[0])
            print('results of SV_C2: ',np.mean(SV_C2['test_accuracy']),Cs[0],toler[1])
            print('results of SV_C3: ',np.mean(SV_C3['test_accuracy']),Cs[1],toler[0]) #WINNER
            print('results of SV_C4: ',np.mean(SV_C4['test_accuracy']),Cs[1],toler[1])
            print('results of SV_C5: ',np.mean(SV_C5['test_accuracy']),Cs[2],toler[0])  
            print('results of SV_C6: ',np.mean(SV_C6['test_accuracy']),Cs[2],toler[1]) 
            print('results of SV_C7: ',np.mean(SV_C7['test_accuracy']),Cs[3],toler[0])
            print('results of SV_C8: ',np.mean(SV_C8['test_accuracy']),Cs[3],toler[1])
            print('results of SV_C8 gamma auto: ',np.mean(SV_C9['test_accuracy']),Cs[3],toler[1])
            
            #Test Data 
            test_model = SVC(kernel= 'linear',C = 2, tol = 0.003) 
            test_model.fit(self.x_test,self.y_test)
            
            
            print(accuracy_score(self.y_test, test_model.predict(self.x_test)))
            # depths = [1,2,3,4]
            # min_samples_leaves = [1,2,3,4]
            # cv_results_DT1 = cross_validate(DecisionTreeClassifier(max_depth=depths[0],min_samples_leaf=min_samples_leaves[0]),
            #                                 self.x_train, self.y_train, cv=10,
            #                                 scoring=['accuracy'],return_train_score=True)
            # cv_results_DT2 = cross_validate(DecisionTreeClassifier(max_depth=depths[1],min_samples_leaf=min_samples_leaves[1]),
            #                                 self.x_train, self.y_train, cv=10,
            #                                 scoring=['accuracy'],return_train_score=True)
            # cv_results_DT3 = cross_validate(DecisionTreeClassifier(max_depth=depths[2],min_samples_leaf=min_samples_leaves[2]),
            #                                 self.x_train, self.y_train, cv=10,
            #                                 scoring=['accuracy'],return_train_score=True)
            # cv_results_DT4 = cross_validate(DecisionTreeClassifier(max_depth=depths[3],min_samples_leaf=min_samples_leaves[3]),
            #                                 self.x_train, self.y_train, cv=10,
            #                                 scoring=['accuracy'],return_train_score=True) 
            # print('results of DT1: ',np.mean(cv_results_DT1['test_accuracy']))
            # print('results of DT2: ',np.mean(cv_results_DT2['test_accuracy']))
            # print('results of DT3: ',np.mean(cv_results_DT3['test_accuracy']))
            # print('results of DT4: ',np.mean(cv_results_DT4['test_accuracy']))
            # output_DT = [np.mean(cv_results_DT1['test_accuracy']), np.mean(cv_results_DT2['test_accuracy']),
            #                     np.mean(cv_results_DT3['test_accuracy']),np.mean(cv_results_DT4['test_accuracy'])]
            # max_value_DT = max(output_DT)
            # max_index_DT = output_DT. index(max_value_DT)
            # fin_depth_DT = depths[max_index_DT]
            # fin_leaf_DT = min_samples_leaves[max_index_DT]
            
            # #train Hyper params              
            # max_depths = np.arange(1, 5, 1)
            # training_error_model = []
            # training_error_model2 = []
            # training_error_model4 = []
            # for max_depth in max_depths:
            #     self.model = DecisionTreeClassifier(min_samples_leaf = max_depth)
            #     self.model2 = GradientBoostingClassifier(min_samples_leaf = max_depth)
            #     self.model4 = RandomForestClassifier(min_samples_leaf = max_depth)
                
            #     self.model.fit(self.x_train,self.y_train)
            #     self.model2.fit(self.x_train,self.y_train)
            #     self.model4.fit(self.x_train,self.y_train)
                
            #     training_error_model.append(metrics.accuracy_score(self.y_valid, self.model.predict(self.x_valid)))
            #     training_error_model2.append(metrics.accuracy_score(self.y_valid, self.model2.predict(self.x_valid)))
            #     training_error_model4.append(metrics.accuracy_score(self.y_valid, self.model4.predict(self.x_valid)))

            # fig, axs = plt.subplots(1, 3)
            # axs[0].plot(max_depths, training_error_model, color='blue', label='validation error GradientBoostingClassifier - tree depth')
            # axs[1].plot(max_depths, training_error_model2, color='red', label='validation error RandomForestClassifier - tree depth')
            # axs[2].plot(max_depths, training_error_model4, color='black', label='validation error DecisionClassifier - tree depth')
            # axs[0].set_xlabel('Tree depth')
            # axs[0].set_ylabel('Root Mean squared error')
            # axs[0].set_title('GradientBoostingClassifier', pad=15, size=15)
            # axs[1].set_xlabel('Tree depth')
            # axs[1].set_ylabel('Root Mean squared error')
            # axs[1].set_title('RandomForestClassifier', pad=15, size=15)
            # axs[2].set_xlabel('Tree depth')
            # axs[2].set_ylabel('Root Mean squared error')
            # axs[2].set_title('DecisionClassifier', pad=15, size=15)
            # plt.show()        
        # print("DecisionTreeClassifier Accuracy:",metrics.accuracy_score(y, y_pred1))
        # print("GradientBoostingClassifier Accuracy:",metrics.accuracy_score(y, y_pred2))
        # print("SVC Accuracy:",metrics.accuracy_score(y, y_pred3))
        # print("GaussianNB Accuracy:",metrics.accuracy_score(y, y_pred4))
        # print("RandomForestClassifier Accuracy:",metrics.accuracy_score(y, y_pred5))
        # print("LogisticRegression Accuracy:",metrics.accuracy_score(y, y_pred6))

        # DTC = metrics.accuracy_score(y, y_pred1)
        # GBC = metrics.accuracy_score(y, y_pred2)
        # SVC_output = metrics.accuracy_score(y, y_pred3)
        # RFC = metrics.accuracy_score(y, y_pred5)
        # LogR = metrics.accuracy_score(y, y_pred6)
        
        

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
    
    # DTC_train, GBC_train, SVC_train, RFC_train, LogR_train = classify.model_test_classify(classify.x_train,classify.y_train,False) #train
    # DTC_valid, GBC_valid, SVC_valid, RFC_valid, LogR_valid =classify.model_test_classify(classify.x_valid,classify.y_valid,True) #valid
    # DTC_test, GBC_test, SVC_test, RFC_test, LogR_test =classify.model_test_classify(classify.x_test,classify.y_test,True) #test
    
    # cols = ['DecisionTreeClassifier','GradientBoostingClassifier','SVC','RandomForestClassifier',
    #         'LogisticRegression']
    # train_list = [DTC_train, GBC_train, SVC_train, RFC_train, LogR_train]
    # valid_list = [DTC_valid, GBC_valid, SVC_valid, RFC_valid, LogR_valid]
    # test_list = [DTC_test, GBC_test, SVC_test, RFC_test, LogR_test]
    # final_lst = []
    # final_lst.append(train_list)
    # final_lst.append(valid_list)
    # final_lst.append(test_list)
    # acc = pd.DataFrame(final_lst,columns=cols)
    # print(acc.to_csv('accuracy_class.csv'))
    
    # lst = []
    # cols = ['acc']
    #classify.plot_feature_importance_classify()
    # h_para = 100
    # for i in range(0,20):
    #     save_list =  classify.model_test_classify(h_para)
    #     lst.append(save_list)
    #     h_para =+ 5 
    # acc = pd.DataFrame(lst,columns=cols)
    # hvplot.show(acc.plot())