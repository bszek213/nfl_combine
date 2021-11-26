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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import time
from sklearn.model_selection import cross_validate

class nflCombineRegressor:

    def __init__(self):
        snaps_cum_2013 = pd.Series(dtype = float)
        snaps_cum_2014 = pd.Series(dtype = float)
        snaps_cum_2015 = pd.Series(dtype = float)
        snaps_cum_2016 = pd.Series(dtype = float)
        snaps_cum_2017 = pd.Series(dtype = float)

    def read_in(self,path): #change this to be relative via argparse()
        self.pd_2013 = pd.read_excel('/home/bszekely/Desktop/ProjectsResearch/nfl_combine/NFL 2013_edit.xlsx')
        self.pd_2014 = pd.read_excel('/home/bszekely/Desktop/ProjectsResearch/nfl_combine/NFL 2014_edit.xlsx')
        self.pd_2015 = pd.read_excel('/home/bszekely/Desktop/ProjectsResearch/nfl_combine/NFL 2015_edit.xlsx')
        self.pd_2016 = pd.read_excel('/home/bszekely/Desktop/ProjectsResearch/nfl_combine/NFL 2016_edit.xlsx')
        self.pd_2017 = pd.read_excel('/home/bszekely/Desktop/ProjectsResearch/nfl_combine/NFL 2017_edit.xlsx')
        
        self.snaps_2013 = pd.read_excel("/home/bszekely/Desktop/ProjectsResearch/nfl_combine/NFL 2013_edit.xlsx",
                                       sheet_name="Snaps")
        self.snaps_2014 = pd.read_excel("/home/bszekely/Desktop/ProjectsResearch/nfl_combine/NFL 2014_edit.xlsx",
                                       sheet_name="Snaps")
        self.snaps_2015 = pd.read_excel("/home/bszekely/Desktop/ProjectsResearch/nfl_combine/NFL 2015_edit.xlsx",
                                       sheet_name="Snaps")
        self.snaps_2016 = pd.read_excel("/home/bszekely/Desktop/ProjectsResearch/nfl_combine/NFL 2016_edit.xlsx",
                                       sheet_name="Snaps")
        self.snaps_2017 = pd.read_excel("/home/bszekely/Desktop/ProjectsResearch/nfl_combine/NFL 2017_edit.xlsx",
                                       sheet_name="Snaps")

    def cumulative_snaps(self):
        """
        Sums all data across seasons,defense,offense, and special teams.
        This will change the the NANs to zeros
        """
        self.snaps_cum_2013 = self.snaps_2013.sum(axis = 1)
        self.snaps_cum_2014 = self.snaps_2014.sum(axis = 1)
        self.snaps_cum_2015 = self.snaps_2015.sum(axis = 1)
        self.snaps_cum_2016 = self.snaps_2016.sum(axis = 1)
        self.snaps_cum_2017 = self.snaps_2017.sum(axis = 1)
        
        print(len(self.snaps_cum_2013), "Samples started with - 2013")
        print(len(self.snaps_cum_2014), "Samples started with - 2014")
        print(len(self.snaps_cum_2015), "Samples started with - 2015")
        print(len(self.snaps_cum_2016), "Samples started with - 2016")
        print(len(self.snaps_cum_2017), "Samples started with - 2017")
    
    def split_test(self):
        index_nonzero_13 = self.snaps_cum_2013[self.snaps_cum_2013 !=0 ].index.tolist()
        index_nonzero_14 = self.snaps_cum_2014[self.snaps_cum_2014 !=0 ].index.tolist()
        index_nonzero_15 = self.snaps_cum_2015[self.snaps_cum_2015 !=0 ].index.tolist()
        index_nonzero_16 = self.snaps_cum_2016[self.snaps_cum_2016 !=0 ].index.tolist()
        index_nonzero_17 = self.snaps_cum_2017[self.snaps_cum_2017 !=0 ].index.tolist()

        snaps_parse_13 = self.snaps_cum_2013.iloc[index_nonzero_13]
        snaps_parse_14 = self.snaps_cum_2014.iloc[index_nonzero_14]
        snaps_parse_15 = self.snaps_cum_2015.iloc[index_nonzero_15]
        snaps_parse_16 = self.snaps_cum_2016.iloc[index_nonzero_16]
        snaps_parse_17 = self.snaps_cum_2017.iloc[index_nonzero_17]
        
        pd_2013_nozero = self.pd_2013.iloc[index_nonzero_13,:]
        pd_2014_nozero = self.pd_2014.iloc[index_nonzero_14,:]
        pd_2015_nozero = self.pd_2015.iloc[index_nonzero_15,:]
        pd_2016_nozero = self.pd_2016.iloc[index_nonzero_16,:]
        pd_2017_nozero = self.pd_2017.iloc[index_nonzero_17,:]
        
        cols =['40yd','Vertical','BP','Broad Jump',
                          'Shuttle','3Cone']
        
        x_data_13_ = pd_2013_nozero[['40yd','Vertical','BP','Broad Jump',
                          'Shuttle','3Cone']]
        x_data_14_ = pd_2014_nozero[['40yd','Vertical','BP','Broad Jump',
                          'Shuttle','3Cone']]
        x_data_15_ = pd_2015_nozero[['40yd','Vertical','BP','Broad Jump',
                          'Shuttle','3Cone']]
        x_data_16_ = pd_2016_nozero[['40yd','Vertical','BP','Broad Jump',
                          'Shuttle','3Cone']]
        x_data_17_ = pd_2017_nozero[['40yd','Vertical','BP','Broad Jump',
                          'Shuttle','3Cone']]
        index_nan_13 = x_data_13_.dropna().index.tolist()
        index_nan_14 = x_data_14_.dropna().index.tolist()
        index_nan_15 = x_data_15_.dropna().index.tolist()
        index_nan_16 = x_data_16_.dropna().index.tolist()
        index_nan_17 = x_data_17_.dropna().index.tolist()
    
        x_data_13_nonan = x_data_13_.loc[index_nan_13]
        x_data_14_nonan = x_data_14_.loc[index_nan_14]
        x_data_15_nonan = x_data_15_.loc[index_nan_15]
        x_data_16_nonan = x_data_16_.loc[index_nan_16]
        x_data_17_nonan = x_data_17_.loc[index_nan_17]
        
        y_data_13_nonan = snaps_parse_13.loc[index_nan_13]
        y_data_14_nonan = snaps_parse_14.loc[index_nan_14]
        y_data_15_nonan = snaps_parse_15.loc[index_nan_15]
        y_data_16_nonan = snaps_parse_16.loc[index_nan_16]
        y_data_17_nonan = snaps_parse_17.loc[index_nan_17]

        scaler = StandardScaler()
        x_data_13 = scaler.fit_transform(x_data_13_nonan)
        x_data_14 = scaler.fit_transform(x_data_14_nonan) 
        x_data_15 = scaler.fit_transform(x_data_15_nonan)
        x_data_16 = scaler.fit_transform(x_data_16_nonan)
        x_data_17 = scaler.fit_transform(x_data_17_nonan)
        
        df_13 = pd.DataFrame(x_data_13, columns = cols)
        df_14 = pd.DataFrame(x_data_14, columns = cols)
        df_15 = pd.DataFrame(x_data_15, columns = cols)
        df_16 = pd.DataFrame(x_data_16, columns = cols)
        df_17 = pd.DataFrame(x_data_17, columns = cols)
        
        x = pd.concat([df_13, df_14, df_15, df_16, df_17]) 
        y = pd.concat([y_data_13_nonan, y_data_14_nonan, y_data_15_nonan, y_data_16_nonan,
                        y_data_17_nonan])     

        print(len(x_data_13_nonan), "Samples started with - 2013")
        print(len(x_data_14_nonan), "Samples started with - 2014")
        print(len(x_data_15_nonan), "Samples started with - 2015")
        #print(len(x_data_16_nonan), "Samples started with - 2016")
        print(len(x_data_17_nonan), "Samples started with - 2017")

        self.x_train, self.x_rem, self.y_train, self.y_rem = train_test_split(x,y, train_size=0.8)
        self.x_valid, self.x_test, self.y_valid, self.y_test = train_test_split(self.x_rem,self.y_rem, test_size=0.5)
     
    def model_test(self):

        # self.model = GradientBoostingRegressor()
        # self.model2 = RandomForestRegressor()
        # self.model3 = LinearRegression()
        # self.model4 = DecisionTreeRegressor()
        # self.model5 = SVR(kernel = 'linear')
        
        #Cross validate all to find one that works the best
        GB = cross_validate(GradientBoostingRegressor(),
                                        self.x_train, self.y_train, cv=10,
                                        scoring=['neg_root_mean_squared_error'],return_train_score=True)
        RF = cross_validate(RandomForestRegressor(),
                                        self.x_train, self.y_train, cv=10,
                                        scoring=['neg_root_mean_squared_error'],return_train_score=True)
        LR = cross_validate(LinearRegression(),
                                        self.x_train, self.y_train, cv=10,
                                        scoring=['neg_root_mean_squared_error'],return_train_score=True)
        DT = cross_validate(DecisionTreeRegressor(),
                                        self.x_train, self.y_train, cv=10,
                                        scoring=['neg_root_mean_squared_error'],return_train_score=True)
        SV_R = cross_validate(SVR(),
                                        self.x_train, self.y_train, cv=10,
                                        scoring=['neg_root_mean_squared_error'],return_train_score=True)
        
        print('results of DT: ',np.abs(np.mean(DT['test_neg_root_mean_squared_error'])))
        print('results of GB: ',np.abs(np.mean(GB['test_neg_root_mean_squared_error'])))
        print('results of SV_R: ',np.abs(np.mean(SV_R['test_neg_root_mean_squared_error'])))
        print('results of RF: ',np.abs(np.mean(RF['test_neg_root_mean_squared_error'])))
        print('results of LR: ',np.abs(np.mean(LR['test_neg_root_mean_squared_error']))) #winner
        
        final_model = LinearRegression()
        final_model.fit(self.x_test,self.y_test)
        print('RMSE on test data',mean_squared_error(self.y_test, final_model.predict(self.x_test), squared=False))
        
        return final_model
        # self.model.fit(self.x_train,self.y_train)
        # self.model2.fit(self.x_train,self.y_train)
        # self.model3.fit(self.x_train,self.y_train)
        # self.model4.fit(self.x_train,self.y_train)
        # self.model5.fit(self.x_train,self.y_train)

        # y_pred = self.model.predict(self.x_train)
        # y_pred1 = self.model2.predict(self.x_train)
        # y_pred2 = self.model3.predict(self.x_train)
        # y_pred3 = self.model4.predict(self.x_train)
        # y_pred4 = self.model5.predict(self.x_train)

        #Perform validation
        #train Hyper params              
        # max_depths = np.arange(1, 25, 1)
        # training_error_model = []
        # training_error_model2 = []
        # training_error_model4 = []
        # for max_depth in max_depths:
        #     self.model = GradientBoostingRegressor(max_depth = max_depth)
        #     self.model2 = RandomForestRegressor(max_depth = max_depth)
        #     self.model4 = DecisionTreeRegressor(max_depth = max_depth)
            
        #     self.model.fit(self.x_train,self.y_train)
        #     self.model2.fit(self.x_train,self.y_train)
        #     self.model4.fit(self.x_train,self.y_train)
        #     training_error_model.append(mean_squared_error(self.y_train, self.model.predict(self.x_train),squared=False))
        #     training_error_model2.append(mean_squared_error(self.y_valid, self.model2.predict(self.x_valid)))
        #     training_error_model4.append(mean_squared_error(self.y_valid, self.model4.predict(self.x_valid)))
            
        # p_test2 = {'max_depth':[2,3,4,5,6,7] }

        # tuning = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.01,n_estimators=1500, min_samples_split=2, min_samples_leaf=1, subsample=1,max_features='sqrt', random_state=10), 
        #     param_grid = p_test2, scoring='accuracy',n_jobs=4, cv=5)
        # tuning.fit(self.x_train,self.y_train)
        # print(tuning.best_params_, tuning.best_score_)
        #Plot parameter selection
        # fig, axs = plt.subplots(1, 3)
        # axs[0].plot(max_depths, training_error_model, color='blue', label='validation error GradientBoostingRegressor - tree depth')
        # axs[1].plot(max_depths, training_error_model2, color='red', label='validation error RandomForestRegressor - tree depth')
        # axs[2].plot(max_depths, training_error_model4, color='black', label='validation error DecisionTreeRegressor - tree depth')
        # axs[0].set_xlabel('Tree depth')
        # axs[0].set_ylabel('Root Mean squared error')
        # axs[0].set_title('GradientBoostingRegressor', pad=15, size=15)
        # axs[1].set_xlabel('Tree depth')
        # axs[1].set_ylabel('Root Mean squared error')
        # axs[1].set_title('RandomForestRegressor', pad=15, size=15)
        # axs[2].set_xlabel('Tree depth')
        # axs[2].set_ylabel('Root Mean squared error')
        # axs[2].set_title('DecisionTreeRegressor', pad=15, size=15)
        # plt.show()
        
        # self.model3 = LinearRegression()
        # self.model5 = SVR(kernel = 'linear')
        
        # self.model3.fit(self.x_valid,self.y_valid)
        # self.model5.fit(self.x_valid,self.y_valid)
        
        # y_pred3 = self.model4.predict(self.x_test)
        # y_pred4 = self.model5.predict(self.x_test)
        
        #Calculate error
        # mse = r2_score(self.y_test, y_pred)
        # mse1 = r2_score(self.y_test, y_pred1)
        # mse2 = r2_score(self.y_test, y_pred2)
        # mse3 = r2_score(self.y_test, y_pred3)
        # mse4 = r2_score(self.y_test, y_pred4)
        # print("GradientBoostingRegressor: ",mse)
        # print("RandomForestRegressor: ",mse1)
        # print("LinearRegression: ",mse2)
        # print("DecisionTreeRegressor: ",mse3)
        # print("SVR: ",mse4)



        
    def plot_feature_importance(self, final_model):
        importance = final_model.coef_
        print(importance[1])
        # summarize feature importance final_model.coef_[0]
        for i,v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i,v))
        #Calculate feature importance 
        feature_imp = pd.Series(np.abs(importance),index=self.x_test.columns).sort_values(ascending=False)
        # feature_imp = pd.Series(self.model.feature_importances_,index=self.X_test.columns).sort_values(ascending=False)
        # feature_imp2 = pd.Series(self.model2.feature_importances_,index=self.X_test.columns).sort_values(ascending=False)
        # feature_imp3 = pd.Series(self.model3.coef_[0],index=self.X_test.columns).sort_values(ascending=False)
        # feature_imp4 = pd.Series(self.model4.feature_importances_,index=self.X_test.columns).sort_values(ascending=False)
        # feature_imp5 = pd.Series(self.model5.coef_[0],index=self.X_test.columns).sort_values(ascending=False)
        fig, axs = plt.subplots(1, 1)
        # axs = axs.flatten()
        sns.barplot(x=feature_imp,y=feature_imp.index)
        # sns.barplot(ax=axs[1],x=feature_imp2,y=feature_imp2.index)
        # sns.barplot(ax=axs[2],x=feature_imp4,y=feature_imp4.index)
        # sns.barplot(ax=axs[3],x=feature_imp5,y=feature_imp5.index)
        # sns.barplot(ax=axs[4],x=feature_imp3,y=feature_imp3.index)
        # # plt.xlabel('Feature Importance')
        axs.set_title('Linear Regression Feature Importances', fontsize=20)
        axs.tick_params(axis='both', which='major', labelsize=16)
        axs.tick_params(axis='both', which='minor', labelsize=16)
        # axs[1].set_title('RandomForestRegressor')
        # axs[2].set_title('DecisionTreeRegressor')
        # axs[3].set_title('SVC')
        # axs[4].set_title('LinearRegression')
        plt.draw()
        plt.show()
        
        
        
if __name__ == '__main__':
    start_time = time.time()
    nfl = nflCombineRegressor()
    nfl.read_in("")
    nfl.cumulative_snaps()
    nfl.split_test()
    final_model = nfl.model_test()
    nfl.plot_feature_importance(final_model)
    # cols = ['Gradient_error', 'RFR_error', 'Linear_error','DT_error',
    #          'SVM_error']
    # lst = []

    # nfl.model_test(nfl.x_valid, nfl.y_valid, True)
    #nfl.model_test(nfl.x_test, nfl.y_test, True)
    # error = pd.DataFrame(lst,columns=cols)
    #nfl.plot_feature_importance()
    print("--- %s seconds ---" % (time.time() - start_time))
    #hvplot.show(error.plot())
    


