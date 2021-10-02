#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sept  28 11:16:50 2021
TO DO: remove data that does not have all combine features,
convert all snap data to binary
run classifiers:
1.Logistic Regression
2. Naive Bayes
3. K-Nearest Neighbors
4. Decision Tree
5. Support Vector Machines

@author: bszekely
"""

from nfl_combine import nflCombine
import pandas as pd

class nflCombineClassify(nflCombine):
    
    def __init__(self,path):
        super().__init__()
        super().read_in(path)
        super().cumulative_snaps()
        
    def snaps_to_binary(self):
        print(self.snaps_cum_2013)
        
if __name__ == '__main__':
    classify = nflCombineClassify('')
    classify.snaps_to_binary()