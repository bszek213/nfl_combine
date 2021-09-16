import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
class nflCombine:

    def __init__(self):
        pd_2013 = pd.DataFrame()
        snaps_2013 = pd.DataFrame()
        snaps_cum = pd.Series(dtype = float)

    def read_in(self,path):
        self.pd_2013 = pd.read_excel('/home/bszekely/Desktop/ProjectsResearch/nfl_combine/NFL 2013_edit.xlsx')
        self.snaps_2013 = pd.read_excel("/home/bszekely/Desktop/ProjectsResearch/nfl_combine/NFL 2013_edit.xlsx",
                                       sheet_name="Snaps")

    def cumulative_snaps(self):
        self.snaps_cum = self.snaps_2013.sum(axis = 1,skipna = True)

    def cluster_analysis(self):
        temp_pd = self.snaps_cum[self.snaps_cum != 0]
        snaps = temp_pd.to_numpy()
        pick = self.pd_2013['Pick'].dropna().to_numpy()
        features = np.matrix([snaps, pick])
        print(features.shape())

if __name__ == '__main__':

    nfl = nflCombine()
    nfl.read_in("")
    nfl.cumulative_snaps()
    nfl.cluster_analysis()
