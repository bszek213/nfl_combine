import pandas as pd

class nflCombine:

    def __init__(self):
        pd_2013 = pd.DataFrame()
        snaps_2013 = pd.DataFrame()

    def read_in(self,path):
        self.pd_2013 = pd.read_excel('/home/bszekely/Desktop/ProjectsResearch/nfl_combine/NFL 2013_edit.xlsx')
        self.snaps_2013 = pd.read_excel("/home/bszekely/Desktop/ProjectsResearch/nfl_combine/NFL 2013_edit.xlsx",
                                       sheet_name="Snaps")
        print(self.snaps_2013)

    def cumulative_snaps(self):

    def cluster_analysis(self):

if __name__ == '__main__':

    nfl = nflCombine()
    nfl.read_in("")
