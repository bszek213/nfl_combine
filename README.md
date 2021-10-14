# NFL Success Prediction with NFL Combine Metrics

## Installation

```bash
$ conda env create -f nfl_combine.yml
$ conda activate nfl_combine
$ python nfl_combine_regressor.py    
```

## Usage
Change the import files line to your directory. I am going to make this dynamic 
```python
def read_in(self,path):
        self.pd_2013 = pd.read_excel('C:/Users/Bharath/Desktop/nfl_combine/NFL 2013_edit.xlsx')
        self.pd_2014 = pd.read_excel('C:/Users/Bharath/Desktop/nfl_combine/NFL 2014_edit.xlsx')
        self.snaps_2013 = pd.read_excel("C:/Users/Bharath/Desktop/nfl_combine/NFL 2013_edit.xlsx",
                                       sheet_name="Snaps")
        self.snaps_2014 = pd.read_excel("C:/Users/Bharath/Desktop/nfl_combine/NFL 2014_edit.xlsx",
                                       sheet_name="Snaps")
```

## Contributing


## License
