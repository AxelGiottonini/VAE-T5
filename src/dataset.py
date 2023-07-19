import os
import typing
import pandas as pd
from torch.utils.data import Dataset as __Dataset__

class Dataset(__Dataset__):
    def __init__(self, path, min_length=None, max_length=None):
        if not os.path.isfile(path):
            raise FileNotFoundError

        df = pd.read_csv(path)

        if not min_length is None:
            df = df[df.iloc[:,-1].apply(lambda seq: min_length < len(seq))]
        if not max_length is None:
            df = df[df.iloc[:,-1].apply(lambda seq: max_length > len(seq))]

        self.df = df.reset_index(drop=True)

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        return self.df.iloc[index, -1]