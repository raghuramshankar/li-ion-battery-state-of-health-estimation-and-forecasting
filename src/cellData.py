import scipy.io
import numpy as np
import pandas as pd
from dataclasses import dataclass


class cellData:
    def __init__(self):
        """loads the battery degrdataion dataset"""
        datasetPath = "datasets/pmattia-dataset/"
        self.dataset = pd.read_csv(
            self.datasetPath + "2017-05-12_6C-50per_3_6C_CH36.csv"
        )
        self.dataset = self.dataset.iloc[:-2]
