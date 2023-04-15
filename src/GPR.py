import gpytorch
import torch
import pandas as pd
from dataclasses import dataclass
import src.cellData as cellData


@dataclass
class GPR:
    dfPath: str

    def __post_init__(self):
        self.df = pd.read_csv(self.dfPath + "2017-05-12_6C-50per_3_6C_CH36.csv")
        self.df = self.df.iloc[:-2]

    def plotDFResistance(self):
        self.df.plot.line(x="Data_Point", y="Internal_Resistance")
