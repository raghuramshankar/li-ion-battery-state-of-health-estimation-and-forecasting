import gpytorch
import pandas as pd
from dataclasses import dataclass
import gpytorch
import numpy as np


@dataclass
class gpr(gpytorch.models.ExactGP):
    df_path: str

    def __post_init__(self):
        # self.df = pd.read_csv(self.df_path, skiprows=19)
        self.df = pd.read_csv(self.df_path)
        self.df = self.df.iloc[:-2]
        # self.df["Ageing_Time"] = pd.to_datetime(self.df["DateTime"])
        self.df["Ageing_Time"] = self.df["Data_Point"]
        # super(GPR, self).__init__(
        #     train_inputs=self.df["Ageing_Time"], train_targets=self.df[""]
        # )
        self.soc = np.linspace(0, 1, 100)

    def ocv_lookup(self, soc_lookup):
        return np.argmin(self.soc - soc_lookup)

    # def cell_model(self, soc):
    #     ocv_lookup =

    def plot_df_resistance(self):
        self.df.plot.line(x="Ageing_Time", y="Internal_Resistance")
