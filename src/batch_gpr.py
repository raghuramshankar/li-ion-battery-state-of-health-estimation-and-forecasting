import gpytorch
from src.cell_data import cell_data
import numpy as np
from dataclasses import dataclass


@dataclass
class batch_gpr(gpytorch.models.ExactGP, cell_data):
    def ocv_lookup(self, soc_lookup):
        return self.df_OCV_SOC.iloc[
            np.argmin(np.abs(self.df_OCV_SOC["SOC"] - soc_lookup))
        ]["OCV"]

    def model(self, soc, I, r0):
        return self.ocv_lookup(soc) + I * r0

    def fit_r0(self):
        self.fit_df = self.df.loc[
            (
                (self.df["Cycle_Index"] > 1)
                & (self.df["Cycle_Index"] < 50)
                & (self.df["Current"] < -1e-3)
            )
        ]
        self.fit_df["SOC"] = 1 - self.fit_df["Discharge_Capacity"] / self.capacity
        self.fit_df["OCV"] = self.fit_df["SOC"].apply(lambda x: self.ocv_lookup(x))
