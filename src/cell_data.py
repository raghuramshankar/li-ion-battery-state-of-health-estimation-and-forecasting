import pandas as pd
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class cell_data:
    df_path: str

    def __post_init__(self):
        self.df = pd.read_csv(self.df_path)
        self.df = self.df.iloc[:-2]

        # add ageing time
        self.df["Ageing_Time"] = self.df["Data_Point"]

    def get_OCV_SOC(self):
        """get the OCV SOC curve from the first cycle data"""
        # get first cycle for OCV
        first_cycle_df = self.df[self.df["Cycle_Index"] == 0]

        # get chg OCV
        self.df_chgOCV_SOC = first_cycle_df[first_cycle_df["Step_Index"] == 5][
            ["Voltage", "Charge_Capacity"]
        ]

        # get dchg OCV
        self.df_dchgOCV_SOC = first_cycle_df[first_cycle_df["Step_Index"] == 6][
            ["Voltage", "Discharge_Capacity"]
        ]

        # get discharge capacity
        self.capacity = (
            self.df_dchgOCV_SOC["Discharge_Capacity"].iloc[-1]
            - self.df_dchgOCV_SOC["Discharge_Capacity"].iloc[0]
        )

        # get OCV as average of dchg and chg OCV
        if len(self.df_chgOCV_SOC) > len(self.df_dchgOCV_SOC):
            OCV = (
                (self.df_chgOCV_SOC["Voltage"].to_numpy())[0 : len(self.df_dchgOCV_SOC)]
                + np.flip(self.df_dchgOCV_SOC["Voltage"].to_numpy())
            ) / 2
        else:
            OCV = (
                (self.df_chgOCV_SOC["Voltage"].to_numpy())
                + np.flip(self.df_dchgOCV_SOC["Voltage"].to_numpy())[
                    0 : len(self.df_chgOCV_SOC)
                ]
            ) / 2

        # get SOC points from OCV
        SOC = np.linspace(0, 1, len(OCV))

        # store OCV SOC as a df
        self.df_OCV_SOC = pd.DataFrame({"OCV": OCV, "SOC": SOC})

    def plot_df_resistance(self):
        """plot the Internal_Resistance of dataset"""
        self.df.plot.line(x="Ageing_Time", y="Internal_Resistance")

    def plot_ocv_soc(self):
        """plot the OCV SOC curve extracted from dataset"""
        _, ax = plt.subplots()
        self.df_OCV_SOC.plot.line(x="SOC", y="OCV", ax=ax)
        plt.plot(
            self.df_OCV_SOC["SOC"],
            self.df_chgOCV_SOC["Voltage"][0 : len(self.df_OCV_SOC)],
            label="Charge",
        )
        plt.plot(
            self.df_OCV_SOC["SOC"],
            np.flip(self.df_dchgOCV_SOC["Voltage"]),
            label="Disharge",
        )
        plt.legend()
        plt.ylabel("Voltage")

    def ocv_lookup(self, soc_lookup):
        """lookup OCV based on SOC input"""
        return self.df_OCV_SOC.iloc[
            np.argmin(np.abs(self.df_OCV_SOC["SOC"] - soc_lookup))
        ]["OCV"]

    def model(self, row):
        """return total resistance"""
        total_resistance = (row["Voltage"] - row["OCV"]) / row["Current"]
        if total_resistance > 5e-1 or total_resistance < 1e-5:
            return np.nan
        return total_resistance

    def fit_r0(self):
        """get the training df with total resistance"""
        self.fit_df = self.df.loc[
            (
                (self.df["Cycle_Index"] > 1)
                & (self.df["Cycle_Index"] < 1000)
                & (self.df["Current"] < -1e-3)
            )
        ]
        self.fit_df["SOC"] = 1 - self.fit_df["Discharge_Capacity"] / self.capacity
        self.fit_df["OCV"] = self.fit_df["SOC"].apply(lambda x: self.ocv_lookup(x))
        self.fit_df["Total_Resistance_Estimate"] = self.fit_df.apply(self.model, axis=1)

        # average the total resistance across SOC during each discharge
        self.train_df = self.fit_df.groupby("Cycle_Index").mean()
