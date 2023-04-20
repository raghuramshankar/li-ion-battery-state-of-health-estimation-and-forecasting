import pandas as pd
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class cellData:
    df_path: str

    def __post_init__(self):
        self.df = pd.read_csv(self.df_path)
        self.df = self.df.iloc[:-2]
        self.df["Ageing_Time"] = self.df["Data_Point"]

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
        OCV = (
            (self.df_chgOCV_SOC["Voltage"].to_numpy())[0 : len(self.df_dchgOCV_SOC)]
            + np.flip(self.df_dchgOCV_SOC["Voltage"].to_numpy())
        ) / 2

        # get SOC points from OCV
        SOC = np.linspace(0, 1, len(OCV))

        # store OCV SOC as a df
        self.df_OCV_SOC = pd.DataFrame({"OCV": OCV, "SOC": SOC})

    def plot_df_resistance(self):
        self.df.plot.line(x="Ageing_Time", y="Internal_Resistance")

    def plot_ocv_soc(self):
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
        return self.df_OCV_SOC.iloc[
            np.argmin(np.abs(self.df_OCV_SOC["SOC"] - soc_lookup))
        ]["OCV"]

    def model(self, soc, I, r0):
        return self.ocv_lookup(soc) + I * r0
