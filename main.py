# %%
import os
import sys
from src.batch_gpr import batch_gpr
from src.kalman_gpr import kalman_gpr
import matplotlib.pyplot as plt

# if "__ipython__":
#     sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(os.getcwd()))))
#     %load_ext autoreload
#     %autoreload 2
#     %matplotlib widget

if __name__ == "__main__":
    gpr_obj = batch_gpr(
        df_path="datasets/pmattia-dataset/2017-06-30_4_65C-44per_5C_CH22.csv"
    )
    gpr_obj.fit_r0()

    # gpr_obj.plot_df_resistance()
    # gpr_obj.plot_ocv_soc()

    # gpr_obj.df.loc[
    #     (
    #         (gpr_obj.df["Cycle_Index"] > 1)
    #         & (gpr_obj.df["Cycle_Index"] < 5000)
    #         & (gpr_obj.df["Current"] < -1)
    #     )
    # ].plot.scatter(x="Data_Point", y="Internal_Resistance", figsize=(8, 5))

    gpr_obj.fit_df.plot.line(
        x="Ageing_Time", y=["Voltage", "OCV"], style=".", figsize=(8, 5)
    )
    plt.show()

    print("Done")
