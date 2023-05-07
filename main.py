# %%
import os
import sys
from src.batch_gpr import batch_gpr
from src.kalman_gpr import kalman_gpr
import matplotlib.pyplot as plt

if __name__ == "__main__":
    gpr_obj = batch_gpr(
        df_path="datasets/pmattia-dataset/2017-06-30_3_6C-80per_3_6C_CH2.csv"
    )
    gpr_obj.get_OCV_SOC()
    gpr_obj.fit_r0()

    gpr_obj.train_df.plot.line(
        x="Ageing_Time",
        y=["Total_Resistance_Estimate"],
        style=".",
        figsize=(8, 5),
    )

    plt.show()

    print("Done")
