# %%
import os
import sys
from src.batch_gpr import batch_gpr
from src.kalman_gpr import kalman_gpr
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataset_path = "datasets/pmattia-dataset/"
    file_path = "2017-06-30_6C-40per_4C_CH45.csv"
    gpr_obj = batch_gpr(df_path=dataset_path + file_path)
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
