# %%
import os
import sys
from src.gpr import gpr

if "__ipython__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(os.getcwd()))))
    %load_ext autoreload
    %autoreload 2

if __name__ == "__main__":
    gpr_obj = gpr(df_path="datasets/pmattia-dataset/2017-06-30_4_65C-44per_5C_CH22.csv")
    gpr_obj.plot_df_resistance()
    gpr_obj.plot_ocv_soc()

    print("Done")