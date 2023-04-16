# %%
import os
import sys
from src.GPR import gpr

if "__ipython__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(os.getcwd()))))
    %matplotlib inline
    %matplotlib widget
    %load_ext autoreload
    %autoreload 2

if __name__ == "__main__":
    gpr_obj = gpr(df_path="datasets/pmattia-dataset/2017-05-12_6C-50per_3_6C_CH36.csv")
    # gpr_obj = gpr(df_path="datasets/moura-fast-charging-dataset/Test180.csv")
    gpr_obj.plot_df_resistance()

    print("Done")
