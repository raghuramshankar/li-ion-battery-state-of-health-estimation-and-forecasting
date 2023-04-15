# %%
import os
import sys
from src.cellData import cellData
from src.GPR import GPR

if "__ipython__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(os.getcwd()))))
    %matplotlib widget
    %load_ext autoreload
    %autoreload 2

if __name__ == "__main__":
    GPRObj = GPR(dfPath="datasets/pmattia-dataset/")
    

    print("Done")
