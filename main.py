# %%
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import os
import sys
from src.cellData import cellData
import numpy as np
import polars as pl

if "__ipython__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(os.getcwd()))))
    %matplotlib widget
    %load_ext autoreload
    %autoreload 2

if __name__ == "__main__":
    cellDataObj = cellData()
    cellDataObj.loadPMAttiaDataset()
    cellDataObj.plotPMAttiaDataset()

    print("Done")
