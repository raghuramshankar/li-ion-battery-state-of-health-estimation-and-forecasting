import gpytorch
from src.cell_data import cell_data
import numpy as np
from dataclasses import dataclass


@dataclass
class batch_gpr(gpytorch.models.ExactGP, cell_data):
    pass
