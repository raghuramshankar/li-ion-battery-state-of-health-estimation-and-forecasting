import gpytorch
from dataclasses import dataclass
from src.cell_data import cell_data


@dataclass
class batch_gpr(gpytorch.models.ExactGP, cell_data):
    pass
