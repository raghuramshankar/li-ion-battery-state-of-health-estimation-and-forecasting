import gpytorch
from dataclasses import dataclass
from src.cellData import cellData


@dataclass
class gpr(gpytorch.models.ExactGP, cellData):
    pass
