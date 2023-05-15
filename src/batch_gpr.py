import gpytorch
from src.cell_data import cell_data
import numpy as np
import torch


class batch_gpr(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        gpytorch.models.ExactGP.__init__(self, train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.LinearKernel()
        )

    def forward(self, x) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def train_model(
        self, train_x, train_y, likelihood=gpytorch.likelihoods.GaussianLikelihood()
    ):
        self.likelihood = likelihood
        self.model = batch_gpr(
            train_x=train_x, train_y=train_y, likelihood=self.likelihood
        )

        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.5
        )  # Includes GaussianLikelihood parameters

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        training_iter = 50
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()
            # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            #     i + 1, training_iter, loss.item(),
            #     model.covar_module.base_kernel.lengthscale.item(),
            #     model.likelihood.noise.item()
            # ))
            optimizer.step()

        # Get into evaluation (predictive posterior) mode
        self.model.eval()
        self.likelihood.eval()
