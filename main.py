# %%
import os
import sys
from src.batch_gpr import batch_gpr
from src.kalman_gpr import kalman_gpr
from src.cell_data import cell_data
import matplotlib.pyplot as plt
import ray
import time
import gpytorch
import torch

ray.init(ignore_reinit_error=True)

# if '__ipython__':
#     %load_ext autoreload
#     %autoreload 2

if __name__ == "__main__":
    start_time = time.time()
    dataset_path = "datasets/pmattia-dataset/"
    file_paths = [file for file in os.listdir(dataset_path)][0:1]
    cell_data_objs = [
        cell_data(df_path=dataset_path + file_path) for file_path in file_paths
    ]
    futures = [
        cell_data_obj.fit_r0_loop.remote(cell_data_obj)
        for cell_data_obj in cell_data_objs
    ]
    results = ray.get(futures)

    _, ax = plt.subplots()
    for train_df in results:
        train_df.plot.line(
            # x="Cycle_Index",
            y=["Total_Resistance_Estimate"],
            style=".",
            figsize=(8, 5),
            ax=ax,
        )

    train_x = torch.tensor(results[0].index.values)
    train_y = torch.tensor(results[0]["Total_Resistance_Estimate"].values)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = batch_gpr(
        train_x=train_x,
        train_y=train_y,
        likelihood=likelihood,
    )
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.1
    )  # Includes GaussianLikelihood parameters

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    training_iter = 200
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
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
    model.eval()
    likelihood.eval()

    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    # with torch.no_grad(), gpytorch.settings.fast_pred_var():
    with torch.no_grad():
        test_x = train_x * 1.1
        observed_pred = likelihood(model(test_x))

        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(8, 5))

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.plot(train_x.numpy(), train_y.numpy(), "k*")
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), "b")
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        # ax.set_ylim([min(train_y) - 1e-2, max(train_y) + 1e-2])
        ax.legend(["Observed Data", "Mean", "Confidence"])

    plt.show()
    print("Total Time = %f" % (time.time() - start_time))
