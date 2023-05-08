# %%
import os
import sys
from src.batch_gpr import batch_gpr
from src.kalman_gpr import kalman_gpr
import matplotlib.pyplot as plt
import ray
import time

ray.init(ignore_reinit_error=True)

if __name__ == "__main__":
    start_time = time.time()
    dataset_path = "datasets/pmattia-dataset/"
    file_paths = [file for file in os.listdir(dataset_path)]
    gpr_objs = [batch_gpr(df_path=dataset_path + file_path) for file_path in file_paths]
    futures = [gpr_obj.fit_r0_loop.remote(gpr_obj) for gpr_obj in gpr_objs]
    results = ray.get(futures)

    _, ax = plt.subplots()
    for train_df in results:
        train_df.plot.line(
            x="Ageing_Time",
            y=["Total_Resistance_Estimate"],
            style=".",
            figsize=(8, 5),
            ax=ax,
        )
    plt.show()

    print("Total Time = %f" % (time.time() - start_time))
