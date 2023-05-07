# Li-ion Battery State of Health Estimation and Forecasting using Gaussian Process Regression

## Open Source Datasets Used:
### [Data-driven prediction of battery cycle life before capacity degradation](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204)
- Severson, K.A., Attia, P.M., Jin, N. et al. Data-driven prediction of battery cycle life before capacity degradation. Nat Energy 4, 383–391 (2019). https://doi.org/10.1038/s41560-019-0356-8

## References:
- Aitio, A., Jöst, D., Sauer, D.U. and Howey, D.A., 2023. Learning battery model parameter dynamics from data with recursive Gaussian process regression. arXiv preprint arXiv:2304.13666
- S. Sarkka, A. Solin and J. Hartikainen, "Spatiotemporal Learning via Infinite-Dimensional Bayesian Filtering and Smoothing: A Look at Gaussian Process Regression Through Kalman Filtering," in IEEE Signal Processing Magazine, vol. 30, no. 4, pp. 51-61, July 2013, doi: 10.1109/MSP.2013.2246292.
- Antti Aitio, David A. Howey, Predicting battery end of life from solar off-grid system field data using machine learning, Joule, Volume 5, Issue 12, 2021, Pages 3204-3220, ISSN 2542-4351, https://doi.org/10.1016/j.joule.2021.11.006

## Gaussian Process Regression
- Batch GPR using [GpyTorch](https://github.com/cornellius-gp/gpytorch)
- Recursive GPR using Kalman Filter and RTS Smoother (See references)