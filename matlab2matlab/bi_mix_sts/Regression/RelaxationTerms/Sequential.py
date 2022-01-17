# https://scikit-learn.org/stable/auto_examples/feature_selection/plot_select_from_model_diabetes.html#sphx-glr-auto-examples-feature-selection-plot-select-from-model-diabetes-py

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector
from time import time
from sklearn.ensemble import ExtraTreesRegressor
import pandas as pd

with open('../../data/dataset_N2N_rhs.dat.OK') as f:
    lines = (line for line in f if not line.startswith('#'))
    data = np.loadtxt(lines, skiprows=0)

X = data[:,0:56]   # x_s, time_s, Temp, ni_n, na_n, rho, v, p, E, H
y = data[:,56:57]  # rhs[0:50]

print(data.shape)
print("x=",X.shape)
print("y=",y.shape)

feature_names = [f"feature {i}" for i in range(X.shape[1])]

est = ExtraTreesRegressor(n_estimators=50)
est = est.fit(X, y.ravel())

tic_fwd = time()
sfs_forward = SequentialFeatureSelector(est, n_features_to_select=2, direction="forward").fit(X, y.ravel())
toc_fwd = time()

tic_bwd = time()
sfs_backward = SequentialFeatureSelector(est, n_features_to_select=2, direction="backward").fit(X, y.ravel())
toc_bwd = time()

print(
    "Features selected by forward sequential selection: "
    f"{feature_names[sfs_forward.get_support()]}"
)
print(f"Done in {toc_fwd - tic_fwd:.3f}s")
print(
    "Features selected by backward sequential selection: "
    f"{feature_names[sfs_backward.get_support()]}"
)
print(f"Done in {toc_bwd - tic_bwd:.3f}s")
