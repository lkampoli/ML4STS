# https://scikit-learn.org/stable/auto_examples/feature_selection/plot_select_from_model_diabetes.html#sphx-glr-auto-examples-feature-selection-plot-select-from-model-diabetes-py

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from time import time
from sklearn.ensemble import ExtraTreesRegressor
import pandas as pd

with open('../../db/datasets/shear_viscosity.txt') as f:
    lines = (line for line in f if not line.startswith('#'))
    dataset = np.loadtxt(lines, skiprows=0)

X = dataset[:,0:51] # P, T, x_ci[mol+at]
y = dataset[:,51:]  # shear viscosity

from sklearn.feature_selection import SequentialFeatureSelector

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
