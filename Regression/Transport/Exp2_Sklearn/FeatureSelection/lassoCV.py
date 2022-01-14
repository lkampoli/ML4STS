# https://scikit-learn.org/stable/auto_examples/feature_selection/plot_select_from_model_diabetes.html#sphx-glr-auto-examples-feature-selection-plot-select-from-model-diabetes-py

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from time import time

with open('../../db/datasets/shear_viscosity.txt') as f:
    lines = (line for line in f if not line.startswith('#'))
    dataset = np.loadtxt(lines, skiprows=0)

X = dataset[:,0:51] # P, T, x_ci[mol+at]
y = dataset[:,51:]  # shear viscosity

feature_names = [f"feature {i}" for i in range(X.shape[1])]

lasso = LassoCV().fit(X, y.ravel())
importance = np.abs(lasso.coef_)
plt.bar(height=importance, x=feature_names)
plt.title("Feature importances via coefficients")
plt.show()
print(importance)

threshold = np.sort(importance)[-3] + .01

tic = time()
sfm = SelectFromModel(estimator=lasso, threshold=threshold).fit(X, y.ravel())
toc = time()
#print(f"Features selected by SelectFromModel: {feature_names[sfm.get_support()]}")
print(f"Done in {toc - tic:.3f}s")

print(sfm.estimator_.coef_)
print(sfm.threshold_)
print(sfm.get_support)
print(sfm.transform(X))
