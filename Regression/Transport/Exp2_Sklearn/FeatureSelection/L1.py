# https://scikit-learn.org/stable/modules/feature_selection.html#rfe
# https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#sphx-glr-auto-examples-ensemble-plot-forest-importances-py

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
import time
from sklearn.ensemble import ExtraTreesRegressor
import pandas as pd

with open('../../db/datasets/shear_viscosity.txt') as f:
    lines = (line for line in f if not line.startswith('#'))
    dataset = np.loadtxt(lines, skiprows=0)

X = dataset[:,0:51] # P, T, x_ci[mol+at]
y = dataset[:,51:]  # shear viscosity

feature_names = [f"feature {i}" for i in range(X.shape[1])]

est = ExtraTreesRegressor(n_estimators=50)
est = est.fit(X, y.ravel())

print(est.feature_importances_)

model = SelectFromModel(est, prefit=True)
X_new = model.transform(X)
print(X_new.shape)


start_time = time.time()
importances = est.feature_importances_
# Feature importance based on mean decrease in impurity
std = np.std([tree.feature_importances_ for tree in est.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()


# Feature importance based on feature permutation
from sklearn.inspection import permutation_importance

start_time = time.time()
result = permutation_importance(
    est, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

forest_importances = pd.Series(result.importances_mean, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()
