# https://scikit-learn.org/stable/auto_examples/feature_selection/plot_select_from_model_diabetes.html#sphx-glr-auto-examples-feature-selection-plot-select-from-model-diabetes-py

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from time import time

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score

from sklearn.preprocessing import StandardScaler


with open('../../data/dataset_N2N_rhs.dat.OK') as f:
    lines = (line for line in f if not line.startswith('#'))
    data = np.loadtxt(lines, skiprows=0)

X = data[:,0:56]   # x_s, time_s, Temp, ni_n, na_n, rho, v, p, E, H
y = data[:,56:57]  # rhs[0:50]

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=69)

sc_x = StandardScaler()
sc_y = StandardScaler()

# fit scaler
sc_x.fit(x_train)

# transform training dataset
x_train = sc_x.transform(x_train)
X = x_train 

# transform test dataset
x_test = sc_x.transform(x_test)

# fit scaler on training dataset
sc_y.fit(y_train)

# transform training dataset
y_train = sc_y.transform(y_train)
y = y_train

# transform test dataset
y_test = sc_y.transform(y_test)

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
