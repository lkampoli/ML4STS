# https://scikit-learn.org/stable/modules/feature_selection.html#rfe
# https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_digits.html#sphx-glr-auto-examples-feature-selection-plot-rfe-digits-py
# https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#sphx-glr-auto-examples-feature-selection-plot-rfe-with-cross-validation-py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif # classification
from sklearn.feature_selection import f_regression, mutual_info_regression # regression
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVR, SVR
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

with open('../../data/dataset_N2N_rhs.dat.OK') as f:
    lines = (line for line in f if not line.startswith('#'))
    data = np.loadtxt(lines, skiprows=0)

X = data[:,0:56]   # x_s, time_s, Temp, ni_n, na_n, rho, v, p, E, H
y = data[:,56:57]  # rhs[0:50]

print(data.shape)
print("x=",X.shape)
print("y=",y.shape)

#x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=69)
#
sc_x = StandardScaler()
#sc_y = StandardScaler()
#X = sc_x.fit_transform(X)
#
## fit scaler
#sc_x.fit(x_train)
#
## transform training dataset
#x_train = sc_x.transform(x_train)
#
## transform test dataset
#x_test = sc_x.transform(x_test)
#
## fit scaler on training dataset
#sc_y.fit(y_train)
#
## transform training dataset
#y_train = sc_y.transform(y_train)
#
## transform test dataset
#y_test = sc_y.transform(y_test)

variances = pd.Series(X.var(axis=0))
fig, ax = plt.subplots(figsize=(7,6))
variances.sort_values().plot(kind='barh', ax=ax)
ax.vlines(0.1, ymin=-1, ymax=25, colors='red')
ax.set_title('Variances of dummy features');
plt.show()

# Create the RFE object and rank each pixel
svc = SVR(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
rfe.fit(X, y.ravel())
ranking = rfe.ranking_.reshape(X[0].shape)
print(ranking)
print(rfe.support_)
print(rfe.ranking_)

# Plot pixel ranking
#plt.matshow(ranking, cmap=plt.cm.Blues)
#plt.colorbar()
#plt.title("Ranking of pixels with RFE")
#plt.show()

# Create the RFE object and compute a cross-validated score.
svc = SVR(kernel="linear")

min_features_to_select = 1  # Minimum number of features to consider
rfecv = RFECV(
    estimator=svc,
    step=1,
    cv=KFold(n_splits=5),
    scoring="r2",
    min_features_to_select=min_features_to_select,
)
rfecv.fit(X, y.ravel())

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (r2)")
plt.plot(
    range(min_features_to_select, len(rfecv.grid_scores_) + min_features_to_select),
    rfecv.grid_scores_,
)
plt.show()
