# https://scikit-learn.org/stable/modules/feature_selection.html#rfe
# https://scikit-learn.org/stable/auto_examples/feature_selection/plot_f_test_vs_mi.html#sphx-glr-auto-examples-feature-selection-plot-f-test-vs-mi-py
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html#sklearn.feature_selection.f_regression
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html#sklearn.feature_selection.mutual_info_regression
# https://scikit-learn.org/stable/auto_examples/feature_selection/plot_feature_selection.html#sphx-glr-auto-examples-feature-selection-plot-feature-selection-py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import chi2, f_classif, mutual_info_classif # classification
from sklearn.feature_selection import f_regression, mutual_info_regression # regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline

with open('../../data/dataset_N2N_rhs.dat.OK') as f:
    lines = (line for line in f if not line.startswith('#'))
    data = np.loadtxt(lines, skiprows=0)

X = data[:,0:56]   # x_s, time_s, Temp, ni_n, na_n, rho, v, p, E, H
y = data[:,56:57]  # rhs[0:50]

print(data.shape)
print("x=",X.shape)
print("y=",y.shape)

f_new, _ = f_regression(X, y.ravel())
f_new /= np.max(f_new)
print(f_new)

mi_new = mutual_info_regression(X, y.ravel())
mi_new /= np.max(mi_new)
print(mi_new)

plt.figure(figsize=(15, 5))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.scatter(X[:, i], y, edgecolor="black", s=20)
    plt.xlabel("$x_{}$".format(i + 1), fontsize=14)
    if i == 0:
        plt.ylabel("$y$", fontsize=14)
    plt.title("F-test={:.2f}, MI={:.2f}".format(f_new[i], mi_new[i]), fontsize=16)
plt.show()

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=69)

# #####################################################################
# Univariate feature selection with F-test for feature scoring
# We use the default selection function to select the four
# most significant features
X_indices = np.arange(X.shape[-1])
selector = SelectKBest(f_regression, k=4)
#selector.fit(X, y.ravel())
selector.fit(x_train, y_train.ravel())
scores = -np.log10(selector.pvalues_)
scores /= scores.max()
plt.bar(X_indices - 0.45, scores, width=0.2, label=r"Univariate score ($-Log(p_{value})$)")

# #####################################################################
# Compare to the weights of an SVM
rgr = make_pipeline(MinMaxScaler(), LinearSVR())
rgr.fit(x_train, y_train.ravel())
print(
    "Regression score without selecting features: {:.3f}".format(
        rgr.score(x_test, y_test.ravel())
    )
)

svm_weights = np.abs(rgr[-1].coef_).sum(axis=0)
svm_weights /= svm_weights.sum()

plt.bar(X_indices - 0.25, svm_weights, width=0.2, label="SVM weight")

rgr_selected = make_pipeline(SelectKBest(f_regression, k=4), MinMaxScaler(), LinearSVR())
rgr_selected.fit(x_train, y_train.ravel())
print(
    "Regression score after univariate feature selection: {:.3f}".format(
        rgr_selected.score(x_test, y_test.ravel())
    )
)

svm_weights_selected = np.abs(rgr_selected[-1].coef_).sum(axis=0)
svm_weights_selected /= svm_weights_selected.sum()

plt.bar(
    X_indices[selector.get_support()] - 0.05,
    svm_weights_selected,
    width=0.2,
    label="SVM weights after selection",
)

plt.title("Comparing feature selection")
plt.xlabel("Feature number")
plt.yticks(())
plt.axis("tight")
plt.legend(loc="upper right")
plt.show()
