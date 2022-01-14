#!/usr/bin/env python

# https://stackoverflow.com/questions/45074698/how-to-pass-elegantly-sklearns-gridseachcvs-best-parameters-to-another-model
# https://medium.com/@alexstrebeck/training-and-testing-machine-learning-models-e1f27dc9b3cb
# https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv

import time
import sys
sys.path.insert(0, './')

#from plotting import newfig, savefig
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import pandas as pd
import seaborn as sns

import operator
import itertools

from sklearn import metrics
from sklearn.metrics import *
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score

from sklearn.inspection import permutation_importance

from sklearn import ensemble
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.decomposition import PCA, KernelPCA, FastICA

from joblib import dump, load
import pickle

n_jobs = 2

with open('../data/boltzmann/shear_viscosity.txt') as f:
    lines = (line for line in f if not line.startswith('#'))
    dataB = np.loadtxt(lines, skiprows=0)

print(dataB.shape)
xB = dataB[:,0:51] # P, T, x_ci[mol+at]
yB = dataB[:,51:]  # shear viscosity
print("x=",xB.shape)
print("y=",yB.shape)

# Let's consider simply the shear viscosity file ...
with open('../data/treanor_marrone/shear_viscosity.txt') as f:
    lines = (line for line in f if not line.startswith('#'))
    dataTM = np.loadtxt(lines, skiprows=0)

print(dataTM.shape)
xTM = dataTM[:,0:52] # P, T, Tv, x_ci[mol+at]
yTM = dataTM[:,52:]  # shear viscosity
print("x=",xTM.shape)
print("y=",yTM.shape)

#plt.scatter(xTM[:,1], xTM[:,10], s=5, c='k', marker='o', label='treanor-marrone')
#plt.scatter(xB[:,1], xB[:,9], s=5, c='r', marker='+', label='boltzmann')
#plt.ylabel(r'$\eta$ [Pa·s]')
#ax.xlabel('T [K] ')
#plt.yscale('log')
#plt.xscale('log')
#plt.legend()
#plt.show()
#plt.close()

x_train, x_test, y_train, y_test = train_test_split(xTM, yTM, train_size=0.75, test_size=0.25, random_state=69)
#x_train, x_test, y_train, y_test = train_test_split(xB, yB, train_size=0.75, test_size=0.25, random_state=69)

print("x=",x_train.shape)
print("y=",y_train.shape)

sc_x = StandardScaler()
sc_y = StandardScaler()

# fit scaler
sc_x.fit(x_train)

# transform training dataset
x_train = sc_x.transform(x_train)

# transform test dataset
x_test = sc_x.transform(x_test)

# fit scaler on training dataset
sc_y.fit(y_train.reshape(-1,1))

# transform training dataset
y_train = sc_y.transform(y_train)

# transform test dataset
y_test = sc_y.transform(y_test)

dump(sc_x, open('scaler_x_shear.pkl', 'wb'))
dump(sc_y, open('scaler_y_shear.pkl', 'wb'))

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', x_test.shape)
print('Testing Labels Shape:', y_test.shape)

hyper_params = [{
#                 'n_estimators': (1, 50, 100,),
#                 'min_weight_fraction_leaf': (0.0, 0.25, 0.5,),
#                 'max_features': ('auto',),
#                 'max_features': ('sqrt', 'log2', 'auto',),
#                 'bootstrap': (True, False,),
#                 'oob_score': (True, False,),
#                 'warm_start': (True, False,),
#                 'criterion': ('mse', 'mae',),
#                 'max_depth': (1, 10, 100, None,),
#                 'max_leaf_nodes': (2, 100,),
#                 'min_samples_split': (2, 5, 10,),
#                 'min_impurity_decrease': (0.1, 0.2, 0.3, 0.5,),
#                 'min_samples_leaf': (1, 10, 100,),
}]

est=ensemble.RandomForestRegressor()
gs = GridSearchCV(est, cv=2, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')
#gs = est 

t0 = time.time()
gs.fit(x_train, y_train.ravel())
#gs.fit(x_train, y_train)
runtime = time.time() - t0
print("Complexity and bandwidth selected and model fitted in %.6f s" % runtime)

train_score_mse = mean_squared_error(      sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_mae = mean_absolute_error(     sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_evs = explained_variance_score(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
#train_score_me = max_error(               sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_r2  = r2_score(                sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))

test_score_mse  = mean_squared_error(      sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
test_score_mae  = mean_absolute_error(     sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
test_score_evs  = explained_variance_score(sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
#test_score_me  = max_error(               sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
test_score_r2   = r2_score(                sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))

print()
print("The model performance for training set")
print("--------------------------------------")
print('MAE is {}'.format(train_score_mae))
print('MSE is {}'.format(train_score_mse))
print('EVS is {}'.format(train_score_evs))
#print('ME is {}'.format(train_score_me))
print('R2 score is {}'.format(train_score_r2))
print()
print("The model performance for testing set")
print("--------------------------------------")
print('MAE is {}'.format(test_score_mae))
print('MSE is {}'.format(test_score_mse))
print('EVS is {}'.format(test_score_evs))
#print('ME is {}'.format(test_score_me))
print('R2 score is {}'.format(test_score_r2))
print()
print("Best parameters set found on development set:")
print(gs.best_params_)
print()

# Re-train with best parameters
regr = ExtraTreesRegressor(**gs.best_params_)
#regr = ExtraTreesRegressor()

# Training
##########
t0 = time.time()
regr.fit(x_train, y_train.ravel())
#regr.fit(x_train, y_train)
regr_fit = time.time() - t0
print("Complexity and bandwidth selected and model fitted in %.6f s" % regr_fit)

# perform permutation importance
################################
# https://scikit-learn.org/stable/modules/permutation_importance.html#permutation-importance
# https://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_pcr_vs_pls.html#sphx-glr-auto-examples-cross-decomposition-plot-pcr-vs-pls-py
results = permutation_importance(regr, x_train, y_train, scoring='neg_mean_squared_error')

# get importance
importance = results.importances_mean

# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))

# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.savefig("importance.pdf", dpi=150, crop='true')
plt.show()
plt.close()

print("importances_mea = ", results.importances_mean)
print("importances_std = ", results.importances_std)

# Principal Component Analysis (PCA)
####################################
# https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html#sphx-glr-auto-examples-preprocessing-plot-scaling-importance-py
#pca = PCA(n_components=2, svd_solver='full')
# pca = PCA(n_components=2, svd_solver='arpack')
pca = PCA(n_components=None)
pca.fit(x_train)

# Get the eigenvalues
print("Eigenvalues:")
print(pca.singular_values_)
print()

# Get explained variances
print("Variances (Percentage):")
print(pca.explained_variance_ratio_ * 100)
print()

# Make the scree plot
plt.plot(np.cumsum(pca.explained_variance_ratio_ * 100))
plt.xlabel("Number of components (Dimensions)")
plt.ylabel("Explained variance (%)")
plt.savefig("pca.pdf", dpi=150, crop='true')
plt.show()
plt.close()

#x_train_pca, x_test_pca, y_train_pca, y_test_pca = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=69)
#
## Fit to data and predict using pipelined GNB and PCA.
#unscaled_regr = make_pipeline(PCA(n_components=2), regr)
#unscaled_regr.fit(x_train_pca, y_train_pca)
#pred_test = unscaled_regr.predict(x_test_pca)
#
## Fit to data and predict using pipelined scaling, GNB and PCA.
#std_regr = make_pipeline(StandardScaler(), PCA(n_components=2), regr)
#std_regr.fit(x_train_pca, y_train_pca)
#pred_test_std = std_regr.predict(x_test_pca)
#
## Show prediction accuracies in scaled and unscaled data.
#print("\nPrediction score for the normal test dataset with PCA")
#print("{:.2%}\n".format(metrics.r2_score(y_test_pca, pred_test)))
#
#print("\nPrediction score for the standardized test dataset with PCA")
#print("{:.2%}\n".format(metrics.r2_score(y_test_pca, pred_test_std)))
#
#plt.scatter(x_train_pca[:, 0], x_train_pca[:, 1], alpha=0.3, label="samples")
#for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
#    comp = comp * var  # scale component by its variance explanation power
#    plt.plot(
#        [0, comp[0]],
#        [0, comp[1]],
#        label=f"Component {i}",
#        linewidth=5,
#        color=f"C{i + 2}",
#    )
#plt.gca().set(
#    aspect="equal",
#    title="2-dimensional dataset with principal components",
#    xlabel="first feature",
#    ylabel="second feature",
#)
#plt.legend()
#plt.show()
#
## Extract PCA from pipeline
#pca = unscaled_regr.named_steps["pca"]
#pca_std = std_regr.named_steps["pca"]
#
## Show first principal components
#print("\nPC 1 without scaling:\n", pca.components_[0])
#print("\nPC 1 with scaling:\n", pca_std.components_[0])
#
## Use PCA without and with scale on X_train data for visualization.
#x_train_pca_transformed = pca.transform(x_train_pca)
#scaler = std_regr.named_steps["standardscaler"]
#x_train_pca_std_transformed = pca_std.transform(scaler.transform(x_train_pca))
#
## visualize standardized vs. untouched dataset with PCA performed
#FIG_SIZE = (10, 7)
#fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=FIG_SIZE)
#
#
#for l, c, m in zip(range(0, 3), ("blue", "red", "green"), ("^", "s", "o")):
#    ax1.scatter(
#        x_train_pca_transformed[y_train == l, 0],
#        x_train_pca_transformed[y_train == l, 1],
#        color=c,
#        label="class %s" % l,
#        alpha=0.5,
#        marker=m,
#    )
#
#for l, c, m in zip(range(0, 3), ("blue", "red", "green"), ("^", "s", "o")):
#    ax2.scatter(
#        x_train_pca_std_transformed[y_train == l, 0],
#        x_train_pca_std_transformed[y_train == l, 1],
#        color=c,
#        label="class %s" % l,
#        alpha=0.5,
#        marker=m,
#    )
#
#ax1.set_title("Training dataset after PCA")
#ax2.set_title("Standardized training dataset after PCA")
#
#for ax in (ax1, ax2):
#    ax.set_xlabel("1st principal component")
#    ax.set_ylabel("2nd principal component")
#    ax.legend(loc="upper right")
#    ax.grid()
#
#plt.tight_layout()
#plt.show()

# Prediction
############
t0 = time.time()
y_regr = regr.predict(x_test)
regr_predict = time.time() - t0
print("Prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict))

#with open('output.log', 'w') as f:
#    print("Training time: %.6f s" % regr_fit, file=f)
#    print("Prediction time: %.6f s" % regr_predict, file=f)
#    print(" ", file=f)
#    print("The model performance for training set", file=f)
#    print("--------------------------------------", file=f)
#    print('MAE is {}'.format(train_score_mae), file=f)
#    print('MSE is {}'.format(train_score_mse), file=f)
#    print('EVS is {}'.format(train_score_evs), file=f)
#    print('ME is {}'.format(train_score_me), file=f)
#    print('R2 score is {}'.format(train_score_r2), file=f)
#    print(" ", file=f)
#    print("The model performance for testing set", file=f)
#    print("--------------------------------------", file=f)
#    print('MAE is {}'.format(test_score_mae), file=f)
#    print('MSE is {}'.format(test_score_mse), file=f)
#    print('EVS is {}'.format(test_score_evs), file=f)
#    print('ME is {}'.format(test_score_me), file=f)
#    print('R2 score is {}'.format(test_score_r2), file=f)
#    print(" ", file=f)
#    print("Best parameters set found on development set:", file=f)
#    print(gs.best_params_, file=f)

x_test_dim = sc_x.inverse_transform(x_test)
y_test_dim = sc_y.inverse_transform(y_test)
y_regr_dim = sc_y.inverse_transform(y_regr)

# pressure
#plt.scatter(x_test_dim[:,0],  y_test_dim[:], s=2, c='k', marker='o', label='KAPPA')
#plt.scatter(x_test_dim[:,0],  y_regr_dim[:], s=2, c='b', marker='+', label='ExtraTrees')

# temperature
plt.scatter(x_test_dim[:,1],  y_test_dim[:], s=2, c='k', marker='o', label='KAPPA')
plt.scatter(x_test_dim[:,1],  y_regr_dim[:], s=2, c='g', marker='+', label='ExtraTrees')

# molar fractions
#plt.scatter(x_test_dim[:,10], y_test_dim[:], s=2, c='k', marker='o', label='KAPPA')
#plt.scatter(x_test_dim[:,10], y_regr_dim[:], s=2, c='c', marker='+', label='ExtraTrees')
#plt.scatter(x_test_dim[:,20], y_test_dim[:], s=2, c='y', marker='o', label='KAPPA')
#plt.scatter(x_test_dim[:,20], y_regr_dim[:], s=2, c='g', marker='+', label='ExtraTrees')
#plt.scatter(x_test_dim[:,30], y_test_dim[:], s=2, c='k', marker='o', label='KAPPA')
#plt.scatter(x_test_dim[:,30], y_regr_dim[:], s=2, c='m', marker='+', label='ExtraTrees')
plt.ylabel('shear viscosity [Pa·s]')
plt.xlabel('T [K] ')
plt.legend()
plt.tight_layout()
plt.savefig("shear.pdf", dpi=150, crop='true')
plt.show()
plt.close()

# save the model to disk
########################
dump(gs, 'model_shear.sav')
