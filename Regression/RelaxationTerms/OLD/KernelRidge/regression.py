#!/usr/bin/env python

import time
import sys
sys.path.insert(0, '../../Utilities/')
from plotting import newfig, savefig
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import operator
import itertools
from sklearn import metrics
from sklearn.metrics import *
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn import kernel_ridge
from sklearn.kernel_ridge import KernelRidge
from joblib import dump, load
import pickle
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

n_jobs = 1
trial  = 1

#dataset=np.loadtxt("../data/datarelax.txt")
dataset=np.loadtxt("../data/datasetDR.txt")
#dataset=np.loadtxt("../data/datasetVT.txt")
#dataset=np.loadtxt("../data/datasetVV.txt")
x=dataset[:,2:3]   # 0: x [m], 1: t [s], 2: T [K]
y=dataset[:,9:10]  # Rci (relaxation source terms)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=69, shuffle=True)

# https://stackoverflow.com/questions/43675665/when-scale-the-data-why-the-train-dataset-use-fit-and-transform-but-the-te
# https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/

#sc_x = MinMaxScaler(feature_range=(0, 1))
sc_x = StandardScaler()
sc_y = StandardScaler()

# fit scaler
sc_x.fit(x_train)
# transform training datasetx
x_train = sc_x.transform(x_train)
# transform test dataset
x_test = sc_x.transform(x_test)

#y_train = y_train.reshape(len(y_train), 1)
#y_test = y_test.reshape(len(y_train), 1)

# fit scaler on training dataset
sc_y.fit(y_train)
# transform training dataset
y_train = sc_y.transform(y_train)
# transform test dataset
y_test = sc_y.transform(y_test)

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', x_test.shape)
print('Testing Labels Shape:', y_test.shape)

hyper_params = [{'kernel': ('poly', 'rbf',),
                 'alpha': (1e-3, 1e-2, 1e-1, 0.0, 0.25, 0.5, 0.75, 1.,),
                 'gamma': (0.1, 1, 10, 100, 1000,),}]

est=kernel_ridge.KernelRidge()
gs = GridSearchCV(est, cv=10, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')

t0 = time.time()
gs.fit(x_train, y_train.ravel())
runtime = time.time() - t0
print("KR complexity and bandwidth selected and model fitted in %.6f s" % runtime)

train_score_mse = mean_squared_error(      sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_mae = mean_absolute_error(     sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_evs = explained_variance_score(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_me  = max_error(               sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))

test_score_mse  = mean_squared_error(      sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
test_score_mae  = mean_absolute_error(     sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
test_score_evs  = explained_variance_score(sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
test_score_me   = max_error(               sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
test_score_r2   = r2_score(                sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))

print("The model performance for testing set")
print("--------------------------------------")
print('MAE is {}'.format(test_score_mae))
print('MSE is {}'.format(test_score_mse))
print('EVS is {}'.format(test_score_evs))
print('ME is {}'.format(test_score_me))
print('R2 score is {}'.format(test_score_r2))

sorted_grid_params = sorted(gs.best_params_.items(), key=operator.itemgetter(0))

out_text = '\t'.join(['regression',
                      str(trial),
                      str(sorted_grid_params).replace('\n',','),
                      str(train_score_mse),
                      str(train_score_mae),
                      str(train_score_evs),
                      str(train_score_me),
                      str(test_score_mse),
                      str(test_score_mae),
                      str(test_score_evs),
                      str(test_score_me),
                      str(runtime)])
print(out_text)
sys.stdout.flush()

best_kernel = gs.best_params_['kernel']
best_alpha  = gs.best_params_['alpha']
best_gamma  = gs.best_params_['gamma']

outF = open("output.txt", "w")
print('best_kernel = ', best_kernel, file=outF)
print('best_alpha = ', best_alpha, file=outF)
print('best_gamma = ', best_gamma, file=outF)
print('R2 score is {}'.format(test_score_r2))
outF.close()

regr = KernelRidge(kernel=best_kernel, gamma=best_gamma, alpha=best_alpha)

t0 = time.time()
regr.fit(x_train, y_train.ravel())
regr_fit = time.time() - t0
print("Complexity and bandwidth selected and model fitted in %.6f s" % regr_fit)

t0 = time.time()
y_regr = regr.predict(x_test)
regr_predict = time.time() - t0
print("Prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict))

# open a file to append
outF = open("output.txt", "a")
print("KR complexity and bandwidth selected and model fitted in %.6f s" % regr_fit, file=outF)
print("KR prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict),file=outF)
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_regr), file=outF)
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_regr), file=outF)
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_regr)), file=outF)
outF.close()

print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_regr))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_regr))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_regr)))


x_test_dim = sc_x.inverse_transform(x_test)
y_test_dim = sc_y.inverse_transform(y_test)
y_regr_dim = sc_y.inverse_transform(y_regr)

plt.scatter(x_test_dim, y_test_dim, s=2, c='k', marker='o', label='Matlab')
plt.scatter(x_test_dim, y_regr_dim, s=2, c='r', marker='+', label='KernelRidge')
#plt.title('Relaxation term $R_{ci}$ regression')
plt.ylabel('$R_{ci}$ $[J/m^3/s]$')
plt.xlabel('T [K] ')
plt.legend()
plt.tight_layout()
#plt.savefig("regression_KR.eps", dpi=150, crop='false')
#plt.savefig("regression_KR.pdf", dpi=150, crop='false')
plt.show()

## Look at the results
#gs_ind = gs.best_estimator_.support_
##plt.scatter(x[gs_ind], y[gs_ind], c='r', s=50, label='KR',   zorder=2, edgecolors=(0, 0, 0))
#plt.scatter(x_test_dim, y_test_dim, c='k',       label='data', zorder=1, edgecolors=(0, 0, 0))
#plt.plot(x_test_dim, y_regr_dim, c='r', label='KR (fit: %.6fs, predict: %.6fs)' % (regr_fit, regr_predict))
##plt.plot(X_plot, y_kr, c='g', label='KRR (fit: %.3fs, predict: %.3fs)' % (regr_fit, regr_predict))
#plt.xlabel('data')
#plt.ylabel('target')
##plt.title('SVR versus Kernel Ridge')
#plt.legend()

# Visualize learning curves
plt.figure()
#svr = SVR(kernel='rbf', C=1e1, gamma=0.1)
#kr = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.1)
#train_sizes, train_scores_svr, test_scores_svr = \
#    learning_curve(svr, X[:100], y[:100], train_sizes=np.linspace(0.1, 1, 10),
#                   scoring="neg_mean_squared_error", cv=10)
train_sizes_abs, train_scores_kr, test_scores_kr, fit_times, score_times = learning_curve(regr,
                                                                                          x_train, #x_test_dim,
                                                                                          y_train, #y_test_dim,
                                                                                          train_sizes=np.linspace(0.1, 1, 10),
                                                                                          scoring="neg_mean_squared_error",
                                                                                          cv=10,
                                                                                          return_times=True)
train_scores_mean = np.mean(train_scores, axis = 1)
train_scores_std  = np.std(train_scores,  axis = 1)
test_scores_mean  = np.mean(test_scores,  axis = 1)
test_scores_std   = np.std(test_scores,   axis = 1)
fit_times_mean    = np.mean(fit_times,    axis = 1)
fit_times_std     = np.std(fit_times,     axis = 1)

#plt.plot(train_sizes, -test_scores_svr.mean(1), 'o-', color="r", label="SVR")
plt.plot(train_sizes, -test_scores_kr.mean(1), 'o-', color="g", label="KRR")
plt.xlabel("Train size")
plt.ylabel("Mean Squared Error")
plt.title('Learning curves')
plt.legend(loc="best")
plt.show()

fig, axes = plt.subplots(3, 1, figsize=(10, 15))

#X, y = load_digits(return_X_y=True)

#title = "Learning Curves (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
#cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

#estimator = GaussianNB()
#plot_learning_curve(estimator, title, X, y, axes=axes[:, 0], ylim=(0.7, 1.01), cv=cv, n_jobs=4)

title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# SVC is more expensive so we do a lower number of CV iterations:
#cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
#estimator = SVC(gamma=0.001)
plot_learning_curve(regr, title, x_train, y_train, axes=axes[:, 1], ylim=(0.7, 1.01), cv=10, n_jobs=-1)
plt.show()

# save the model to disk
#dump(gs, 'model_KR.sav')
