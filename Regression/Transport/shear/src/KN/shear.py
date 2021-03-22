#!/usr/bin/env python
# coding: utf-8

# https://towardsdatascience.com/explaining-feature-importance-by-example-of-a-random-forest-d9166011959e
# https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/

import time
import sys
sys.path.insert(0, '../../../../../Utilities/')

from plotting import newfig, savefig
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

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV

import joblib
from joblib import dump, load
import pickle

from sklearn.inspection import permutation_importance

from sklearn.tree import DecisionTreeRegressor
#
from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.neighbors import NearestNeighbors
#
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import RFE

from sklearn.decomposition import PCA

n_jobs = 1
trial  = 1

# Import database
# https://pandas.pydata.org/pandas-docs/version/0.25.1/reference/api/pandas.DataFrame.to_numpy.html#pandas.DataFrame.to_numpy
#data = pd.read_fwf("../../../Data/TCs_air5.txt").to_numpy()
#x    = data[:,0:7]
#y    = data[:,7:8]

with open('../../../../Data/TCs_air5.txt') as f:
    lines = (line for line in f if not line.startswith('#'))
    dataset = np.loadtxt(lines, skiprows=1)

#dataset = np.loadtxt("../../../Data/TCs_air5.txt")
x = dataset[:,0:7] # T, P, x_N2, x_O2, x_NO, x_N, x_O
y = dataset[:,7:8] # shear

# create the PCA instance
pca = PCA(2)

# fit on data
pca.fit(x)

# access values and vectors
print(pca.components_)
print(pca.explained_variance_)

# transform data
xPCA = pca.transform(x)
print(xPCA)
print(xPCA.shape)

# The data is then split into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=69)

# define feature selection
fs = SelectKBest(score_func=f_regression, k=7)

# apply feature selection
x_selected = fs.fit_transform(x, y)
print(x_selected.shape)

# define RFE
#rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=5)
#
# fit RFE
#rfe.fit(X, y)
#
# summarize all features
#for i in range(X.shape[1]):
#	print('Column: %d, Selected=%s, Rank: %d' % (i, rfe.support_[i], rfe.ranking_[i]))

sc_x = StandardScaler()
sc_y = StandardScaler()

sc_x.fit(x_train)
x_train = sc_x.transform(x_train)
x_test  = sc_x.transform(x_test)

sc_y.fit(y_train)
y_train = sc_y.transform(y_train)
y_test  = sc_y.transform(y_test)


dump(sc_x, open('../../scaler/scaler_x_shear.pkl', 'wb'))
dump(sc_y, open('../../scaler/scaler_y_shear.pkl', 'wb'))

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:',   y_train.shape)
print('Testing Features Shape:',  x_test.shape)
print('Testing Labels Shape:',    y_test.shape)

#hyper_params = [{'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute',),
#                 'n_neighbors': (1,2,3,4,5,6,7,8,9,10,),
#                 'leaf_size': (1, 10, 25, 50, 75, 100,),
#                 'weights': ('uniform', 'distance',),
#                 'p': (1,2,),
#}]
hyper_params = [{'algorithm': ('auto',),
                 'n_neighbors': (1,10,),
                 'leaf_size': (1, 100,),
                 'weights': ('uniform', 'distance',),
                 'p': (1,2,),
}]

est=neighbors.KNeighborsRegressor()
gs  = GridSearchCV(est, cv=3, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')

t0 = time.time()
gs.fit(x_train, y_train)
runtime = time.time() - t0
print("Complexity and bandwidth selected and model fitted in %.6f s" % runtime)

train_score_mse = mean_squared_error(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_mae = mean_absolute_error(sc_y.inverse_transform(y_train),sc_y.inverse_transform(gs.predict(x_train)))
train_score_evs = explained_variance_score(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_me  = max_error(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_msle = mean_squared_log_error(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))

test_score_mse = mean_squared_error(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_mae = mean_absolute_error(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_evs = explained_variance_score(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_me  = max_error(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_msle = mean_squared_log_error(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_r2  = r2_score(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))

print("The model performance for testing set")
print("--------------------------------------")
print('MAE is {}'.format(test_score_mae))
print('MSE is {}'.format(test_score_mse))
print('EVS is {}'.format(test_score_evs))
print('ME is {}'.format(test_score_me))
print('MSLE is {}'.format(test_score_msle))
print('R2 score is {}'.format(test_score_r2))

sorted_grid_params = sorted(gs.best_params_.items(), key=operator.itemgetter(0))

out_text = '\t'.join(['regression',
                      str(trial),
                      str(sorted_grid_params).replace('\n',','),
                      str(train_score_mse),
                      str(train_score_mae),
                      str(train_score_evs),
                      str(train_score_me),
                      str(train_score_msle),
                      str(test_score_mse),
                      str(test_score_mae),
                      str(test_score_evs),
                      str(test_score_me),
                      str(test_score_msle),
                      str(runtime)])
print(out_text)
sys.stdout.flush()

best_algorithm = gs.best_params_['algorithm']
best_n_neighbors = gs.best_params_['n_neighbors']
best_leaf_size = gs.best_params_['leaf_size']
best_weights = gs.best_params_['weights']
best_p = gs.best_params_['p']

outF = open("output_shear.txt", "w")
print('best_algorithm = ', best_algorithm, file=outF)
print('best_n_neighbors = ', best_n_neighbors, file=outF)
print('best_leaf_size = ', best_leaf_size, file=outF)
print('best_weights = ', best_weights, file=outF)
print('best_p = ', best_p, file=outF)
outF.close()

regr = KNeighborsRegressor(n_neighbors=best_n_neighbors,
                           algorithm=best_algorithm,
                           leaf_size=best_leaf_size,
                           weights=best_weights,
                           p=best_p)

t0 = time.time()
regr.fit(x_train, y_train.ravel())
regr_fit = time.time() - t0
print("Complexity and bandwidth selected and model fitted in %.6f s" % regr_fit)

#print('R^2 Training Score: {:.2f} \nOOB Score: {:.2f} \nR^2 Validation Score: {:.2f}'.format(regr.score(x_train, y_train.ravel()),
#                                                                                            regr.oob_score_,
#                                                                                             regr.score(x_test, y_test.ravel())))

#importance = regr.feature_importances_
#
## summarize feature importance
#for i,v in enumerate(importance):
#	print('Feature: %0d, Score: %.5f' % (i,v))
#
## plot feature importance
#plt.title("Feature importances")
#features = np.array(['T', 'P', '$X_{N2}$', '$X_{O2}$', '$X_{NO}$', '$X_N$', '$X_O$'])
#plt.bar(features, importance)
##plt.bar([x for x in range(len(importance))], importance)
#plt.savefig("../../pdf/importance_shear.pdf", dpi=150, crop='false')
#plt.show()
#plt.close()

# perform permutation importance
results = permutation_importance(regr, x_train, y_train, scoring='neg_mean_squared_error')

# get importance
importance = results.importances_mean

# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))

# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()

t0 = time.time()
y_regr = regr.predict(x_test)
regr_predict = time.time() - t0
print("Prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict))

# open a file to append
outF = open("output_shear.txt", "a")
print("Complexity and bandwidth selected and model fitted in %.6f s" % regr_fit, file=outF)
print("Prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict),file=outF)
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

plt.scatter(x_test_dim[:,0], y_test_dim[:], s=5, c='k', marker='o', label='KAPPA')
plt.scatter(x_test_dim[:,0], y_regr_dim[:], s=2.5, c='r', marker='o', label='DecisionTree')
#plt.title('Shear viscosity regression with kNN')
plt.ylabel(r'$\eta$ [Pa·s]')
plt.xlabel('T [K] ')
plt.legend()
plt.tight_layout()
plt.savefig("../../pdf/shear.pdf", dpi=150, crop='false')
plt.show()
plt.close()

# save the model to disk
dump(gs, '../../model/model_shear.sav')
