#!/usr/bin/env python
# coding: utf-8

# In[35]:


import time
import sys
sys.path.insert(0, '../../../Utilities/')
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
from sklearn.model_selection import train_test_split, GridSearchCV

n_jobs = 1
trial  = 1

# Import database
dataset=np.loadtxt("../../data/dataset_lite.csv", delimiter=",")
x=dataset[:,0:2]
y=dataset[:,2] # 0: X, 1: T, 2: shear, 3: bulk, 4: conductivity

# Plot dataset
#plt.scatter(x[:,1], dataset[:,2], s=0.5)
#plt.title('Shear viscosity')
#plt.xlabel('T [K]')
#plt.ylabel(r'$\eta$')
#plt.show()

#plt.scatter(x[:,1], dataset[:,3], s=0.5)
#plt.title('Bulk viscosity')
#plt.xlabel('T [K]')
#plt.ylabel(r'$\zeta$')
#plt.show()

#plt.scatter(x[:,1], dataset[:,4], s=0.5)
#plt.title('Thermal conductivity')
#plt.xlabel('T [K]')
#plt.ylabel(r'$\lambda$')
#plt.show()

y=np.reshape(y, (-1,1))
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(x)
Y = sc_y.fit_transform(y)

# The data is then split into training and test data
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.75, test_size=0.25, random_state=42)

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', x_test.shape)
print('Testing Labels Shape:', y_test.shape)


# In[36]:


### KR ###

from sklearn import kernel_ridge
from sklearn.kernel_ridge import KernelRidge

hyper_params = [{'kernel': ('poly','rbf',), 'alpha': (1e-4,1e-2,0.1,1,10,), 'gamma': (0.01,0.1,1,10,100,),}]
est=kernel_ridge.KernelRidge()
kr = GridSearchCV(est, cv=5, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')

# Train
t0 = time.time()
kr.fit(x_train, y_train.ravel())
kr_fit = time.time() - t0
print("KR complexity and bandwidth selected and model fitted in %.6f s" % kr_fit)

# Predict
#t0 = time.time()
#y_kr = kr.predict(x_test)
#kr_predict = time.time() - t0
#print("KR prediction for %d inputs in %.6f s" % (x_test.shape[0], kr_predict))

#print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_kr))
#print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_kr))
#print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_kr)))

train_score_mse = mean_squared_error(      sc_y.inverse_transform(y_train), sc_y.inverse_transform(kr.predict(x_train)))
train_score_mae = mean_absolute_error(     sc_y.inverse_transform(y_train), sc_y.inverse_transform(kr.predict(x_train)))
train_score_evs = explained_variance_score(sc_y.inverse_transform(y_train), sc_y.inverse_transform(kr.predict(x_train)))
train_score_me  = max_error(               sc_y.inverse_transform(y_train), sc_y.inverse_transform(kr.predict(x_train)))

test_score_mse  = mean_squared_error(      sc_y.inverse_transform(y_test),  sc_y.inverse_transform(kr.predict(x_test)))
test_score_mae  = mean_absolute_error(     sc_y.inverse_transform(y_test),  sc_y.inverse_transform(kr.predict(x_test)))
test_score_evs  = explained_variance_score(sc_y.inverse_transform(y_test),  sc_y.inverse_transform(kr.predict(x_test)))
test_score_me   = max_error(               sc_y.inverse_transform(y_test),  sc_y.inverse_transform(kr.predict(x_test)))

sorted_grid_params = sorted(kr.best_params_.items(), key=operator.itemgetter(0))

out_text = '\t'.join(['kernel-ridge',
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
                      str(kr_fit)])
print(out_text)
sys.stdout.flush()

best_kernel = kr.best_params_['kernel']
best_alpha  = kr.best_params_['alpha']
best_gamma  = kr.best_params_['gamma']

outF = open("KR.txt", "w")
print('best_kernel = ', best_kernel, file=outF)
print('best_alpha = ',  best_alpha,  file=outF)
print('best_gamma = ',  best_gamma,  file=outF)
outF.close()

kr = KernelRidge(kernel=best_kernel, gamma=best_gamma, alpha=best_alpha)

t0 = time.time()
kr.fit(x_train, y_train.ravel())
kr_fit = time.time() - t0
print("KR complexity and bandwidth selected and model fitted in %.6f s" % kr_fit)

t0 = time.time()
y_kr = kr.predict(x_test)
kr_predict = time.time() - t0
print("KR prediction for %d inputs in %.6f s" % (x_test.shape[0], kr_predict))

outF = open("KR.txt", "a")
print("KR complexity and bandwidth selected and model fitted in %.6f s" % kr_fit, file=outF)
print("KR prediction for %d inputs in %.6f s" % (x_test.shape[0], kr_predict),file=outF)
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_kr), file=outF)
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_kr), file=outF)
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_kr)), file=outF)
outF.close()

print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_kr))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_kr))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_kr)))

y_kr_dim = sc_y.inverse_transform(y_kr)


# In[37]:


### RF ###

from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor

hyper_params = [{'n_estimators': (10, 100, 1000),
                 'min_weight_fraction_leaf': (0.0, 0.25, 0.5),
                 'max_features': ('sqrt','log2',None),
}]

est=ensemble.RandomForestRegressor()
rf = GridSearchCV(est, cv=5, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')

# Train
t0 = time.time()
rf.fit(x_train, y_train.ravel())
rf_fit = time.time() - t0
print("RF complexity and bandwidth selected and model fitted in %.6f s" % rf_fit)

# Predict
#t0 = time.time()
#y_rf = rf.predict(x_test)
#rf_predict = time.time() - t0
#print("RF prediction for %d inputs in %.3f s" % (x_test.shape[0], rf_predict))

#print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_rf))
#print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_rf))
#print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_rf)))

train_score_mse = mean_squared_error(      sc_y.inverse_transform(y_train), sc_y.inverse_transform(rf.predict(x_train)))
train_score_mae = mean_absolute_error(     sc_y.inverse_transform(y_train), sc_y.inverse_transform(rf.predict(x_train)))
train_score_evs = explained_variance_score(sc_y.inverse_transform(y_train), sc_y.inverse_transform(rf.predict(x_train)))
train_score_me  = max_error(               sc_y.inverse_transform(y_train), sc_y.inverse_transform(rf.predict(x_train)))

test_score_mse  = mean_squared_error(      sc_y.inverse_transform(y_test),  sc_y.inverse_transform(rf.predict(x_test)))
test_score_mae  = mean_absolute_error(     sc_y.inverse_transform(y_test),  sc_y.inverse_transform(rf.predict(x_test)))
test_score_evs  = explained_variance_score(sc_y.inverse_transform(y_test),  sc_y.inverse_transform(rf.predict(x_test)))
test_score_me   = max_error(               sc_y.inverse_transform(y_test),  sc_y.inverse_transform(rf.predict(x_test)))

sorted_grid_params = sorted(rf.best_params_.items(), key=operator.itemgetter(0))

out_text = '\t'.join(['random-forest',
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
                      str(rf_fit)])
print(out_text)
sys.stdout.flush()

best_n_estimators = rf.best_params_['n_estimators']
best_min_weight_fraction_leaf = rf.best_params_['min_weight_fraction_leaf']
best_max_features = rf.best_params_['max_features']

outF = open("RF.txt", "w")
print('best_n_estimators = ', best_n_estimators, file=outF)
print('best_min_weight_fraction_leaf = ', best_min_weight_fraction_leaf, file=outF)
print('best_max_features = ', best_max_features, file=outF)
outF.close()

rf = RandomForestRegressor(n_estimators=best_n_estimators,
                           min_weight_fraction_leaf=best_min_weight_fraction_leaf,
                           max_features=best_max_features)

t0 = time.time()
rf.fit(x_train, y_train.ravel())
rf_fit = time.time() - t0
print("RF complexity and bandwidth selected and model fitted in %.6f s" % rf_fit)

t0 = time.time()
y_rf = rf.predict(x_test)
rf_predict = time.time() - t0
print("RF prediction for %d inputs in %.6f s" % (x_test.shape[0], rf_predict))

outF = open("RF.txt", "a")
print("RF complexity and bandwidth selected and model fitted in %.6f s" % rf_fit, file=outF)
print("RF prediction for %d inputs in %.6f s" % (x_test.shape[0], rf_predict),file=outF)
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_rf), file=outF)
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_rf), file=outF)
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_rf)), file=outF)
outF.close()

print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_rf))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_rf))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_rf)))

y_rf_dim = sc_y.inverse_transform(y_rf)


# In[ ]:


### kNN ###

from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor

hyper_params = [{'algorithm': ('ball_tree', 'kd_tree', 'brute',), 'n_neighbors': (1,2,3,4,5,6,7,8,9,10,),
                 'leaf_size': (1, 10, 20, 30, 100,), 'weights': ('uniform', 'distance',), 'p': (1,2,),}]

est=neighbors.KNeighborsRegressor()
kn = GridSearchCV(est, cv=5, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')

# Train
t0 = time.time()
kn.fit(x_train, y_train.ravel())
kn_fit = time.time() - t0
print("KN complexity and bandwidth selected and model fitted in %.6f s" % kn_fit)

# Predict
#t0 = time.time()
#y_kn = kn.predict(x_test)
#kn_predict = time.time() - t0
#print("KN prediction for %d inputs in %.6f s" % (x_test.shape[0], kn_predict))

#print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_kn))
#print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_kn))
#print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_kn)))

train_score_mse = mean_squared_error(      sc_y.inverse_transform(y_train), sc_y.inverse_transform(kn.predict(x_train)))
train_score_mae = mean_absolute_error(     sc_y.inverse_transform(y_train), sc_y.inverse_transform(kn.predict(x_train)))
train_score_evs = explained_variance_score(sc_y.inverse_transform(y_train), sc_y.inverse_transform(kn.predict(x_train)))
train_score_me  = max_error(               sc_y.inverse_transform(y_train), sc_y.inverse_transform(kn.predict(x_train)))

test_score_mse  = mean_squared_error(      sc_y.inverse_transform(y_test),  sc_y.inverse_transform(kn.predict(x_test)))
test_score_mae  = mean_absolute_error(     sc_y.inverse_transform(y_test),  sc_y.inverse_transform(kn.predict(x_test)))
test_score_evs  = explained_variance_score(sc_y.inverse_transform(y_test),  sc_y.inverse_transform(kn.predict(x_test)))
test_score_me   = max_error(               sc_y.inverse_transform(y_test),  sc_y.inverse_transform(kn.predict(x_test)))

sorted_grid_params = sorted(kn.best_params_.items(), key=operator.itemgetter(0))

out_text = '\t'.join(['k-nearest-neighbour',
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
                      str(kn_fit)])
print(out_text)
sys.stdout.flush()

best_algorithm = kn.best_params_['algorithm']
best_n_neighbors = kn.best_params_['n_neighbors']
best_leaf_size = kn.best_params_['leaf_size']
best_weights = kn.best_params_['weights']
best_p = kn.best_params_['p']

outF = open("KN.txt", "w")
print('best_algorithm = ', best_algorithm, file=outF)
print('best_n_neighbors = ', best_n_neighbors, file=outF)
print('best_leaf_size = ', best_leaf_size, file=outF)
print('best_weights = ', best_weights, file=outF)
print('best_p = ', best_p, file=outF)
outF.close()

kn = KNeighborsRegressor(n_neighbors=best_n_neighbors, algorithm=best_algorithm,
                         leaf_size=best_leaf_size, weights=best_weights, p=best_p)

t0 = time.time()
kn.fit(x_train, y_train.ravel())
kn_fit = time.time() - t0
print("kNN complexity and bandwidth selected and model fitted in %.6f s" % kn_fit)

t0 = time.time()
y_kn = kn.predict(x_test)
kn_predict = time.time() - t0
print("KN prediction for %d inputs in %.6f s" % (x_test.shape[0], kn_predict))

outF = open("KN.txt", "a")
print("kNN complexity and bandwidth selected and model fitted in %.6f s" % kn_fit, file=outF)
print("kNN prediction for %d inputs in %.6f s" % (x_test.shape[0], kn_predict),file=outF)
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_kn), file=outF)
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_kn), file=outF)
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_kn)), file=outF)
outF.close()

print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_kn))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_kn))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_kn)))

y_kn_dim = sc_y.inverse_transform(y_kn)


# In[ ]:


### SVR ###

from sklearn import svm
from sklearn.svm import SVR

hyper_params = [{'kernel': ('poly', 'rbf',), 'gamma': ('scale', 'auto',),
                 'C': (1e-1, 1e0, 1e1,), 'epsilon': (1e-1, 1e0, 1e1,), }]

est=svm.SVR()
svr = GridSearchCV(est, cv=5, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')

# Train
t0 = time.time()
svr.fit(x_train, y_train.ravel())
svr_fit = time.time() - t0
print("SVR complexity and bandwidth selected and model fitted in %.6f s" % svr_fit)

# Predict
#t0 = time.time()
#y_svr = svr.predict(x_test)
#svr_predict = time.time() - t0
#print("SVR prediction for %d inputs in %.6f s" % (x_test.shape[0], svr_predict))

train_score_mse = mean_squared_error(      sc_y.inverse_transform(y_train), sc_y.inverse_transform(svr.predict(x_train)))
train_score_mae = mean_absolute_error(     sc_y.inverse_transform(y_train), sc_y.inverse_transform(svr.predict(x_train)))
train_score_evs = explained_variance_score(sc_y.inverse_transform(y_train), sc_y.inverse_transform(svr.predict(x_train)))
train_score_me  = max_error(               sc_y.inverse_transform(y_train), sc_y.inverse_transform(svr.predict(x_train)))

test_score_mse  = mean_squared_error(      sc_y.inverse_transform(y_test),  sc_y.inverse_transform(svr.predict(x_test)))
test_score_mae  = mean_absolute_error(     sc_y.inverse_transform(y_test),  sc_y.inverse_transform(svr.predict(x_test)))
test_score_evs  = explained_variance_score(sc_y.inverse_transform(y_test),  sc_y.inverse_transform(svr.predict(x_test)))
test_score_me   = max_error(               sc_y.inverse_transform(y_test),  sc_y.inverse_transform(svr.predict(x_test)))

sorted_grid_params = sorted(svr.best_params_.items(), key=operator.itemgetter(0))

out_text = '\t'.join(['svr',
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
                      str(svr_fit)])
print(out_text)
sys.stdout.flush()

best_kernel = svr.best_params_['kernel']
best_gamma = svr.best_params_['gamma']
best_C = svr.best_params_['C']
best_epsilon = svr.best_params_['epsilon']
#best_coef0 = svr.best_params_['coef0']

outF = open("SVR.txt", "w")
print('best_kernel = ', best_kernel, file=outF)
print('best_gamma = ', best_gamma, file=outF)
print('best_C = ', best_C, file=outF)
print('best_epsilon = ', best_epsilon, file=outF)
#print('best_coef0 = ', best_coef0, file=outF)
outF.close()

svr = SVR(kernel=best_kernel, epsilon=best_epsilon, C=best_C, gamma=best_gamma)

t0 = time.time()
svr.fit(x_train, y_train.ravel())
svr_fit = time.time() - t0
print("SVR complexity and bandwidth selected and model fitted in %.6f s" % svr_fit)

t0 = time.time()
y_svr = svr.predict(x_test)
svr_predict = time.time() - t0
print("SVR prediction for %d inputs in %.6f s" % (x_test.shape[0], svr_predict))

outF = open("SVR.txt", "a")
print("SVR complexity and bandwidth selected and model fitted in %.6f s" % svr_fit, file=outF)
print("SVR prediction for %d inputs in %.6f s" % (x_test.shape[0], svr_predict),file=outF)
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_svr), file=outF)
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_svr), file=outF)
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_svr)), file=outF)
outF.close()

print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_svr))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_svr))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_svr)))

y_svr_dim = sc_y.inverse_transform(y_svr)


# In[ ]:


### MLP

from sklearn.neural_network import MLPRegressor

hyper_params = [
    {
        'activation' : ('logistic', 'tanh', 'relu',),
        'solver' : ('lbfgs','adam','sgd',),
        'learning_rate' : ('constant', 'invscaling', 'adaptive',),
        #'hidden_layer_sizes': [(50, 50), (100,100), (150,150), (200,200),],
        #'early_stopping': (True, False),
    },
]

est=MLPRegressor()
mlp = GridSearchCV(est, cv=5, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')

# Train
t0 = time.time()
mlp.fit(x_train, y_train.ravel())
mlp_fit = time.time() - t0
print("MLP complexity and bandwidth selected and model fitted in %.6f s" % mlp_fit)

# Predict
#t0 = time.time()
#y_mlp = svr.predict(x_test)
#mlp_predict = time.time() - t0
#print("MLP prediction for %d inputs in %.6f s" % (x_test.shape[0], mlp_predict))

train_score_mse = mean_squared_error(      sc_y.inverse_transform(y_train), sc_y.inverse_transform(mlp.predict(x_train)))
train_score_mae = mean_absolute_error(     sc_y.inverse_transform(y_train), sc_y.inverse_transform(mlp.predict(x_train)))
train_score_evs = explained_variance_score(sc_y.inverse_transform(y_train), sc_y.inverse_transform(mlp.predict(x_train)))
train_score_me  = max_error(               sc_y.inverse_transform(y_train), sc_y.inverse_transform(mlp.predict(x_train)))

test_score_mse  = mean_squared_error(      sc_y.inverse_transform(y_test),  sc_y.inverse_transform(mlp.predict(x_test)))
test_score_mae  = mean_absolute_error(     sc_y.inverse_transform(y_test),  sc_y.inverse_transform(mlp.predict(x_test)))
test_score_evs  = explained_variance_score(sc_y.inverse_transform(y_test),  sc_y.inverse_transform(mlp.predict(x_test)))
test_score_me   = max_error(               sc_y.inverse_transform(y_test),  sc_y.inverse_transform(mlp.predict(x_test)))

sorted_grid_params = sorted(mlp.best_params_.items(), key=operator.itemgetter(0))

out_text = '\t'.join(['mlp',
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
                      str(mlp_fit)])
print(out_text)
sys.stdout.flush()

best_activation = mlp.best_params_['activation']
best_solver = mlp.best_params_['solver']
best_learning_rate = mlp.best_params_['learning_rate']

outF = open("MLP.txt", "w")
print('best_activation = ', best_activation, file=outF)
print('best_solver = ', best_solver, file=outF)
print('best_learning_rate = ', best_learning_rate, file=outF)
outF.close()

mlp = MLPRegressor(activation=best_activation, solver=best_solver, learning_rate=best_learning_rate)

t0 = time.time()
mlp.fit(x_train, y_train.ravel())
mlp_fit = time.time() - t0
print("MLP complexity and bandwidth selected and model fitted in %.6f s" % mlp_fit)

t0 = time.time()
y_mlp = mlp.predict(x_test)
mlp_predict = time.time() - t0
print("MLP prediction for %d inputs in %.6f s" % (x_test.shape[0], mlp_predict))

outF = open("MLP.txt", "a")
print("MLP complexity and bandwidth selected and model fitted in %.6f s" % mlp_fit, file=outF)
print("MLP prediction for %d inputs in %.6f s" % (x_test.shape[0], mlp_predict),file=outF)
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_mlp), file=outF)
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_mlp), file=outF)
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_mlp)), file=outF)
outF.close()

print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_mlp))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_mlp))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_mlp)))

y_mlp_dim = sc_y.inverse_transform(y_mlp)


# In[39]:


### GP ###

from sklearn import gaussian_process 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, DotProduct, RBF, RationalQuadratic, ConstantKernel, Matern

hyper_params = [{#'n_restarts_optimizer': (0,1,10,100,),
                 #'alpha': (1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3,),
                 'kernel': (1.0 * RBF(1.0), 
                            #ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)),
                            #ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)),
                            #DotProduct() + WhiteKernel(),
                            #ConstantKernel(0.1, (0.01, 10.0)) * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2),
                            #1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5),
                            #1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1), 
                            #1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)),
                            #1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0,                                     
                            #                     length_scale_bounds=(0.1, 10.0),       
                            #                     periodicity_bounds=(1.0, 10.0)),
                           ),}]


est = gaussian_process.GaussianProcessRegressor()                                                                                   
gp = GridSearchCV(est, cv=5, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')

# Train
t0 = time.time()
gp.fit(x_train, y_train.ravel())
gp_fit = time.time() - t0
print("GP complexity and bandwidth selected and model fitted in %.6f s" % gp_fit)

# Predict
#t0 = time.time()
#y_mlp = svr.predict(x_test)
#mlp_predict = time.time() - t0
#print("MLP prediction for %d inputs in %.6f s" % (x_test.shape[0], mlp_predict))

train_score_mse = mean_squared_error(      sc_y.inverse_transform(y_train), sc_y.inverse_transform(gp.predict(x_train)))
train_score_mae = mean_absolute_error(     sc_y.inverse_transform(y_train), sc_y.inverse_transform(gp.predict(x_train)))
train_score_evs = explained_variance_score(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gp.predict(x_train)))
train_score_me  = max_error(               sc_y.inverse_transform(y_train), sc_y.inverse_transform(gp.predict(x_train)))

test_score_mse  = mean_squared_error(      sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gp.predict(x_test)))
test_score_mae  = mean_absolute_error(     sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gp.predict(x_test)))
test_score_evs  = explained_variance_score(sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gp.predict(x_test)))
test_score_me   = max_error(               sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gp.predict(x_test)))

sorted_grid_params = sorted(gp.best_params_.items(), key=operator.itemgetter(0))

out_text = '\t'.join(['gp',
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
                      str(gp_fit)])
print(out_text)
sys.stdout.flush()

best_kernel = gp.best_params_['kernel']                                                                                             
#best_alpha = gp.best_params_['alpha']                                                                                               
#best_n_restarts_optimizer = gp.best_params_['n_restarts_optimizer']                                                                 
#best_gamma = gp.best_params_['gamma']                                                                                              
#best_C = gp.best_params_['C']                                                                                                      
#best_epsilon = gp.best_params_['epsilon']
    
outF = open("GP.txt", "w")
print('best_kernel = ', best_kernel, file=outF)                                                                                    
#print('best_alpha = ', best_alpha, file=outF)                                                                                      
#print('best_n_restarts_optimizer = ', best_n_restarts_optimizer, file=outF)                                                        
outF.close()

#gp = GaussianProcessRegressor(kernel=best_kernel, alpha=best_alpha, n_restarts_optimizer=best_n_restarts_optimizer)
gp = GaussianProcessRegressor(kernel=best_kernel)

t0 = time.time()
gp.fit(x_train, y_train.ravel())
gp_fit = time.time() - t0
print("GP complexity and bandwidth selected and model fitted in %.6f s" % gp_fit)

t0 = time.time()
y_gp = mlp.predict(x_test)
gp_predict = time.time() - t0
print("GP prediction for %d inputs in %.6f s" % (x_test.shape[0], gp_predict))

outF = open("GP.txt", "a")
print("GP complexity and bandwidth selected and model fitted in %.6f s" % gp_fit, file=outF)
print("GP prediction for %d inputs in %.6f s" % (x_test.shape[0], gp_predict),file=outF)
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_gp), file=outF)
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_gp), file=outF)
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_gp)), file=outF)
outF.close()

print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_gp))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_gp))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_gp)))

y_gp_dim = sc_y.inverse_transform(y_gp)


# In[40]:


### Global print

x_test_dim = sc_x.inverse_transform(x_test)
y_test_dim = sc_y.inverse_transform(y_test)

plt.scatter(x_test_dim[:,1], y_test_dim[:], s=5, c='red', marker='o', label='KAPPA')
#plt.scatter(x_test_dim[:,1], y_svr_dim[:], s=1, facecolors='none', edgecolors='k', marker='p', label='Support Vector Machine')
#plt.scatter(x_test_dim[:,1], y_kr_dim[:],  s=1, facecolors='none', edgecolors='k', marker='p', label='Kernel Ridge')
#plt.scatter(x_test_dim[:,1], y_rf_dim[:],  s=1, facecolors='none', edgecolors='k', marker='p', label='Random Forest')
#plt.scatter(x_test_dim[:,1], y_kn_dim[:],  s=1, facecolors='none', edgecolors='k', marker='p', label='k-Nearest Neighbour')
#plt.scatter(x_test_dim[:,1], y_mlp_dim[:], s=1, facecolors='none', edgecolors='k', marker='p', label='Multi-layer Perceptron')
plt.scatter(x_test_dim[:,1], y_gp_dim[:],  s=1, facecolors='none', edgecolors='k', marker='p', label='Gaussian Process')
plt.title('Shear viscosity regression')
plt.ylabel(r'$\eta$ [Pa·s]')
plt.xlabel('T [K] ')
plt.legend()
plt.tight_layout()
#plt.savefig("eta.pdf",     dpi=150, crop='false')
#plt.savefig("eta_SVR.pdf", dpi=150, crop='false')
#plt.savefig("eta_KR.pdf",  dpi=150, crop='false')
#plt.savefig("eta_RF.pdf",  dpi=150, crop='false')
#plt.savefig("eta_KN.pdf",  dpi=150, crop='false')
#plt.savefig("eta_MLP.pdf", dpi=150, crop='false')
plt.savefig("eta_GP.pdf", dpi=150, crop='false')
plt.show()

