
import time
import sys
#sys.path.insert(0, '../../../Utilities/')
#from plotting import newfig, savefig
#import matplotlib as mpl
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
#import operator
#import itertools
#from sklearn import metrics
#from sklearn.metrics import *
#from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler#, MinMaxScaler
#from sklearn.model_selection import train_test_split, GridSearchCV#, learning_curve, cross_val_score
#from sklearn import kernel_ridge
#from sklearn.kernel_ridge import KernelRidge
#import xgboost
from sklearn.neural_network import MLPRegressor

Xin = sys.argv[1]
Tin = sys.argv[2]

#n_jobs = 1
#trial  = 1

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
#x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.75, test_size=0.25, random_state=42)

#print('Training Features Shape:', x_train.shape)
#print('Training Labels Shape:', y_train.shape)
#print('Testing Features Shape:', x_test.shape)
#print('Testing Labels Shape:', y_test.shape)

# MPL
#hyper_params = [
#    {
#        'activation' : ('logistic', 'tanh', 'relu',),
#        'solver' : ('lbfgs','adam','sgd',),
#        'learning_rate' : ('constant', 'invscaling', 'adaptive',),
#    },
#]

#est=MLPRegressor()
#grid_clf = GridSearchCV(est, cv=5, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')

#t0 = time.time()
#grid_clf.fit(x_train, y_train.ravel())
#runtime = time.time() - t0
#print("MLP complexity and bandwidth selected and model fitted in %.3f s" % runtime)

#train_score_mse  = mean_squared_error(sc_y.inverse_transform(y_train), sc_y.inverse_transform(grid_clf.predict(x_train)))
#train_score_mae  = mean_absolute_error(sc_y.inverse_transform(y_train),sc_y.inverse_transform(grid_clf.predict(x_train)))
#train_score_evs  = explained_variance_score(sc_y.inverse_transform(y_train), sc_y.inverse_transform(grid_clf.predict(x_train)))
#train_score_me   = max_error(sc_y.inverse_transform(y_train), sc_y.inverse_transform(grid_clf.predict(x_train)))
#train_score_msle = mean_squared_log_error(sc_y.inverse_transform(y_train), sc_y.inverse_transform(grid_clf.predict(x_train)))

#test_score_mse  = mean_squared_error(sc_y.inverse_transform(y_test),  sc_y.inverse_transform(grid_clf.predict(x_test)))
#test_score_mae  = mean_absolute_error(sc_y.inverse_transform(y_test), sc_y.inverse_transform(grid_clf.predict(x_test)))
#test_score_evs  = explained_variance_score(sc_y.inverse_transform(y_test),  sc_y.inverse_transform(grid_clf.predict(x_test)))
#test_score_me   = max_error(sc_y.inverse_transform(y_test),  sc_y.inverse_transform(grid_clf.predict(x_test)))
#test_score_msle = mean_squared_log_error(sc_y.inverse_transform(y_test),  sc_y.inverse_transform(grid_clf.predict(x_test)))

#sorted_grid_params = sorted(grid_clf.best_params_.items(), key=operator.itemgetter(0))

#out_text = '\t'.join(['mpl',
#                      str(trial),
#                      str(sorted_grid_params).replace('\n',','),
#                      str(train_score_mse),
#                      str(train_score_mae),
#                      str(train_score_evs),
#                      str(train_score_me),
#                      str(train_score_msle),
#                      str(test_score_mse),
#                      str(test_score_mae),
#                      str(test_score_evs),
#                      str(test_score_me),
#                      str(test_score_msle),
#                      str(runtime)])
#print(out_text)
#sys.stdout.flush()

#best_activation = grid_clf.best_params_['activation']
#best_solver = grid_clf.best_params_['solver']
#best_learning_rate = grid_clf.best_params_['learning_rate']

# open a (new) file to write
#outF = open("output.txt", "w")
#print('best_activation = ', best_activation, file=outF)
#print('best_solver = ', best_solver, file=outF)
#print('best_learning_rate = ', best_learning_rate, file=outF)
#outF.close()

#mlp = MLPRegressor(activation=best_activation, solver=best_solver, learning_rate=best_learning_rate)
#mlp = MLPRegressor(activation='relu', solver='lbfgs', learning_rate='constant')

#t0 = time.time()
#mlp.fit(x_train, y_train.ravel())
#mlp_fit = time.time() - t0
#print("MLP complexity and bandwidth selected and model fitted in %.3f s" % mlp_fit)

# https://scikit-learn.org/stable/modules/model_persistence.html
# https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
# https://medium.com/datadriveninvestor/machine-learning-how-to-save-and-load-scikit-learn-models-d7b99bc32c27
# https://www.geeksforgeeks.org/saving-a-machine-learning-model/
#import pickle
#save_mlp = pickle.dumps(mlp)
#load_mlp = pickle.loads(save_mlp)

from joblib import load#, dump
#dump(mlp, 'mlp.joblib')
mlp = load('mlp.joblib')

Xinput = np.array([[float(Xin), int(Tin)]])
Xinput = sc_x.transform(Xinput)

t0 = time.time()
#y_mlp = mlp.predict(x_test)
y_mlp = mlp.predict(Xinput)
mlp_predict = time.time() - t0
#print("MLP prediction for %d inputs in %.3f s" % (x_test.shape[0], mlp_predict))
###print("MLP prediction for %d inputs in %.6f s" % (Xinput.shape[0], mlp_predict))

# open a file to append
#outF = open("output.txt", "a")
#print("MLP complexity and bandwidth selected and model fitted in %.3f s" % mlp_fit, file=outF)
#print("MLP prediction for %d inputs in %.3f s" % (x_test.shape[0], mlp_predict),file=outF)
#print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_mlp), file=outF)
#print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_mlp), file=outF)
#print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_mlp)), file=outF)
#outF.close()

#print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_mlp))
#print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_mlp))
#print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_mlp)))

#x_test_dim = sc_x.inverse_transform(x_test)
#y_test_dim = sc_y.inverse_transform(y_test)

Xinput     = sc_x.inverse_transform(Xinput)
y_mlp_dim  = sc_y.inverse_transform(y_mlp)
#print(y_mlp_dim.astype(float, order='F', casting='unsafe', subok=False, copy=False))

#outF = open("result.txt", "w")
#print(y_mlp_dim.astype(float), file=outF)
#outF.close()

#np.savetxt(sys.stdout, y_mlp_dim)
np.savetxt('out', y_mlp_dim)

#plt.scatter(x[:,1], y[:], s=5, c='red',     marker='o', label='KAPPA')
##plt.scatter(x_test_dim[:,1], y_test_dim[:], s=5, c='red',     marker='o', label='KAPPA')
##plt.scatter(x_test_dim[:,1], y_svr_dim[:],  s=2, c='blue',    marker='+', label='Support Vector Machine')
##plt.scatter(x_test_dim[:,1], y_kr_dim[:],   s=2, c='green',   marker='p', label='Kernel Ridge')
##plt.scatter(x_test_dim[:,1], y_rf_dim[:],   s=2, c='cyan',    marker='*', label='Random Forest')
##plt.scatter(x_test_dim[:,1], y_kn_dim[:],   s=2, c='magenta', marker='d', label='k-Nearest Neighbour')
##plt.scatter(x_test_dim[:,1], y_gp_dim[:],   s=2, c='orange',  marker='^', label='Gaussian Process')
##plt.scatter(x_test_dim[:,1], y_sgd_dim[:],  s=2, c='yellow',  marker='*', label='Stochastic Gradient Descent')
##plt.scatter(x_test_dim[:,1], y_mlp_dim[:],  s=2, c='orange',   marker='*', label='Multi-layer Perceptron')
#plt.scatter(Xinput[:,1], y_mlp_dim[:],  s=2, c='b',  marker='*', label='Multi-layer Perceptron')
#plt.title('Shear viscosity regression with MLP')
#plt.ylabel(r'$\eta$ [Pa·s]')
#plt.xlabel('T [K] ')
#plt.legend()
#plt.tight_layout()
##plt.savefig("eta_MLP.pdf", dpi=150, crop='false')
#plt.show()
