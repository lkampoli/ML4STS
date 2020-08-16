
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
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, cross_val_score
from sklearn import gaussian_process
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, DotProduct, RBF, RationalQuadratic, ConstantKernel, Matern

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

#kernel = DotProduct() + WhiteKernel()
#kernel = ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) + WhiteKernel(1e-1)
#kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
#kernel = 1.0 * RBF(1.0)

hyper_params = [{'alpha': (1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3,),
                 'n_restarts_optimizer': (0,1,10,100,),
                 'kernel': (DotProduct() + WhiteKernel(),),}]
#                 'kernel': ([1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)),
#                              1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1),
#                              1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0,
#                                                   length_scale_bounds=(0.1, 10.0),
#                                                   periodicity_bounds=(1.0, 10.0)),
#                              ConstantKernel(0.1, (0.01, 10.0))
#                              * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2),
#                              1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)],), }]

#                 "kernel": ([ExpSineSquared(l, p)
#                           for l in np.logspace(-2, 2, 10)
#                           for p in np.logspace(0, 2, 10)],
#                           [DotProduct() + WhiteKernel()],
#                           [1.0 * RBF(1.0)],
#                           [ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))],),}]

#kernels = [1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)),
#           1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1),
#           1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0,
#                                length_scale_bounds=(0.1, 10.0),
#                                periodicity_bounds=(1.0, 10.0)),
#           ConstantKernel(0.1, (0.01, 10.0))
#               * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2),
#           1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0),
#                        nu=1.5)]

# kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))

est = gaussian_process.GaussianProcessRegressor()
gp = GridSearchCV(est, cv=5, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')

#gp_kernel = ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) + WhiteKernel(1e-1)
###gp_kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
#gp_kernel = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
#gp_kernel = 1.0 * RBF(1.0)
###gp = GaussianProcessRegressor(kernel=gp_kernel)

t0 = time.time()
gp.fit(x_train, y_train.ravel())
gp_fit = time.time() - t0
print("GP complexity and bandwidth selected and model fitted in %.6f s" % gp_fit)

#t0 = time.time()
#y_gp, sigma = gp.predict(x_test, return_std=True)
#gp_predict = time.time() - t0
#print("GP prediction for %d inputs in %.3f s" % (x_test.shape[0], gp_predict))

#print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_gp))
#print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_gp))
#print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_gp)))

#x_test_dim = sc_x.inverse_transform(x_test)
#y_test_dim = sc_y.inverse_transform(y_test)
#y_svr_dim  = sc_y.inverse_transform(y_svr)
#y_kr_dim   = sc_y.inverse_transform(y_kr)
#y_rf_dim   = sc_y.inverse_transform(y_rf)
#y_kn_dim   = sc_y.inverse_transform(y_kn)
#y_gp_dim   = sc_y.inverse_transform(y_gp)
#y_sgd_dim  = sc_y.inverse_transform(y_sgd)

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
best_alpha = gp.best_params_['alpha']
best_n_restarts_optimizer = gp.best_params_['n_restarts_optimizer']
#best_gamma = gp.best_params_['gamma']
#best_C = gp.best_params_['C']
#best_epsilon = gp.best_params_['epsilon']

# open a (new) file to write
#outF = open("output.txt", "w")
#print('best_kernel = ', best_kernel, file=outF)
#print('best_alpha = ', best_alpha, file=outF)
#print('best_n_restarts_optimizer = ', best_n_restarts_optimizer, file=outF)
#print('best_C = ', best_C, file=outF)
#print('best_epsilon = ', best_epsilon, file=outF)
#print('best_coef0 = ', best_coef0, file=outF)
#print('best_gamma = ', best_gamma, file=outF)
#outF.close()

gp = GaussianProcessRegressor(kernel=best_kernel, alpha=best_alpha, n_restarts_optimizer=best_n_restarts_optimizer)

t0 = time.time()
gp.fit(x_train, y_train.ravel())
gp_fit = time.time() - t0
print("GP complexity and bandwidth selected and model fitted in %.6f s" % gp_fit)

t0 = time.time()
y_gp = gp.predict(x_test)
gp_predict = time.time() - t0
print("GP prediction for %d inputs in %.6f s" % (x_test.shape[0], gp_predict))

# open a file to append
outF = open("output.txt", "a")
print("GP complexity and bandwidth selected and model fitted in %.6f s" % gp_fit, file=outF)
print("GP prediction for %d inputs in %.6f s" % (x_test.shape[0], gp_predict),file=outF)
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_gp), file=outF)
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_gp), file=outF)
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_gp)), file=outF)
outF.close()

print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_gp))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_gp))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_gp)))

x_test_dim = sc_x.inverse_transform(x_test)
y_test_dim = sc_y.inverse_transform(y_test)
y_gp_dim   = sc_y.inverse_transform(y_gp)

plt.scatter(x_test_dim[:,1], y_test_dim[:], s=5, c='red',     marker='o', label='KAPPA')
#plt.scatter(x_test_dim[:,1], y_svr_dim[:],  s=2, c='blue',    marker='+', label='Support Vector Machine')
#plt.scatter(x_test_dim[:,1], y_kr_dim[:],   s=2, c='green',   marker='p', label='Kernel Ridge')
#plt.scatter(x_test_dim[:,1], y_rf_dim[:],   s=2, c='cyan',    marker='*', label='Random Forest')
#plt.scatter(x_test_dim[:,1], y_kn_dim[:],   s=2, c='magenta', marker='d', label='k-Nearest Neighbour')
plt.scatter(x_test_dim[:,1], y_gp_dim[:],   s=2, c='orange',  marker='^', label='Gaussian Process')
#plt.scatter(x_test_dim[:,1], y_sgd_dim[:],  s=2, c='yellow',  marker='*', label='Stochastic Gradient Descent')
#plt.title('Shear viscosity regression with SVR, KR, RF, kN, GP')
plt.title('Shear viscosity regression with GP')
plt.ylabel(r'$\eta$ [Pa·s]')
plt.xlabel('T [K] ')
plt.legend()
plt.tight_layout()
plt.savefig("eta_GP.pdf", dpi=150, crop='false')
plt.show()
