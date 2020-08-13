# https://medium.com/datadriveninvestor/part-ii-support-vector-machines-regression-b4d4559ba2c
# https://towardsdatascience.com/an-introduction-to-support-vector-regression-svr-a3ebc1672c2
# https://github.com/tomsharp/SVR/blob/master/SVR.ipynb
# https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html#sphx-glr-auto-examples-svm-plot-svm-regression-py
# https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_kernel_ridge_regression.html#sphx-glr-auto-examples-miscellaneous-plot-kernel-ridge-regression-py
# https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/
# https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html#sklearn.gaussian_process.kernels.RBF
# https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html#sphx-glr-auto-examples-gaussian-process-plot-gpr-noisy-targets-py
# https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy.html#sphx-glr-auto-examples-gaussian-process-plot-gpr-noisy-py
# https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html#sklearn.kernel_ridge.KernelRidge
# https://stackoverflow.com/questions/29819428/normalization-or-standardization-data-input-for-svm-scikitlearn
## https://stackoverflow.com/questions/50789508/random-forest-regression-how-do-i-analyse-its-performance-python-sklearn

import time

import sys
sys.path.insert(0, '../../Utilities/')

from plotting import newfig, savefig

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from sklearn import metrics
from sklearn import preprocessing

from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve

from sklearn.kernel_ridge import KernelRidge

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, DotProduct, RBF, RationalQuadratic, ConstantKernel

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor

# Import database
dataset=np.loadtxt("../data/dataset_lite.csv", delimiter=",")
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
#sc_x = MinMaxScaler()
#sc_y = MinMaxScaler()
X = sc_x.fit_transform(x)
Y = sc_y.fit_transform(y)
#y=np.reshape(y, (-1,1))
#sc_x = MinMaxScaler()
#sc_y = MinMaxScaler()
#print(sc_x.fit(x))
#X=sc_x.transform(x)
#print(sc_y.fit(y))
#Y=sc_y.transform(y)

# The data is then split into training and test data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', x_test.shape)
print('Testing Labels Shape:', y_test.shape)

# Fit regression model

# gamma{‘scale’, ‘auto’} or float, default=’scale’
#
#    Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
#        if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,
#        if ‘auto’, uses 1 / n_features.

# Cfloat, default=1.0
#
#    Regularization parameter. The strength of the regularization is inversely proportional to C.
#    Must be strictly positive. The penalty is a squared l2 penalty.

# epsilonfloat, default=0.1

#    Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is
#    associated in the training loss function with points predicted within a distance epsilon from the
#    actual value.

# SVR
#pipeline = make_pipeline(preprocessing.StandardScaler(), SVR(kernel='rbf', epsilon=0.01, C=100, gamma = 0.01))
#pipeline = make_pipeline(SVR(kernel='rbf', epsilon=0.01, C=100, gamma = 'scale'))
#pipeline = make_pipeline(SVR(kernel='linear', C=100, gamma='auto'))
#pipeline = make_pipeline(SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1))
svr = SVR(kernel='rbf', epsilon=0.01, C=100, gamma = 'auto')
#svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1),
#                  param_grid={"C": [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
#                               "epsilon": [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
#                               "gamma": np.logspace(-2, 2, 5)})

# KRR
kr = KernelRidge(kernel='rbf', gamma=0.1)
#kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1),
#                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
#                              "gamma": np.logspace(-2, 2, 5)})

rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

kn = KNeighborsRegressor(n_neighbors=1, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski',
                         metric_params=None, n_jobs=None)

#gp_kernel = ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) + WhiteKernel(1e-1)
gp_kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
#gp_kernel = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
#gp_kernel = 1.0 * RBF(1.0)
gp = GaussianProcessRegressor(kernel=gp_kernel)

t0 = time.time()
svr.fit(x_train, y_train)
svr_fit = time.time() - t0
print("SVR complexity and bandwidth selected and model fitted in %.3f s" % svr_fit)

t0 = time.time()
kr.fit(x_train, y_train)
kr_fit = time.time() - t0
print("KRR complexity and bandwidth selected and model fitted in %.3f s" % kr_fit)

t0 = time.time()
rf.fit(x_train, y_train)
rf_fit = time.time() - t0
print("RF complexity and bandwidth selected and model fitted in %.3f s" % rf_fit)

t0 = time.time()
kn.fit(x_train, y_train)
kn_fit = time.time() - t0
print("KN complexity and bandwidth selected and model fitted in %.3f s" % kn_fit)

t0 = time.time()
gp.fit(x_train, y_train)
gp_fit = time.time() - t0
print("GP complexity and bandwidth selected and model fitted in %.3f s" % gp_fit)

t0 = time.time()
y_svr = svr.predict(x_test)
svr_predict = time.time() - t0
print("SVR prediction for %d inputs in %.3f s" % (x_test.shape[0], svr_predict))

print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_svr))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_svr))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_svr)))

t0 = time.time()
y_kr = kr.predict(x_test)
kr_predict = time.time() - t0
print("KRR prediction for %d inputs in %.3f s" % (x_test.shape[0], kr_predict))

print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_kr))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_kr))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_kr)))

t0 = time.time()
y_rf = rf.predict(x_test)
rf_predict = time.time() - t0
print("RF prediction for %d inputs in %.3f s" % (x_test.shape[0], rf_predict))

print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_rf))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_rf))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_rf)))

t0 = time.time()
y_kn = kn.predict(x_test)
kn_predict = time.time() - t0
print("KN prediction for %d inputs in %.3f s" % (x_test.shape[0], kn_predict))

print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_kn))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_kn))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_kn)))

t0 = time.time()
y_gp, sigma = gp.predict(x_test, return_std=True)
gp_predict = time.time() - t0
print("GP prediction for %d inputs in %.3f s" % (x_test.shape[0], gp_predict))

print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_gp))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_gp))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_gp)))

x_test_dim = sc_x.inverse_transform(x_test)
y_test_dim = sc_y.inverse_transform(y_test)
y_svr_dim  = sc_y.inverse_transform(y_svr)
y_kr_dim   = sc_y.inverse_transform(y_kr)
y_rf_dim   = sc_y.inverse_transform(y_rf)
y_kn_dim   = sc_y.inverse_transform(y_kn)
y_gp_dim   = sc_y.inverse_transform(y_gp)

plt.scatter(x_test_dim[:,1], y_test_dim[:], s=5, c='red',     marker='o', label='KAPPA')
plt.scatter(x_test_dim[:,1], y_svr_dim[:],  s=2, c='blue',    marker='+', label='Support Vector Machine')
plt.scatter(x_test_dim[:,1], y_kr_dim[:],   s=2, c='green',   marker='p', label='Kernel Ridge')
plt.scatter(x_test_dim[:,1], y_rf_dim[:],   s=2, c='cyan',    marker='*', label='Random Forest')
plt.scatter(x_test_dim[:,1], y_kn_dim[:],   s=2, c='magenta', marker='d', label='k-Nearest Neighbour')
plt.scatter(x_test_dim[:,1], y_gp_dim[:],   s=2, c='orange',  marker='^', label='Gaussian Process')
plt.title('Shear viscosity regression with SVR, KR, RF, kN, GP')
plt.ylabel(r'$\eta$ [Pa·s]')
plt.xlabel('T [K] ')
plt.legend()
plt.tight_layout()
plt.savefig("eta_SVR_KR_RF_kN_GP.pdf", dpi=150, crop='false')
plt.show()
