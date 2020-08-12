# https://towardsdatascience.com/quick-start-to-gaussian-process-regression-36d838810319
# http://krasserm.github.io/2018/03/19/gaussian-processes/
# https://github.com/krasserm/bayesian-machine-learning
# https://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-regression-gpr
# https://juanitorduz.github.io/gaussian_process_reg/
# https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html#sphx-glr-auto-examples-gaussian-process-plot-gpr-noisy-targets-py
#

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import metrics
from gaussian_processes_util import plot_gp

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

# The data is then split into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', x_test.shape)
print('Testing Labels Shape:', y_test.shape)

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, DotProduct, WhiteKernel

noise = 0.1

# kernel = DotProduct() + WhiteKernel()
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
gpr = GaussianProcessRegressor(kernel=kernel, alpha=noise**2)
# model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)
params = kernel.get_params()

# Reuse training data from previous 1D example
gpr.fit(x_train, y_train)
gpr.score(x_train, y_train)

# Compute posterior predictive mean and covariance
#mu_s, cov_s = gpr.predict(x_test, return_cov=True)
y_pred, sigma = gpr.predict(x_test, return_std=True)

# Obtain optimized kernel parameters
l = gpr.kernel_.k2.get_params()['length_scale']
sigma_f = np.sqrt(gpr.kernel_.k1.get_params()['constant_value'])

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
plt.figure()
plt.plot(x_train, y_train, 'r:', label='Train')
plt.plot(x_train, y_train, 'r.', markersize=10, label='Observations')
plt.plot(x_test, y_pred, 'b-', label='Prediction')
plt.fill(np.concatenate([x_test, x_test[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')
plt.savefig("eta_gaussianprocess.pdf", dpi=150, crop='false')
plt.show()

# Plot the results
#plot_gp(mu_s, cov_s, x_test, x_train=x_train, y_train=y_train)

# Calculate metrics
#print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_pred))
#print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_pred))
#print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#plt.scatter(x[3602:4001,1], y[3602:4001], s=0.5, label='KAPPA')
#plt.scatter(x[:,1], y[:], s=0.5, label='KAPPA')
#plt.plot(x_test[:,1], y_pred[:], 'o', color='red', label='predicted', linewidth=2, markersize=2)
#plt.title(' Shear viscosity for molar fraction = 0.9 ')
#plt.ylabel(r'$\eta$ [Pa·s]')
#plt.xlabel('T [K] ')
#plt.legend()
#plt.tight_layout()
#plt.savefig("eta_randomforest.pdf", dpi=150, crop='false')
#plt.show()
