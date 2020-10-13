#!/usr/bin/env python
# coding: utf-8

# Physics Informed Neural Network (PINN) in Tensorflow

import time
import sys
sys.path.insert(0, './utilities')

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

import scipy.io
from scipy.interpolate import griddata

from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import product, combinations
import matplotlib.gridspec as gridspec

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection  import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

# Tensorflow random seed for initialization
np.random.seed(1234)
tf.compat.v1.set_random_seed(1234)

# Define the Class PINN which we are going to use
class PINN:

    # Initialize the class
    def __init__(self, x, ni, rho, u, p, E, Ri, layers):

        # Create Input Matrix for the given training data point
        X = np.concatenate([x], 1)

        # min & max for normalization
        self.lb = X.min(0)
        self.ub = X.max(0)

        self.X  = X

        # class attribute definitions
        self.x   = x
        self.ni  = ni
        self.rho = rho
        self.u   = u
        self.p   = p
        self.E   = E
        self.Ri  = Ri
        self.layers = layers

        # Initialize NN
        # initialize_NN is another class method which is used to assign random
        # weights and bias terms to the network. This not only initializes the
        # network but also structures the sizes and values of all the weights and
        # biases that would be so required for the network defined by layers.
        self.weights, self.biases = self.initialize_NN(layers)

        # Define a session to run
        # tf placeholders and graph
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                                                         log_device_placement=True))

        # Define tensors for each variable using tf.placeholder, with shape
        # similar to their numpy counterparts variable_Name
        self.x_tf   = tf.compat.v1.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.rho_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.rho.shape[1]])
        self.u_tf   = tf.compat.v1.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.p_tf   = tf.compat.v1.placeholder(tf.float32, shape=[None, self.p.shape[1]])
        self.E_tf   = tf.compat.v1.placeholder(tf.float32, shape=[None, self.E.shape[1]])
        self.ni_tf  = tf.compat.v1.placeholder(tf.float32, shape=[None, self.ni.shape[1]])
        self.Ri_tf  = tf.compat.v1.placeholder(tf.float32, shape=[None, self.Ri.shape[1]])

        # Predict the values of output by a single forward propagation.
        # Also get AutoDiff coefficients from the same class method: net_Euler_STS
        [self.ni_pred, self.rho_pred, self.u_pred, self.p_pred, self.E_pred, self.Ri_pred,
         self.e1, self.e2, self.e3, self.e4, self.eni] = self.net_Euler_STS(self.x_tf)

        # MSE Normalization
        # The initial normalization terms are necessary to ensure that the
        # gradients don't get driven towards either the residual aquared errors
        # or the MSE of the outputs. Basically, to ensure equal weightage to it
        # being 'trained to training data' as well as being 'Physics informed
        rho_norm = np.amax(rho)
        u_norm   = np.amax(u)
        p_norm   = np.amax(p)
        E_norm   = np.amax(E)
        e1_norm  = rho_norm*u_norm        #*S_norm # e1 is continuity residual
        e2_norm  = p_norm                 #*S_norm # e2 is momentum   residual
        e3_norm  = E_norm*rho_norm*u_norm #*S_norm # e3 is energy     residual
        ni_norm  = np.amax(ni)
        eni_norm = ni_norm *u_norm

        # Weight factor... let's see its impact by varying it w = [0:100].
        # If is it 0, then PINN -> NN and we do not physically inform the NN.
        w = 0.

        # Define Cost function or the Loss
        # In this case I have set the mean squared error of the ouputs to be
        # the loss and commented the PINN residual arguements. Uncommenting the
        # 4 residual expressions will result in a true Phyics Informed Neural
        # Network, otherwise, it is just a data trained Neural network
        self.loss =   tf.reduce_sum(tf.square(self.u_tf   - self.u_pred))  /(u_norm**2)     + \
                      tf.reduce_sum(tf.square(self.rho_tf - self.rho_pred))/(rho_norm**2)   + \
                      tf.reduce_sum(tf.square(self.p_tf   - self.p_pred))  /(p_norm**2)     + \
                      tf.reduce_sum(tf.square(self.E_tf   - self.E_pred))  /(E_norm**2)     + \
                      tf.reduce_sum(tf.square(self.ni_tf  - self.ni_pred)) /(ni_norm**2)    + \
                    w*tf.reduce_sum(tf.square(self.e1))  /(e1_norm**2)                      + \
                    w*tf.reduce_sum(tf.square(self.e2))  /(e2_norm**2)                      + \
                    w*tf.reduce_sum(tf.square(self.e3))  /(e3_norm**2)                      + \
                    w*tf.reduce_sum(tf.square(self.e4))  /(p_norm**2)                       + \
                    w*tf.reduce_sum(tf.square(self.eni)) /(eni_norm**2)

        # Define optimizers
        # There are 2 optimizers used: external by Scipy (L-BFGS-B) and internal
        # by Tensorflow (which is Adam). The external optimizer gives an extra
        # push after the internal has done its job. No need to change the default
        # options of the optimizers. We have used Adam optimizer in this case,
        # since, it is the most common and generally the fastest known converger
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method = 'L-BFGS-B',
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        # Adam
        self.optimizer_Adam = tf.compat.v1.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # Run the session after variable initialization
        init = tf.global_variables_initializer()
        self.sess.run(init)

    # Class methods

    # These are basic initialization functions to create the weigths and biases
    # tensor variables and assign random values to start with code snippet
    # iterates over the layers vector to generate the tensors as stated
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    # Code for a single forward propagation pass taking in weights, biases and
    # input matrix X. Note the normalization step on X as H before passing on to
    # the network
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    # This is the differentiating code snippet which does the Automatic
    # Differential method to find the coefficients of the necessary gradients (or
    # equivalently, residuals) to be used in the MSE (mean squared error) in the
    # cost function. The step is to reference the earlier function neural_net and
    # gain the outputs as a matrix. The matrix is then sliced into individual
    # components to get pressure, density, speed and specific energy. We define
    # the cross section area contour S to be used in the following Physical
    # expressions for 1D Nozzle flow. Next we find the residuals by AutoDiff.
    # The autodiff function provided by Tensorflow is tf.gradients as above. The
    # mass_flow_grad, momentum_grad and energy_grad are actually the residuals of
    # the three Compressible Physical expressions. Return all the variables back
    # to the class attributes.

    def net_Euler_STS(self, x):

        nci_rho_u_p_E = self.neural_net(tf.concat([x], 1), self.weights, self.biases)

        ni  = nci_rho_u_p_E[:,0:48]
        rho = nci_rho_u_p_E[:,48:49]
        u   = nci_rho_u_p_E[:,49:50]
        p   = nci_rho_u_p_E[:,50:51]
        E   = nci_rho_u_p_E[:,51:52]
        Ri  = nci_rho_u_p_E[:,52:100]

        # temporal derivatives
        #ni_t   = tf.gradients(ni,   t)[0]
        #rho_t   = tf.gradients(rho,   t)[0]
        #rho_u_t = tf.gradients(rho*u, t)[0]
        #rho_E_t = tf.gradients(rho*E, t)[0]

        ni_u_x  = tf.gradients(ni *u, x)[0]

        # autodiff gradient #1
        mass_flow_grad = tf.gradients(rho*u, x)[0]

        # autodiff gradient #2
        momentum_grad = tf.gradients((rho*u*u + p), x)[0]

        # autodiff gradient #3
        energy_grad = tf.gradients((rho*E + p)*u, x)[0]

        # state residual
        gamma = 1.4
        state_res = p - rho*(gamma-1.0)*(E-0.5*gamma*u*u)

        eqni = ni_u_x - Ri

        eq1 = mass_flow_grad
        eq2 = momentum_grad
        eq3 = energy_grad
        eq4 = state_res

        return ni, rho, u, p, E, Ri, eq1, eq2, eq3, eq4, eqni

    # callback method just prints the current loss (cost) value of the network.
    def callback(self, loss):
        print('Loss: %.3e' % (loss))

    # Train method actually trains the network weights based on the target
    # of minimizing the loss. tf_dict is defined as the set of input and
    # ideal output parameters for the given data in loop. For the given
    # iterations 'nIter' (variable), the train_op_Adam session is run.
    def train(self, nIter):

        tf_dict = {self.x_tf:   self.x,
                   self.ni_tf:  self.ni,
                   self.rho_tf: self.rho,
                   self.u_tf:   self.u,
                   self.p_tf:   self.p,
                   self.E_tf:   self.E,
                   self.Ri_tf:  self.Ri}

        global loss_vector
        loss_vector = []

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            loss_value = self.sess.run(self.loss, tf_dict)
            loss_vector.append(loss_value)

            # Print
            if it % 100 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                res1 = self.sess.run(self.e1, tf_dict)
                res2 = self.sess.run(self.e2, tf_dict)
                res3 = self.sess.run(self.e3, tf_dict)

                print('Iter: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                print('Mass Residual: %f\t\tMomentum Residual: %f\tEnergy Residual: %f'
                    %(sum(map(lambda a:a*a,res1))/len(res1), sum(map(lambda a:a*a,res2))/len(res2), sum(map(lambda a:a*a,res3))/len(res3)))
                start_time = time.time()

        # The following is external optimizer.
        # Uncomment it to see in action. It runs indefinitely till
        # convergence. Even after the iterations are finished, the optimizer
        # continues to minimize the loss as compelled to by the statement
        # self.optimizer.minimize, passing tf_dict to the loss expression defined
        # as an attribute earlier in the class.
        self.optimizer.minimize(self.sess,
                feed_dict = tf_dict,
                fetches = [self.loss],
                loss_callback = self.callback)

        return loss_vector

    # Predict method is used to predict the output values when test data is
    # passed into the netowrk after the training is completed. All the values are
    # returned to the call
    def predict(self, x_test):

        tf_dict  = {self.x_tf: x_test}

        ni_test  = self.sess.run(self.ni_pred,  tf_dict)
        rho_test = self.sess.run(self.rho_pred, tf_dict)
        u_test   = self.sess.run(self.u_pred,   tf_dict)
        p_test   = self.sess.run(self.p_pred,   tf_dict)
        E_test   = self.sess.run(self.E_pred,   tf_dict)
        Ri_test  = self.sess.run(self.Ri_pred,  tf_dict)

        return ni_test, rho_test, u_test, p_test, E_test, Ri_test

def plot_solution(X_star, u_star, index):

    lb = X_star.min(0)
    ub = X_star.max(0)

    nn = 500

    x = np.linspace(lb[0], ub[0], nn)
    t = np.linspace(lb[1], ub[1], nn)
    X, T = np.meshgrid(x,t)

    U_star = griddata(X_star, u_star.flatten(), (X, T), method='cubic')

    plt.figure(index)
    plt.pcolor(X,T,U_star, cmap = 'jet')
    plt.colorbar()
    plt.show()

# The class PINN is initialized by using the standard init function from object
# oriented python. There are a total of 7 variables fed to initialize:
#
# Inputs:
# x (distance from the shock or through the 1D nozzle length)
# t (time)
#
# Outputs:
# rho (density at x,t),
# u (speed at x,t),
# p (pressure at x,t),
# E (specific energy at x,t),
# layers (layers vector).
#
# We are providing x and t as inputs to the NN and output is P, rho, u, E
#
# X is the net input matrix which is formed by concatenating x and t for the
# training data point. Additional '1' is concatenated for incorporating bias terms.
#
# lb and ub are the lower and upper bound respectively of X which would be later
# used to normalize the value of X before passing it onto the neural network.
# This is done to avoid explosion of network output values due to large training
# data values of X.

# Main function, inside which there are the training and testing commands.
if __name__ == "__main__":

    # Neural Network Architecture
    # layers is a vector of all the node in each of the neural network layers
    # First value, 1 respresents the input layer with 1 parameter (x) while
    # last value 100 is the number of outputs desired
    layers = [1, 10, 25, 15, 100]
    #layers = [1, 20, 20, 20, 20, 20, 20, 20, 20, 20, 100]
    #layers = [1, 40, 40, 40, 40, 100]

    # Load Data
    # The Matlab generated data was stored in file name 'dataset_STS.txt', which
    # is read by numpy and stored as a new variable by name data.
    data = np.loadtxt('data/dataset_STS.txt')

    # The training set length N_train is taken to be 85% of the entire dataset.
    # The rest 15% will be used as test set to validate the result of the training.
    N_train = int(0.85*data.shape[0])

    # idx is a random numbers vector, which will be used to randomly pick 85% of
    # the data from the dataset.
    idx = np.random.choice(range(data.shape[0]), size=(N_train,), replace=False)

    # The rest is mere slicing of dataset to get all required parameters.

    x   = data[:,0:1].flatten()[:,None]
    ni  = data[:,1:49]#.flatten()[:,None]
    rho = data[:,49:50].flatten()[:,None]
    u   = data[:,50:51].flatten()[:,None]
    p   = data[:,51:52].flatten()[:,None]
    E   = data[:,52:53].flatten()[:,None]
    Ri  = data[:,53:101]#.flatten()[:,None]

    [x_train,   x_test,
     ni_train,  ni_test,
     rho_train, rho_test,
     u_train,   u_test,
     p_train,   p_test,
     E_train,   E_test,
     Ri_train,  Ri_test] = train_test_split(x, ni, rho, u, p, E, Ri, test_size=0.20, random_state=69)

    sc_x   = MinMaxScaler() #StandardScaler() RobustScaler() MaxAbsScaler()
    #sc_ni = np.zeros(ni.shape[1])
    #for i in range(ni.shape[1]):
    sc_ni  = MinMaxScaler()
    sc_rho = MinMaxScaler()
    sc_u   = MinMaxScaler()
    sc_p   = MinMaxScaler()
    sc_E   = MinMaxScaler()
    sc_Ri  = MinMaxScaler()

    # Training set
    x_train   = sc_x.fit_transform(x_train)
    ni_train  = sc_ni.fit_transform(ni_train)
    rho_train = sc_rho.fit_transform(rho_train)
    u_train   = sc_u.fit_transform(u_train)
    p_train   = sc_p.fit_transform(p_train)
    E_train   = sc_E.fit_transform(E_train)
    Ri_train  = sc_Ri.fit_transform(Ri_train)

    # Testing set
    x_test   = sc_x.fit_transform(x_test)
    ni_test  = sc_ni.fit_transform(ni_test)
    rho_test = sc_rho.fit_transform(rho_test)
    u_test   = sc_u.fit_transform(u_test)
    p_test   = sc_p.fit_transform(p_test)
    E_test   = sc_E.fit_transform(E_test)
    Ri_test  = sc_Ri.fit_transform(Ri_test)

    # Training the NN based on the training set, randomly chosen above model
    # = PINN(..) passes the necessary training data to the 'NN' class (model
    # here being an instance of the NN class) in order to initialize all the
    # parameters as well as the NN architecture including random initialization
    # of weights and biases.
    model = PINN(x_train, ni_train, rho_train, u_train, p_train, E_train, Ri_train, layers)

    model.train(10000)

    # Plotting Loss
    plt.plot(loss_vector, label='Loss value')
    plt.legend()
    plt.title('Loss value over iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig('./figures/Loss', crop = False)

    # Prediction
    # The input parameters of the test set are used to predict the pressure, density, speed and specific energy for the
    # given x and t by using the .predict method.
    [ni_pred, rho_pred, u_pred, p_pred, E_pred, Ri_pred] = model.predict(x_test)

# Error
# Normal relative error is printed for each variable
error_ni  = np.linalg.norm(ni_test -ni_pred ,2)/np.linalg.norm(ni_test ,2)
error_rho = np.linalg.norm(rho_test-rho_pred,2)/np.linalg.norm(rho_test,2)
error_u   = np.linalg.norm(u_test-u_pred,2)/np.linalg.norm(u_test,2)
error_p   = np.linalg.norm(p_test-p_pred,2)/np.linalg.norm(p_test,2)
error_E   = np.linalg.norm(E_test-E_pred,2)/np.linalg.norm(E_test,2)
error_Ri  = np.linalg.norm(Ri_test -Ri_pred ,2)/np.linalg.norm(Ri_test ,2)

print("Test Error in ni: "+str(error_ni))
print("Test Error in rho: "+str(error_rho))
print("Test Error in u: "+str(error_u))
print("Test Error in p: "+str(error_p))
print("Test Error in E: "+str(error_E))
print("Test Error in Ri: "+str(error_Ri))

x_train_sb   = sc_x.inverse_transform(x_train)
ni_train_sb  = sc_ni.inverse_transform(ni_train)
rho_train_sb = sc_rho.inverse_transform(rho_train)
u_train_sb   = sc_u.inverse_transform(u_train)
p_train_sb   = sc_p.inverse_transform(p_train)
E_train_sb   = sc_E.inverse_transform(E_train)
Ri_train_sb  = sc_Ri.inverse_transform(Ri_train)

x_test_sb   = sc_x.inverse_transform(x_test)
ni_test_sb  = sc_ni.inverse_transform(ni_test)
rho_test_sb = sc_rho.inverse_transform(rho_test)
u_test_sb   = sc_u.inverse_transform(u_test)
p_test_sb   = sc_p.inverse_transform(p_test)
E_test_sb   = sc_E.inverse_transform(E_test)
Ri_test_sb  = sc_Ri.inverse_transform(Ri_test)

ni_pred_sb  = sc_ni.inverse_transform(ni_pred)
rho_pred_sb = sc_rho.inverse_transform(rho_pred)
u_pred_sb   = sc_u.inverse_transform(u_pred)
p_pred_sb   = sc_p.inverse_transform(p_pred)
E_pred_sb   = sc_E.inverse_transform(E_pred)
Ri_pred_sb  = sc_Ri.inverse_transform(Ri_pred)

# Plot Nci
plt.plot(x_test_sb, ni_pred_sb, 'o', color='black', label='NN', linewidth=2, markersize=5, fillstyle='none')
plt.plot(x_test_sb, ni_test_sb, 'o', color='red',   label='Exact', markersize=4)
plt.title('Comparison of NN and Exact solution for Molecular Number Density')
plt.xlabel('X [m]')
plt.ylabel('$x_{ci}$ $[-]$')
#plt.legend()
plt.tight_layout()
#savefig('./figures/Nci', crop = False)
plt.show()

# Plot Nat
#plt.plot(x_test_sb, nat_pred_sb, 'o', color='black', label='NN', linewidth=2, markersize=5, fillstyle='none')
#plt.plot(x_test_sb, nat_test_sb, 'o', color='red',   label='Exact', markersize=4)
#plt.title('Comparison of NN and Exact solution for Atomic Number Density')
#plt.xlabel('X []')
#plt.ylabel('$n_{at}$ $[]$')
#plt.legend()
#plt.tight_layout()
#savefig('./figures/Nat', crop = False)
#plt.show()

# Plot RHO
plt.plot(x_test_sb, rho_pred_sb, 'o', color='black', label='NN', linewidth=2, markersize=5, fillstyle='none')
plt.plot(x_test_sb, rho_test_sb, 'o', color='red',   label='Exact', markersize=4)
plt.title('Comparison of NN and Exact solution for Density')
plt.xlabel('X [m]')
plt.ylabel(r'$\rho$ $[kg/m^3]$')
plt.legend()
plt.tight_layout()
#savefig('./figures/RHO', crop = False)
plt.show()

# Plot P
plt.plot(x_test_sb, p_pred_sb, 'o', color='black', label='NN', linewidth=2, markersize=5, fillstyle='none')
plt.plot(x_test_sb, p_test_sb, 'o', color='red',   label='Exact', markersize=4)
plt.title('Comparison of NN and Exact solution for Pressure')
plt.xlabel('X [m]')
plt.ylabel('P [Pa]')
plt.legend()
plt.tight_layout()
#savefig('./figures/P', crop = False)
plt.show()

# Plot U
plt.plot(x_test_sb, u_pred_sb, 'o', color='black', label='NN', linewidth=2, markersize=5, fillstyle='none')
plt.plot(x_test_sb, u_test_sb, 'o', color='red',   label='Exact', markersize=4)
plt.title('Comparison of NN and Exact solution for Velocity')
plt.xlabel('X [m]')
plt.ylabel('U [m/s]')
plt.legend()
plt.tight_layout()
#savefig('./figures/U', crop = False)
plt.show()

# Plot E
plt.plot(x_test_sb, E_pred_sb, 'o', color='black', label='NN', linewidth=2, markersize=5, fillstyle='none')
plt.plot(x_test_sb, E_test_sb, 'o', color='red',   label='Exact', markersize=4)
plt.title('Comparison of NN and Exact solution for Energy')
plt.xlabel('X [m]')
plt.ylabel('E [J]')
plt.legend()
plt.tight_layout()
#savefig('./figures/E', crop = False)
plt.show()

# Plot Rci
plt.plot(x_test_sb, Ri_pred_sb, 'o', color='black', label='NN', linewidth=2, markersize=5, fillstyle='none')
plt.plot(x_test_sb, Ri_test_sb, 'o', color='red',   label='Exact', markersize=4)
plt.title('Comparison of NN and Exact solution for $R_{ci}$')
plt.xlabel('X [m]')
plt.ylabel('$R_{ci}$ $[J/m^3/s]$')
#plt.legend()
plt.tight_layout()
#savefig('./figures/Rci', crop = False)
plt.show()

# Plot Rat
#plt.plot(x_test_sb, Rat_pred_sb, 'o', color='black', label='NN', linewidth=2, markersize=5, fillstyle='none')
#plt.plot(x_test_sb, Rat_test_sb, 'o', color='red',   label='Exact', markersize=4 )
#plt.title('Comparison of NN and Exact solution for $R_{at}$')
#plt.xlabel('X []')
#plt.ylabel(r'$R_{at}$ $[]$')
#plt.legend()
#plt.tight_layout()
##savefig('./figures/Rat', crop='false')
#plt.show()
