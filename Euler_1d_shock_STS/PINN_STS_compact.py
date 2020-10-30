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

from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import RobustScaler
#from sklearn.preprocessing import MaxAbsScaler
#from sklearn.preprocessing import Normalizer
#from sklearn.preprocessing import QuantileTransformer
#from sklearn.preprocessing import PowerTransformer

# Tensorflow random seed for initialization
np.random.seed(1234)
tf.compat.v1.set_random_seed(1234)

# Define the Class PINN which we are going to use
class PINN:

    # Initialize the class
    def __init__(self, x, nci, nat, rho, u, p, E, Rci, Rat, layers):

        # Create Input Matrix for the given training data point
        X = np.concatenate([x], 1)

        # min & max for normalization
        self.lb = X.min(0)
        self.ub = X.max(0)

        self.X  = X

        # class attribute definitions
        self.x   = x

        self.nci = nci
        self.nat = nat

        self.rho = rho
        self.u   = u
        self.p   = p
        self.E   = E

        self.Rci = Rci
        self.Rat = Rat

        self.layers = layers

        # Initialize_NN is another class method which is used to assign random
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

        self.nci_tf = tf.placeholder(tf.float32, shape=[None, self.nci.shape[1]])
        self.nat_tf = tf.placeholder(tf.float32, shape=[None, self.nat.shape[1]])

        self.Rci_tf = tf.placeholder(tf.float32, shape=[None, self.Rci.shape[1]])
        self.Rat_tf = tf.placeholder(tf.float32, shape=[None, self.Rat.shape[1]])

        # Predict the values of output by a single forward propagation.
        # Also get AutoDiff coefficients from the same class method: net_Euler_STS
        [self.nci_pred, self.nat_pred, self.rho_pred, self.u_pred, self.p_pred, self.E_pred, self.Rci_pred, self.Rat_pred,
         self.e1, self.e2, self.e3, self.e4, self.enci, self.enat] = self.net_Euler_STS(self.x_tf)

        # MSE Normalization
        # The initial normalization terms are necessary to ensure that the
        # gradients don't get driven towards either the residual squared errors
        # or the MSE of the outputs. Basically, to ensure equal weightage to it
        # being 'trained to training data' as well as being 'Physics informed'
        rho_norm = np.amax(rho)
        u_norm   = np.amax(u)
        p_norm   = np.amax(p)
        E_norm   = np.amax(E)

        e1_norm  = rho_norm*u_norm        # e1 is continuity residual
        e2_norm  = p_norm                 # e2 is momentum   residual
        e3_norm  = E_norm*rho_norm*u_norm # e3 is energy     residual

        nci_norm = np.amax(nci)
        nat_norm = np.amax(nat)

        enci_norm = nci_norm *u_norm
        enat_norm = nat_norm*u_norm

        Rci_norm = np.amax(Rci)
        Rat_norm = np.amax(Rat)

        # Weight factor... let's see its impact by varying it w = [0:100].
        # If is it 0, then PINN -> NN and we do not physically inform the NN.
        w = 0.

        # Define Cost function or the Loss
        # In this case I have set the mean squared error of the ouputs to be
        # the loss and commented the PINN residual arguements. Uncommenting the
        # residual expressions will result in a true Phyics Informed Neural
        # Network, otherwise, it is just a data trained Neural network
        self.loss = tf.reduce_sum(tf.square(self.u_tf   - self.u_pred)) /(u_norm**2) + \
                    tf.reduce_sum(tf.square(self.rho_tf - self.rho_pred))/(rho_norm**2) + \
                    tf.reduce_sum(tf.square(self.p_tf   - self.p_pred)) /(p_norm**2) + \
                    tf.reduce_sum(tf.square(self.E_tf   - self.E_pred)) /(E_norm**2) + \
                    tf.reduce_sum(tf.square(self.nci_tf - self.nci_pred))/(nci_norm**2) + \
                    tf.reduce_sum(tf.square(self.nat_tf - self.nat_pred))/(nat_norm**2) + \
                    tf.reduce_sum(tf.square(self.Rci_tf - self.Rci_pred))/(Rci_norm**2) + \
                    tf.reduce_sum(tf.square(self.Rat_tf - self.Rat_pred))/(Rat_norm**2) + \
                    w*tf.reduce_sum(tf.square(self.e1))/(e1_norm**2)                    + \
                    w*tf.reduce_sum(tf.square(self.e2))/(e2_norm**2)                    + \
                    w*tf.reduce_sum(tf.square(self.e3))/(e3_norm**2)                    + \
                    w*tf.reduce_sum(tf.square(self.e4))/(p_norm**2)                     + \
                    w*tf.reduce_sum(tf.square(self.enci))/(enci_norm**2) + \
                    w*tf.reduce_sum(tf.square(self.enat))/(enat_norm**2)

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

        nci = nci_rho_u_p_E[:,0:47]
        nat = nci_rho_u_p_E[:,47:48]

        rho = nci_rho_u_p_E[:,48:49]
        u   = nci_rho_u_p_E[:,49:50]
        p   = nci_rho_u_p_E[:,50:51]
        E   = nci_rho_u_p_E[:,51:52]

        Rci = nci_rho_u_p_E[:,52:99]
        Rat = nci_rho_u_p_E[:,99:100]

        nci_u_x = tf.gradients(nci*u, x)[0]
        nat_u_x = tf.gradients(nat*u, x)[0]

        # autodiff gradient #1
        mass_flow_grad = tf.gradients(rho*u, x)[0]

        # autodiff gradient #2
        momentum_grad = tf.gradients((rho*u*u + p), x)[0]

        # autodiff gradient #3
        energy_grad = tf.gradients((rho*E + p)*u, x)[0]

        # state residual
        gamma = 1.4
        state_res = p - rho*(gamma-1.0)*(E-0.5*gamma*u*u)

        eqnci  =  nci_u_x  - Rci
        eqnat =  nat_u_x - Rat

        eq1 =  mass_flow_grad
        eq2 =  momentum_grad
        eq3 =  energy_grad
        eq4 =  state_res

        return nci, nat, rho, u, p, E, Rci, Rat, eq1, eq2, eq3, eq4, eqnci, eqnat

    # callback method just prints the current loss (cost) value of the network.
    def callback(self, loss):
        print('Loss: %.3e' % (loss))

    tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
    baseline=None, restore_best_weights=False
    )

    # Train method actually trains the network weights based on the target
    # of minimizing the loss. tf_dict is defined as the set of input and
    # ideal output parameters for the given data in loop. For the given
    # iterations 'nIter' (variable), the train_op_Adam session is run.
    def train(self, nIter):

        tf_dict = {self.x_tf: self.x, self.nci_tf: self.nci, self.nat_tf: self.nat,
                   self.rho_tf: self.rho, self.u_tf: self.u, self.p_tf: self.p, self.E_tf: self.E,
                   self.Rci_tf: self.Rci, self.Rat_tf: self.Rat}

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
                #res1 = self.sess.run(self.e1, tf_dict)
                #res2 = self.sess.run(self.e2, tf_dict)
                #res3 = self.sess.run(self.e3, tf_dict)

                print('Iter: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                #print('Mass Residual: %f\t\tMomentum Residual: %f\tEnergy Residual: %f'
                #    %(sum(map(lambda a:a*a,res1))/len(res1), sum(map(lambda a:a*a,res2))/len(res2), sum(map(lambda a:a*a,res3))/len(res3)))
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

        nci_test = self.sess.run(self.nci_pred, tf_dict)
        nat_test = self.sess.run(self.nat_pred, tf_dict)

        rho_test = self.sess.run(self.rho_pred, tf_dict)
        u_test   = self.sess.run(self.u_pred,   tf_dict)
        p_test   = self.sess.run(self.p_pred,   tf_dict)
        E_test   = self.sess.run(self.E_pred,   tf_dict)

        Rci_test = self.sess.run(self.Rci_pred, tf_dict)
        Rat_test = self.sess.run(self.Rat_pred, tf_dict)

        return nci_test, nat_test, rho_test, u_test, p_test, E_test, Rci_test, Rat_test

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

# Main function, inside which there are the training and testing commands.
if __name__ == "__main__":

    # Neural Network Architecture
    # layers is a vector of all the node in each of the neural network layers
    # First value, 1 respresents the input layer with 1 parameter (x) while
    # last value 100 is the number of outputs desired
    layers = [1, 100, 100]
    ###layers = [1, 10, 25, 15, 100]
    #layers = [1, 20, 20, 20, 20, 20, 20, 20, 20, 20, 100]
    #layers = [1, 40, 40, 40, 40, 100]
    #layers = [1, 10, 10, 10, 100]
    #layers = [1, 15, 25, 25, 15, 100]
    #layers = [1, 10,10,10,10,10,10,10, 100]

    # Load Data
    # The Matlab generated data was stored in file name 'dataset_STS.txt', which
    # is read by numpy and stored as a new variable by name data.
    data = np.loadtxt('data/dataset_STS.txt')

    # The training set length N_train is taken to be 85% of the entire dataset.
    # The rest 15% will be used as test set to validate the result of the training.
    #train_frac = 0.01
    #N_train = int(train_frac*data.shape[0])

    # idx is a random numbers vector, which will be used to randomly pick 85% of
    # the data from the dataset.
    #idx = np.random.choice(range(data.shape[0]), size=(N_train,), replace=False)

    # The rest is mere slicing of dataset to get all required parameters.

    x   = data[:,0:1].flatten()[:,None]
    nci = data[:,1:48].flatten()[:,None]
    nat = data[:,48:49].flatten()[:,None]

    rho = data[:,49:50].flatten()[:,None]
    u   = data[:,50:51].flatten()[:,None]
    p   = data[:,51:52].flatten()[:,None]
    E   = data[:,52:53].flatten()[:,None]

    Rci = data[:,53:100].flatten()[:,None]
    Rat = data[:,100:101].flatten()[:,None]

    [x_train,   x_test,
     nci_train, nci_test,
     nat_train, nat_test,
     rho_train, rho_test,
     u_train,   u_test,
     p_train,   p_test,
     E_train,   E_test,
     Rci_train, Rci_test,
     Rat_train, Rat_test] = train_test_split(x, nci, nat, rho, u, p, E, Rci, Rat, test_size=0.15, random_state=0)

    sc_x   = MinMaxScaler(); sc_x.fit(x_train)    ; x_train   = sc_x.transform(x_train)
    sc_nci = MinMaxScaler(); sc_nci.fit(nci_train); nci_train = sc_nci.transform(nci_train)
    sc_nat = MinMaxScaler(); sc_nat.fit(nat_train); nat_train = sc_nat.transform(nat_train)
    sc_rho = MinMaxScaler(); sc_rho.fit(rho_train); rho_train = sc_rho.transform(rho_train)
    sc_u   = MinMaxScaler(); sc_u.fit(u_train)    ; u_train   = sc_u.transform(u_train)
    sc_p   = MinMaxScaler(); sc_p.fit(p_train)    ; p_train   = sc_p.transform(p_train)
    sc_E   = MinMaxScaler(); sc_E.fit(E_train)    ; E_train   = sc_E.transform(E_train)
    sc_Rci = MinMaxScaler(); sc_Rci.fit(Rci_train); Rci_train = sc_Rci.transform(Rci_train)
    sc_Rat = MinMaxScaler(); sc_Rat.fit(Rat_train); Rat_train = sc_Rat.transform(Rat_train)

    x_test   = sc_x.transform(x_test)
    nci_test = sc_nci.transform(nci_test)
    nat_test = sc_nat.transform(nat_test)
    rho_test = sc_rho.transform(rho_test)
    u_test   = sc_u.transform(u_test)
    p_test   = sc_p.transform(p_test)
    E_test   = sc_E.transform(E_test)
    Rci_test = sc_Rci.transform(Rci_test)
    Rat_test = sc_Rat.transform(Rat_test)

    # Training the NN based on the training set, randomly chosen above model
    # = PINN(..) passes the necessary training data to the 'NN' class (model
    # here being an instance of the NN class) in order to initialize all the
    # parameters as well as the NN architecture including random initialization
    # of weights and biases.
    model = PINN(x_train,
                 nci_train, nat_train, rho_train, u_train, p_train, E_train, Rci_train, Rat_train, layers)

    model.train(1000)

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
    [nci_pred, nat_pred, rho_pred, u_pred, p_pred, E_pred, Rci_pred, Rat_pred] = model.predict(x_test)

# Error
# Normal relative error is printed for each variable
error_nci = np.linalg.norm(nci_test -nci_pred ,2)/np.linalg.norm(nci_test ,2)
error_nat = np.linalg.norm(nat_test-nat_pred,2)/np.linalg.norm(nat_test,2)
error_rho = np.linalg.norm(rho_test-rho_pred,2)/np.linalg.norm(rho_test,2)
error_u   = np.linalg.norm(u_test-u_pred,2)/np.linalg.norm(u_test,2)
error_p   = np.linalg.norm(p_test-p_pred,2)/np.linalg.norm(p_test,2)
error_E   = np.linalg.norm(E_test-E_pred,2)/np.linalg.norm(E_test,2)
error_Rci = np.linalg.norm(Rci_test -Rci_pred ,2)/np.linalg.norm(Rci_test ,2)
error_Rat = np.linalg.norm(Rat_test-Rat_pred,2)/np.linalg.norm(Rat_test,2)

print("Test Error in nci: "+str(error_nci))
print("Test Error in rho: "+str(error_rho))
print("Test Error in u: "+str(error_u))
print("Test Error in p: "+str(error_p))
print("Test Error in E: "+str(error_E))
print("Test Error in Rci: "+str(error_Rci))

x_train_sb   = sc_x.inverse_transform(x_train)
nci_train_sb = sc_nci.inverse_transform(nci_train)
nat_train_sb = sc_nat.inverse_transform(nat_train)
rho_train_sb = sc_rho.inverse_transform(rho_train)
u_train_sb   = sc_u.inverse_transform(u_train)
p_train_sb   = sc_p.inverse_transform(p_train)
E_train_sb   = sc_E.inverse_transform(E_train)
Rci_train_sb = sc_Rci.inverse_transform(Rci_train)
Rat_train_sb = sc_Rat.inverse_transform(Rat_train)

x_test_sb   = sc_x.inverse_transform(x_test)
nci_test_sb = sc_nci.inverse_transform(nci_test)
nat_test_sb = sc_nat.inverse_transform(nat_test)
rho_test_sb = sc_rho.inverse_transform(rho_test)
u_test_sb   = sc_u.inverse_transform(u_test)
p_test_sb   = sc_p.inverse_transform(p_test)
E_test_sb   = sc_E.inverse_transform(E_test)
Rci_test_sb = sc_Rci.inverse_transform(Rci_test)
Rat_test_sb = sc_Rat.inverse_transform(Rat_test)

nci_pred_sb = sc_nci.inverse_transform(nci_pred)
nat_pred_sb = sc_nat.inverse_transform(nat_pred)
rho_pred_sb = sc_rho.inverse_transform(rho_pred)
u_pred_sb   = sc_u.inverse_transform(u_pred)
p_pred_sb   = sc_p.inverse_transform(p_pred)
E_pred_sb   = sc_E.inverse_transform(E_pred)
Rci_pred_sb = sc_Rci.inverse_transform(Rci_pred)
Rat_pred_sb = sc_Rat.inverse_transform(Rat_pred)

# Plot Nci
plt.plot(x_test_sb, n3_pred_sb, 'o', color='black', label='NN, i=3', linewidth=4, markersize=5, fillstyle='none')
plt.plot(x_test_sb, n3_test_sb, 'o', color='red',   label='Exact, i=3', markersize=4)
plt.plot(x_test_sb, n6_pred_sb, 'o', color='black', label='NN, i=6', linewidth=4, markersize=5, fillstyle='none')
plt.plot(x_test_sb, n6_test_sb, 'o', color='blue',   label='Exact, i=6', markersize=4)
plt.plot(x_test_sb, n9_pred_sb, 'o', color='black', label='NN, i=9', linewidth=4, markersize=5, fillstyle='none')
plt.plot(x_test_sb, n9_test_sb, 'o', color='green',   label='Exact, i=9', markersize=4)
plt.plot(x_test_sb, n12_pred_sb, 'o', color='black', label='NN, i=12', linewidth=4, markersize=5, fillstyle='none')
plt.plot(x_test_sb, n12_test_sb, 'o', color='magenta',   label='Exact, i=12', markersize=4)
plt.plot(x_test_sb, n15_pred_sb, 'o', color='black', label='NN, i=15', linewidth=4, markersize=5, fillstyle='none')
plt.plot(x_test_sb, n15_test_sb, 'o', color='yellow',   label='Exact, i=15', markersize=4)
#plt.title('Comparison of NN and Exact solution for Molecular Number Density')
plt.xlabel('X [mm]')
plt.ylabel('$n_{ci}$ $[m^-3]$')
#plt.legend()
plt.tight_layout()
savefig('./figures/Nci', crop = False)
plt.show()

# Plot Nat
plt.plot(x_test_sb, nat_pred_sb, 'o', color='black', label='NN', linewidth=2, markersize=5, fillstyle='none')
plt.plot(x_test_sb, nat_test_sb, 'o', color='red',   label='Exact', markersize=4)
#plt.title('Comparison of NN and Exact solution for Atomic Number Density')
plt.xlabel('X [mm]')
plt.ylabel('$n_{at}$ $[m^-3]$')
plt.legend()
plt.tight_layout()
savefig('./figures/Nat', crop = False)
plt.show()

# Plot RHO
plt.plot(x_test_sb, rho_pred_sb, 'o', color='black', label='NN', linewidth=2, markersize=5, fillstyle='none')
plt.plot(x_test_sb, rho_test_sb, 'o', color='red',   label='Exact', markersize=4)
#plt.title('Comparison of NN and Exact solution for Density')
plt.xlabel('X [mm]')
plt.ylabel(r'$\rho$ $[kg/m^3]$')
plt.legend()
plt.tight_layout()
savefig('./figures/RHO', crop = False)
plt.show()

# Plot P
plt.plot(x_test_sb, p_pred_sb, 'o', color='black', label='NN', linewidth=2, markersize=5, fillstyle='none')
plt.plot(x_test_sb, p_test_sb, 'o', color='red',   label='Exact', markersize=4)
#plt.title('Comparison of NN and Exact solution for Pressure')
plt.xlabel('X [mm]')
plt.ylabel('P [Pa]')
plt.legend()
plt.tight_layout()
savefig('./figures/P', crop = False)
plt.show()

# Plot U
plt.plot(x_test_sb, u_pred_sb, 'o', color='black', label='NN', linewidth=2, markersize=5, fillstyle='none')
plt.plot(x_test_sb, u_test_sb, 'o', color='red',   label='Exact', markersize=4)
#plt.title('Comparison of NN and Exact solution for Velocity')
plt.xlabel('X [mm]')
plt.ylabel('U [m/s]')
plt.legend()
plt.tight_layout()
savefig('./figures/U', crop = False)
plt.show()

# Plot E
plt.plot(x_test_sb, E_pred_sb, 'o', color='black', label='NN', linewidth=2, markersize=5, fillstyle='none')
plt.plot(x_test_sb, E_test_sb, 'o', color='red',   label='Exact', markersize=4)
#plt.title('Comparison of NN and Exact solution for Energy')
plt.xlabel('X [mm]')
plt.ylabel('E [eV]')
plt.legend()
plt.tight_layout()
savefig('./figures/E', crop = False)
plt.show()

# Plot Rci
plt.plot(x_test_sb, R3_pred_sb,  'o', color='black',   label='NN, i=3',  linewidth=4,  markersize=5, fillstyle='none')
plt.plot(x_test_sb, R3_test_sb,  'o', color='red',     label='Exact, i=3',            markersize=4)
plt.plot(x_test_sb, R6_pred_sb,  'o', color='black',   label='NN, i=6',  linewidth=4,  markersize=5, fillstyle='none')
plt.plot(x_test_sb, R6_test_sb,  'o', color='blue',    label='Exact, i=6',            markersize=4)
plt.plot(x_test_sb, R9_pred_sb,  'o', color='black',   label='NN, i=9',  linewidth=4,  markersize=5, fillstyle='none')
plt.plot(x_test_sb, R9_test_sb,  'o', color='green',   label='Exact, i=9',            markersize=4)
plt.plot(x_test_sb, R12_pred_sb, 'o', color='black',   label='NN, i=12', linewidth=4, markersize=5, fillstyle='none')
plt.plot(x_test_sb, R12_test_sb, 'o', color='magenta', label='Exact, i=12',           markersize=4)
plt.plot(x_test_sb, R15_pred_sb, 'o', color='black',   label='NN, i=15', linewidth=4, markersize=5, fillstyle='none')
plt.plot(x_test_sb, R15_test_sb, 'o', color='yellow',  label='Exact, i=15',           markersize=4)
#plt.title('Comparison of NN and Exact solution for $R_{ci}$')
plt.xlabel('X [mm]')
plt.ylabel(r'$R_{ci} [J/m^3/s]$')
#plt.legend()
plt.tight_layout()
savefig('./figures/Rci', crop = False)
plt.show()

# Plot Rat
plt.plot(x_test_sb, Rat_pred_sb, 'o', color='black', label='NN', linewidth=2, markersize=5, fillstyle='none')
plt.plot(x_test_sb, Rat_test_sb, 'o', color='red',   label='Exact',           markersize=4 )
#plt.title('Comparison of NN and Exact solution for $R_{at}$')
plt.xlabel('X [mm]')
plt.ylabel(r'$R_{at} [J/m^3/s]$')
plt.legend()
plt.tight_layout()
savefig('./figures/Rat', crop='false')
plt.show()
