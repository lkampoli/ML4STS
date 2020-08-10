# Physics Informed Neural Network (PINN) in Tensorflow

# Importing necessary libraries Note that there are 2 important data handling and
# numerical calculation libraries: **numpy** and **scipy** alongside tensorflow.
# *Matplotlib* is necessary to plot and visualize data
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
import time
from pyDOE import lhs
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import csv

# Tensorflow random seed for initialization
np.random.seed(1234)
tf.compat.v1.set_random_seed(1234)

# Define the Class PINN which we are going to use
class PINN:

    # Initialize the class
    def __init__(self, x, t, rho, u, p, E, layers):

        # Create Input Matrix for the given training data point
        X = np.concatenate([x, t], 1)

        # Domain Boundary (min & max for normalization)
        self.lb = X.min(0)
        self.ub = X.max(0)

        self.X   = X

        # Training Data (class attribute definitions)
        self.x   = x
        self.t   = t
        self.rho = rho
        self.u   = u
        self.p   = p
        self.E   = E

        # Layers of NN
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
        self.t_tf   = tf.compat.v1.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.rho_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.rho.shape[1]])
        self.u_tf   = tf.compat.v1.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.p_tf   = tf.compat.v1.placeholder(tf.float32, shape=[None, self.p.shape[1]])
        self.E_tf   = tf.compat.v1.placeholder(tf.float32, shape=[None, self.E.shape[1]])

        # ... not used ...
        self.dummy_tf = tf.placeholder(tf.float32, shape=(None, layers[-1]))
        self.learning_rate = tf.placeholder(tf.float32, shape=[])

        # Predict the values of output by a single forward propagation. Also get
        # AutoDiff coefficients from the same class method: net_Euler
        [self.rho_pred,
        self.u_pred,
        self.p_pred,
        self.E_pred,
        self.e1,
        self.e2,
        self.e3,
        self.e4] = self.net_Euler(self.x_tf, self.t_tf)

        # MSE Normalization
        # The initial normalization terms are necessary to ensure that the
        # gradients don't get driven towards either the residual squared errors
        # or the MSE of the outputs. Basically, to ensure equal weightage to it
        # being 'trained to training data' as well as being 'Physics informed
        rho_norm = np.amax(rho)
        u_norm   = np.amax(u)
        p_norm   = np.amax(p)
        E_norm   = np.amax(E)
        #S_norm  = 1.0
        e1_norm  = rho_norm*u_norm        #*S_norm # e1 is continuity residual
        e2_norm  = p_norm                 #*S_norm # e2 is momentum   residual
        e3_norm  = E_norm*rho_norm*u_norm #*S_norm # e3 is energy     residual

        # Weight factor... let's see its impact by varying it w = [0:100].
        # If is it 0, then PINN -> NN and we do not physically inform the NN.
        w = 0.

        # Define Cost function or the Loss
        # In this case I have set the mean squared error of the ouputs to be
        # the loss and commented the PINN residual arguements. Uncommenting the
        # 4 residual expressions will result in a true Phyics Informed Neural
        # Network, otherwise, it is just a data trained Neural network
        # tf.reduce_mean(tf.pow(res_true - res_mom_u, 2)) TODO: try it!
        self.loss = tf.reduce_sum(tf.square(self.p_tf   - self.p_pred))  /(p_norm**2)   + \
                    tf.reduce_sum(tf.square(self.rho_tf - self.rho_pred))/(rho_norm**2) + \
                    tf.reduce_sum(tf.square(self.u_tf   - self.u_pred))  /(u_norm**2)   + \
                    tf.reduce_sum(tf.square(self.E_tf   - self.E_pred))  /(E_norm**2)   + \
                    w*tf.reduce_sum(tf.square(self.e2))/(e2_norm**2) + \
                    w*tf.reduce_sum(tf.square(self.e3))/(e3_norm**2) + \
                    w*tf.reduce_sum(tf.square(self.e1))/(e1_norm**2) + \
                    w*tf.reduce_sum(tf.square(self.e4))/(p_norm**2)

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
        #self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
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
    def net_Euler(self, x, t):

        rho_u_p_E = self.neural_net(tf.concat([x,t], 1), self.weights, self.biases)

        rho = rho_u_p_E[:,0:1]
        u   = rho_u_p_E[:,1:2]
        p   = rho_u_p_E[:,2:3]
        E   = rho_u_p_E[:,3:4]

        # temporal derivatives
        rho_t   = tf.gradients(rho,   t)[0]
        rho_u_t = tf.gradients(rho*u, t)[0]
        rho_E_t = tf.gradients(rho*E, t)[0]

        # spatial derivatives
        mass_flow_grad = tf.gradients(rho*u, x)[0]
        momentum_grad  = tf.gradients((rho*u*u + p), x)[0]
        energy_grad    = tf.gradients((rho*E + p)*u, x)[0]

        # state residual
        gamma = 1.4
        state_res = p - rho*(gamma-1.0)*(E-0.5*gamma*u*u)

        eq1 = rho_t   + mass_flow_grad
        eq2 = rho_u_t + momentum_grad
        eq3 = rho_E_t + energy_grad
        eq4 =           state_res

        return rho, u, p, E, eq1, eq2, eq3, eq4

    # callback method just prints the current loss (cost) value of the network.
    def callback(self, loss):
        print('Loss: %.3e' % (loss))

    # Train method actually trains the network weights based on the target
    # of minimizing the loss. tf_dict is defined as the set of input and
    # ideal output parameters for the given data in loop. For the given
    # iterations 'nIter' (variable), the train_op_Adam session is run.
    def train(self, nIter):

        tf_dict = {self.x_tf:   self.x,
                   self.t_tf:   self.t,
                   self.rho_tf: self.rho,
                   self.u_tf:   self.u,
                   self.p_tf:   self.p,
                   self.E_tf:   self.E}

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
                #loss_value = self.sess.run(self.loss, tf_dict)
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
    def predict(self, x_test, t_test):

        tf_dict  = {self.x_tf: x_test, self.t_tf: t_test}

        rho_test = self.sess.run(self.rho_pred, tf_dict)
        u_test   = self.sess.run(self.u_pred,   tf_dict)
        p_test   = self.sess.run(self.p_pred,   tf_dict)
        E_test   = self.sess.run(self.E_pred,   tf_dict)

        return rho_test, u_test, p_test, E_test


def plot_solution(X_star, u_star, index):

    lb = X_star.min(0)
    ub = X_star.max(0)

    nn = 1000

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
    # First value, 2 respresents the input layer with 2 parameters while
    # last value 4 is the number of outputs desired
    layers = [2, 10, 4]
    #layers = [2, 10, 25, 15, 4]
    #layers = [2, 20, 20, 20, 20, 20, 20, 20, 4]
    #layers = [2, 40, 40, 40, 40, 4]
    #layers = [2, 100, 100, 100, 100, 100, 100, 100, 100, 4]

    rho = []
    u   = []
    p   = []
    E   = []

    # Load Data
    # The Fortran generated data was stored in file name 'datashcok1d.txt', which
    # is read by numpy and stored as a new variable by name data.
    #data = np.loadtxt('data/datashock1d.txt') # too big
    data = np.loadtxt('data/datashock1d001.txt')
    #data = np.loadtxt('data/datashock1dlite.txt') # too small

    # The training set length N_train is taken to be 85% of the entire dataset.
    # The rest 15% will be used as test set to validate the result of the training.
    N_train = int(0.85*data.shape[0])

    # idx is a random numbers vector, which will be used to randomly pick 85% of
    # the data from the dataset.
    idx = np.random.choice(range(data.shape[0]), size=(N_train,), replace=False)

    # The rest is mere slicing of dataset to get all required parameters.
    # x
    x_train  = data[idx,0:1].flatten()[:,None]
    t_train  = data[idx,1:2].flatten()[:,None]
    XT_train = np.concatenate([x_train, t_train], 1)
    # y
    rho_train = data[idx,2:3].flatten()[:,None]
    u_train   = data[idx,3:4].flatten()[:,None]
    p_train   = data[idx,4:5].flatten()[:,None]
    E_train   = data[idx,5:6].flatten()[:,None]

    # Training the NN based on the training set, randomly chosen above model
    # = PINN(..) passes the necessary training data to the 'NN' class (model
    # here being an instance of the NN class) in order to initialize all the
    # parameters as well as the NN architecture including random initialization
    # of weights and biases.
    model = PINN(x_train, t_train, rho_train, u_train, p_train, E_train, layers)
    model.train(20000)

    #model.train2(num_epochs = 200, batch_size = 100, learning_rate=1e-3)
    #model.train2(num_epochs = 300, batch_size = 100, learning_rate=1e-4)
    #model.train2(num_epochs = 300, batch_size = 100, learning_rate=1e-5)
    #model.train2(num_epochs = 200, batch_size = 100, learning_rate=1e-6)

    # Plotting Loss
    plt.plot(loss_vector, label='Loss value')
    plt.legend()
    plt.title('Loss value over iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig('./figures/Loss', crop = False)

    # Test Data
    # Test the neural network performance using Test dataset "data1" generated
    # by eliminating the initally randomly selected rows from the data. The test
    # dataset "data1" is then sliced to get test parameter values.
    data1 = data
    data1 = np.delete(data1, idx, 0)
    # x
    x_test  = data1[:,0:1].flatten()[:,None]
    t_test  = data1[:,1:2].flatten()[:,None]
    XT_test = np.concatenate([x_test, t_test], 1)
    # y
    rho_test = data1[:,2:3].flatten()[:,None]
    u_test   = data1[:,3:4].flatten()[:,None]
    p_test   = data1[:,4:5].flatten()[:,None]
    E_test   = data1[:,5:6].flatten()[:,None]

    # Prediction
    # The input parameters of the test set are used to predict the pressure,
    # density, speed and specific energy for the given x and t by using the
    # .predict method.
    rho_pred, u_pred, p_pred, E_pred = model.predict(x_test, t_test)

    #RHO_pred = griddata(x_test, rho_pred.flatten(), (X, T), method='cubic')


# Error
# Normal relative error is printed for each variable
error_rho = np.linalg.norm(rho_test-rho_pred,2)/np.linalg.norm(rho_test,2)
error_u   = np.linalg.norm(u_test-u_pred,2)/np.linalg.norm(u_test,2)
error_p   = np.linalg.norm(p_test-p_pred,2)/np.linalg.norm(p_test,2)
error_E   = np.linalg.norm(E_test-E_pred,2)/np.linalg.norm(E_test,2)
print("Test Error in rho: "+str(error_rho))
print("Test Error in u: "+str(error_u))
print("Test Error in p: "+str(error_p))
print("Test Error in E: "+str(error_E))

#errors = np.sqrt((Y_test-Y_pred)**2/Y_test**2)
#mean_errors = np.mean(errors,0)
#std_errors = np.std(errors,0)

#Plotting
#a = 201
#for i in range (0, 3046, 1608):
#plt.plot(x_test[0:3046:a], rho_pred[0:3046:a], 'm', label='NN')
#plt.plot(x_test[0:3046:a], rho_test[0:3046:a], 'g', label='Exact')
#plt.title('Comparison of NN and Exact solution for Density')
#plt.xlabel('x')
#plt.ylabel('value')
#plt.legend()
#plt.show()
#savefig('./figures/Error', crop = False)


# Plotting
plot_solution(XT_test, rho_pred, 1)
savefig('./figures/rho_pred', crop = False)
plot_solution(XT_test, u_pred, 1)
savefig('./figures/u_pred', crop = False)
plot_solution(XT_test, p_pred, 1)
savefig('./figures/p_pred', crop = False)
plot_solution(XT_test, E_pred, 1)
savefig('./figures/E_pred', crop = False)

#    ############################# Plotting ###############################
#
#    fig, ax = newfig(1.0, 1.4)
#    ax.axis('off')
#
#    ####### Row 0: rho(t,x) ##################
#    gs0 = gridspec.GridSpec(1, 2)
#    gs0.update(top=1-0.06, bottom=1-1.0/3.0+0.06, left=0.15, right=0.85, wspace=0)
#    ax = plt.subplot(gs0[:, :])
#
#    h = ax.imshow(RR_star.T, interpolation='nearest', cmap='rainbow',
#                  #extent=[lb[0], ub[0], lb[1], ub[1]], #t_test.min(),t_test.max()
#                  #extent=[lb[1], ub[1], lb[0], ub[0]], #t_test.min(),t_test.max()
#                  extent=[t_test.min(),t_test.max(), lb[0], ub[0]],
#                  origin='lower', aspect='auto')
#    divider = make_axes_locatable(ax)
#    cax = divider.append_axes("right", size="5%", pad=0.05)
#    fig.colorbar(h, cax=cax)
#
#    ax.plot(XT_train[:,0], XT_train[:,1], 'kx', label = 'Data (%d points)' % (rho_train.shape[0]), markersize = 2, clip_on = False)
#
#    line = np.linspace(x_train.min(), x_train.max(), 2)[:,None]
#    #ax.plot(t_test[5]*np.ones((2,1)), line, 'w-', linewidth = 1)
#    #ax.plot(t_test[15]*np.ones((2,1)), line, 'w-', linewidth = 1)
#    #ax.plot(t_test[30]*np.ones((2,1)), line, 'w-', linewidth = 1)
#
#    ax.set_xlabel('$x$')
#    ax.set_ylabel('$t$')
#    ax.legend(loc='upper center', bbox_to_anchor=(1.0, -0.125), ncol=5, frameon=False)
#    #leg = ax.legend(frameon=False, loc = 'best')
#    #plt.setp(leg.get_texts(), color='w')
#    ax.set_title(r'$\rho(x,t)$', fontsize = 10)
#
#    ####### Row 1: rho(t,x) slices ##################
#    gs1 = gridspec.GridSpec(1, 4)
#    gs1.update(top=1-1.0/3.0-0.1, bottom=1.0-2.0/3.0, left=0.1, right=0.9, wspace=0.5)
#
#    ax = plt.subplot(gs1[0, 0])
#    # ax.plot(x_test,Exact[25,:], 'b-', linewidth = 2, label = 'Exact')
#    # ax.plot(x_test,rho_pred[25,:], 'r--', linewidth = 2, label = 'Prediction')
#    ax.set_xlabel('$x$')
#    ax.set_ylabel('Density')
#    ax.set_title('$t = 2.0$', fontsize = 10)
#    ax.axis('square')
#    ax.set_xlim([0.0,1.0])
#    ax.set_ylim([0.0,2.0])
#
#    ax = plt.subplot(gs1[0, 1])
#    # ax.plot(x_test,Exact[50,:], 'b-', linewidth = 2, label = 'Exact')
#    # ax.plot(x,U_pred[50,:], 'r--', linewidth = 2, label = 'Prediction')
#    ax.set_xlabel('$x$')
#    ax.set_ylabel('Velocity')
#    ax.axis('square')
#    ax.set_xlim([0.0,1.0])
#    ax.set_ylim([0.0,2.0])
#    ax.set_title('$t = 2.0$', fontsize = 10)
#    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
#
#    ax = plt.subplot(gs1[0, 2])
#    #ax.plot(x,Exact[75,:], 'b-', linewidth = 2, label = 'Exact')
#    #ax.plot(x,U_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
#    ax.set_xlabel('$x$')
#    ax.set_ylabel('Pressure')
#    ax.axis('square')
#    ax.set_xlim([0.0,1.0])
#    ax.set_ylim([0.0, 2.0])
#    ax.set_title('$t = 2.0$', fontsize = 10)
#
#    ax = plt.subplot(gs1[0, 3])
#    # ax.plot(x_test,Exact[50,:], 'b-', linewidth = 2, label = 'Exact')
#    # ax.plot(x,U_pred[50,:], 'r--', linewidth = 2, label = 'Prediction')
#    ax.set_xlabel('$x$')
#    ax.set_ylabel('Energy')
#    ax.axis('square')
#    ax.set_xlim([0.0,1.0])
#    ax.set_ylim([0.0,2.0])
#    ax.set_title('$t = 2.0$', fontsize = 10)
#    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
#
#    savefig('./figures/Euler_identification')

# Visualize and compare the results against CFD generated outputs: plt_u and
# plt_l are the upper and lower bounds of a visualizing window respectively.
# This defines a specific set of inputs for which the results will be plotted.
# Again, we predict the output values for the specific input window. We now
# plot the predicted ouputs by the network against the exact solution values
# by CFD using matplotlib

# x
x_test_plt  = data[:, 0:1].flatten()[:,None]
t_test_plt  = data[:, 1:2].flatten()[:,None]

# y
rho_test_plt = data[:, 2:3].flatten()[:,None]
u_test_plt   = data[:, 3:4].flatten()[:,None]
p_test_plt   = data[:, 4:5].flatten()[:,None]
E_test_plt   = data[:, 5:6].flatten()[:,None]

# Prediction (for plotting)
rho_pred_plt, u_pred_plt, p_pred_plt, E_pred_plt = model.predict(x_test_plt, t_test_plt)

# Note that these value should be changed if using a different dataset or a different time
c = 1
b = 200*c
a = 201

# Plot RHO
plt.plot(x_test_plt[0+b:len(x_test_plt)-1:a], rho_pred_plt[0+b:len(x_test_plt)-1:a], 'm', label='NN')
plt.plot(x_test_plt[0+b:len(x_test_plt)-1:a], rho_test_plt[0+b:len(x_test_plt)-1:a], 'g', label='Exact')
plt.title('Comparison of NN and Exact solution for Density at t = ' +str(t_test_plt[b]))
plt.xlabel('x')
plt.ylabel('value')
plt.legend()
plt.show()
savefig('./figures/predictedRHO.png', crop = False)

# Plot P
plt.plot(x_test_plt[0+b:len(x_test_plt)-1:a], p_pred_plt[0+b:len(x_test_plt)-1:a], 'm', label='NN')
plt.plot(x_test_plt[0+b:len(x_test_plt)-1:a], p_test_plt[0+b:len(x_test_plt)-1:a], 'g', label='Exact')
plt.title('Comparison of NN and Exact solution for Pressure at t = ' +str(t_test_plt[b]))
plt.xlabel('x')
plt.ylabel('value')
plt.legend()
plt.show()
savefig('./figures/predictedP', crop = False)

# Plot U
plt.plot(x_test_plt[0+b:len(x_test_plt)-1:a], u_pred_plt[0+b:len(x_test_plt)-1:a], 'm', label='NN')
plt.plot(x_test_plt[0+b:len(x_test_plt)-1:a], u_test_plt[0+b:len(x_test_plt)-1:a], 'g', label='Exact')
plt.title('Comparison of NN and Exact solution for Velocity at t = ' +str(t_test_plt[b]))
plt.xlabel('x')
plt.ylabel('value')
plt.legend()
plt.show()
savefig('./figures/predictedU', crop = False)

# Plot E
plt.plot(x_test_plt[0+b:len(x_test_plt)-1:a], E_pred_plt[0+b:len(x_test_plt)-1:a], 'm', label='NN')
plt.plot(x_test_plt[0+b:len(x_test_plt)-1:a], E_test_plt[0+b:len(x_test_plt)-1:a], 'g', label='Exact')
plt.title('Comparison of NN and Exact solution for Energy at t = ' +str(t_test_plt[b]))
plt.xlabel('x')
plt.ylabel('value')
plt.legend()
plt.show()
savefig('./figures/predictedE', crop = False)
