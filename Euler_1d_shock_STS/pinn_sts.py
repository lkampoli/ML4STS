
import time
import sys
sys.path.insert(0, './')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import product, combinations
import matplotlib.gridspec as gridspec

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Tensorflow random seed for initialization
np.random.seed(1234)
tf.set_random_seed(1234)


# Define the Class PINN which we are going to use
class PINN:

    # Initialize the class
    def __init__(self, x, n, rho, u, p, E, R, layers):

        # Create Input Matrix for the given training data point
        X = np.concatenate([x], 1)

        length = len(n)
        # min & max for normalization
        self.lb = X.min(0)
        self.ub = X.max(0)

        self.X  = X

        # class attribute definitions
        self.x = x

        self.n = []
        i=0
        while i<length:
          self.n.append(n[i])
          i=i+1

        self.rho = rho
        self.u   = u
        self.p   = p
        self.E   = E

        self.R = []
        i = 0
        while i<length:
          self.R.append(R[i])
          i=i+1

        self.layers = layers

        # Initialize_NN is another class method which is used to assign random
        # weights and bias terms to the network. This not only initializes the
        # network but also structures the sizes and values of all the weights and
        # biases that would be so required for the network defined by layers.
        self.weights, self.biases = self.initialize_NN(layers)

        # Define a session to run
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                                         log_device_placement=True))

        # Define tensors for each variable using tf.placeholder, with shape
        # similar to their numpy counterparts variable_Name
        self.x_tf   = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])

        self.rho_tf = tf.placeholder(tf.float32, shape=[None, self.rho.shape[1]])
        self.u_tf   = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.p_tf   = tf.placeholder(tf.float32, shape=[None, self.p.shape[1]])
        self.E_tf   = tf.placeholder(tf.float32, shape=[None, self.E.shape[1]])

        self.n_tf = [tf.placeholder(tf.float32, shape=[None, self.n[i].shape[1]]) for i in range(int(48))]
        self.R_tf = [tf.placeholder(tf.float32, shape=[None, self.R[i].shape[1]]) for i in range(int(48))]

        # Predict the values of output by a single forward propagation.
        # Also get AutoDiff coefficients from the same class method: net_Euler_STS
        [self.n_pred,
         #####
         self.rho_pred, self.u_pred, self.p_pred, self.E_pred,
         #####
         self.R_pred,
         #####
         self.e1, self.e2, self.e3, self.e4,
         #####
         self.en] = self.net_Euler_STS(self.x_tf)

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

        n_norm = [np.amax(n[i]) for i in range(int(48))]
        en_norm = [n_norm[i]*u_norm for i in range(int(48))]
        R_norm = [np.amax(R[i]) for i in range(int(48))]


        # Weight factor... let's see its impact by varying it w = [0:100].
        # If is it 0, then PINN -> NN and we do not physically inform the NN.
        w = 0.0

        # Define Cost function or the Loss
        # In this case I have set the mean squared error of the ouputs to be
        # the loss and commented the PINN residual arguements. Uncommenting the
        # residual expressions will result in a true Phyics Informed Neural
        # Network, otherwise, it is just a data trained Neural network
        self.loss = tf.reduce_sum(tf.square(self.u_tf   - self.u_pred)) /(u_norm**2) + \
                    tf.reduce_sum(tf.square(self.rho_tf - self.rho_pred)) /(rho_norm**2) + \
                    tf.reduce_sum(tf.square(self.p_tf   - self.p_pred)) /(p_norm**2) + \
                    tf.reduce_sum(tf.square(self.E_tf   - self.E_pred)) /(E_norm**2) + \
                    w*tf.reduce_sum(tf.square(self.e1))/(e1_norm**2) + \
                    w*tf.reduce_sum(tf.square(self.e2))/(e2_norm**2) + \
                    w*tf.reduce_sum(tf.square(self.e3))/(e3_norm**2) + \
                    w*tf.reduce_sum(tf.square(self.e4))/(p_norm**2)
        i = 0
        while i<48:
          self.loss = self.loss + tf.reduce_sum(tf.square(self.n_tf[i] - self.n_pred[i]))/(n_norm[i]**2) + \
                                  tf.reduce_sum(tf.square(self.R_tf[i] - self.R_pred[i]))/(R_norm[i]**2) + \
                                  w*tf.reduce_sum(tf.square(self.en[i]))/(en_norm[i]**2)
          i = i + 1

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
        self.optimizer_Adam = tf.train.AdamOptimizer()
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
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

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

        n = [nci_rho_u_p_E[:,i:i+1] for i in range(int(48))]

        rho = nci_rho_u_p_E[:,48:49]
        u   = nci_rho_u_p_E[:,49:50]
        p   = nci_rho_u_p_E[:,50:51]
        E   = nci_rho_u_p_E[:,51:52]

        R = [nci_rho_u_p_E[:,i+52:i+53] for i in range(int(48))]

        n_u_x = [tf.gradients(n[i] *u, x)[0] for i in range(int(48))]

        # autodiff gradient #1
        mass_flow_grad = tf.gradients(rho*u, x)[0]

        # autodiff gradient #2
        momentum_grad = tf.gradients((rho*u*u + p), x)[0]

        # autodiff gradient #3
        energy_grad = tf.gradients((rho*E + p)*u, x)[0]

        # state residual
        gamma = 1.4
        state_res = p - rho*(gamma-1.0)*(E-0.5*gamma*u*u)

        eqn = [n_u_x[i] - R[i] for i in range(int(48))]

        eq1 =  mass_flow_grad
        eq2 =  momentum_grad
        eq3 =  energy_grad
        eq4 =  state_res

        return n, \
               rho, u, p, E, \
               R, \
               eq1, eq2, eq3, eq4, \
               eqn

    # Callback method prints the current loss (cost) value of the NN
    def callback(self, loss):
        print('Loss: %.3e' % (loss))

    def train(self, nIter):
    #def train(self, nIter, learning_rate):

        tf_dict = {self.x_tf: self.x}
        tf_dict.update(dict(zip(self.n_tf, self.n)))
        tf_dict.update({self.rho_tf: self.rho, self.u_tf: self.u, self.p_tf: self.p, self.E_tf: self.E})
        tf_dict.update(dict(zip(self.R_tf, self.R)))

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
    # passed into the network after the training is completed. All the values are
    # returned to the call.
    def predict(self, x_test):

      tf_dict  = {self.x_tf: x_test}

      n_test   = [self.sess.run(self.n_pred[i], tf_dict) for i in range(int(48))]
      rho_test = self.sess.run(self.rho_pred, tf_dict)
      u_test   = self.sess.run(self.u_pred, tf_dict)
      p_test   = self.sess.run(self.p_pred, tf_dict)
      E_test   = self.sess.run(self.E_pred, tf_dict)
      R_test   = [self.sess.run(self.R_pred[i], tf_dict) for i in range(int(48))]

      return n_test, rho_test, u_test, p_test, E_test, R_test


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

# MAIN function
if __name__ == "__main__":

    # NN Architecture
    #layers = [1, 100, 100]
    layers = [1, 100, 100, 100, 100, 100]
    #layers = [1, 25, 50, 75, 50, 25, 100]

    # Loading data
    data = np.loadtxt("data/dataset_STS.txt")
    #data = np.loadtxt("data/bi_mix_sts/dataset_STS.txt")

    # Slicing of dataset to get all required parameters
    x   = data[:,0:1].flatten()[:,None]
    n   = [data[:,i:i+1].flatten()[:,None] for i in range(int(48))]
    rho = data[:,49:50].flatten()[:,None]
    u   = data[:,50:51].flatten()[:,None]
    p   = data[:,51:52].flatten()[:,None]
    E   = data[:,52:53].flatten()[:,None]
    R   = [data[:,i+53:i+54].flatten()[:,None] for i in range(int(48))]

#-------------------------------------------------------------------------------
    tts = train_test_split(x, *n, rho, u, p, E, *R, test_size=0.15, random_state=0)
    tts = np.array(tts)
    x_train   = tts[0]
    x_test    = tts[1]
    rho_train = tts[98]
    rho_test  = tts[99]
    u_train   = tts[100]
    u_test    = tts[101]
    p_train   = tts[102]
    p_test    = tts[103]
    E_train   = tts[104]
    E_test    = tts[105]
    i = 0
    j = 0
    n_train = []
    n_test  = []
    R_train = []
    R_test  = []
    while i<48:
      n_train.append(tts[j+2])
      n_test.append(tts[j+3])
      R_train.append(tts[j+106])
      R_test.append(tts[j+107])
      i = i + 1
      j = j + 2
# -------------do not touch it anymore------------------------------------------
    sc_x = MinMaxScaler(); sc_x.fit(x_train); x_train = sc_x.transform(x_train)
    sc_n = [MinMaxScaler() for i in range(int(48))]

    i = 0
    while i<48:
      sc_n[i].fit(n_train[i])
      n_train[i] = sc_n[i].transform(n_train[i])
      i = i + 1

    sc_rho = MinMaxScaler(); sc_rho.fit(rho_train); rho_train = sc_rho.transform(rho_train)
    sc_u   = MinMaxScaler(); sc_u.fit(u_train)    ; u_train   = sc_u.transform(u_train)
    sc_p   = MinMaxScaler(); sc_p.fit(p_train)    ; p_train   = sc_p.transform(p_train)
    sc_E   = MinMaxScaler(); sc_E.fit(E_train)    ; E_train   = sc_E.transform(E_train)
    sc_R   = [MinMaxScaler() for i in range(int(48))]

    i = 0
    while i<48:
      sc_R[i].fit(R_train[i])
      R_train[i] = sc_R[i].transform(R_train[i])
      i = i + 1

    x_test   = sc_x.transform(x_test)
    n_test   = [sc_n[i].transform(n_test[i]) for i in range(int(48))]
    rho_test = sc_rho.transform(rho_test)
    u_test   = sc_u.transform(u_test)
    p_test   = sc_p.transform(p_test)
    E_test   = sc_E.transform(E_test)
    R_test   = [sc_R[i].transform(R_test[i]) for i in range(int(48))]

    # Training the NN
    model = PINN(x_train,
                 n_train,
                 rho_train, u_train, p_train, E_train,
                 R_train, layers)

    model.train(1000)

    # Plotting Loss
    plt.plot(loss_vector, label='Loss value')
    plt.legend()
    plt.title('Loss value over iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()
    #plt.savefig(f"{images_dir}/Loss", crop = False)

    # Prediction
    [n_pred, rho_pred, u_pred, p_pred, E_pred, R_pred] = model.predict(x_test)

    # Normal relative error is printed for each variable
    error_n   = [np.linalg.norm(n_test[i] - n_pred[i], 2)/np.linalg.norm(n_test[i],2) for i in range(int(48))]
    error_rho = np.linalg.norm(rho_test-rho_pred,2)/np.linalg.norm(rho_test,2)
    error_u   = np.linalg.norm(u_test-u_pred,2)/np.linalg.norm(u_test,2)
    error_p   = np.linalg.norm(p_test-p_pred,2)/np.linalg.norm(p_test,2)
    error_E   = np.linalg.norm(E_test-E_pred,2)/np.linalg.norm(E_test,2)
    error_R   = [np.linalg.norm(R_test[i] - R_pred[i], 2)/np.linalg.norm(R_test[i],2) for i in range(int(48))]

    i = 0
    while i<48:
        errorN = error_n[i]
        errorR = error_R[i]
        print("Test Error in nci: "+str(errorN))
        print("Test Error in Rci: "+str(errorR))
        i=i+1
    #print("Test Error in nci: "+str(error_n[i] for i in range(int(48))))
    print("Test Error in rho: "+str(error_rho))
    print("Test Error in u: "+str(error_u))
    print("Test Error in p: "+str(error_p))
    print("Test Error in E: "+str(error_E))
    #print("Test Error in Rci: "+str(error_R[i] for i in range(int(48))))

    # inverse transform
    x_train_sb   = sc_x.inverse_transform(x_train)
    n_train_sb   = [sc_n[i].inverse_transform(n_train[i]) for i in range(int(48))]
    rho_train_sb = sc_rho.inverse_transform(rho_train)
    u_train_sb   = sc_u.inverse_transform(u_train)
    p_train_sb   = sc_p.inverse_transform(p_train)
    E_train_sb   = sc_E.inverse_transform(E_train)
    R_train_sb   = [sc_R[i].inverse_transform(R_train[i]) for i in range(int(48))]
    x_test_sb    = sc_x.inverse_transform(x_test)
    n_test_sb    = [sc_n[i].inverse_transform(n_test[i]) for i in range(int(48))]
    rho_test_sb  = sc_rho.inverse_transform(rho_test)
    u_test_sb    = sc_u.inverse_transform(u_test)
    p_test_sb    = sc_p.inverse_transform(p_test)
    E_test_sb    = sc_E.inverse_transform(E_test)
    R_test_sb    = [sc_R[i].inverse_transform(R_test[i]) for i in range(int(48))]
    n_pred_sb    = [sc_n[i].inverse_transform(n_pred[i]) for i in range(int(48))]
    rho_pred_sb  = sc_rho.inverse_transform(rho_pred)
    u_pred_sb    = sc_u.inverse_transform(u_pred)
    p_pred_sb    = sc_p.inverse_transform(p_pred)
    E_pred_sb    = sc_E.inverse_transform(E_pred)
    R_pred_sb    = [sc_R[i].inverse_transform(R_pred[i]) for i in range(int(48))]

    # Plot Nci
    plt.plot(x_test_sb, n_pred_sb[2], 'o', color='black', label='NN, i=3', linewidth=4, markersize=5, fillstyle='none')
    plt.plot(x_test_sb, n_test_sb[2], 'o', color='red',   label='Exact, i=3', markersize=4)
    plt.plot(x_test_sb, n_pred_sb[5], 'o', color='black', label='NN, i=6', linewidth=4, markersize=5, fillstyle='none')
    plt.plot(x_test_sb, n_test_sb[5], 'o', color='blue',   label='Exact, i=6', markersize=4)
    plt.plot(x_test_sb, n_pred_sb[8], 'o', color='black', label='NN, i=9', linewidth=4, markersize=5, fillstyle='none')
    plt.plot(x_test_sb, n_test_sb[8], 'o', color='green',   label='Exact, i=9', markersize=4)
    plt.plot(x_test_sb, n_pred_sb[11], 'o', color='black', label='NN, i=12', linewidth=4, markersize=5, fillstyle='none')
    plt.plot(x_test_sb, n_test_sb[11], 'o', color='magenta',   label='Exact, i=12', markersize=4)
    plt.plot(x_test_sb, n_pred_sb[14], 'o', color='black', label='NN, i=15', linewidth=4, markersize=5, fillstyle='none')
    plt.plot(x_test_sb, n_test_sb[14], 'o', color='yellow',   label='Exact, i=15', markersize=4)
    #plt.title('Comparison of NN and Exact solution for Molecular Number Density')
    plt.xlabel('X [mm]')
    plt.ylabel('$n_{ci}$ $[m^-3]$')
    #plt.legend()
    plt.tight_layout()
    #plt.savefig(f"{images_dir}/Nci", crop='false')
    plt.show()

    # Plot Nat
    plt.plot(x_test_sb, n_pred_sb[47], 'o', color='black', label='NN', linewidth=2, markersize=5, fillstyle='none')
    plt.plot(x_test_sb, n_test_sb[47], 'o', color='red',   label='Exact', markersize=4)
    #plt.title('Comparison of NN and Exact solution for Atomic Number Density')
    plt.xlabel('X [mm]')
    plt.ylabel('$n_{at}$ $[m^-3]$')
    plt.legend()
    plt.tight_layout()
    #plt.savefig(f"{images_dir}/Nat", crop='false')
    plt.show()

    # Plot RHO
    plt.plot(x_test_sb, rho_pred_sb, 'o', color='black', label='NN', linewidth=2, markersize=5, fillstyle='none')
    plt.plot(x_test_sb, rho_test_sb, 'o', color='red',   label='Exact', markersize=4)
    #plt.title('Comparison of NN and Exact solution for Density')
    plt.xlabel('X [mm]')
    plt.ylabel(r'$\rho$ $[kg/m^3]$')
    plt.legend()
    plt.tight_layout()
    #plt.savefig(f"{images_dir}/RHO", crop='false')
    plt.show()

    # Plot P
    plt.plot(x_test_sb, p_pred_sb, 'o', color='black', label='NN', linewidth=2, markersize=5, fillstyle='none')
    plt.plot(x_test_sb, p_test_sb, 'o', color='red',   label='Exact', markersize=4)
    #plt.title('Comparison of NN and Exact solution for Pressure')
    plt.xlabel('X [mm]')
    plt.ylabel('P [Pa]')
    plt.legend()
    plt.tight_layout()
    #plt.savefig(f"{images_dir}/P", crop='false')
    plt.show()

    # Plot U
    plt.plot(x_test_sb, u_pred_sb, 'o', color='black', label='NN', linewidth=2, markersize=5, fillstyle='none')
    plt.plot(x_test_sb, u_test_sb, 'o', color='red',   label='Exact', markersize=4)
    #plt.title('Comparison of NN and Exact solution for Velocity')
    plt.xlabel('X [mm]')
    plt.ylabel('U [m/s]')
    plt.legend()
    plt.tight_layout()
    #plt.savefig(f"{images_dir}/U", crop='false')
    plt.show()

    # Plot E
    plt.plot(x_test_sb, E_pred_sb, 'o', color='black', label='NN', linewidth=2, markersize=5, fillstyle='none')
    plt.plot(x_test_sb, E_test_sb, 'o', color='red',   label='Exact', markersize=4)
    #plt.title('Comparison of NN and Exact solution for Energy')
    plt.xlabel('X [mm]')
    plt.ylabel('E [eV]')
    plt.legend()
    plt.tight_layout()
    #plt.savefig(f"{images_dir}/E", crop='false')
    plt.show()

    # Plot Rci
    plt.plot(x_test_sb, R_pred_sb[2],  'o', color='black',   label='NN, i=3',  linewidth=4,  markersize=5, fillstyle='none')
    plt.plot(x_test_sb, R_test_sb[2],  'o', color='red',     label='Exact, i=3',            markersize=4)
    plt.plot(x_test_sb, R_pred_sb[5],  'o', color='black',   label='NN, i=6',  linewidth=4,  markersize=5, fillstyle='none')
    plt.plot(x_test_sb, R_test_sb[5],  'o', color='blue',    label='Exact, i=6',            markersize=4)
    plt.plot(x_test_sb, R_pred_sb[8],  'o', color='black',   label='NN, i=9',  linewidth=4,  markersize=5, fillstyle='none')
    plt.plot(x_test_sb, R_test_sb[8],  'o', color='green',   label='Exact, i=9',            markersize=4)
    plt.plot(x_test_sb, R_pred_sb[11], 'o', color='black',   label='NN, i=12', linewidth=4, markersize=5, fillstyle='none')
    plt.plot(x_test_sb, R_test_sb[11], 'o', color='magenta', label='Exact, i=12',           markersize=4)
    plt.plot(x_test_sb, R_pred_sb[14], 'o', color='black',   label='NN, i=15', linewidth=4, markersize=5, fillstyle='none')
    plt.plot(x_test_sb, R_test_sb[14], 'o', color='yellow',  label='Exact, i=15',           markersize=4)
    #plt.title('Comparison of NN and Exact solution for $R_{ci}$')
    plt.xlabel('X [mm]')
    plt.ylabel(r'$R_{ci} [J/m^3/s]$')
    #plt.legend()
    plt.tight_layout()
    #plt.savefig(f"{images_dir}/Rci", crop='false')
    plt.show()

    # Plot Rat
    plt.plot(x_test_sb, R_pred_sb[47], 'o', color='black', label='NN', linewidth=2, markersize=5, fillstyle='none')
    plt.plot(x_test_sb, R_test_sb[47], 'o', color='red',   label='Exact',           markersize=4 )
    #plt.title('Comparison of NN and Exact solution for $R_{at}$')
    plt.xlabel('X [mm]')
    plt.ylabel(r'$R_{at} [J/m^3/s]$')
    plt.legend()
    plt.tight_layout()
    #plt.savefig(f"{images_dir}/Rat", crop='false')
    plt.show()
