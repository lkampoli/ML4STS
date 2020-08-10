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

# Tensorflow random seed for initialization
np.random.seed(1234)
tf.compat.v1.set_random_seed(1234)

# Define the Class PINN which we are going to use
class PINN:

    # Initialize the class
    def __init__(self,
                 x, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16, n17, n18,
                 n19, n20, n21, n22, n23, n24, n25, n26, n27, n28, n29, n30, n31, n32, n33, n34,
                 n35, n36, n37, n38, n39, n40, n41, n42, n43, n44, n45, n46, n47, nat,
                 rho, u, p, E,
                 R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16, R17, R18,
                 R19, R20, R21, R22, R23, R24, R25, R26, R27, R28, R29, R30, R31, R32, R33, R34,
                 R35, R36, R37, R38, R39, R40, R41, R42, R43, R44, R45, R46, R47, Rat, layers):

        # Create Input Matrix for the given training data point
        X = np.concatenate([x], 1)

        # min & max for normalization
        self.lb = X.min(0)
        self.ub = X.max(0)

        self.X  = X

        # class attribute definitions
        self.x   = x

        self.n1  = n1
        self.n2  = n2
        self.n3  = n3
        self.n4  = n4
        self.n5  = n5
        self.n6  = n6
        self.n7  = n7
        self.n8  = n8
        self.n9  = n9
        self.n10 = n10
        self.n11 = n11
        self.n12 = n12
        self.n13 = n13
        self.n14 = n14
        self.n15 = n15
        self.n16 = n16
        self.n17 = n17
        self.n18 = n18
        self.n19 = n19
        self.n20 = n20
        self.n21 = n21
        self.n22 = n22
        self.n23 = n23
        self.n24 = n24
        self.n25 = n25
        self.n26 = n26
        self.n27 = n27
        self.n28 = n28
        self.n29 = n29
        self.n30 = n30
        self.n31 = n31
        self.n32 = n32
        self.n33 = n33
        self.n34 = n34
        self.n35 = n35
        self.n36 = n36
        self.n37 = n37
        self.n38 = n38
        self.n39 = n39
        self.n40 = n40
        self.n41 = n41
        self.n42 = n42
        self.n43 = n43
        self.n44 = n44
        self.n45 = n45
        self.n46 = n46
        self.n47 = n47
        self.nat = nat

        self.rho = rho
        self.u   = u
        self.p   = p
        self.E   = E

        self.R1  = R1
        self.R2  = R2
        self.R3  = R3
        self.R4  = R4
        self.R5  = R5
        self.R6  = R6
        self.R7  = R7
        self.R8  = R8
        self.R9  = R9
        self.R10 = R10
        self.R11 = R11
        self.R12 = R12
        self.R13 = R13
        self.R14 = R14
        self.R15 = R15
        self.R16 = R16
        self.R17 = R17
        self.R18 = R18
        self.R19 = R19
        self.R20 = R20
        self.R21 = R21
        self.R22 = R22
        self.R23 = R23
        self.R24 = R24
        self.R25 = R25
        self.R26 = R26
        self.R27 = R27
        self.R28 = R28
        self.R29 = R29
        self.R30 = R30
        self.R31 = R31
        self.R32 = R32
        self.R33 = R33
        self.R34 = R34
        self.R35 = R35
        self.R36 = R36
        self.R37 = R37
        self.R38 = R38
        self.R39 = R39
        self.R40 = R40
        self.R41 = R41
        self.R42 = R42
        self.R43 = R43
        self.R44 = R44
        self.R45 = R45
        self.R46 = R46
        self.R47 = R47
        self.Rat = Rat

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

        self.n1_tf  = tf.placeholder(tf.float32, shape=[None, self.n1.shape[1]])
        self.n2_tf  = tf.placeholder(tf.float32, shape=[None, self.n2.shape[1]])
        self.n3_tf  = tf.placeholder(tf.float32, shape=[None, self.n3.shape[1]])
        self.n4_tf  = tf.placeholder(tf.float32, shape=[None, self.n4.shape[1]])
        self.n5_tf  = tf.placeholder(tf.float32, shape=[None, self.n5.shape[1]])
        self.n6_tf  = tf.placeholder(tf.float32, shape=[None, self.n6.shape[1]])
        self.n7_tf  = tf.placeholder(tf.float32, shape=[None, self.n7.shape[1]])
        self.n8_tf  = tf.placeholder(tf.float32, shape=[None, self.n8.shape[1]])
        self.n9_tf  = tf.placeholder(tf.float32, shape=[None, self.n9.shape[1]])
        self.n10_tf = tf.placeholder(tf.float32, shape=[None, self.n10.shape[1]])
        self.n11_tf = tf.placeholder(tf.float32, shape=[None, self.n11.shape[1]])
        self.n12_tf = tf.placeholder(tf.float32, shape=[None, self.n12.shape[1]])
        self.n13_tf = tf.placeholder(tf.float32, shape=[None, self.n13.shape[1]])
        self.n14_tf = tf.placeholder(tf.float32, shape=[None, self.n14.shape[1]])
        self.n15_tf = tf.placeholder(tf.float32, shape=[None, self.n15.shape[1]])
        self.n16_tf = tf.placeholder(tf.float32, shape=[None, self.n16.shape[1]])
        self.n17_tf = tf.placeholder(tf.float32, shape=[None, self.n17.shape[1]])
        self.n18_tf = tf.placeholder(tf.float32, shape=[None, self.n18.shape[1]])
        self.n19_tf = tf.placeholder(tf.float32, shape=[None, self.n19.shape[1]])
        self.n20_tf = tf.placeholder(tf.float32, shape=[None, self.n20.shape[1]])
        self.n21_tf = tf.placeholder(tf.float32, shape=[None, self.n21.shape[1]])
        self.n22_tf = tf.placeholder(tf.float32, shape=[None, self.n22.shape[1]])
        self.n23_tf = tf.placeholder(tf.float32, shape=[None, self.n23.shape[1]])
        self.n24_tf = tf.placeholder(tf.float32, shape=[None, self.n24.shape[1]])
        self.n25_tf = tf.placeholder(tf.float32, shape=[None, self.n25.shape[1]])
        self.n26_tf = tf.placeholder(tf.float32, shape=[None, self.n26.shape[1]])
        self.n27_tf = tf.placeholder(tf.float32, shape=[None, self.n27.shape[1]])
        self.n28_tf = tf.placeholder(tf.float32, shape=[None, self.n28.shape[1]])
        self.n29_tf = tf.placeholder(tf.float32, shape=[None, self.n29.shape[1]])
        self.n30_tf = tf.placeholder(tf.float32, shape=[None, self.n30.shape[1]])
        self.n31_tf = tf.placeholder(tf.float32, shape=[None, self.n31.shape[1]])
        self.n32_tf = tf.placeholder(tf.float32, shape=[None, self.n32.shape[1]])
        self.n33_tf = tf.placeholder(tf.float32, shape=[None, self.n33.shape[1]])
        self.n34_tf = tf.placeholder(tf.float32, shape=[None, self.n34.shape[1]])
        self.n35_tf = tf.placeholder(tf.float32, shape=[None, self.n35.shape[1]])
        self.n36_tf = tf.placeholder(tf.float32, shape=[None, self.n36.shape[1]])
        self.n37_tf = tf.placeholder(tf.float32, shape=[None, self.n37.shape[1]])
        self.n38_tf = tf.placeholder(tf.float32, shape=[None, self.n38.shape[1]])
        self.n39_tf = tf.placeholder(tf.float32, shape=[None, self.n39.shape[1]])
        self.n40_tf = tf.placeholder(tf.float32, shape=[None, self.n40.shape[1]])
        self.n41_tf = tf.placeholder(tf.float32, shape=[None, self.n41.shape[1]])
        self.n42_tf = tf.placeholder(tf.float32, shape=[None, self.n42.shape[1]])
        self.n43_tf = tf.placeholder(tf.float32, shape=[None, self.n43.shape[1]])
        self.n44_tf = tf.placeholder(tf.float32, shape=[None, self.n44.shape[1]])
        self.n45_tf = tf.placeholder(tf.float32, shape=[None, self.n45.shape[1]])
        self.n46_tf = tf.placeholder(tf.float32, shape=[None, self.n46.shape[1]])
        self.n47_tf = tf.placeholder(tf.float32, shape=[None, self.n47.shape[1]])
        self.nat_tf = tf.placeholder(tf.float32, shape=[None, self.nat.shape[1]])

        self.R1_tf  = tf.placeholder(tf.float32, shape=[None, self.R1.shape[1]])
        self.R2_tf  = tf.placeholder(tf.float32, shape=[None, self.R2.shape[1]])
        self.R3_tf  = tf.placeholder(tf.float32, shape=[None, self.R3.shape[1]])
        self.R4_tf  = tf.placeholder(tf.float32, shape=[None, self.R4.shape[1]])
        self.R5_tf  = tf.placeholder(tf.float32, shape=[None, self.R5.shape[1]])
        self.R6_tf  = tf.placeholder(tf.float32, shape=[None, self.R6.shape[1]])
        self.R7_tf  = tf.placeholder(tf.float32, shape=[None, self.R7.shape[1]])
        self.R8_tf  = tf.placeholder(tf.float32, shape=[None, self.R8.shape[1]])
        self.R9_tf  = tf.placeholder(tf.float32, shape=[None, self.R9.shape[1]])
        self.R10_tf = tf.placeholder(tf.float32, shape=[None, self.R10.shape[1]])
        self.R11_tf = tf.placeholder(tf.float32, shape=[None, self.R11.shape[1]])
        self.R12_tf = tf.placeholder(tf.float32, shape=[None, self.R12.shape[1]])
        self.R13_tf = tf.placeholder(tf.float32, shape=[None, self.R13.shape[1]])
        self.R14_tf = tf.placeholder(tf.float32, shape=[None, self.R14.shape[1]])
        self.R15_tf = tf.placeholder(tf.float32, shape=[None, self.R15.shape[1]])
        self.R16_tf = tf.placeholder(tf.float32, shape=[None, self.R16.shape[1]])
        self.R17_tf = tf.placeholder(tf.float32, shape=[None, self.R17.shape[1]])
        self.R18_tf = tf.placeholder(tf.float32, shape=[None, self.R18.shape[1]])
        self.R19_tf = tf.placeholder(tf.float32, shape=[None, self.R19.shape[1]])
        self.R20_tf = tf.placeholder(tf.float32, shape=[None, self.R20.shape[1]])
        self.R21_tf = tf.placeholder(tf.float32, shape=[None, self.R21.shape[1]])
        self.R22_tf = tf.placeholder(tf.float32, shape=[None, self.R22.shape[1]])
        self.R23_tf = tf.placeholder(tf.float32, shape=[None, self.R23.shape[1]])
        self.R24_tf = tf.placeholder(tf.float32, shape=[None, self.R24.shape[1]])
        self.R25_tf = tf.placeholder(tf.float32, shape=[None, self.R25.shape[1]])
        self.R26_tf = tf.placeholder(tf.float32, shape=[None, self.R26.shape[1]])
        self.R27_tf = tf.placeholder(tf.float32, shape=[None, self.R27.shape[1]])
        self.R28_tf = tf.placeholder(tf.float32, shape=[None, self.R28.shape[1]])
        self.R29_tf = tf.placeholder(tf.float32, shape=[None, self.R29.shape[1]])
        self.R30_tf = tf.placeholder(tf.float32, shape=[None, self.R30.shape[1]])
        self.R31_tf = tf.placeholder(tf.float32, shape=[None, self.R31.shape[1]])
        self.R32_tf = tf.placeholder(tf.float32, shape=[None, self.R32.shape[1]])
        self.R33_tf = tf.placeholder(tf.float32, shape=[None, self.R33.shape[1]])
        self.R34_tf = tf.placeholder(tf.float32, shape=[None, self.R34.shape[1]])
        self.R35_tf = tf.placeholder(tf.float32, shape=[None, self.R35.shape[1]])
        self.R36_tf = tf.placeholder(tf.float32, shape=[None, self.R36.shape[1]])
        self.R37_tf = tf.placeholder(tf.float32, shape=[None, self.R37.shape[1]])
        self.R38_tf = tf.placeholder(tf.float32, shape=[None, self.R38.shape[1]])
        self.R39_tf = tf.placeholder(tf.float32, shape=[None, self.R39.shape[1]])
        self.R40_tf = tf.placeholder(tf.float32, shape=[None, self.R40.shape[1]])
        self.R41_tf = tf.placeholder(tf.float32, shape=[None, self.R41.shape[1]])
        self.R42_tf = tf.placeholder(tf.float32, shape=[None, self.R42.shape[1]])
        self.R43_tf = tf.placeholder(tf.float32, shape=[None, self.R43.shape[1]])
        self.R44_tf = tf.placeholder(tf.float32, shape=[None, self.R44.shape[1]])
        self.R45_tf = tf.placeholder(tf.float32, shape=[None, self.R45.shape[1]])
        self.R46_tf = tf.placeholder(tf.float32, shape=[None, self.R46.shape[1]])
        self.R47_tf = tf.placeholder(tf.float32, shape=[None, self.R47.shape[1]])
        self.Rat_tf = tf.placeholder(tf.float32, shape=[None, self.Rat.shape[1]])

        # Predict the values of output by a single forward propagation.
        # Also get AutoDiff coefficients from the same class method: net_Euler_STS
        [self.n1_pred,  self.n2_pred,  self.n3_pred,  self.n4_pred,  self.n5_pred,  self.n6_pred,  self.n7_pred,  self.n8_pred,
         self.n9_pred,  self.n10_pred, self.n11_pred, self.n12_pred, self.n13_pred, self.n14_pred, self.n15_pred, self.n16_pred,
         self.n17_pred, self.n18_pred, self.n19_pred, self.n20_pred, self.n21_pred, self.n22_pred, self.n23_pred, self.n24_pred,
         self.n25_pred, self.n26_pred, self.n27_pred, self.n28_pred, self.n29_pred, self.n30_pred, self.n31_pred, self.n32_pred,
         self.n33_pred, self.n34_pred, self.n35_pred, self.n36_pred, self.n37_pred, self.n38_pred, self.n39_pred, self.n40_pred,
         self.n41_pred, self.n42_pred, self.n43_pred, self.n44_pred, self.n45_pred, self.n46_pred, self.n47_pred, self.nat_pred,
         self.rho_pred, self.u_pred,   self.p_pred,   self.E_pred,
         self.e1,   self.e2,   self.e3,   self.e4,
         self.en1,  self.en2,  self.en3,  self.en4,  self.en5,  self.en6,  self.en7,  self.en8,  self.en9,  self.en10, self.en11,
         self.en12, self.en13, self.en14, self.en15, self.en16, self.en17, self.en18, self.en19, self.en20, self.en21, self.en22,
         self.en23, self.en24, self.en25, self.en26, self.en27, self.en28, self.en29, self.en30, self.en31, self.en32, self.en33,
         self.en34, self.en35, self.en36, self.en37, self.en38, self.en39, self.en40, self.en41, self.en42, self.en43, self.en44,
         self.en45, self.en46, self.en47, self.enat] = self.net_Euler_STS(self.x_tf)

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

        n1_norm  = np.amax(n1)
        n2_norm  = np.amax(n2)
        n3_norm  = np.amax(n3)
        n4_norm  = np.amax(n4)
        n5_norm  = np.amax(n5)
        n6_norm  = np.amax(n6)
        n7_norm  = np.amax(n7)
        n8_norm  = np.amax(n8)
        n9_norm  = np.amax(n9)
        n10_norm = np.amax(n10)
        n11_norm = np.amax(n11)
        n12_norm = np.amax(n12)
        n13_norm = np.amax(n13)
        n14_norm = np.amax(n14)
        n15_norm = np.amax(n15)
        n16_norm = np.amax(n16)
        n17_norm = np.amax(n17)
        n18_norm = np.amax(n18)
        n19_norm = np.amax(n19)
        n20_norm = np.amax(n20)
        n21_norm = np.amax(n21)
        n22_norm = np.amax(n22)
        n23_norm = np.amax(n23)
        n24_norm = np.amax(n24)
        n25_norm = np.amax(n25)
        n26_norm = np.amax(n26)
        n27_norm = np.amax(n27)
        n28_norm = np.amax(n28)
        n29_norm = np.amax(n29)
        n30_norm = np.amax(n30)
        n31_norm = np.amax(n31)
        n32_norm = np.amax(n32)
        n33_norm = np.amax(n33)
        n34_norm = np.amax(n34)
        n35_norm = np.amax(n35)
        n36_norm = np.amax(n36)
        n37_norm = np.amax(n37)
        n38_norm = np.amax(n38)
        n39_norm = np.amax(n39)
        n40_norm = np.amax(n40)
        n41_norm = np.amax(n41)
        n42_norm = np.amax(n42)
        n43_norm = np.amax(n43)
        n44_norm = np.amax(n44)
        n45_norm = np.amax(n45)
        n46_norm = np.amax(n46)
        n47_norm = np.amax(n47)
        nat_norm = np.amax(nat)

        en1_norm  = n1_norm *u_norm
        en2_norm  = n2_norm *u_norm
        en3_norm  = n3_norm *u_norm
        en4_norm  = n4_norm *u_norm
        en5_norm  = n5_norm *u_norm
        en6_norm  = n6_norm *u_norm
        en7_norm  = n7_norm *u_norm
        en8_norm  = n8_norm *u_norm
        en9_norm  = n9_norm *u_norm
        en10_norm = n10_norm*u_norm
        en11_norm = n11_norm*u_norm
        en12_norm = n12_norm*u_norm
        en13_norm = n13_norm*u_norm
        en14_norm = n14_norm*u_norm
        en15_norm = n15_norm*u_norm
        en16_norm = n16_norm*u_norm
        en17_norm = n17_norm*u_norm
        en18_norm = n18_norm*u_norm
        en19_norm = n19_norm*u_norm
        en20_norm = n20_norm*u_norm
        en21_norm = n21_norm*u_norm
        en22_norm = n22_norm*u_norm
        en23_norm = n23_norm*u_norm
        en24_norm = n24_norm*u_norm
        en25_norm = n25_norm*u_norm
        en26_norm = n26_norm*u_norm
        en27_norm = n27_norm*u_norm
        en28_norm = n28_norm*u_norm
        en29_norm = n29_norm*u_norm
        en30_norm = n30_norm*u_norm
        en31_norm = n31_norm*u_norm
        en32_norm = n32_norm*u_norm
        en33_norm = n33_norm*u_norm
        en34_norm = n34_norm*u_norm
        en35_norm = n35_norm*u_norm
        en36_norm = n36_norm*u_norm
        en37_norm = n37_norm*u_norm
        en38_norm = n38_norm*u_norm
        en39_norm = n39_norm*u_norm
        en40_norm = n40_norm*u_norm
        en41_norm = n41_norm*u_norm
        en42_norm = n42_norm*u_norm
        en43_norm = n43_norm*u_norm
        en44_norm = n44_norm*u_norm
        en45_norm = n45_norm*u_norm
        en46_norm = n46_norm*u_norm
        en47_norm = n47_norm*u_norm
        enat_norm = nat_norm*u_norm

        # Weight factor... let's see its impact by varying it w = [0:100].
        # If is it 0, then PINN -> NN and we do not physically inform the NN.
        w = 1

        # Define Cost function or the Loss
        # In this case I have set the mean squared error of the ouputs to be
        # the loss and commented the PINN residual arguements. Uncommenting the
        # 4 residual expressions will result in a true Phyics Informed Neural
        # Network, otherwise, it is just a data trained Neural network
        self.loss = tf.reduce_sum(tf.square(self.u_tf   - self.u_pred)) /(u_norm**2) + \
                    tf.reduce_sum(tf.square(self.rho_tf - self.rho_pred))/(rho_norm**2) + \
                    tf.reduce_sum(tf.square(self.p_tf   - self.p_pred)) /(p_norm**2) + \
                    tf.reduce_sum(tf.square(self.E_tf   - self.E_pred)) /(E_norm**2) + \
                    tf.reduce_sum(tf.square(self.n1_tf  - self.n1_pred))/(n1_norm**2) + \
                    tf.reduce_sum(tf.square(self.n2_tf  - self.n2_pred))/(n2_norm**2) + \
                    tf.reduce_sum(tf.square(self.n3_tf  - self.n3_pred))/(n3_norm**2) + \
                    tf.reduce_sum(tf.square(self.n4_tf  - self.n4_pred))/(n4_norm**2) + \
                    tf.reduce_sum(tf.square(self.n5_tf  - self.n5_pred))/(n5_norm**2) + \
                    tf.reduce_sum(tf.square(self.n6_tf  - self.n6_pred))/(n6_norm**2) + \
                    tf.reduce_sum(tf.square(self.n7_tf  - self.n7_pred))/(n7_norm**2) + \
                    tf.reduce_sum(tf.square(self.n8_tf  - self.n8_pred))/(n8_norm**2) + \
                    tf.reduce_sum(tf.square(self.n9_tf  - self.n9_pred))/(n9_norm**2) + \
                    tf.reduce_sum(tf.square(self.n10_tf - self.n10_pred))/(n10_norm**2) + \
                    tf.reduce_sum(tf.square(self.n11_tf - self.n11_pred))/(n11_norm**2) + \
                    tf.reduce_sum(tf.square(self.n12_tf - self.n12_pred))/(n12_norm**2) + \
                    tf.reduce_sum(tf.square(self.n13_tf - self.n13_pred))/(n13_norm**2) + \
                    tf.reduce_sum(tf.square(self.n14_tf - self.n14_pred))/(n14_norm**2) + \
                    tf.reduce_sum(tf.square(self.n15_tf - self.n15_pred))/(n15_norm**2) + \
                    tf.reduce_sum(tf.square(self.n16_tf - self.n16_pred))/(n16_norm**2) + \
                    tf.reduce_sum(tf.square(self.n17_tf - self.n17_pred))/(n17_norm**2) + \
                    tf.reduce_sum(tf.square(self.n18_tf - self.n18_pred))/(n18_norm**2) + \
                    tf.reduce_sum(tf.square(self.n19_tf - self.n19_pred))/(n19_norm**2) + \
                    tf.reduce_sum(tf.square(self.n20_tf - self.n20_pred))/(n20_norm**2) + \
                    tf.reduce_sum(tf.square(self.n21_tf - self.n21_pred))/(n21_norm**2) + \
                    tf.reduce_sum(tf.square(self.n22_tf - self.n22_pred))/(n22_norm**2) + \
                    tf.reduce_sum(tf.square(self.n23_tf - self.n23_pred))/(n23_norm**2) + \
                    tf.reduce_sum(tf.square(self.n24_tf - self.n24_pred))/(n24_norm**2) + \
                    tf.reduce_sum(tf.square(self.n25_tf - self.n25_pred))/(n25_norm**2) + \
                    tf.reduce_sum(tf.square(self.n26_tf - self.n26_pred))/(n26_norm**2) + \
                    tf.reduce_sum(tf.square(self.n27_tf - self.n27_pred))/(n27_norm**2) + \
                    tf.reduce_sum(tf.square(self.n28_tf - self.n28_pred))/(n28_norm**2) + \
                    tf.reduce_sum(tf.square(self.n29_tf - self.n29_pred))/(n29_norm**2) + \
                    tf.reduce_sum(tf.square(self.n30_tf - self.n30_pred))/(n30_norm**2) + \
                    tf.reduce_sum(tf.square(self.n31_tf - self.n31_pred))/(n31_norm**2) + \
                    tf.reduce_sum(tf.square(self.n32_tf - self.n32_pred))/(n32_norm**2) + \
                    tf.reduce_sum(tf.square(self.n33_tf - self.n33_pred))/(n33_norm**2) + \
                    tf.reduce_sum(tf.square(self.n34_tf - self.n34_pred))/(n34_norm**2) + \
                    tf.reduce_sum(tf.square(self.n35_tf - self.n35_pred))/(n35_norm**2) + \
                    tf.reduce_sum(tf.square(self.n36_tf - self.n36_pred))/(n36_norm**2) + \
                    tf.reduce_sum(tf.square(self.n37_tf - self.n37_pred))/(n37_norm**2) + \
                    tf.reduce_sum(tf.square(self.n38_tf - self.n38_pred))/(n38_norm**2) + \
                    tf.reduce_sum(tf.square(self.n39_tf - self.n39_pred))/(n39_norm**2) + \
                    tf.reduce_sum(tf.square(self.n40_tf - self.n40_pred))/(n40_norm**2) + \
                    tf.reduce_sum(tf.square(self.n41_tf - self.n41_pred))/(n41_norm**2) + \
                    tf.reduce_sum(tf.square(self.n42_tf - self.n42_pred))/(n42_norm**2) + \
                    tf.reduce_sum(tf.square(self.n43_tf - self.n43_pred))/(n43_norm**2) + \
                    tf.reduce_sum(tf.square(self.n44_tf - self.n44_pred))/(n44_norm**2) + \
                    tf.reduce_sum(tf.square(self.n45_tf - self.n45_pred))/(n45_norm**2) + \
                    tf.reduce_sum(tf.square(self.n46_tf - self.n46_pred))/(n46_norm**2) + \
                    tf.reduce_sum(tf.square(self.n47_tf - self.n47_pred))/(n47_norm**2) + \
                    tf.reduce_sum(tf.square(self.nat_tf - self.nat_pred))/(nat_norm**2) #+ \
                    #w*tf.reduce_sum(tf.square(self.e1))/(e1_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.e2))/(e2_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.e3))/(e3_norm**2) +
                    #w*tf.reduce_sum(tf.square(self.e4))/(p_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en1))/(en1_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en2))/(en2_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en3))/(en3_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en4))/(en4_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en5))/(en5_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en6))/(en6_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en7))/(en7_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en8))/(en8_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en9))/(en9_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en10))/(en10_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en11))/(en11_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en12))/(en12_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en13))/(en13_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en14))/(en14_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en15))/(en15_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en16))/(en16_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en17))/(en17_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en18))/(en18_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en19))/(en19_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en20))/(en20_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en21))/(en21_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en22))/(en22_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en23))/(en23_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en24))/(en24_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en25))/(en25_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en26))/(en26_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en27))/(en27_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en28))/(en28_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en29))/(en29_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en30))/(en30_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en31))/(en31_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en32))/(en32_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en33))/(en33_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en34))/(en34_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en35))/(en35_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en36))/(en36_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en37))/(en37_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en38))/(en38_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en39))/(en39_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en40))/(en40_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en41))/(en41_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en42))/(en42_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en43))/(en43_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en44))/(en44_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en45))/(en45_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en46))/(en46_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.en47))/(en47_norm**2) + \
                    #w*tf.reduce_sum(tf.square(self.enat))/(enat_norm**2)

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

        n1  = nci_rho_u_p_E[:,0:1]
        n2  = nci_rho_u_p_E[:,1:2]
        n3  = nci_rho_u_p_E[:,2:3]
        n4  = nci_rho_u_p_E[:,3:4]
        n5  = nci_rho_u_p_E[:,4:5]
        n6  = nci_rho_u_p_E[:,5:6]
        n7  = nci_rho_u_p_E[:,6:7]
        n8  = nci_rho_u_p_E[:,7:8]
        n9  = nci_rho_u_p_E[:,8:9]
        n10  = nci_rho_u_p_E[:,9:10]
        n11  = nci_rho_u_p_E[:,10:11]
        n12  = nci_rho_u_p_E[:,11:12]
        n13  = nci_rho_u_p_E[:,12:13]
        n14  = nci_rho_u_p_E[:,13:14]
        n15  = nci_rho_u_p_E[:,14:15]
        n16  = nci_rho_u_p_E[:,15:16]
        n17  = nci_rho_u_p_E[:,16:17]
        n18  = nci_rho_u_p_E[:,17:18]
        n19  = nci_rho_u_p_E[:,18:19]
        n20  = nci_rho_u_p_E[:,19:20]
        n21  = nci_rho_u_p_E[:,20:21]
        n22  = nci_rho_u_p_E[:,21:22]
        n23  = nci_rho_u_p_E[:,22:23]
        n24  = nci_rho_u_p_E[:,23:24]
        n25  = nci_rho_u_p_E[:,24:25]
        n26  = nci_rho_u_p_E[:,25:26]
        n27  = nci_rho_u_p_E[:,26:27]
        n28  = nci_rho_u_p_E[:,27:28]
        n29  = nci_rho_u_p_E[:,28:29]
        n30  = nci_rho_u_p_E[:,29:30]
        n31  = nci_rho_u_p_E[:,30:31]
        n32  = nci_rho_u_p_E[:,31:32]
        n33  = nci_rho_u_p_E[:,32:33]
        n34  = nci_rho_u_p_E[:,33:34]
        n35  = nci_rho_u_p_E[:,34:35]
        n36  = nci_rho_u_p_E[:,35:36]
        n37  = nci_rho_u_p_E[:,36:37]
        n38  = nci_rho_u_p_E[:,37:38]
        n39  = nci_rho_u_p_E[:,38:39]
        n40  = nci_rho_u_p_E[:,39:40]
        n41  = nci_rho_u_p_E[:,40:41]
        n42  = nci_rho_u_p_E[:,41:42]
        n43  = nci_rho_u_p_E[:,42:43]
        n44  = nci_rho_u_p_E[:,43:44]
        n45  = nci_rho_u_p_E[:,44:45]
        n46  = nci_rho_u_p_E[:,45:46]
        n47  = nci_rho_u_p_E[:,46:47]
        nat  = nci_rho_u_p_E[:,47:48]

        rho = nci_rho_u_p_E[:,48:49]
        u   = nci_rho_u_p_E[:,49:50]
        p   = nci_rho_u_p_E[:,50:51]
        E   = nci_rho_u_p_E[:,51:52]

        R1   = nci_rho_u_p_E[:,52:53]
        R2   = nci_rho_u_p_E[:,53:54]
        R3   = nci_rho_u_p_E[:,54:55]
        R4   = nci_rho_u_p_E[:,55:56]
        R5   = nci_rho_u_p_E[:,56:57]
        R6   = nci_rho_u_p_E[:,57:58]
        R7   = nci_rho_u_p_E[:,58:59]
        R8   = nci_rho_u_p_E[:,59:60]
        R9   = nci_rho_u_p_E[:,60:61]
        R10  = nci_rho_u_p_E[:,61:62]
        R11  = nci_rho_u_p_E[:,62:63]
        R12  = nci_rho_u_p_E[:,63:64]
        R13  = nci_rho_u_p_E[:,64:65]
        R14  = nci_rho_u_p_E[:,65:66]
        R15  = nci_rho_u_p_E[:,66:67]
        R16  = nci_rho_u_p_E[:,67:68]
        R17  = nci_rho_u_p_E[:,68:69]
        R18  = nci_rho_u_p_E[:,69:70]
        R19  = nci_rho_u_p_E[:,70:71]
        R20  = nci_rho_u_p_E[:,71:72]
        R21  = nci_rho_u_p_E[:,72:73]
        R22  = nci_rho_u_p_E[:,73:74]
        R23  = nci_rho_u_p_E[:,74:75]
        R24  = nci_rho_u_p_E[:,75:76]
        R25  = nci_rho_u_p_E[:,76:77]
        R26  = nci_rho_u_p_E[:,77:78]
        R27  = nci_rho_u_p_E[:,78:79]
        R28  = nci_rho_u_p_E[:,79:80]
        R29  = nci_rho_u_p_E[:,80:81]
        R30  = nci_rho_u_p_E[:,81:82]
        R31  = nci_rho_u_p_E[:,82:83]
        R32  = nci_rho_u_p_E[:,83:84]
        R33  = nci_rho_u_p_E[:,84:85]
        R34  = nci_rho_u_p_E[:,85:86]
        R35  = nci_rho_u_p_E[:,86:87]
        R36  = nci_rho_u_p_E[:,87:88]
        R37  = nci_rho_u_p_E[:,88:89]
        R38  = nci_rho_u_p_E[:,89:90]
        R39  = nci_rho_u_p_E[:,90:91]
        R40  = nci_rho_u_p_E[:,91:92]
        R41  = nci_rho_u_p_E[:,92:93]
        R42  = nci_rho_u_p_E[:,93:94]
        R43  = nci_rho_u_p_E[:,94:95]
        R44  = nci_rho_u_p_E[:,95:96]
        R45  = nci_rho_u_p_E[:,96:97]
        R46  = nci_rho_u_p_E[:,97:98]
        R47  = nci_rho_u_p_E[:,98:99]
        Rat  = nci_rho_u_p_E[:,99:100]

        # temporal derivatives
        #n1_t   = tf.gradients(n1,   t)[0]
        #n2_t   = tf.gradients(n2,   t)[0]
        #n3_t   = tf.gradients(n3,   t)[0]
        #n4_t   = tf.gradients(n4,   t)[0]
        #n5_t   = tf.gradients(n5,   t)[0]
        #n6_t   = tf.gradients(n6,   t)[0]
        #n7_t   = tf.gradients(n7,   t)[0]
        #n8_t   = tf.gradients(n8,   t)[0]
        #n9_t   = tf.gradients(n9,   t)[0]
        #n10_t   = tf.gradients(n10,   t)[0]
        #n11_t   = tf.gradients(n11,   t)[0]
        #n12_t   = tf.gradients(n12,   t)[0]
        #n13_t   = tf.gradients(n13,   t)[0]
        #n14_t   = tf.gradients(n14,   t)[0]
        #n15_t   = tf.gradients(n15,   t)[0]
        #n16_t   = tf.gradients(n16,   t)[0]
        #n17_t   = tf.gradients(n17,   t)[0]
        #n18_t   = tf.gradients(n18,   t)[0]
        #n19_t   = tf.gradients(n19,   t)[0]
        #n20_t   = tf.gradients(n20,   t)[0]
        #n21_t   = tf.gradients(n21,   t)[0]
        #n22_t   = tf.gradients(n22,   t)[0]
        #n23_t   = tf.gradients(n23,   t)[0]
        #n24_t   = tf.gradients(n24,   t)[0]
        #n25_t   = tf.gradients(n25,   t)[0]
        #n26_t   = tf.gradients(n26,   t)[0]
        #n27_t   = tf.gradients(n27,   t)[0]
        #n28_t   = tf.gradients(n28,   t)[0]
        #n29_t   = tf.gradients(n29,   t)[0]
        #n30_t   = tf.gradients(n30,   t)[0]
        #n31_t   = tf.gradients(n31,   t)[0]
        #n32_t   = tf.gradients(n32,   t)[0]
        #n33_t   = tf.gradients(n33,   t)[0]
        #n34_t   = tf.gradients(n34,   t)[0]
        #n35_t   = tf.gradients(n35,   t)[0]
        #n36_t   = tf.gradients(n36,   t)[0]
        #n37_t   = tf.gradients(n37,   t)[0]
        #n38_t   = tf.gradients(n38,   t)[0]
        #n39_t   = tf.gradients(n39,   t)[0]
        #n40_t   = tf.gradients(n40,   t)[0]
        #n41_t   = tf.gradients(n41,   t)[0]
        #n42_t   = tf.gradients(n42,   t)[0]
        #n43_t   = tf.gradients(n43,   t)[0]
        #n44_t   = tf.gradients(n44,   t)[0]
        #n45_t   = tf.gradients(n45,   t)[0]
        #n46_t   = tf.gradients(n46,   t)[0]
        #n47_t   = tf.gradients(n47,   t)[0]
        #nat_t   = tf.gradients(nat,   t)[0]

        #rho_t   = tf.gradients(rho,   t)[0]
        #rho_u_t = tf.gradients(rho*u, t)[0]
        #rho_E_t = tf.gradients(rho*E, t)[0]

        n1_u_x  = tf.gradients(n1 *u, x)[0]
        n2_u_x  = tf.gradients(n2 *u, x)[0]
        n3_u_x  = tf.gradients(n3 *u, x)[0]
        n4_u_x  = tf.gradients(n4 *u, x)[0]
        n5_u_x  = tf.gradients(n5 *u, x)[0]
        n6_u_x  = tf.gradients(n6 *u, x)[0]
        n7_u_x  = tf.gradients(n7 *u, x)[0]
        n8_u_x  = tf.gradients(n8 *u, x)[0]
        n9_u_x  = tf.gradients(n9 *u, x)[0]
        n10_u_x = tf.gradients(n10*u, x)[0]
        n11_u_x = tf.gradients(n11*u, x)[0]
        n12_u_x = tf.gradients(n12*u, x)[0]
        n13_u_x = tf.gradients(n13*u, x)[0]
        n14_u_x = tf.gradients(n14*u, x)[0]
        n15_u_x = tf.gradients(n15*u, x)[0]
        n16_u_x = tf.gradients(n16*u, x)[0]
        n17_u_x = tf.gradients(n17*u, x)[0]
        n18_u_x = tf.gradients(n18*u, x)[0]
        n19_u_x = tf.gradients(n19*u, x)[0]
        n20_u_x = tf.gradients(n20*u, x)[0]
        n21_u_x = tf.gradients(n21*u, x)[0]
        n22_u_x = tf.gradients(n22*u, x)[0]
        n23_u_x = tf.gradients(n23*u, x)[0]
        n24_u_x = tf.gradients(n24*u, x)[0]
        n25_u_x = tf.gradients(n25*u, x)[0]
        n26_u_x = tf.gradients(n26*u, x)[0]
        n27_u_x = tf.gradients(n27*u, x)[0]
        n28_u_x = tf.gradients(n28*u, x)[0]
        n29_u_x = tf.gradients(n29*u, x)[0]
        n30_u_x = tf.gradients(n30*u, x)[0]
        n31_u_x = tf.gradients(n31*u, x)[0]
        n32_u_x = tf.gradients(n32*u, x)[0]
        n33_u_x = tf.gradients(n33*u, x)[0]
        n34_u_x = tf.gradients(n34*u, x)[0]
        n35_u_x = tf.gradients(n35*u, x)[0]
        n36_u_x = tf.gradients(n36*u, x)[0]
        n37_u_x = tf.gradients(n37*u, x)[0]
        n38_u_x = tf.gradients(n38*u, x)[0]
        n39_u_x = tf.gradients(n39*u, x)[0]
        n40_u_x = tf.gradients(n40*u, x)[0]
        n41_u_x = tf.gradients(n41*u, x)[0]
        n42_u_x = tf.gradients(n42*u, x)[0]
        n43_u_x = tf.gradients(n43*u, x)[0]
        n44_u_x = tf.gradients(n44*u, x)[0]
        n45_u_x = tf.gradients(n45*u, x)[0]
        n46_u_x = tf.gradients(n46*u, x)[0]
        n47_u_x = tf.gradients(n47*u, x)[0]
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

        eqn1  =  n1_u_x  - R1
        eqn2  =  n2_u_x  - R2
        eqn3  =  n3_u_x  - R3
        eqn4  =  n4_u_x  - R4
        eqn5  =  n5_u_x  - R5
        eqn6  =  n6_u_x  - R6
        eqn7  =  n7_u_x  - R7
        eqn8  =  n8_u_x  - R8
        eqn9  =  n9_u_x  - R9
        eqn10 =  n10_u_x - R10
        eqn11 =  n11_u_x - R11
        eqn12 =  n12_u_x - R12
        eqn13 =  n13_u_x - R13
        eqn14 =  n14_u_x - R14
        eqn15 =  n15_u_x - R15
        eqn16 =  n16_u_x - R16
        eqn17 =  n17_u_x - R17
        eqn18 =  n18_u_x - R18
        eqn19 =  n19_u_x - R19
        eqn20 =  n20_u_x - R20
        eqn21 =  n21_u_x - R21
        eqn22 =  n22_u_x - R22
        eqn23 =  n23_u_x - R23
        eqn24 =  n24_u_x - R24
        eqn25 =  n25_u_x - R25
        eqn26 =  n26_u_x - R26
        eqn27 =  n27_u_x - R27
        eqn28 =  n28_u_x - R28
        eqn29 =  n29_u_x - R29
        eqn30 =  n30_u_x - R30
        eqn31 =  n31_u_x - R31
        eqn32 =  n32_u_x - R32
        eqn33 =  n33_u_x - R33
        eqn34 =  n34_u_x - R34
        eqn35 =  n35_u_x - R35
        eqn36 =  n36_u_x - R36
        eqn37 =  n37_u_x - R37
        eqn38 =  n38_u_x - R38
        eqn39 =  n39_u_x - R39
        eqn40 =  n40_u_x - R40
        eqn41 =  n41_u_x - R41
        eqn42 =  n42_u_x - R42
        eqn43 =  n43_u_x - R43
        eqn44 =  n44_u_x - R44
        eqn45 =  n45_u_x - R45
        eqn46 =  n46_u_x - R46
        eqn47 =  n47_u_x - R47
        eqnat =  nat_u_x - Rat

        eq1 =  mass_flow_grad
        eq2 =  momentum_grad
        eq3 =  energy_grad
        eq4 =  state_res

        return  n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, \
                n11, n12, n13, n14, n15, n16, n17, n18, n19, n20, \
                n21, n22, n23, n24, n25, n26, n27, n28, n29, n30, \
                n31, n32, n33, n34, n35, n36, n37, n38, n39, n40, \
                n41, n42, n43, n44, n45, n46, n47, nat, \
                rho, u, p, E, \
                R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, \
                R11, R12, R13, R14, R15, R16, R17, R18, R19, R20, \
                R21, R22, R23, R24, R25, R26, R27, R28, R29, R30, \
                R31, R32, R33, R34, R35, R36, R37, R38, R39, R40, \
                R41, R42, R43, R44, R45, R46, R47, Rat, \
                eq1, eq2, eq3, eq4#, \
                #eqn1, eqn2, eqn3, eqn4, eqn5, eqn6, eqn7, eqn8, eqn9, eqn10, \
                #eqn11, eqn12, eqn13, eqn14, eqn15, eqn16, eqn17, eqn18, eqn19, eqn20, \
                #eqn21, eqn22, eqn23, eqn24, eqn25, eqn26, eqn27, eqn28, eqn29, eqn30, \
                #eqn31, eqn32, eqn33, eqn34, eqn35, eqn36, eqn37, eqn38, eqn39, eqn40, \
                ##eqn41, eqn42, eqn43, eqn44, eqn45, eqn46, eqn47, eqnat

    # callback method just prints the current loss (cost) value of the network.
    def callback(self, loss):
        print('Loss: %.3e' % (loss))

    # Train method actually trains the network weights based on the target
    # of minimizing the loss. tf_dict is defined as the set of input and
    # ideal output parameters for the given data in loop. For the given
    # iterations 'nIter' (variable), the train_op_Adam session is run.
    def train(self, nIter):

        tf_dict = {self.x_tf:   self.x,
                   self.n1_tf:  self.n1,  self.n2_tf:  self.n2,  self.n3_tf:  self.n3,  self.n4_tf:  self.n4,
                   self.n5_tf:  self.n5,  self.n6_tf:  self.n6,  self.n7_tf:  self.n7,  self.n8_tf:  self.n8,
                   self.n9_tf:  self.n9,  self.n10_tf: self.n10, self.n11_tf: self.n11, self.n12_tf: self.n12,
                   self.n13_tf: self.n13, self.n14_tf: self.n14, self.n15_tf: self.n15, self.n16_tf: self.n16,
                   self.n17_tf: self.n17, self.n18_tf: self.n18, self.n19_tf: self.n19, self.n20_tf: self.n20,
                   self.n21_tf: self.n21, self.n22_tf: self.n22, self.n23_tf: self.n23, self.n24_tf: self.n24,
                   self.n25_tf: self.n25, self.n26_tf: self.n26, self.n27_tf: self.n27, self.n28_tf: self.n28,
                   self.n29_tf: self.n29, self.n30_tf: self.n30, self.n31_tf: self.n31, self.n32_tf: self.n32,
                   self.n33_tf: self.n33, self.n34_tf: self.n34, self.n35_tf: self.n35, self.n36_tf: self.n36,
                   self.n37_tf: self.n37, self.n38_tf: self.n38, self.n39_tf: self.n39, self.n40_tf: self.n40,
                   self.n41_tf: self.n41, self.n42_tf: self.n42, self.n43_tf: self.n43, self.n44_tf: self.n44,
                   self.n45_tf: self.n45, self.n46_tf: self.n46, self.n47_tf: self.n47, self.nat_tf: self.nat,
                   self.rho_tf: self.rho, self.u_tf:   self.u,   self.p_tf:   self.p,   self.E_tf:   self.E,
                   self.n1_tf:  self.n1,  self.n2_tf:  self.n2,  self.n3_tf:  self.n3,  self.n4_tf:  self.n4,
                   self.n5_tf:  self.n5,  self.n6_tf:  self.n6,  self.n7_tf:  self.n7,  self.n8_tf:  self.n8,
                   self.n9_tf:  self.n9,  self.n10_tf: self.n10, self.n11_tf: self.n11, self.n12_tf: self.n12,
                   self.n13_tf: self.n13, self.n14_tf: self.n14, self.n15_tf: self.n15, self.n16_tf: self.n16,
                   self.n17_tf: self.n17, self.n18_tf: self.n18, self.n19_tf: self.n19, self.n20_tf: self.n20,
                   self.n21_tf: self.n21, self.n22_tf: self.n22, self.n23_tf: self.n23, self.n24_tf: self.n24,
                   self.n25_tf: self.n25, self.n26_tf: self.n26, self.n27_tf: self.n27, self.n28_tf: self.n28,
                   self.n29_tf: self.n29, self.n30_tf: self.n30, self.n31_tf: self.n31, self.n32_tf: self.n32,
                   self.n33_tf: self.n33, self.n34_tf: self.n34, self.n35_tf: self.n35, self.n36_tf: self.n36,
                   self.n37_tf: self.n37, self.n38_tf: self.n38, self.n39_tf: self.n39, self.n40_tf: self.n40,
                   self.n41_tf: self.n41, self.n42_tf: self.n42, self.n43_tf: self.n43, self.n44_tf: self.R44,
                   self.n45_tf: self.R45, self.n46_tf: self.R46, self.n47_tf: self.R47, self.nat_tf: self.Rat}

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

        n1_test  = self.sess.run(self.n1_pred,  tf_dict)
        n2_test  = self.sess.run(self.n2_pred,  tf_dict)
        n3_test  = self.sess.run(self.n3_pred,  tf_dict)
        n4_test  = self.sess.run(self.n4_pred,  tf_dict)
        n5_test  = self.sess.run(self.n5_pred,  tf_dict)
        n6_test  = self.sess.run(self.n6_pred,  tf_dict)
        n7_test  = self.sess.run(self.n7_pred,  tf_dict)
        n8_test  = self.sess.run(self.n8_pred,  tf_dict)
        n9_test  = self.sess.run(self.n9_pred,  tf_dict)
        n10_test = self.sess.run(self.n10_pred, tf_dict)
        n11_test = self.sess.run(self.n11_pred, tf_dict)
        n12_test = self.sess.run(self.n12_pred, tf_dict)
        n13_test = self.sess.run(self.n13_pred, tf_dict)
        n14_test = self.sess.run(self.n14_pred, tf_dict)
        n15_test = self.sess.run(self.n15_pred, tf_dict)
        n16_test = self.sess.run(self.n16_pred, tf_dict)
        n17_test = self.sess.run(self.n17_pred, tf_dict)
        n18_test = self.sess.run(self.n18_pred, tf_dict)
        n19_test = self.sess.run(self.n19_pred, tf_dict)
        n20_test = self.sess.run(self.n20_pred, tf_dict)
        n21_test = self.sess.run(self.n21_pred, tf_dict)
        n22_test = self.sess.run(self.n22_pred, tf_dict)
        n23_test = self.sess.run(self.n23_pred, tf_dict)
        n24_test = self.sess.run(self.n24_pred, tf_dict)
        n25_test = self.sess.run(self.n25_pred, tf_dict)
        n26_test = self.sess.run(self.n26_pred, tf_dict)
        n27_test = self.sess.run(self.n27_pred, tf_dict)
        n28_test = self.sess.run(self.n28_pred, tf_dict)
        n29_test = self.sess.run(self.n29_pred, tf_dict)
        n30_test = self.sess.run(self.n30_pred, tf_dict)
        n31_test = self.sess.run(self.n31_pred, tf_dict)
        n32_test = self.sess.run(self.n32_pred, tf_dict)
        n33_test = self.sess.run(self.n33_pred, tf_dict)
        n34_test = self.sess.run(self.n34_pred, tf_dict)
        n35_test = self.sess.run(self.n35_pred, tf_dict)
        n36_test = self.sess.run(self.n36_pred, tf_dict)
        n37_test = self.sess.run(self.n37_pred, tf_dict)
        n38_test = self.sess.run(self.n38_pred, tf_dict)
        n39_test = self.sess.run(self.n39_pred, tf_dict)
        n40_test = self.sess.run(self.n40_pred, tf_dict)
        n41_test = self.sess.run(self.n41_pred, tf_dict)
        n42_test = self.sess.run(self.n42_pred, tf_dict)
        n43_test = self.sess.run(self.n43_pred, tf_dict)
        n44_test = self.sess.run(self.n44_pred, tf_dict)
        n45_test = self.sess.run(self.n45_pred, tf_dict)
        n46_test = self.sess.run(self.n46_pred, tf_dict)
        n47_test = self.sess.run(self.n47_pred, tf_dict)
        nat_test = self.sess.run(self.nat_pred, tf_dict)

        rho_test = self.sess.run(self.rho_pred, tf_dict)
        u_test   = self.sess.run(self.u_pred,   tf_dict)
        p_test   = self.sess.run(self.p_pred,   tf_dict)
        E_test   = self.sess.run(self.E_pred,   tf_dict)

        R1_test  = self.sess.run(self.R1_pred,  tf_dict)
        R2_test  = self.sess.run(self.R2_pred,  tf_dict)
        R3_test  = self.sess.run(self.R3_pred,  tf_dict)
        R4_test  = self.sess.run(self.R4_pred,  tf_dict)
        R5_test  = self.sess.run(self.R5_pred,  tf_dict)
        R6_test  = self.sess.run(self.R6_pred,  tf_dict)
        R7_test  = self.sess.run(self.R7_pred,  tf_dict)
        R8_test  = self.sess.run(self.R8_pred,  tf_dict)
        R9_test  = self.sess.run(self.R9_pred,  tf_dict)
        R10_test = self.sess.run(self.R10_pred, tf_dict)
        R11_test = self.sess.run(self.R11_pred, tf_dict)
        R12_test = self.sess.run(self.R12_pred, tf_dict)
        R13_test = self.sess.run(self.R13_pred, tf_dict)
        R14_test = self.sess.run(self.R14_pred, tf_dict)
        R15_test = self.sess.run(self.R15_pred, tf_dict)
        R16_test = self.sess.run(self.R16_pred, tf_dict)
        R17_test = self.sess.run(self.R17_pred, tf_dict)
        R18_test = self.sess.run(self.R18_pred, tf_dict)
        R19_test = self.sess.run(self.R19_pred, tf_dict)
        R20_test = self.sess.run(self.R20_pred, tf_dict)
        R21_test = self.sess.run(self.R21_pred, tf_dict)
        R22_test = self.sess.run(self.R22_pred, tf_dict)
        R23_test = self.sess.run(self.R23_pred, tf_dict)
        R24_test = self.sess.run(self.R24_pred, tf_dict)
        R25_test = self.sess.run(self.R25_pred, tf_dict)
        R26_test = self.sess.run(self.R26_pred, tf_dict)
        R27_test = self.sess.run(self.R27_pred, tf_dict)
        R28_test = self.sess.run(self.R28_pred, tf_dict)
        R29_test = self.sess.run(self.R29_pred, tf_dict)
        R30_test = self.sess.run(self.R30_pred, tf_dict)
        R31_test = self.sess.run(self.R31_pred, tf_dict)
        R32_test = self.sess.run(self.R32_pred, tf_dict)
        R33_test = self.sess.run(self.R33_pred, tf_dict)
        R34_test = self.sess.run(self.R34_pred, tf_dict)
        R35_test = self.sess.run(self.R35_pred, tf_dict)
        R36_test = self.sess.run(self.R36_pred, tf_dict)
        R37_test = self.sess.run(self.R37_pred, tf_dict)
        R38_test = self.sess.run(self.R38_pred, tf_dict)
        R39_test = self.sess.run(self.R39_pred, tf_dict)
        R40_test = self.sess.run(self.R40_pred, tf_dict)
        R41_test = self.sess.run(self.R41_pred, tf_dict)
        R42_test = self.sess.run(self.R42_pred, tf_dict)
        R43_test = self.sess.run(self.R43_pred, tf_dict)
        R44_test = self.sess.run(self.R44_pred, tf_dict)
        R45_test = self.sess.run(self.R45_pred, tf_dict)
        R46_test = self.sess.run(self.R46_pred, tf_dict)
        R47_test = self.sess.run(self.R47_pred, tf_dict)
        Rat_test = self.sess.run(self.Rat_pred, tf_dict)

        return n1_test, n2_test, n3_test, n4_test, n5_test, n6_test, n7_test, n8_test, n9_test, \
               n10_test, n11_test, n12_test, n13_test, n14_test, n15_test, n16_test, n17_test, n18_test, \
               n19_test, n20_test, n21_test, n22_test, n23_test, n24_test, n25_test, n26_test, n27_test, \
               n28_test, n29_test, n30_test, n31_test, n32_test, n33_test, n34_test, n35_test, n36_test, \
               n37_test, n38_test, n39_test, n40_test, n41_test, n42_test, n43_test, n44_test, n45_test, \
               n46_test, n47_test, nat_test, rho_test, u_test, p_test, E_test, \
               R1_test, R2_test, R3_test, R4_test, R5_test, R6_test, R7_test, R8_test, R9_test, \
               R10_test, R11_test, R12_test, R13_test, R14_test, R15_test, R16_test, R17_test, R18_test, \
               R19_test, R20_test, R21_test, R22_test, R23_test, R24_test, R25_test, R26_test, R27_test, \
               R28_test, R29_test, R30_test, R31_test, R32_test, R33_test, R34_test, R35_test, R36_test, \
               R37_test, R38_test, R39_test, R40_test, R41_test, R42_test, R43_test, R44_test, R45_test, \
               R46_test, R47_test, Rat_test

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
    #layers = [1, 20, 20, 20, 20, 20, 20, 20, 100]
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

    # x
    x_train   = data[idx,0:1].flatten()[:,None]

    # y
    n1_train  = data[idx,1:2].flatten()[:,None]
    n2_train  = data[idx,2:3].flatten()[:,None]
    n3_train  = data[idx,3:4].flatten()[:,None]
    n4_train  = data[idx,4:5].flatten()[:,None]
    n5_train  = data[idx,5:6].flatten()[:,None]
    n6_train  = data[idx,6:7].flatten()[:,None]
    n7_train  = data[idx,7:8].flatten()[:,None]
    n8_train  = data[idx,8:9].flatten()[:,None]
    n9_train  = data[idx,9:10].flatten()[:,None]
    n10_train = data[idx,10:11].flatten()[:,None]
    n11_train = data[idx,11:12].flatten()[:,None]
    n12_train = data[idx,12:13].flatten()[:,None]
    n13_train = data[idx,13:14].flatten()[:,None]
    n14_train = data[idx,14:15].flatten()[:,None]
    n15_train = data[idx,15:16].flatten()[:,None]
    n16_train = data[idx,16:17].flatten()[:,None]
    n17_train = data[idx,17:18].flatten()[:,None]
    n18_train = data[idx,18:19].flatten()[:,None]
    n19_train = data[idx,19:20].flatten()[:,None]
    n20_train = data[idx,20:21].flatten()[:,None]
    n21_train = data[idx,21:22].flatten()[:,None]
    n22_train = data[idx,22:23].flatten()[:,None]
    n23_train = data[idx,23:24].flatten()[:,None]
    n24_train = data[idx,24:25].flatten()[:,None]
    n25_train = data[idx,25:26].flatten()[:,None]
    n26_train = data[idx,26:27].flatten()[:,None]
    n27_train = data[idx,27:28].flatten()[:,None]
    n28_train = data[idx,28:29].flatten()[:,None]
    n29_train = data[idx,29:30].flatten()[:,None]
    n30_train = data[idx,30:31].flatten()[:,None]
    n31_train = data[idx,31:32].flatten()[:,None]
    n32_train = data[idx,32:33].flatten()[:,None]
    n33_train = data[idx,33:34].flatten()[:,None]
    n34_train = data[idx,34:35].flatten()[:,None]
    n35_train = data[idx,35:36].flatten()[:,None]
    n36_train = data[idx,36:37].flatten()[:,None]
    n37_train = data[idx,37:38].flatten()[:,None]
    n38_train = data[idx,38:39].flatten()[:,None]
    n39_train = data[idx,39:40].flatten()[:,None]
    n40_train = data[idx,40:41].flatten()[:,None]
    n41_train = data[idx,41:42].flatten()[:,None]
    n42_train = data[idx,42:43].flatten()[:,None]
    n43_train = data[idx,43:44].flatten()[:,None]
    n44_train = data[idx,44:45].flatten()[:,None]
    n45_train = data[idx,45:46].flatten()[:,None]
    n46_train = data[idx,46:47].flatten()[:,None]
    n47_train = data[idx,47:48].flatten()[:,None]
    nat_train = data[idx,48:49].flatten()[:,None]

    rho_train = data[idx,49:50].flatten()[:,None]
    u_train   = data[idx,50:51].flatten()[:,None]
    p_train   = data[idx,51:52].flatten()[:,None]
    E_train   = data[idx,52:53].flatten()[:,None]

    R1_train  = data[idx,53:54].flatten()[:,None]
    R2_train  = data[idx,54:55].flatten()[:,None]
    R3_train  = data[idx,55:56].flatten()[:,None]
    R4_train  = data[idx,56:57].flatten()[:,None]
    R5_train  = data[idx,57:58].flatten()[:,None]
    R6_train  = data[idx,58:59].flatten()[:,None]
    R7_train  = data[idx,59:60].flatten()[:,None]
    R8_train  = data[idx,60:61].flatten()[:,None]
    R9_train  = data[idx,61:62].flatten()[:,None]
    R10_train = data[idx,62:63].flatten()[:,None]
    R11_train = data[idx,63:64].flatten()[:,None]
    R12_train = data[idx,64:65].flatten()[:,None]
    R13_train = data[idx,65:66].flatten()[:,None]
    R14_train = data[idx,66:67].flatten()[:,None]
    R15_train = data[idx,67:68].flatten()[:,None]
    R16_train = data[idx,68:69].flatten()[:,None]
    R17_train = data[idx,69:70].flatten()[:,None]
    R18_train = data[idx,70:71].flatten()[:,None]
    R19_train = data[idx,71:72].flatten()[:,None]
    R20_train = data[idx,72:73].flatten()[:,None]
    R21_train = data[idx,73:74].flatten()[:,None]
    R22_train = data[idx,74:75].flatten()[:,None]
    R23_train = data[idx,75:76].flatten()[:,None]
    R24_train = data[idx,76:77].flatten()[:,None]
    R25_train = data[idx,77:78].flatten()[:,None]
    R26_train = data[idx,78:79].flatten()[:,None]
    R27_train = data[idx,79:80].flatten()[:,None]
    R28_train = data[idx,80:81].flatten()[:,None]
    R29_train = data[idx,81:82].flatten()[:,None]
    R30_train = data[idx,82:83].flatten()[:,None]
    R31_train = data[idx,83:84].flatten()[:,None]
    R32_train = data[idx,84:85].flatten()[:,None]
    R33_train = data[idx,85:86].flatten()[:,None]
    R34_train = data[idx,86:87].flatten()[:,None]
    R35_train = data[idx,87:88].flatten()[:,None]
    R36_train = data[idx,88:89].flatten()[:,None]
    R37_train = data[idx,89:90].flatten()[:,None]
    R38_train = data[idx,90:91].flatten()[:,None]
    R39_train = data[idx,91:92].flatten()[:,None]
    R40_train = data[idx,92:93].flatten()[:,None]
    R41_train = data[idx,93:94].flatten()[:,None]
    R42_train = data[idx,94:95].flatten()[:,None]
    R43_train = data[idx,95:96].flatten()[:,None]
    R44_train = data[idx,96:97].flatten()[:,None]
    R45_train = data[idx,97:98].flatten()[:,None]
    R46_train = data[idx,98:99].flatten()[:,None]
    R47_train = data[idx,99:100].flatten()[:,None]
    Rat_train = data[idx,100:101].flatten()[:,None]

    print(rho_train)

    # Training the NN based on the training set, randomly chosen above model
    # = PINN(..) passes the necessary training data to the 'NN' class (model
    # here being an instance of the NN class) in order to initialize all the
    # parameters as well as the NN architecture including random initialization
    # of weights and biases.
    model = PINN(x_train,
                 n1_train,  n2_train,  n3_train,  n4_train,  n5_train,  n6_train,  n7_train,  n8_train,  n9_train,  n10_train,
                 n11_train, n12_train, n13_train, n14_train, n15_train, n16_train, n17_train, n18_train, n19_train, n20_train,
                 n21_train, n22_train, n23_train, n24_train, n25_train, n26_train, n27_train, n28_train, n29_train, n30_train,
                 n31_train, n32_train, n33_train, n34_train, n35_train, n36_train, n37_train, n38_train, n39_train, n40_train,
                 n41_train, n42_train, n43_train, n44_train, n45_train, n46_train, n47_train, nat_train,
                 rho_train, u_train, p_train, E_train,
                 R1_train,  R2_train,  R3_train,  R4_train,  R5_train,  R6_train,  R7_train,  R8_train,  R9_train,  R10_train,
                 R11_train, R12_train, R13_train, R14_train, R15_train, R16_train, R17_train, R18_train, R19_train, R20_train,
                 R21_train, R22_train, R23_train, R24_train, R25_train, R26_train, R27_train, R28_train, R29_train, R30_train,
                 R31_train, R32_train, R33_train, R34_train, R35_train, R36_train, R37_train, R38_train, R39_train, R40_train,
                 R41_train, R42_train, R43_train, R44_train, R45_train, R46_train, R47_train, Rat_train, layers)

    model.train(10000)

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

    # y
    n1_test  = data1[:,1:2].flatten()[:,None]
    n2_test  = data1[:,2:3].flatten()[:,None]
    n3_test  = data1[:,3:4].flatten()[:,None]
    n4_test  = data1[:,4:5].flatten()[:,None]
    n5_test  = data1[:,5:6].flatten()[:,None]
    n6_test  = data1[:,6:7].flatten()[:,None]
    n7_test  = data1[:,7:8].flatten()[:,None]
    n8_test  = data1[:,8:9].flatten()[:,None]
    n9_test  = data1[:,9:10].flatten()[:,None]
    n10_test = data1[:,10:11].flatten()[:,None]
    n11_test = data1[:,11:12].flatten()[:,None]
    n12_test = data1[:,12:13].flatten()[:,None]
    n13_test = data1[:,13:14].flatten()[:,None]
    n14_test = data1[:,14:15].flatten()[:,None]
    n15_test = data1[:,15:16].flatten()[:,None]
    n16_test = data1[:,16:17].flatten()[:,None]
    n17_test = data1[:,17:18].flatten()[:,None]
    n18_test = data1[:,18:19].flatten()[:,None]
    n19_test = data1[:,19:20].flatten()[:,None]
    n20_test = data1[:,20:21].flatten()[:,None]
    n21_test = data1[:,21:22].flatten()[:,None]
    n22_test = data1[:,22:23].flatten()[:,None]
    n23_test = data1[:,23:24].flatten()[:,None]
    n24_test = data1[:,24:25].flatten()[:,None]
    n25_test = data1[:,25:26].flatten()[:,None]
    n26_test = data1[:,26:27].flatten()[:,None]
    n27_test = data1[:,27:28].flatten()[:,None]
    n28_test = data1[:,28:29].flatten()[:,None]
    n29_test = data1[:,29:30].flatten()[:,None]
    n30_test = data1[:,30:31].flatten()[:,None]
    n31_test = data1[:,31:32].flatten()[:,None]
    n32_test = data1[:,32:33].flatten()[:,None]
    n33_test = data1[:,33:34].flatten()[:,None]
    n34_test = data1[:,34:35].flatten()[:,None]
    n35_test = data1[:,35:36].flatten()[:,None]
    n36_test = data1[:,36:37].flatten()[:,None]
    n37_test = data1[:,37:38].flatten()[:,None]
    n38_test = data1[:,38:39].flatten()[:,None]
    n39_test = data1[:,39:40].flatten()[:,None]
    n40_test = data1[:,40:41].flatten()[:,None]
    n41_test = data1[:,41:42].flatten()[:,None]
    n42_test = data1[:,42:43].flatten()[:,None]
    n43_test = data1[:,43:44].flatten()[:,None]
    n44_test = data1[:,44:45].flatten()[:,None]
    n45_test = data1[:,45:46].flatten()[:,None]
    n46_test = data1[:,46:47].flatten()[:,None]
    n47_test = data1[:,47:48].flatten()[:,None]
    nat_test = data1[:,48:49].flatten()[:,None]

    rho_test = data1[:,49:50].flatten()[:,None]
    u_test   = data1[:,50:51].flatten()[:,None]
    p_test   = data1[:,51:52].flatten()[:,None]
    E_test   = data1[:,52:53].flatten()[:,None]

    R1_test  = data1[:,53:54].flatten()[:,None]
    R2_test  = data1[:,54:55].flatten()[:,None]
    R3_test  = data1[:,55:56].flatten()[:,None]
    R4_test  = data1[:,56:57].flatten()[:,None]
    R5_test  = data1[:,57:58].flatten()[:,None]
    R6_test  = data1[:,58:59].flatten()[:,None]
    R7_test  = data1[:,59:60].flatten()[:,None]
    R8_test  = data1[:,60:61].flatten()[:,None]
    R9_test  = data1[:,61:62].flatten()[:,None]
    R10_test = data1[:,62:63].flatten()[:,None]
    R11_test = data1[:,63:64].flatten()[:,None]
    R12_test = data1[:,64:65].flatten()[:,None]
    R13_test = data1[:,65:66].flatten()[:,None]
    R14_test = data1[:,66:67].flatten()[:,None]
    R15_test = data1[:,67:68].flatten()[:,None]
    R16_test = data1[:,68:69].flatten()[:,None]
    R17_test = data1[:,69:70].flatten()[:,None]
    R18_test = data1[:,70:71].flatten()[:,None]
    R19_test = data1[:,71:72].flatten()[:,None]
    R20_test = data1[:,72:73].flatten()[:,None]
    R21_test = data1[:,73:74].flatten()[:,None]
    R22_test = data1[:,74:75].flatten()[:,None]
    R23_test = data1[:,75:76].flatten()[:,None]
    R24_test = data1[:,76:77].flatten()[:,None]
    R25_test = data1[:,77:78].flatten()[:,None]
    R26_test = data1[:,78:79].flatten()[:,None]
    R27_test = data1[:,79:80].flatten()[:,None]
    R28_test = data1[:,80:81].flatten()[:,None]
    R29_test = data1[:,81:82].flatten()[:,None]
    R30_test = data1[:,82:83].flatten()[:,None]
    R31_test = data1[:,83:84].flatten()[:,None]
    R32_test = data1[:,84:85].flatten()[:,None]
    R33_test = data1[:,85:86].flatten()[:,None]
    R34_test = data1[:,86:87].flatten()[:,None]
    R35_test = data1[:,87:88].flatten()[:,None]
    R36_test = data1[:,88:89].flatten()[:,None]
    R37_test = data1[:,89:90].flatten()[:,None]
    R38_test = data1[:,90:91].flatten()[:,None]
    R39_test = data1[:,91:92].flatten()[:,None]
    R40_test = data1[:,92:93].flatten()[:,None]
    R41_test = data1[:,93:94].flatten()[:,None]
    R42_test = data1[:,94:95].flatten()[:,None]
    R43_test = data1[:,95:96].flatten()[:,None]
    R44_test = data1[:,96:97].flatten()[:,None]
    R45_test = data1[:,97:98].flatten()[:,None]
    R46_test = data1[:,98:99].flatten()[:,None]
    R47_test = data1[:,99:100].flatten()[:,None]
    Rat_test = data1[:,100:101].flatten()[:,None]

    # Prediction
    # The input parameters of the test set are used to predict the pressure, density, speed and specific energy for the
    # given x and t by using the .predict method.
    [n1_pred, n2_pred,  n3_pred,  n4_pred,  n5_pred,  n6_pred,  n7_pred,  n8_pred,  n9_pred,  n10_pred, n11_pred, n12_pred,
    n13_pred, n14_pred, n15_pred, n16_pred, n17_pred, n18_pred, n19_pred, n20_pred, n21_pred, n22_pred, n23_pred, n24_pred,
    n25_pred, n26_pred, n27_pred, n28_pred, n29_pred, n30_pred, n31_pred, n32_pred, n33_pred, n34_pred, n35_pred, n36_pred,
    n37_pred, n38_pred, n39_pred, n40_pred, n41_pred, n42_pred, n43_pred, n44_pred, n45_pred, n46_pred, n47_pred, nat_pred,
    rho_pred, u_pred, p_pred, E_pred,
    R1_pred,  R2_pred,  R3_pred,  R4_pred,  R5_pred,  R6_pred,  R7_pred,  R8_pred,  R9_pred,  R10_pred, R11_pred, R12_pred,
    R13_pred, R14_pred, R15_pred, R16_pred, R17_pred, R18_pred, R19_pred, R20_pred, R21_pred, R22_pred, R23_pred, R24_pred,
    R25_pred, R26_pred, R27_pred, R28_pred, R29_pred, R30_pred, R31_pred, R32_pred, R33_pred, R34_pred, R35_pred, R36_pred,
    R37_pred, R38_pred, R39_pred, R40_pred, R41_pred, R42_pred, R43_pred, R44_pred, R45_pred, R46_pred, R47_pred,
    Rat_pred] = model.predict(x_test)

# Error
# Normal relative error is printed for each variable
error_n1 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
error_n2 = np.linalg.norm(n2_test-n2_pred,2)/np.linalg.norm(n2_test,2)
#error_n3 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n4 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n5 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n6 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n7 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n8 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n9 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n10 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n11 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n12 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n13 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n14 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n15 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n16 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n17 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n18 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n19 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n10 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n20 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n21 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n22 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n23 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n24 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n25 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n26 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n27 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n28 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n29 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n30 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n31 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n32 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n33 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n34 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n35 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n36 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n37 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n38 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n39 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n40 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n41 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n42 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n43 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n44 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n45 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n46 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_n47 = np.linalg.norm(n1_test-n1_pred,2)/np.linalg.norm(n1_test,2)
#error_nat = np.linalg.norm(nat_test-nat_pred,2)/np.linalg.norm(nat_test,2)

error_rho = np.linalg.norm(rho_test-rho_pred,2)/np.linalg.norm(rho_test,2)
error_u   = np.linalg.norm(u_test-u_pred,2)/np.linalg.norm(u_test,2)
error_p   = np.linalg.norm(p_test-p_pred,2)/np.linalg.norm(p_test,2)
error_E   = np.linalg.norm(E_test-E_pred,2)/np.linalg.norm(E_test,2)
print("Test Error in n1: "+str(error_n1))
print("Test Error in n2: "+str(error_n2))
# ...
print("Test Error in rho: "+str(error_rho))
print("Test Error in u: "+str(error_u))
print("Test Error in p: "+str(error_p))
print("Test Error in E: "+str(error_E))

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

#plot_solution(XT_test, n1_pred, 1)
#savefig('./figures/n1_pred', crop = False)
# ...
#plot_solution(XT_test, rho_pred, 1)
#savefig('./figures/rho_pred', crop = False)
#plot_solution(XT_test, u_pred, 1)
#savefig('./figures/u_pred', crop = False)
#plot_solution(XT_test, p_pred, 1)
#savefig('./figures/p_pred', crop = False)
#plot_solution(XT_test, E_pred, 1)
#savefig('./figures/E_pred', crop = False)

# Predict for plotting
#lb = XT_test.min(0)
#ub = XT_test.max(0)
#nn = 100
#x = np.linspace(lb[0], ub[0], nn)
#t = np.linspace(lb[1], ub[1], nn)
#X, T = np.meshgrid(x,t)

#RR_star = griddata(XT_test, rho_pred.flatten(), (X, T), method='cubic')
#UU_star = griddata(XT_test, u_pred.flatten(),   (X, T), method='cubic')
#PP_star = griddata(XT_test, p_pred.flatten(),   (X, T), method='cubic')
#EE_star = griddata(XT_test, E_pred.flatten(),   (X, T), method='cubic')

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
x_test_plt  = data[:,0:1].flatten()[:,None]

# y
n1_test_plt  = data[:,1:2].flatten()[:,None]
n2_test_plt  = data[:,2:3].flatten()[:,None]
n3_test_plt  = data[:,3:4].flatten()[:,None]
n4_test_plt  = data[:,4:5].flatten()[:,None]
n5_test_plt  = data[:,5:6].flatten()[:,None]
n6_test_plt  = data[:,6:7].flatten()[:,None]
n7_test_plt  = data[:,7:8].flatten()[:,None]
n8_test_plt  = data[:,8:9].flatten()[:,None]
n9_test_plt  = data[:,9:10].flatten()[:,None]
n10_test_plt = data[:,10:11].flatten()[:,None]
n11_test_plt = data[:,11:12].flatten()[:,None]
n12_test_plt = data[:,12:13].flatten()[:,None]
n13_test_plt = data[:,13:14].flatten()[:,None]
n14_test_plt = data[:,14:15].flatten()[:,None]
n15_test_plt = data[:,15:16].flatten()[:,None]
n16_test_plt = data[:,16:17].flatten()[:,None]
n17_test_plt = data[:,17:18].flatten()[:,None]
n18_test_plt = data[:,18:19].flatten()[:,None]
n19_test_plt = data[:,19:20].flatten()[:,None]
n20_test_plt = data[:,20:21].flatten()[:,None]
n21_test_plt = data[:,21:22].flatten()[:,None]
n22_test_plt = data[:,22:23].flatten()[:,None]
n23_test_plt = data[:,23:24].flatten()[:,None]
n24_test_plt = data[:,24:25].flatten()[:,None]
n25_test_plt = data[:,25:26].flatten()[:,None]
n26_test_plt = data[:,26:27].flatten()[:,None]
n27_test_plt = data[:,27:28].flatten()[:,None]
n28_test_plt = data[:,28:29].flatten()[:,None]
n29_test_plt = data[:,29:30].flatten()[:,None]
n30_test_plt = data[:,30:31].flatten()[:,None]
n31_test_plt = data[:,31:32].flatten()[:,None]
n32_test_plt = data[:,32:33].flatten()[:,None]
n33_test_plt = data[:,33:34].flatten()[:,None]
n34_test_plt = data[:,34:35].flatten()[:,None]
n35_test_plt = data[:,35:36].flatten()[:,None]
n36_test_plt = data[:,36:37].flatten()[:,None]
n37_test_plt = data[:,37:38].flatten()[:,None]
n38_test_plt = data[:,38:39].flatten()[:,None]
n39_test_plt = data[:,39:40].flatten()[:,None]
n40_test_plt = data[:,40:41].flatten()[:,None]
n41_test_plt = data[:,41:42].flatten()[:,None]
n42_test_plt = data[:,42:43].flatten()[:,None]
n43_test_plt = data[:,43:44].flatten()[:,None]
n44_test_plt = data[:,44:45].flatten()[:,None]
n45_test_plt = data[:,45:46].flatten()[:,None]
n46_test_plt = data[:,46:47].flatten()[:,None]
n47_test_plt = data[:,47:48].flatten()[:,None]
nat_test_plt = data[:,48:49].flatten()[:,None]

rho_test_plt = data[:,49:50].flatten()[:,None]
u_test_plt   = data[:,50:51].flatten()[:,None]
p_test_plt   = data[:,51:52].flatten()[:,None]
E_test_plt   = data[:,52:53].flatten()[:,None]

R1_test_plt  = data[:,53:54].flatten()[:,None]
R2_test_plt  = data[:,54:55].flatten()[:,None]
R3_test_plt  = data[:,55:56].flatten()[:,None]
R4_test_plt  = data[:,56:57].flatten()[:,None]
R5_test_plt  = data[:,57:58].flatten()[:,None]
R6_test_plt  = data[:,58:59].flatten()[:,None]
R7_test_plt  = data[:,59:60].flatten()[:,None]
R8_test_plt  = data[:,60:61].flatten()[:,None]
R9_test_plt  = data[:,61:62].flatten()[:,None]
R10_test_plt = data[:,62:63].flatten()[:,None]
R11_test_plt = data[:,63:64].flatten()[:,None]
R12_test_plt = data[:,64:65].flatten()[:,None]
R13_test_plt = data[:,65:66].flatten()[:,None]
R14_test_plt = data[:,66:67].flatten()[:,None]
R15_test_plt = data[:,67:68].flatten()[:,None]
R16_test_plt = data[:,68:69].flatten()[:,None]
R17_test_plt = data[:,69:70].flatten()[:,None]
R18_test_plt = data[:,70:71].flatten()[:,None]
R19_test_plt = data[:,71:72].flatten()[:,None]
R20_test_plt = data[:,72:73].flatten()[:,None]
R21_test_plt = data[:,73:74].flatten()[:,None]
R22_test_plt = data[:,74:75].flatten()[:,None]
R23_test_plt = data[:,75:76].flatten()[:,None]
R24_test_plt = data[:,76:77].flatten()[:,None]
R25_test_plt = data[:,77:78].flatten()[:,None]
R26_test_plt = data[:,78:79].flatten()[:,None]
R27_test_plt = data[:,79:80].flatten()[:,None]
R28_test_plt = data[:,80:81].flatten()[:,None]
R29_test_plt = data[:,81:82].flatten()[:,None]
R30_test_plt = data[:,82:83].flatten()[:,None]
R31_test_plt = data[:,83:84].flatten()[:,None]
R32_test_plt = data[:,84:85].flatten()[:,None]
R33_test_plt = data[:,85:86].flatten()[:,None]
R34_test_plt = data[:,86:87].flatten()[:,None]
R35_test_plt = data[:,87:88].flatten()[:,None]
R36_test_plt = data[:,88:89].flatten()[:,None]
R37_test_plt = data[:,89:90].flatten()[:,None]
R38_test_plt = data[:,90:91].flatten()[:,None]
R39_test_plt = data[:,91:92].flatten()[:,None]
R40_test_plt = data[:,92:93].flatten()[:,None]
R41_test_plt = data[:,93:94].flatten()[:,None]
R42_test_plt = data[:,94:95].flatten()[:,None]
R43_test_plt = data[:,95:96].flatten()[:,None]
R44_test_plt = data[:,96:97].flatten()[:,None]
R45_test_plt = data[:,97:98].flatten()[:,None]
R46_test_plt = data[:,98:99].flatten()[:,None]
R47_test_plt = data[:,99:100].flatten()[:,None]
Rat_test_plt = data[:,100:101].flatten()[:,None]

# Prediction (for plotting)
n1_pred_plt,  n2_pred_plt,  n3_pred_plt,  n4_pred_plt,  n5_pred_plt,  n6_pred_plt,  n7_pred_plt,  n8_pred_plt,  n9_pred_plt,  \
n10_pred_plt, n11_pred_plt, n12_pred_plt, n13_pred_plt, n14_pred_plt, n15_pred_plt, n16_pred_plt, n17_pred_plt, n18_pred_plt, \
n19_pred_plt, n20_pred_plt, n21_pred_plt, n22_pred_plt, n23_pred_plt, n24_pred_plt, n25_pred_plt, n26_pred_plt, n27_pred_plt, \
n28_pred_plt, n29_pred_plt, n30_pred_plt, n31_pred_plt, n32_pred_plt, n33_pred_plt, n34_pred_plt, n35_pred_plt, n36_pred_plt, \
n37_pred_plt, n38_pred_plt, n39_pred_plt, n40_pred_plt, n41_pred_plt, n42_pred_plt, n43_pred_plt, n44_pred_plt, n45_pred_plt, \
n46_pred_plt, n47_pred_plt, nat_pred_plt, rho_pred_plt, u_pred_plt, p_pred_plt, E_pred_plt,                                   \
R1_pred_plt,  R2_pred_plt,  R3_pred_plt,  R4_pred_plt,  R5_pred_plt,  R6_pred_plt,  R7_pred_plt,  R8_pred_plt,  R9_pred_plt,  \
R10_pred_plt, R11_pred_plt, R12_pred_plt, R13_pred_plt, R14_pred_plt, R15_pred_plt, R16_pred_plt, R17_pred_plt, R18_pred_plt, \
R19_pred_plt, R20_pred_plt, R21_pred_plt, R22_pred_plt, R23_pred_plt, R24_pred_plt, R25_pred_plt, R26_pred_plt, R27_pred_plt, \
R28_pred_plt, R29_pred_plt, R30_pred_plt, R31_pred_plt, R32_pred_plt, R33_pred_plt, R34_pred_plt, R35_pred_plt, R36_pred_plt, \
R37_pred_plt, R38_pred_plt, R39_pred_plt, R40_pred_plt, R41_pred_plt, R42_pred_plt, R43_pred_plt, R44_pred_plt, R45_pred_plt, \
R46_pred_plt, R47_pred_plt, Rat_pred_plt = model.predict(x_test_plt)

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

