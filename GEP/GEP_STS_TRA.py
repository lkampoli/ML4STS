#!/usr/bin/env python
# coding: utf-8

import os
import geppy as gep
from deap import creator, base, tools
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

import operator 
import math
import datetime

from numba import jit

# for reproduction
s = 0
random.seed(s)
np.random.seed(s)

# doublecheck the data is there
print(os.listdir("./"))

# read in the data to pandas
data = np.loadtxt("./data/STS/shear_viscosity.txt")
print(data.shape)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score

#x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=69)
#
#sc_x = StandardScaler()
#sc_y = StandardScaler()
#
## fit scaler
#sc_x.fit(x_train)
#
## transform training dataset
#x_train = sc_x.transform(x_train)
#
## transform test dataset
#x_test = sc_x.transform(x_test)
#
## fit scaler on training dataset
#sc_y.fit(y_train.reshape(-1,1))
#
## transform training dataset
#y_train = sc_y.transform(y_train)
#
## transform test dataset
#y_test = sc_y.transform(y_test)

# Split my data into Train and Test chunks, 20/80
msk     = np.random.rand(len(data)) < 0.8
train   = data[msk]
holdout = data[~msk]

# check the number of records we'll validate our MSE with
print("holdout",holdout.shape)

# check the number of records we'll train our algorithm with
print("train",train.shape)

# NOTE: I'm only feeding in the TRAIN values to the algorithms. 
# Later I will independently check the MSE myself using a holdout test dataset

P    = train[:,0]; print("P=",P) # data[:,1].reshape((-1,1))
T    = train[:,1]; print("T=",T)
x_1  = train[:,2]; print("x_1=",x_1) 
x_2  = train[:,3] 
x_3  = train[:,4] 
x_4  = train[:,5]
x_5  = train[:,6]
x_6  = train[:,7]
x_7  = train[:,8] 
x_8  = train[:,9] 
x_9  = train[:,10] 
x_10 = train[:,11] 
x_11 = train[:,12]
x_12 = train[:,13]
x_13 = train[:,14] 
x_14 = train[:,15] 
x_15 = train[:,16] 
x_16 = train[:,17]
x_17 = train[:,18] 
x_18 = train[:,19] 
x_19 = train[:,20] 
x_20 = train[:,21] 
x_21 = train[:,22] 
x_22 = train[:,23] 
x_23 = train[:,24] 
x_24 = train[:,25] 
x_25 = train[:,26] 
x_26 = train[:,27] 
x_27 = train[:,28]
x_28 = train[:,29] 
x_29 = train[:,30] 
x_30 = train[:,31] 
x_31 = train[:,32] 
x_32 = train[:,33] 
x_33 = train[:,34] 
x_34 = train[:,35] 
x_35 = train[:,36]
x_36 = train[:,37] 
x_37 = train[:,38] 
x_38 = train[:,39] 
x_39 = train[:,40]
x_40 = train[:,41] 
x_41 = train[:,42] 
x_42 = train[:,43] 
x_43 = train[:,44] 
x_44 = train[:,45] 
x_45 = train[:,46] 
x_46 = train[:,47] 
x_47 = train[:,48];
x_48 = train[:,49]; print("x_48=",x_48) 
x_at = train[:,50]; print("x_at=",x_at) 
shear= train[:,51]; print("shear=",shear) # target

Y = shear         # this is our target, now mapped to Y

# Creating the primitives set
# The first step in GEP (or GP as well) is to specify the primitive set, 
# which contains the elementary building blocks to formulate the model. 
# For this problem, we have:
# + function set: the standard arithmetic operators:
#   addition (+), subtraction (-), multiplication (*), and division (/).
# + terminal set: only the single input 'x' and random numerical 
#   constants (RNC).
# 
# NOTE:
# 
# - We define a *protected division* to avoid dividing by zero.
# - Even there may be multiple RNCs in the model, we only need 
#   to call `PrimitiveSet.add_rnc` once.

def protected_div(x1, x2):
    if abs(x2) < 1e-6:
       return 1
    return x1 / x2

def protected_div1(x1, x2):
    if abs(x2) < 1e-6:
        return 1
    return x1 / x2

def protected_div2(x1, x2):
    if np.isscalar(x2):
        if abs(x2) < 1e-6:
            x2 = float('nan')
    else:
        x2[x2<1e-6] = float('nan')
    return x1 / x2

def protected_exp(x1):
    try:
        temp = math.exp(x1)
        if temp < 10000:
            ans = temp
        else:
            ans = 0
    except OverflowError:
        ans = 0
    return ans

def root(x1):
    return np.sqrt(x1)
    #return math.sqrt(x1)

def epow(x1):
    return np.exp(x1)
    #return math.exp(x1)

# Map our input data to the GEP variables
# Here we map the input data to the GEP algorithm:
# 
# We do that by listing the field names as "input_names".
# 
# In reviewing geppy code, in the file:  
#   geppy/geppy/core/symbol.py
#   
# we find how terminals in the gene are named correctly to match input
# data.
# 
# Oh - notice, I only mapped in below the input data columes, 
#      and not the TARGET "PE" which is sitting in var Y.
# I didn't notice where to map that - so suggest you force the target 
# variable to "Y" when reading in data.

pset = gep.PrimitiveSet('Main', input_names=['P','T','x_1','x_2','x_3','x_4','x_5','x_6','x_7','x_8','x_9','x_10','x_11','x_12','x_13','x_14','x_15','x_16','x_17','x_18','x_19','x_20','x_21','x_22','x_23','x_24','x_25','x_26','x_27','x_28','x_29','x_30','x_31','x_32','x_33','x_34','x_35','x_36','x_37','x_38','x_39','x_40','x_41','x_42','x_43','x_44','x_45','x_46','x_47','x_48','x_at'])

# Define the operators
# Here we define and pass the operators we'll construct our final 
# symbolic regression function with
pset.add_function(operator.add, 2)
pset.add_function(operator.sub, 2)
pset.add_function(operator.mul, 2)
#pset.add_function(operator.truediv, 2)
pset.add_function(protected_div, 2)
#pset.add_function(protected_exp, 1)
#pset.add_function(root, 1)
#pset.add_function(epow, 1)
pset.add_function(math.sin, 1) 
pset.add_function(math.cos, 1)
pset.add_function(math.tan, 1)
#pset.add_function(math.atan, 1)
#pset.add_function(math.pow, 1)
#pset.add_function(math.sqrt, 1)
#pset.add_function(math.exp, 1)
#pset.add_function(math.log, 1)
pset.add_rnc_terminal()
#pset.add_pow_terminal('X') #attention: Must the same as input in primitive set
#pset.add_pow_terminal('Y')

#pset.add_constant_terminal(1.0)

# Create the individual and population
# Our objective is to **minimize** the MSE (mean squared error) for 
# data fitting.
# Define the indiviudal class, a subclass of *gep.Chromosome*
creator.create("FitnessMin", base.Fitness, weights=(-1,))  # to minimize the objective (fitness)
creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMin)

# Register the individual and population creation operations
# In DEAP, it is recommended to register the operations used in evolution into a *toolbox* to make full use of DEAP functionality. The configuration of individuals in this problem is:
# + head length: 6
# + number of genes in each chromosome: 2
# + RNC array length: 8
# 
# Generally, more complicated problems require a larger head length and longer chromosomes formed with more genes. **The most important is that we should use the `GeneDc` class for genes to make use of the GEP-RNC algorithm.**

# Your Core Settings Defined Here

h = 7            # head length
n_genes = 2      # number of genes in a chromosome
r = h*2 + 1      # length of the RNC array
enable_ls = True # whether to apply the linear scaling technique
N_eval = 1

# **NOTE** Above you define the gene structure which sets out the maximum complexity of the symbolic regression

toolbox = gep.Toolbox()
toolbox.register('rnc_gen', random.randint, a=-10, b=10) # each RNC is random integer within [0, 10]
#toolbox.register('rnc_gen', random.choice, np.arange(0.1,10.0,0.1))
toolbox.register('gene_gen', gep.GeneDc, pset=pset, head_length=h, rnc_gen=toolbox.rnc_gen, rnc_array_length=r)
toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=n_genes, linker=operator.add)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# compile utility: which translates an individual into an executable function (Lambda)
toolbox.register('compile', gep.compile_, pset=pset)

# Define the fitness evaluation function
# In DEAP, the single objective optimization problem is just a special case of 
# more general multiobjective ones. Since *geppy* is built on top of DEAP, it 
# conforms to this convention. **Even if the fitness only contains one measure, 
# keep in mind that DEAP stores it as an iterable.** 
# 
# Knowing that, you can understand why the evaluation function must return a 
# tuple value (even if it is a 1-tuple). That's also why we set:
# ``weights=(-1,)`` when creating the ``FitnessMax`` class.

#@jit
def evaluate(individual):
    """Evalute the fitness of an individual: MAE (mean absolute error)"""
    func = toolbox.compile(individual)
    
    # below call the individual as a function over the inputs
    Yp = np.array(list(map(func,P,T,x_1,x_2,x_3,x_4,x_5,x_6,x_7,x_8,x_9,x_10,x_11,x_12,x_13,x_14,x_15,x_16,x_17,x_18,x_19,x_20,x_21,x_22,x_23,x_24,x_25,x_26,x_27,x_28,x_29,x_30,x_31,x_32,x_33,x_34,x_35,x_36,x_37,x_38,x_39,x_40,x_41,x_42,x_43,x_44,x_45,x_46,x_47,x_48,x_at))) 
    
    # return the MSE as we are evaluating on it anyway,
    # then the stats are more fun to watch...
    return np.mean((Y - Yp) ** 2),


# [optional] Enable the linear scaling technique. It is hard for GP to determine 
# real constants, which are important in regression problems. Thus, we can 
# (implicitly) ask GP to evolve the shape (form) of the model and we help GP to 
# determine constans by applying the simple least squares method (LSM).

#@jit
def evaluate_ls(individual):
    """
    First apply linear scaling (ls) to the individual 
    and then evaluate its fitness: MSE (mean squared error)
    """
    func = toolbox.compile(individual)

    Yp = np.array(list(map(func,P,T,x_1,x_2,x_3,x_4,x_5,x_6,x_7,x_8,x_9,x_10,x_11,x_12,x_13,x_14,x_15,x_16,x_17,x_18,x_19,x_20,x_21,x_22,x_23,x_24,x_25,x_26,x_27,x_28,x_29,x_30,x_31,x_32,x_33,x_34,x_35,x_36,x_37,x_38,x_39,x_40,x_41,x_42,x_43,x_44,x_45,x_46,x_47,x_48,x_at))) 
    
    # special cases which cannot be handled by np.linalg.lstsq: (1) individual has only a terminal 
    #  (2) individual returns the same value for all test cases, like 'x - x + 10'. np.linalg.lstsq will fail in such cases.
    # That is, the predicated value for all the examples remains identical, which may happen in the evolution.
    if isinstance(Yp, np.ndarray):
        Q = np.hstack((np.reshape(Yp, (-1, 1)), np.ones((len(Yp), 1))))
        (individual.a, individual.b), residuals, _, _ = np.linalg.lstsq(Q, Y, rcond=-1)
        #(individual.a, individual.b), residuals, _, _ = np.linalg.lstsq(Q, Y)
        # residuals is the sum of squared errors
        if residuals.size > 0:
            return residuals[0] / len(Y),   # MSE
    
    # regarding the above special cases, the optimal linear scaling w.r.t LSM is just the mean of true target values
    individual.a = 0
    individual.b = np.mean(Y)
    return np.mean((Y - individual.b) ** 2),


if enable_ls:
    toolbox.register('evaluate', evaluate_ls)
else:
    toolbox.register('evaluate', evaluate)


# Register genetic operators
# Compared with GP and other genetic algorithms, GEP has its own set 
# of genetic operators aside from common mutation and crossover. For 
# details, please check the tutorial:
# [Introduction to gene expression programming](https://geppy.readthedocs.io/en/latest/intro_GEP.html).
# 
# In the following code, the selection operator is ``tools.selTournament`` 
# provided by DEAP, while all other operators are specially designed for GEP in *geppy*.

toolbox.register('select', tools.selTournament, tournsize=3)
# 1. general operators
toolbox.register('mut_uniform', gep.mutate_uniform, pset=pset, ind_pb=0.05, pb=1)
toolbox.register('mut_invert', gep.invert, pb=0.1)
toolbox.register('mut_is_transpose', gep.is_transpose, pb=0.1)
toolbox.register('mut_ris_transpose', gep.ris_transpose, pb=0.1)
toolbox.register('mut_gene_transpose', gep.gene_transpose, pb=0.1)
toolbox.register('cx_1p', gep.crossover_one_point, pb=0.3)
toolbox.register('cx_2p', gep.crossover_two_point, pb=0.2)
toolbox.register('cx_gene', gep.crossover_gene, pb=0.1)
# 2. Dc-specific operators
toolbox.register('mut_dc', gep.mutate_uniform_dc, ind_pb=0.05, pb=1)
toolbox.register('mut_invert_dc', gep.invert_dc, pb=0.1)
toolbox.register('mut_transpose_dc', gep.transpose_dc, pb=0.1)
# for some uniform mutations, we can also assign the ind_pb a string to indicate our expected number of point mutations in an individual
toolbox.register('mut_rnc_array_dc', gep.mutate_rnc_array_dc, rnc_gen=toolbox.rnc_gen, ind_pb='0.5p')
toolbox.pbs['mut_rnc_array_dc'] = 1  # we can also give the probability via the pbs property

# Statistics to be inspected
# We often need to monitor of progress of an evolutionary program. 
# DEAP offers two classes to handle the boring work of recording statistics. 
# Details are presented in [Computing statistics](http://deap.readthedocs.io/en/master/tutorials/basic/part3.html). 
# In the following, we are intereted in the average/standard 
# deviation/min/max of all the individuals' fitness in each generation.

stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

Max_evolution = empty_lists = [ [] for i in range(N_eval) ]
Hof_save = empty_lists = [ [] for i in range(N_eval) ]
Max_fit = np.zeros(N_eval)

# Launch evolution
# We make use of *geppy*'s builtin algorithm ``gep_rnc`` here to perform 
# the GEP-RNC evolution. A special class from DEAP, `HallOfFame`, is 
# adopted to store the best individuals ever found. Besides, it should 
# be noted that in GEP [*elitism*](https://en.wikipedia.org/wiki/Genetic_algorithm#Elitism) 
# is highly recommended because some genetic operators in GEP are 
# destructive and may destroy the best individual we have evolved.

# size of population and number of generations
n_pop  = 100
n_gen  = 100
champs = 3

pop = toolbox.population(n=n_pop)
hof = tools.HallOfFame(champs) # only record the best three individuals ever found in all generations

startDT = datetime.datetime.now()
print (str(startDT))

# start evolution
for i in range(N_eval):
    pop, log = gep.gep_simple(pop, toolbox, n_generations=n_gen, n_elites=1,
                              stats=stats, hall_of_fame=hof, verbose=True)

    Max_evolution[i] = log.select("min")
    Hof_save[i] = hof

print ("Evolution times were:\n\nStarted:\t", startDT, "\nEnded:   \t", str(datetime.datetime.now()))

# **Let's check the best individuals ever evolved.**
print(hof[0])

# extract statistics:
maxFitnessValues, meanFitnessValues = log.select("max", "avg")

# plot statistics:
sns.set_style("whitegrid")
plt.plot(maxFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Generation')
plt.ylabel('Max / Average Fitness')
plt.title('Max and Average fitness over Generations')
plt.show()

# Present our Work and Conclusions
# Symbolic simplification of the final solution
# The symbolic tree answer may contain many redundancies, for example, 
# `protected_div(x, x)` is just 1. We can perform symbolic simplification
# of the final result by `geppy.simplify` which depends on `sympy` package. 
# We can also leverage sympy to better present our work to others

# print the best symbolic regression we found
#best_ind = Hof_save[0][0]
best_ind = hof[0]
symplified_best = gep.simplify(best_ind)
best_func = toolbox.compile(best_ind)

# convergence
plt.figure()
plt.plot(Max_evolution[0])

temp = np.zeros(N_eval)
for i in range(N_eval):
    print("Hof_save:",Hof_save[i][0])
    temp[i] = Max_evolution[i][-1]


# display
#shear_test = np.linspace(np.min(shear),np.max(shear),num=10000)
#plt.figure()
#plt.semilogx(df.Pi1,10*np.log10(df.PiF),color='b',marker='.',linestyle=' ')
#for ii in np.unique(df.Pi2):
#    plt.semilogx(Pi1_test,10*np.log10(best_func(Pi1_test,ii)))
#plt.show()


if enable_ls:
    symplified_best = best_ind.a * symplified_best + best_ind.b

key= '''
#Given training examples of
#
#    P = 
#    T = 
#    x_i = 
#
#we trained a computer using Genetic Algorithms to predict the 
#
#    shear = 
#
#Our symbolic regression process found the following equation offers our best prediction:
#
#'''

print('\n', key,'\t', str(symplified_best), '\n\nwhich formally is presented as:\n\n')

from sympy import *
init_printing()
print(symplified_best)
print(str(symplified_best))

# output the top 3 champs
champs = 3
for i in range(champs):
    ind = hof[i]
    symplified_model = gep.simplify(ind)

    print('\nSymplified best individual {}: '.format(i))
    print(symplified_model)
    print("raw indivudal:")
    print(hof[i])

# we want to use symbol labels instead of words in the tree graph
rename_labels = {'add': '+', 'sub': '-', 'mul': '*', 'protected_div': '/'}  
gep.export_expression_tree(best_ind, rename_labels, 'numerical_expression_tree.png')

# As we can see from the above simplified expression, the *truth model* has been successfully found. 
# Due to the existence of Gaussian noise, the minimum mean absolute error （MAE) is still not zero even the best individual represents the true model.

# ## Visualization
# If you are interested in the expression tree corresponding to the individual, i.e., the genotype/phenotype system, *geppy* supports tree visualization by the `graph` and the `export_expression_tree` functions:
# 
# - `graph` only outputs the nodes and links information to describe the tree topology, with which you can render the tree with tools you like;
# - `export_expression_tree` implements tree visualization with data generated by `graph` internally using the `graphviz` package. 
# 
# **Note**: even if the linear scaling is applied, here only the raw individual in GP (i.e., the one without linear scaling) is visualized.

# show the above image here for convenience
from IPython.display import Image
Image(filename='numerical_expression_tree.png')

# # DoubleCheck our final Test Statistics
# Earlier, we split our data into train and test chunks.
# 
# The GEPPY program never saw 20% of our data, so lets doublecheck the reported errors on our holdout test file are accurate:

def CalculateBestModelOutput(P,T,x_1,x_2,x_3,x_4,x_5,x_6,x_7,x_8,x_9,x_10,x_11,x_12,x_13,x_14,x_15,x_16,x_17,x_18,x_19,x_20,x_21,x_22,x_23,x_24,x_25,x_26,x_27,x_28,x_29,x_30,x_31,x_32,x_33,x_34,x_35,x_36,x_37,x_38,x_39,x_40,x_41,x_42,x_43,x_44,x_45,x_46,x_47,x_48,x_at, model):
#    # pass in a string view of the "model" as str(symplified_best)
#    # this string view of the equation may reference any of the other inputs, AT, V, AP, RH we registered
#    # we then use eval of this string to calculate the answer for these inputs
    return eval(model) 

## some previous example outputs
#  2.39245892691119e-5 + 7.81268662658234e-9*(x_18*(T*(x_1 + x_40) + T + x_27) + sin(sin(x_8)))/x_18 
#  2.81267339167932e-5 + 8.63443922749263e-9*(T*(x_13 + x_46) - x_32*(x_13 + x_46) + (x_13 + x_46)*tan(x_2 + cos(tan(x_42)) - 6) + tan(x_6))/(x_13 + x_46) 

P    = holdout[:,0]
T    = holdout[:,1]
x_1  = holdout[:,2]
x_2  = holdout[:,3] 
x_3  = holdout[:,4] 
x_4  = holdout[:,5]
x_5  = holdout[:,6]
x_6  = holdout[:,7]
x_7  = holdout[:,8] 
x_8  = holdout[:,9] 
x_9  = holdout[:,10] 
x_10 = holdout[:,11] 
x_11 = holdout[:,12]
x_12 = holdout[:,13]
x_13 = holdout[:,14] 
x_14 = holdout[:,15] 
x_15 = holdout[:,16] 
x_16 = holdout[:,17]
x_17 = holdout[:,18] 
x_18 = holdout[:,19] 
x_19 = holdout[:,20] 
x_20 = holdout[:,21] 
x_21 = holdout[:,22] 
x_22 = holdout[:,23] 
x_23 = holdout[:,24] 
x_24 = holdout[:,25] 
x_25 = holdout[:,26] 
x_26 = holdout[:,27] 
x_27 = holdout[:,28]
x_28 = holdout[:,29] 
x_29 = holdout[:,30] 
x_30 = holdout[:,31] 
x_31 = holdout[:,32] 
x_32 = holdout[:,33] 
x_33 = holdout[:,34] 
x_34 = holdout[:,35] 
x_35 = holdout[:,36]
x_36 = holdout[:,37] 
x_37 = holdout[:,38] 
x_38 = holdout[:,39] 
x_39 = holdout[:,40]
x_40 = holdout[:,41] 
x_41 = holdout[:,42] 
x_42 = holdout[:,43] 
x_43 = holdout[:,44] 
x_44 = holdout[:,45] 
x_45 = holdout[:,46] 
x_46 = holdout[:,47] 
x_47 = holdout[:,48]
x_48 = holdout[:,49]
x_at = holdout[:,50]
shear= holdout[:,51]

pred_shear = CalculateBestModelOutput(P,T,x_1,x_2,x_3,x_4,x_5,x_6,x_7,x_8,x_9,x_10,x_11,x_12,x_13,x_14,x_15,x_16,x_17,x_18,x_19,x_20,x_21,x_22,x_23,x_24,x_25,x_26,x_27,x_28,x_29,x_30,x_31,x_32,x_33,x_34,x_35,x_36,x_37,x_38,x_39,x_40,x_41,x_42,x_43,x_44,x_45,x_46,x_47,x_48,x_at,str(symplified_best))

print(pred_shear.shape)
print("pred_shear=",pred_shear)

# Validation MSE
from sklearn.metrics import mean_squared_error, r2_score
print("Mean squared error: %.2f" % mean_squared_error(shear, pred_shear))
print("R2 score : %.2f" % r2_score(shear, pred_shear))

# Let's eyeball predicted vs actual data
from matplotlib import pyplot
pyplot.rcParams['figure.figsize'] = [20, 5]
plotlen=200
pyplot.plot(pred_shear) # predictions are in blue
pyplot.plot(shear)      # actual values are in orange
pyplot.savefig("shear.png", dpi=150, crop='true')
pyplot.show()

import matplotlib.pyplot as plt
# shear viscosity vs temperature
#plt.scatter(T, shear,      s=2, c='k', marker='o', label='truth')
#plt.scatter(T, pred_shear, s=2, c='g', marker='+', label='prediction')
plt.plot(T, shear)
plt.plot(T, pred_shear)
plt.ylabel('shear viscosity [Pa·s]')
plt.xlabel('T [K] ')
plt.legend()
plt.tight_layout()
plt.savefig("shear.pdf", dpi=150, crop='true')
plt.show()

# Histogram of prediction Errors on the holdout dataset
pyplot.rcParams['figure.figsize'] = [10, 5]
hfig = pyplot.figure()
ax = hfig.add_subplot(111)

numBins = 100
ax.hist(shear-pred_shear,numBins,color='green',alpha=0.8)
pyplot.show()

best_ind = hof[0]
for gene in best_ind:
    print(gene.kexpression)
