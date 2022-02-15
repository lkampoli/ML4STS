#!/usr/bin/env python
# coding: utf-8

# # Improving symbolic regression with linear scaling
# Generally, if the constants in a mathematical model are continuous real numbers, then it is difficult for both GP/GEP to find the true value of such constants. In many cases, GEP can obtain the true *structure* of the model very quickly, after which it will spend much more time in optimizing the constants. However, even with the [GEP-RNC](numerical_expression_inference-RNC.ipynb) algorithm proposed by Cândida Ferreira, it is still quite difficult for GEP to locate a good value for the constants, even after a lot of efforts have been spent by GEP.
# 
# In the paper *Improving Symbolic Regression with Interval Arithmetic and Linear Scaling* and [*Scaled Symbolic Regression*](https://www.researchgate.net/publication/220286036_Scaled_Symbolic_Regression), a simple yet effective method is developed to improve symbolic regression of arbitrary mathematical expressions. The intuition of this method is that **by rescaling the expression found by GEP/GP to the optimal slope and intercept, it will focus its search on expressions that are close in shape with the target, instead of demanding that first the scale is correct. Thus, GEP/GP will focus on the construction of the model structure, which it is extremely good at, and the linear scaling technique can help find the constant coefficients.**
# 
# We only need to change the evaluation routine when implementing the linear scaling technique. Instead of the original mean squared error (MSE) objective, first rescale the model by determining a pair of weights $(a, b)$ to 
# \begin{equation}
# min_{a,b} \quad \frac{1}{N} \sum_{i=1}^N (y_i - (ay_{pi}+b))^2
# \end{equation}
# where $y_i$ and $y_{pi}$ are the true value and predicted value for the $i^{th}$ example's target variable.
# 
# Thus, if the best model found by GEP finally is $g(x)$, then the scaled model is $ag(x)+b$.

# In[1]:


import geppy as gep
from deap import creator, base, tools
import numpy as np
import random

# for reproduction
s = 10
random.seed(s)
np.random.seed(s)

# whether enable linear scaling
LINEAR_SCALING = True


# # Synthetic dataset
# 
# For this simple task, we first choose a ground truth function $f$ to generate a dataset $D$. Then, 50 input-output exampels are generated randomly.

# In[2]:


def f(x):
    """Ground truth function"""
    return -2.45 * x ** 2 + 9.87 * x  + 14.56


# In[3]:


n_cases = 100
X = np.random.uniform(-10, 10, size=n_cases)   # random numbers in range [-10, 10)
Y = f(X) + np.random.normal(size=n_cases)   # Gaussian noise
Y = f(X)


# # Creating the primitives set
# The first step in GEP (or GP as well) is to specify the primitive set, which contains the elementary building blocks to formulate the model. For this problem, we have:
# + function set: the standard arithmetic operators addition (+), subtraction (-), multiplication (*), and division (/).
# + terminal set: only the single input 'x' and random numerical constants (RNC).
# 
# NOTE:
# 
# - We define a *protected division* to avoid dividing by zero.
# - Even there may be multiple RNCs in the model, we only need to call `PrimitiveSet.add_rnc` once.

# In[4]:


def protected_div(a, b):
    if np.isscalar(b):
        if abs(b) < 1e-6:
            b = 1
    else:
        b[abs(b) < 1e-6] = 1
    return a / b


# In[5]:


import operator 

pset = gep.PrimitiveSet('Main', input_names=['x'])
pset.add_function(operator.add, 2)
pset.add_function(operator.sub, 2)
pset.add_function(operator.mul, 2)
pset.add_function(protected_div, 2)
pset.add_ephemeral_terminal(name='enc', gen=lambda: random.uniform(-5, 5)) # each ENC is a random integer within [-10, 10]


# # Create the individual and population
# Our objective is to **minimize** the MSE (mean squared error) for data fitting.
# ## Define the indiviudal class, a subclass of *gep.Chromosome*

# In[6]:


from deap import creator, base, tools

creator.create("FitnessMin", base.Fitness, weights=(-1,))  # to minimize the objective (fitness)
creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMin, a=float, b=float)


# ## Register the individual and population creation operations
# In DEAP, it is recommended to register the operations used in evolution into a *toolbox* to make full use of DEAP functionality. The configuration of individuals in this problem is:
# + head length: 6
# + number of genes in each chromosome: 2
# 
# Generally, more complicated problems require a larger head length and longer chromosomes formed with more genes.

# In[7]:


h = 7 # head length
n_genes = 2   # number of genes in a chromosome


# In[8]:


toolbox = gep.Toolbox()
toolbox.register('gene_gen', gep.Gene, pset=pset, head_length=h)
toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=n_genes, linker=operator.add)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# compile utility: which translates an individual into an executable function (Lambda)
toolbox.register('compile', gep.compile_, pset=pset)


# # Define the fitness evaluation function
# In DEAP, the single objective optimization problem is just a special case of more general multiobjective ones. Since *geppy* is built on top of DEAP, it conforms to this convention. **Even if the fitness only contains one measure, keep in mind that DEAP stores it as an iterable.** 
# 
# Knowing that, you can understand why the evaluation function must return a tuple value (even if it is a 1-tuple). That's also why we set ``weights=(-1,)`` when creating the ``FitnessMax`` class.

# In[9]:


def evaluate(individual):
    """Evalute the fitness of an individual: MSE (mean squared error)"""
    func = toolbox.compile(individual)
    Yp = func(X)   # predictions with the GEP model
    return np.mean((Y - Yp) ** 2),


# In[10]:


def evaluate_linear_scaling(individual):
    """Evaluate the fitness of an individual with linearly scaled MSE.
    Get a and b by minimizing (a*Yp + b - Y)"""
    func = toolbox.compile(individual)
    Yp = func(X)
    
    # special cases: (1) individual has only a terminal 
    #  (2) individual returns the same value for all test cases, like 'x - x + 10'. np.linalg.lstsq will fail in such cases.
    
    if isinstance(Yp, np.ndarray):
        Q = np.hstack((np.reshape(Yp, (-1, 1)), np.ones((len(Yp), 1))))
        (individual.a, individual.b), residuals, _, _ = np.linalg.lstsq(Q, Y, rcond=None)   
        # residuals is the sum of squared errors
        if residuals.size > 0:
            return residuals[0] / len(Y),   # MSE
    
    # for the above special cases, the optimal linear scaling is just the mean of true target values
    individual.a = 0
    individual.b = np.mean(Y)
    return np.mean((Y - individual.b) ** 2),


# In[11]:


if LINEAR_SCALING:
    toolbox.register('evaluate', evaluate_linear_scaling)
else:
    toolbox.register('evaluate', evaluate)


# # Register genetic operators
# Compared with GP and other genetic algorithms, GEP has its own set of genetic operators aside from common mutation and crossover. For details, please check the tutorial [Introduction to gene expression programming](https://geppy.readthedocs.io/en/latest/intro_GEP.html).
# 
# In the following code, the selection operator is ``tools.selTournament`` provided by DEAP, while all other operators are specially designed for GEP in *geppy*.

# In[12]:


toolbox.register('select', tools.selTournament, tournsize=3)
# 1. general operators
toolbox.register('mut_uniform', gep.mutate_uniform, pset=pset, ind_pb=0.05, pb=1)
toolbox.register('mut_invert', gep.invert, pb=0.1)
toolbox.register('mut_is_transpose', gep.is_transpose, pb=0.1)
toolbox.register('mut_ris_transpose', gep.ris_transpose, pb=0.1)
toolbox.register('mut_gene_transpose', gep.gene_transpose, pb=0.1)
toolbox.register('cx_1p', gep.crossover_one_point, pb=0.4)
toolbox.register('cx_2p', gep.crossover_two_point, pb=0.2)
toolbox.register('cx_gene', gep.crossover_gene, pb=0.1)
toolbox.register('mut_ephemeral', gep.mutate_uniform_ephemeral, ind_pb='1p')  # 1p: expected one point mutation in an individual
toolbox.pbs['mut_ephemeral'] = 1  # we can also give the probability via the pbs property


# # Statistics to be inspected
# We often need to monitor of progress of an evolutionary program. DEAP offers two classes to handle the boring work of recording statistics. Details are presented in [Computing statistics](http://deap.readthedocs.io/en/master/tutorials/basic/part3.html). In the following, we are intereted in the average/standard deviation/min/max of all the individuals' fitness in each generation.

# In[13]:


stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)


# # Launch evolution
# We make use of *geppy*'s builtin algorithm ``gep_rnc`` here to perform the GEP-RNC evolution. A special class from DEAP, `HallOfFame`, is adopted to store the best individuals ever found. Besides, it should be noted that in GEP [*elitism*](https://en.wikipedia.org/wiki/Genetic_algorithm#Elitism) is highly recommended because some genetic operators in GEP are destructive and may destroy the best individual we have evolved.

# In[14]:


# size of population and number of generations
n_pop = 100
n_gen = 200

pop = toolbox.population(n=n_pop)
hof = tools.HallOfFame(3)   # only record the best three individuals ever found in all generations

# start evolution
pop, log = gep.gep_simple(pop, toolbox, n_generations=n_gen, n_elites=1,
                          stats=stats, hall_of_fame=hof, verbose=True)


# **Let's check the best individuals ever evolved.**

# In[15]:


print(hof[0])


# # *[optional]* Post-processing: simplification and visualization
# ## Symbolic simplification of the final solution
# The original solution seems a little complicated, which may contain many redundancies, for example, `protected_div(x, x)` is just 1. We can perform symbolic simplification of the final result by `geppy.simplify` which depends on `sympy` package.

# In[16]:


for i in range(3):
    ind = hof[i]
    symplified_model = gep.simplify(ind)
    if LINEAR_SCALING:
        symplified_model = ind.a * symplified_model + ind.b
    print('Symplified best individual {}: '.format(i))
    print(symplified_model)


# As we can see from the above simplified expression, the *truth model* has been successfully found. Due to the existence of Gaussian noise, the minimum mean absolute error （MAE) is still not zero even the best individual represents the true model.

# ## Visualization
# If you are interested in the expression tree corresponding to the individual, i.e., the genotype/phenotype system, *geppy* supports tree visualization by the `graph` and the `export_expression_tree` functions:
# 
# - `graph` only outputs the nodes and links information to describe the tree topology, with which you can render the tree with tools you like;
# - `export_expression_tree` implements tree visualization with data generated by `graph` internally using the `graphviz` package. 

# In[17]:


# we want use symbol labels instead of words in the tree graph
rename_labels = {'add': '+', 'sub': '-', 'mul': '*', 'protected_div': '/'}  
best_ind = hof[0]
gep.export_expression_tree(best_ind, rename_labels, 'data/numerical_expression_tree.png')


# In[18]:


# show the above image here for convenience
from IPython.display import Image
Image(filename='data/numerical_expression_tree.png') 


# # Discussion
# **With the given no-noise data, a normal GEP can only find a model with $MSE=52.0646$, though the model structure is right. By contrast, linear scaling-enabled GEP obtains a solution with $MSE=2.03544e^{-5}$, which is qualified to be considered as the true model.**
# 
# If only integer constants are involved, then the GEP-RNC algorithm and the ENC-based algorithm can both be adopted. However, generally the GEP-RNC algorithm is more effective.
# 
# In other cases where the model is more complicated involving real coefficients and constants, then both of the above two algorithms will have difficulty in optimizing these constants. A better way is to incorporate a local optimizater into GEP dedicated to constant number optimization, or use other techniques like the linear scaling here. 

# In[ ]:




