from mpmath import polylog, exp
from scipy.optimize import newton

def root(A,B,t):
    return abs(newton(lambda x: (x - A*polylog(1.5, B*exp(-t*x))), 0.0))

