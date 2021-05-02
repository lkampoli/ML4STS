# Generated with SMOP  0.41-beta
from libsmop import *
# kdis.m

    
@function
def kdis(t=None,*args,**kwargs):
    varargin = kdis.varargin
    nargin = kdis.nargin

    # calculation of dissociation rates, collision w mol/at 
# Treanor-Marrone model
# t - temperature
# sw - switch of osc-or model (1 - anh.o./ 2 - har.o.)
    
    global k,e_i,CA,nA,D
    # equil. coef-s
    kd_eq=dot(multiply(CA,t ** nA),exp(- D / t))
# kdis.m:10
    
    # equil. vibr. partition function
    ZvT=sum(exp(- e_i / (dot(t,k))))
# kdis.m:12
    # parameter of TM model
    U=D / 6
# kdis.m:14
    # U = 3*t;
# U = Inf;
    ZvU=sum(exp(e_i / (dot(U,k))))
# kdis.m:17
    # non-equilibrium factor
    Z=dot(ZvT / ZvU,exp(dot(e_i / k,(1 / t + 1 / U))))
# kdis.m:20
    # dis. rates
    kd=dot(kd_eq.T,Z.T)
# kdis.m:23
    
    return kd
    
if __name__ == '__main__':
    pass
    