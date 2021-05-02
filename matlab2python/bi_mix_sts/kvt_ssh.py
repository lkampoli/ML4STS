# Generated with SMOP  0.41-beta
from libsmop import *
# kvt_ssh.m

    
@function
def kvt_ssh(t=None,*args,**kwargs):
    varargin = kvt_ssh.varargin
    nargin = kvt_ssh.nargin

    # function for vib. exch. rates, SHH model, Stupochenko
# t - temperature, K
# sw_o - oscillator switcher
# k_down - array of the rate coef-s of VT exchanges A2(i+1)+M = A2(i)+M
# i+1 - vib. level before collision
# i - vib. level after collision
    
    global k,c,h,h_bar,om_e,om_x_e,l,sw_o,m,r0,em,re
    # constantes
    if sw_o == 1:
        om10=om_e - dot(2,om_x_e)
# kvt_ssh.m:12
    else:
        if sw_o == 2:
            om10=copy(om_e)
# kvt_ssh.m:14
    
    Lmax=l - 1
# kvt_ssh.m:16
    
    i=(arange(0,Lmax - 1))
# kvt_ssh.m:17
    om0=dot(dot(dot(2,pi),c),om10)
# kvt_ssh.m:18
    
    # reduced masses, kg
    mu=dot(m(1),m) / (m(1) + m)
# kvt_ssh.m:21
    # inverse radius in 1st approximation m^-1
    a=17.5 / r0
# kvt_ssh.m:24
    # collision frequency, m^3/sec
    Z=dot(r0 ** 2.0 / sqrt(mu),sqrt(dot(dot(dot(8,pi),k),t)))
# kvt_ssh.m:27
    # firstly, find rates of transition from 1st lev. to 2nd lev.
    chi=(dot(dot(dot(pi ** 2,om0 ** 2) / 2 / k,(mu / a ** 2)),t ** (- 1))) ** (1 / 3)
# kvt_ssh.m:30
    
    
    r=multiply(r0,(dot(0.5,sqrt(1 + (dot(chi,t)) / em)) + 0.5) ** (- 1 / 6))
# kvt_ssh.m:32
    # steric factor
    Z_0=multiply((multiply(a,re)) ** 2,exp(- ((multiply(dot(3,a),re ** 2)) / (dot(8,r)))))
# kvt_ssh.m:34
    Const=dot(dot(dot(dot(1.294 / Z_0,4),pi ** 2),om0) / h_bar,sqrt(dot(4,pi) / 3))
# kvt_ssh.m:36
    
    r_c_r_02=(dot(0.5,sqrt(1 + (dot(chi,t)) / em)) + 0.5) ** (- 1 / 3)
# kvt_ssh.m:38
    
    P10=multiply(dot(multiply(multiply(multiply(Const,r_c_r_02),(1 + dot(1.1,em) / t) ** (- 1)),mu) / a ** 2.0,chi ** 0.5),exp(dot(- 3,chi) + dot(h_bar,om0) / (dot(dot(2,k),t)) + em / t))
# kvt_ssh.m:40
    k10=multiply(Z,P10)
# kvt_ssh.m:42
    
    # secondly, find rates coef-s for transition i+1 -> i
    if sw_o == 2:
        k_down=dot(k10,(i + 1))
# kvt_ssh.m:46
    else:
        if sw_o == 1:
            # anharmonicity factor for VT transitions (Gordietz)
            aA=dot(a,1e-10)
# kvt_ssh.m:49
            # reduced masses
            mu_amu=dot(m(1),m) / (m(1) + m) / 1.6605e-27
# kvt_ssh.m:51
            # adiabatic factor for transition i+1 -> i
    # E(i+1) - E(i)
            diffE=dot(dot((om_e - dot(dot(2,(i + 1)),om_x_e)),h),c)
# kvt_ssh.m:54
            E1=dot((om_e - dot(2,om_x_e)),0.014388)
# kvt_ssh.m:56
            dE=dot(om_x_e,0.014388)
# kvt_ssh.m:57
            gamma_n=multiply(pi / a / h_bar,sqrt(mu / (dot(dot(2,k),t))))
# kvt_ssh.m:59
            gamma_n=dot(gamma_n,diffE)
# kvt_ssh.m:60
            gamma_0=dot(multiply(0.32 / aA,sqrt(mu_amu / t)),E1)
# kvt_ssh.m:62
            delta=zeros(2,length(i))
# kvt_ssh.m:64
            delta[1,arange()]=multiply((dot(dot(4 / 3,gamma_0(1)),dE) / E1) ** (gamma_n(1,arange()) < 20),(dot(dot(4,(gamma_0(1)) ** (2 / 3)),dE) / E1) ** (gamma_n(1,arange()) >= 20))
# kvt_ssh.m:65
            delta[2,arange()]=multiply((dot(dot(4 / 3,gamma_0(1)),dE) / E1) ** (gamma_n(1,arange()) < 20),(dot(dot(4,(gamma_0(1)) ** (2 / 3)),dE) / E1) ** (gamma_n(1,arange()) >= 20))
# kvt_ssh.m:67
            k_down[1,arange()]=multiply(multiply(dot((i + 1),k10(1)),exp(multiply(i,delta(1,arange())))),exp(dot(dot(dot(- i,h),c),om_x_e) / (dot(k,t))))
# kvt_ssh.m:70
            k_down[2,arange()]=multiply(multiply(dot((i + 1),k10(2)),exp(multiply(i,delta(2,arange())))),exp(dot(dot(dot(- i,h),c),om_x_e) / (dot(k,t))))
# kvt_ssh.m:71
    