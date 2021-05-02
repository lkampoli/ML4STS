# Generated with SMOP  0.41-beta
from libsmop import *
# kvv_ssh.m

    
@function
def kvv_ssh(t=None,*args,**kwargs):
    varargin = kvv_ssh.varargin
    nargin = kvv_ssh.nargin

    # rate coef-s of VV exchanges, A2 - A2
# kdown - array of k_(i->i-1)^(j->j+1)
# t - temperature, K
    
    global c,h,k,m,om_e,om_x_e,l,sw_o,r0
    mu=dot(0.5,m(1))
# kvv_ssh.m:8
    
    m_osc=dot(0.25,m(1))
# kvv_ssh.m:9
    
    if sw_o == 1:
        om10=om_e - dot(2,om_x_e)
# kvv_ssh.m:12
    else:
        if sw_o == 2:
            om10=copy(om_e)
# kvv_ssh.m:14
    
    Lmax=l - 1
# kvv_ssh.m:17
    
    om0=dot(dot(dot(2,pi),c),om10)
# kvv_ssh.m:18
    
    
    a=17.5 / r0(1)
# kvv_ssh.m:20
    
    Z=dot(r0(1) ** 2 / sqrt(mu),sqrt(dot(dot(dot(8,pi),k),t)))
# kvv_ssh.m:21
    
    
    Q10=dot(dot(dot(dot(0.5 ** 4 / m_osc,a ** 2) / om0 ** 2,4),k),t)
# kvv_ssh.m:23
    k10=dot(Q10,Z)
# kvv_ssh.m:24
    kdown=zeros(Lmax,Lmax)
# kvv_ssh.m:26
    j_up=arange(0,Lmax - 1)
# kvv_ssh.m:27
    if sw_o == 2:
        for i_down in arange(1,Lmax).reshape(-1):
            kdown[i_down,arange()]=dot(dot(i_down,(j_up + 1)),k10)
# kvv_ssh.m:30
    else:
        if sw_o == 1:
            # anharmonicity factor for VV transitions
            aA=dot(a,1e-10)
# kvv_ssh.m:34
            mu_amu=mu / 1.6605e-27
# kvv_ssh.m:35
            dE=dot(om_x_e,0.014388)
# kvv_ssh.m:36
            delta=dot(dot(0.427 / aA,sqrt(mu_amu / t)),dE)
# kvv_ssh.m:37
            for i_down in arange(1,Lmax).reshape(-1):
                kdown[i_down,arange()]=multiply(multiply(multiply(dot(dot(i_down,(j_up + 1)),k10),exp(multiply(- delta,abs(i_down - 1 - j_up)))),(1.5 - dot(0.5,exp(multiply(- delta,abs(i_down - 1 - j_up)))))),exp(dot(dot(dot((j_up - i_down + 1),h),c),om_x_e) / (dot(k,t))))
# kvv_ssh.m:39
    