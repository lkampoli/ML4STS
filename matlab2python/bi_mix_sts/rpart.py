# Generated with SMOP  0.41-beta
from libsmop import *
# rpart.m

    
@function
def rpart(t=None,y=None,*args,**kwargs):
    varargin = rpart.varargin
    nargin = rpart.nargin

    format('long','e')
    global c,h,k,m,l,e_i,e_0,Be,D,n0,v0,T0,Delta
    Lmax=l - 1
# rpart.m:6
    ni_b=y(arange(1,l))
# rpart.m:8
    na_b=y(l + 1)
# rpart.m:9
    nm_b=sum(ni_b)
# rpart.m:10
    v_b=y(l + 2)
# rpart.m:11
    T_b=y(l + 3)
# rpart.m:12
    xx=dot(t,Delta)
# rpart.m:14
    temp=dot(T_b,T0)
# rpart.m:16
    ef_b=dot(0.5,D) / T0
# rpart.m:18
    ei_b=e_i / (dot(k,T0))
# rpart.m:20
    e0_b=e_0 / (dot(k,T0))
# rpart.m:21
    sigma=2
# rpart.m:23
    Theta_r=dot(dot(Be,h),c) / k
# rpart.m:24
    Z_rot=temp / (multiply(sigma,Theta_r))
# rpart.m:25
    M=sum(m)
# rpart.m:27
    mb=m / M
# rpart.m:28
    # A*X=B
    A=zeros(l + 3,l + 3)
# rpart.m:31
    for i in arange(1,l).reshape(-1):
        A[i,i]=v_b
# rpart.m:34
        A[i,l + 2]=ni_b(i)
# rpart.m:35
    
    A[l + 1,l + 1]=v_b
# rpart.m:38
    A[l + 1,l + 2]=na_b
# rpart.m:39
    for i in arange(1,l + 1).reshape(-1):
        A[l + 2,i]=T_b
# rpart.m:42
    
    A[l + 2,l + 2]=dot(dot(dot(M,v0 ** 2) / k / T0,(dot(mb(1),nm_b) + dot(mb(2),na_b))),v_b)
# rpart.m:44
    A[l + 2,l + 3]=nm_b + na_b
# rpart.m:45
    for i in arange(1,l).reshape(-1):
        A[l + 3,i]=dot(2.5,T_b) + ei_b(i) + e0_b
# rpart.m:48
    
    A[l + 3,l + 1]=dot(1.5,T_b) + ef_b
# rpart.m:50
    A[l + 3,l + 2]=dot(1 / v_b,(dot(dot(3.5,nm_b),T_b) + dot(dot(2.5,na_b),T_b) + sum(multiply((ei_b + e0_b),ni_b)) + dot(ef_b,na_b)))
# rpart.m:51
    A[l + 3,l + 3]=dot(2.5,nm_b) + dot(1.5,na_b)
# rpart.m:53
    AA=sparse(A)
# rpart.m:55
    # B
    B=zeros(l + 3,1)
# rpart.m:58
    Kdr=dot(dot(dot((dot(m(1),h ** 2) / (dot(dot(dot(dot(dot(m(2),m(2)),2),pi),k),temp))) ** (3 / 2),Z_rot),exp(- e_i.T / (dot(k,temp)))),exp(D / temp))
# rpart.m:60
    Kvt=exp((e_i(arange(1,end() - 1)) - e_i(arange(2,end()))) / (dot(k,temp))).T
# rpart.m:63
    kd=dot(dot(kdis(temp),Delta),n0) / v0
# rpart.m:65
    kr=zeros(2,l)
# rpart.m:67
    for iM in arange(1,2).reshape(-1):
        kr[iM,arange()]=dot(multiply(kd(iM,arange()),Kdr),n0)
# rpart.m:69
    
    # VT i+1 -> i
    kvt_down=dot(dot(kvt_ssh(temp),Delta),n0) / v0
# rpart.m:73
    kvt_up=zeros(2,Lmax)
# rpart.m:74
    for ip in arange(1,2).reshape(-1):
        kvt_up[ip,arange()]=multiply(kvt_down(ip,arange()),Kvt)
# rpart.m:76
    
    # VV
    kvv_down=dot(dot(kvv_ssh(temp),Delta),n0) / v0
# rpart.m:80
    kvv_up=zeros(Lmax,Lmax)
# rpart.m:81
    deps=e_i(arange(1,end() - 1)) - e_i(arange(2,end()))
# rpart.m:82
    for ip in arange(1,Lmax).reshape(-1):
        kvv_up[ip,arange()]=multiply(kvv_down(ip,arange()),exp((deps(ip) - deps.T) / (dot(k,temp))))
# rpart.m:84
    
    RD=zeros(l,1)
# rpart.m:87
    RVT=zeros(l,1)
# rpart.m:88
    RVV=zeros(l,1)
# rpart.m:89
    for i1 in arange(1,l).reshape(-1):
        RD[i1]=dot(nm_b,(dot(dot(na_b,na_b),kr(1,i1)) - dot(ni_b(i1),kd(1,i1)))) + dot(na_b,(dot(dot(na_b,na_b),kr(2,i1)) - dot(ni_b(i1),kd(2,i1))))
# rpart.m:93
        if i1 == 1:
            RVT[i1]=dot(nm_b,(dot(ni_b(i1 + 1),kvt_down(1,i1)) - dot(ni_b(i1),kvt_up(1,i1)))) + dot(na_b,(dot(ni_b(i1 + 1),kvt_down(2,i1)) - dot(ni_b(i1),kvt_up(2,i1))))
# rpart.m:98
            RVV[i1]=dot(ni_b(i1 + 1),sum(multiply(ni_b(arange(1,end() - 1)),kvv_down(i1,arange()).T))) - dot(ni_b(i1),sum(multiply(ni_b(arange(2,end())),kvv_up(i1,arange()).T)))
# rpart.m:101
        else:
            if i1 == l:
                RVT[i1]=dot(nm_b,(dot(ni_b(i1 - 1),kvt_up(1,i1 - 1)) - dot(ni_b(i1),kvt_down(1,i1 - 1)))) + dot(na_b,(dot(ni_b(i1 - 1),kvt_up(2,i1 - 1)) - dot(ni_b(i1),kvt_down(2,i1 - 1))))
# rpart.m:106
                RVV[i1]=dot(ni_b(i1 - 1),sum(multiply(ni_b(arange(2,end())),kvv_up(i1 - 1,arange()).T))) - dot(ni_b(i1),sum(multiply(ni_b(arange(1,end() - 1)),kvv_down(i1 - 1,arange()).T)))
# rpart.m:109
            else:
                RVT[i1]=dot(nm_b,(dot(ni_b(i1 + 1),kvt_down(1,i1)) + dot(ni_b(i1 - 1),kvt_up(1,i1 - 1)) - dot(ni_b(i1),(kvt_up(1,i1) + kvt_down(1,i1 - 1))))) + dot(na_b,(dot(ni_b(i1 + 1),kvt_down(2,i1)) + dot(ni_b(i1 - 1),kvt_up(2,i1 - 1)) - dot(ni_b(i1),(kvt_up(2,i1) + kvt_down(2,i1 - 1)))))
# rpart.m:114
                RVV[i1]=dot(ni_b(i1 + 1),sum(multiply(ni_b(arange(1,end() - 1)),kvv_down(i1,arange()).T))) + dot(ni_b(i1 - 1),sum(multiply(ni_b(arange(2,end())),kvv_up(i1 - 1,arange()).T))) - dot(ni_b(i1),(sum(multiply(ni_b(arange(2,end())),kvv_up(i1,arange()).T)) + sum(multiply(ni_b(arange(1,end() - 1)),kvv_down(i1 - 1,arange()).T))))
# rpart.m:119
    
    B[arange(1,l)]=RD
# rpart.m:126
    
    B[l + 1]=dot(- 2,sum(RD))
# rpart.m:127
    dy=dot(AA ** (- 1),B)
# rpart.m:129