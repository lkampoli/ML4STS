# Generated with SMOP  0.41-beta
from libsmop import *
# rpart_post.m

    
@function
def rpart_post(y=None,*args,**kwargs):
    varargin = rpart_post.varargin
    nargin = rpart_post.nargin

    format('long','e')
    global c,h,k,m,l,e_i,Be,D,n0,v0,T0,Delta
    Lmax=l - 1
# rpart_post.m:6
    ni_b=y(arange(1,l))
# rpart_post.m:8
    na_b=y(l + 1)
# rpart_post.m:9
    nm_b=sum(ni_b)
# rpart_post.m:10
    T_b=y(l + 3)
# rpart_post.m:11
    temp=dot(T_b,T0)
# rpart_post.m:13
    sigma=2
# rpart_post.m:15
    Theta_r=dot(dot(Be,h),c) / k
# rpart_post.m:16
    Z_rot=temp / (multiply(sigma,Theta_r))
# rpart_post.m:17
    Kdr=dot(dot(dot((dot(m(1),h ** 2) / (dot(dot(dot(dot(dot(m(2),m(2)),2),pi),k),temp))) ** (3 / 2),Z_rot),exp(- e_i.T / (dot(k,temp)))),exp(D / temp))
# rpart_post.m:19
    Kvt=exp((e_i(arange(1,end() - 1)) - e_i(arange(2,end()))) / (dot(k,temp))).T
# rpart_post.m:22
    kd=dot(dot(kdis(temp),Delta),n0) / v0
# rpart_post.m:24
    kr=zeros(2,l)
# rpart_post.m:26
    for iM in arange(1,2).reshape(-1):
        kr[iM,arange()]=dot(multiply(kd(iM,arange()),Kdr),n0)
# rpart_post.m:28
    
    # VT i+1 -> i
    kvt_down=dot(dot(kvt_ssh(temp),Delta),n0) / v0
# rpart_post.m:32
    kvt_up=zeros(2,Lmax)
# rpart_post.m:33
    for ip in arange(1,2).reshape(-1):
        kvt_up[ip,arange()]=multiply(kvt_down(ip,arange()),Kvt)
# rpart_post.m:35
    
    # VV
    kvv_down=dot(dot(kvv_ssh(temp),Delta),n0) / v0
# rpart_post.m:39
    kvv_up=zeros(Lmax,Lmax)
# rpart_post.m:40
    deps=e_i(arange(1,end() - 1)) - e_i(arange(2,end()))
# rpart_post.m:41
    for ip in arange(1,Lmax).reshape(-1):
        kvv_up[ip,arange()]=multiply(kvv_down(ip,arange()),exp((deps(ip) - deps.T) / (dot(k,temp))))
# rpart_post.m:43
    
    RDm=zeros(l,1)
# rpart_post.m:46
    RDa=zeros(l,1)
# rpart_post.m:47
    RVTm=zeros(l,1)
# rpart_post.m:48
    RVTa=zeros(l,1)
# rpart_post.m:49
    RVV=zeros(l,1)
# rpart_post.m:50
    for i1 in arange(1,l).reshape(-1):
        RDm[i1]=dot(nm_b,(dot(dot(na_b,na_b),kr(1,i1)) - dot(ni_b(i1),kd(1,i1))))
# rpart_post.m:53
        RDa[i1]=dot(na_b,(dot(dot(na_b,na_b),kr(2,i1)) - dot(ni_b(i1),kd(2,i1))))
# rpart_post.m:54
        if i1 == 1:
            RVTm[i1]=dot(nm_b,(dot(ni_b(i1 + 1),kvt_down(1,i1)) - dot(ni_b(i1),kvt_up(1,i1))))
# rpart_post.m:56
            RVTa[i1]=dot(na_b,(dot(ni_b(i1 + 1),kvt_down(2,i1)) - dot(ni_b(i1),kvt_up(2,i1))))
# rpart_post.m:57
            RVV[i1]=dot(ni_b(i1 + 1),sum(multiply(ni_b(arange(1,end() - 1)),kvv_down(i1,arange()).T))) - dot(ni_b(i1),sum(multiply(ni_b(arange(2,end())),kvv_up(i1,arange()).T)))
# rpart_post.m:58
        else:
            if i1 == l:
                RVTm[i1]=dot(nm_b,(dot(ni_b(i1 - 1),kvt_up(1,i1 - 1)) - dot(ni_b(i1),kvt_down(1,i1 - 1))))
# rpart_post.m:62
                RVTa[i1]=dot(na_b,(dot(ni_b(i1 - 1),kvt_up(2,i1 - 1)) - dot(ni_b(i1),kvt_down(2,i1 - 1))))
# rpart_post.m:63
                RVV[i1]=dot(ni_b(i1 - 1),sum(multiply(ni_b(arange(2,end())),kvv_up(i1 - 1,arange()).T))) - dot(ni_b(i1),sum(multiply(ni_b(arange(1,end() - 1)),kvv_down(i1 - 1,arange()).T)))
# rpart_post.m:64
            else:
                RVTm[i1]=dot(nm_b,(dot(ni_b(i1 + 1),kvt_down(1,i1)) + dot(ni_b(i1 - 1),kvt_up(1,i1 - 1)) - dot(ni_b(i1),(kvt_up(1,i1) + kvt_down(1,i1 - 1)))))
# rpart_post.m:68
                RVTa[i1]=dot(na_b,(dot(ni_b(i1 + 1),kvt_down(2,i1)) + dot(ni_b(i1 - 1),kvt_up(2,i1 - 1)) - dot(ni_b(i1),(kvt_up(2,i1) + kvt_down(2,i1 - 1)))))
# rpart_post.m:70
                RVV[i1]=dot(ni_b(i1 + 1),sum(multiply(ni_b(arange(1,end() - 1)),kvv_down(i1,arange()).T))) + dot(ni_b(i1 - 1),sum(multiply(ni_b(arange(2,end())),kvv_up(i1 - 1,arange()).T))) - dot(ni_b(i1),(sum(multiply(ni_b(arange(2,end())),kvv_up(i1,arange()).T)) + sum(multiply(ni_b(arange(1,end() - 1)),kvv_down(i1 - 1,arange()).T))))
# rpart_post.m:72
    
    RDm=dot(dot(RDm.T,n0),v0) / Delta
# rpart_post.m:79
    RDa=dot(dot(RDa.T,n0),v0) / Delta
# rpart_post.m:80
    RVTm=dot(dot(RVTm.T,n0),v0) / Delta
# rpart_post.m:81
    RVTa=dot(dot(RVTa.T,n0),v0) / Delta
# rpart_post.m:82
    RVV=dot(dot(RVV.T,n0),v0) / Delta
# rpart_post.m:83