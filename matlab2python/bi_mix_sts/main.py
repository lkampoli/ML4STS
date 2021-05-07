# Generated with SMOP  0.41-beta
from libsmop import *
    
    sw_sp = 1
    sw_o=1
    xc=[1,0]
    x_w=1000
    c=299000000.0
    h=6.6261e-34
    k=1.3807e-23
    N_a=6.0221e+23
    h_bar=h / (dot(2,pi))
    R=8.3145

    import scipy.io
    mat = scipy.io.loadmat('data_species.mat')
    om_e   = mat['OMEGA'][0]
    om_x_e = mat['OMEGA'][1]
    
    Be     = mat['BE'][sw_sp-1]
    
    D=ED(sw_sp)
    CA=CArr(sw_sp,arange())
    
    nA=NArr(sw_sp,arange())
    l=QN(sw_sp,sw_o)
    e_i=copy(en_vibr)
    e_0=copy(en_vibr_0)
    mu=dot(concat([[MU(sw_sp)],[dot(0.5,MU(sw_sp))]]),0.001)
    m=mu / N_a
    sigma0=dot(pi,R0(sw_sp,1) ** 2)
    r0=concat([[R0(sw_sp,1)],[dot(0.5,(R0(sw_sp,1) + R0(sw_sp,2)))]])
    em=concat([[EM(sw_sp,1)],[sqrt(dot(dot(dot(EM(sw_sp,1),EM(sw_sp,2)),R0(sw_sp,1) ** 6),R0(sw_sp,2) ** 6)) / r0(2) ** 6]])
    re=RE(sw_sp)
    p0=dot(0.8,133.322)
    
    for T in arange(300,300,100).reshape(-1):
        T0=copy(T)
        Tv0=copy(T0)
        M0=13.4
        # Let's assume constant p0
        n0=p0 / (dot(k,T0))
        if xc(1) != 0:
            gamma0=1.4
        else:
            gamma0=5 / 3
        rho0_c=dot(multiply(m,xc),n0)
        rho0=sum(rho0_c)
        mu_mix=sum(rho0_c / mu) / rho0
        R_bar=dot(R,mu_mix)
        a0=sqrt(dot(dot(gamma0,R_bar),T0))
        v0=dot(M0,a0)
        NN=copy(in_con)
        n1=NN(1)
        T1=NN(2)
        v1=NN(3)
        Zvibr_0=sum(exp(- e_i / Tv0 / k))
        Y0_bar=zeros(l + 3,1)
        Y0_bar[arange(1,l)]=dot(dot(xc(1),n1) / Zvibr_0,exp(- e_i / Tv0 / k))
        Y0_bar[l + 1]=dot(xc(2),n1)
        Y0_bar[l + 2]=v1
        Y0_bar[l + 3]=T1
        Delta=1 / (dot(dot(sqrt(2),n0),sigma0))
        xspan=concat([0,x_w]) / Delta
        options=odeset('RelTol',1e-08,'AbsTol',1e-08,'Stats','off')
        X,Y=ode15s(rpart,xspan,Y0_bar,options,nargout=2)
        x_s=dot(dot(X,Delta),100)
        Temp=dot(Y(arange(),l + 3),T0)
        v=dot(Y(arange(),l + 2),v0)
        n_i=dot(Y(arange(),arange(1,l)),n0)
        n_a=dot(Y(arange(),l + 1),n0)
        n_m=sum(n_i,2)
        time_s=dot(X,Delta) / v0
        Npoint=length(X)
        Nall=sum(n_i,2) + n_a
        ni_n=n_i / repmat(Nall,1,l)
        nm_n=sum(ni_n,2)
        na_n=n_a / Nall
        rho=dot(m(1),n_m) + dot(m(2),n_a)
        p=multiply(dot(Nall,k),Temp)
        e_v=multiply(repmat(e_i.T + e_0,Npoint,1),n_i)
        e_v=sum(e_v,2)
        e_v0=dot(dot(n0,xc(1)) / Zvibr_0,sum(multiply(exp(- e_i / Tv0 / k),(e_i + e_0))))
        e_f=dot(dot(dot(0.5,D),n_a),k)
        e_f0=dot(dot(dot(dot(0.5,D),xc(2)),n0),k)
        e_tr=multiply(dot(dot(1.5,Nall),k),Temp)
        e_tr0=multiply(dot(dot(1.5,n0),k),T0)
        e_rot=multiply(dot(n_m,k),Temp)
        e_rot0=multiply(dot(dot(n0,xc(1)),k),T0)
        E=e_tr + e_rot + e_v + e_f
        E0=e_tr0 + e_rot0 + e_v0 + e_f0
        H=(E + p) / rho
        H0=(E0 + p0) / rho0
        u10=dot(rho0,v0)
        u20=dot(rho0,v0 ** 2) + p0
        u30=H0 + v0 ** 2 / 2
        u1=u10 - multiply(rho,v)
        u2=u20 - multiply(rho,v ** 2) - p
        u3=u30 - H - v ** 2 / 2
        d1=max(abs(u1) / u10)
        d2=max(abs(u2) / u20)
        d3=max(abs(u3) / u30)
        disp('Relative error of conservation law of:')
        disp(concat(['mass = ',num2str(d1)]))
        disp(concat(['momentum = ',num2str(d2)]))
        disp(concat(['energy = ',num2str(d3)]))
        RDm=zeros(Npoint,l)
        RDa=zeros(Npoint,l)
        RVTm=zeros(Npoint,l)
        RVTa=zeros(Npoint,l)
        RVV=zeros(Npoint,l)
        for i in arange(1,Npoint).reshape(-1):
            input_=Y(i,arange()).T
            rdm,rda,rvtm,rvta,rvv=rpart_post(input_,nargout=5)
            RDm[i,arange()]=rdm
            RDa[i,arange()]=rda
            RVTm[i,arange()]=rvtm
            RVTa[i,arange()]=rvta
            RVV[i,arange()]=rvv
        RD_mol=RDm + RDa
        RVT=RVTm + RVTa
        RD_at=dot(- 2,sum(RD_mol,2))
        save('database','Temp','x_s','n_i','n_a','rho','v','p','E','RD_mol','RD_at')
        toc
    
