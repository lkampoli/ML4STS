# Generated with SMOP  0.41-beta
from libsmop import *
    
    sw_sp = 1
    sw_o=1
    xc=concat([[1],[0]])
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
# main.m:32
    e_i=copy(en_vibr)
# main.m:34
    e_0=copy(en_vibr_0)
# main.m:35
    mu=dot(concat([[MU(sw_sp)],[dot(0.5,MU(sw_sp))]]),0.001)
# main.m:37
    m=mu / N_a
# main.m:38
    sigma0=dot(pi,R0(sw_sp,1) ** 2)
# main.m:40
    r0=concat([[R0(sw_sp,1)],[dot(0.5,(R0(sw_sp,1) + R0(sw_sp,2)))]])
# main.m:42
    em=concat([[EM(sw_sp,1)],[sqrt(dot(dot(dot(EM(sw_sp,1),EM(sw_sp,2)),R0(sw_sp,1) ** 6),R0(sw_sp,2) ** 6)) / r0(2) ** 6]])
# main.m:44
    re=RE(sw_sp)
# main.m:47
    p0=dot(0.8,133.322)
# main.m:49
    
    #p0 = 0.8*300.; # Pa
    
    for T in arange(300,300,100).reshape(-1):
        T0=copy(T)
# main.m:54
        #T0 = 500;
        Tv0=copy(T0)
# main.m:57
        #    for M = 13:0.1:14
        #M0 = M
        M0=13.4
# main.m:62
        # Let's assume constant p0
        n0=p0 / (dot(k,T0))
# main.m:66
        if xc(1) != 0:
            gamma0=1.4
# main.m:69
        else:
            gamma0=5 / 3
# main.m:71
        rho0_c=dot(multiply(m,xc),n0)
# main.m:74
        rho0=sum(rho0_c)
# main.m:76
        mu_mix=sum(rho0_c / mu) / rho0
# main.m:78
        R_bar=dot(R,mu_mix)
# main.m:80
        a0=sqrt(dot(dot(gamma0,R_bar),T0))
# main.m:82
        v0=dot(M0,a0)
# main.m:84
        NN=copy(in_con)
# main.m:86
        n1=NN(1)
# main.m:87
        T1=NN(2)
# main.m:88
        v1=NN(3)
# main.m:89
        Zvibr_0=sum(exp(- e_i / Tv0 / k))
# main.m:91
        Y0_bar=zeros(l + 3,1)
# main.m:93
        Y0_bar[arange(1,l)]=dot(dot(xc(1),n1) / Zvibr_0,exp(- e_i / Tv0 / k))
# main.m:95
        Y0_bar[l + 1]=dot(xc(2),n1)
# main.m:96
        Y0_bar[l + 2]=v1
# main.m:97
        Y0_bar[l + 3]=T1
# main.m:98
        Delta=1 / (dot(dot(sqrt(2),n0),sigma0))
# main.m:100
        xspan=concat([0,x_w]) / Delta
# main.m:101
        options=odeset('RelTol',1e-08,'AbsTol',1e-08,'Stats','off')
# main.m:103
        X,Y=ode15s(rpart,xspan,Y0_bar,options,nargout=2)
# main.m:104
        x_s=dot(dot(X,Delta),100)
# main.m:106
        Temp=dot(Y(arange(),l + 3),T0)
# main.m:108
        v=dot(Y(arange(),l + 2),v0)
# main.m:110
        n_i=dot(Y(arange(),arange(1,l)),n0)
# main.m:112
        n_a=dot(Y(arange(),l + 1),n0)
# main.m:114
        n_m=sum(n_i,2)
# main.m:116
        time_s=dot(X,Delta) / v0
# main.m:118
        Npoint=length(X)
# main.m:120
        Nall=sum(n_i,2) + n_a
# main.m:122
        ni_n=n_i / repmat(Nall,1,l)
# main.m:124
        nm_n=sum(ni_n,2)
# main.m:126
        na_n=n_a / Nall
# main.m:128
        rho=dot(m(1),n_m) + dot(m(2),n_a)
# main.m:130
        p=multiply(dot(Nall,k),Temp)
# main.m:132
        e_v=multiply(repmat(e_i.T + e_0,Npoint,1),n_i)
# main.m:134
        e_v=sum(e_v,2)
# main.m:135
        e_v0=dot(dot(n0,xc(1)) / Zvibr_0,sum(multiply(exp(- e_i / Tv0 / k),(e_i + e_0))))
# main.m:137
        e_f=dot(dot(dot(0.5,D),n_a),k)
# main.m:139
        e_f0=dot(dot(dot(dot(0.5,D),xc(2)),n0),k)
# main.m:140
        e_tr=multiply(dot(dot(1.5,Nall),k),Temp)
# main.m:142
        e_tr0=multiply(dot(dot(1.5,n0),k),T0)
# main.m:143
        e_rot=multiply(dot(n_m,k),Temp)
# main.m:145
        e_rot0=multiply(dot(dot(n0,xc(1)),k),T0)
# main.m:146
        E=e_tr + e_rot + e_v + e_f
# main.m:148
        E0=e_tr0 + e_rot0 + e_v0 + e_f0
# main.m:149
        H=(E + p) / rho
# main.m:151
        H0=(E0 + p0) / rho0
# main.m:152
        u10=dot(rho0,v0)
# main.m:154
        u20=dot(rho0,v0 ** 2) + p0
# main.m:155
        u30=H0 + v0 ** 2 / 2
# main.m:156
        u1=u10 - multiply(rho,v)
# main.m:158
        u2=u20 - multiply(rho,v ** 2) - p
# main.m:159
        u3=u30 - H - v ** 2 / 2
# main.m:160
        d1=max(abs(u1) / u10)
# main.m:162
        d2=max(abs(u2) / u20)
# main.m:163
        d3=max(abs(u3) / u30)
# main.m:164
        disp('Relative error of conservation law of:')
        disp(concat(['mass = ',num2str(d1)]))
        disp(concat(['momentum = ',num2str(d2)]))
        disp(concat(['energy = ',num2str(d3)]))
        RDm=zeros(Npoint,l)
# main.m:171
        RDa=zeros(Npoint,l)
# main.m:172
        RVTm=zeros(Npoint,l)
# main.m:173
        RVTa=zeros(Npoint,l)
# main.m:174
        RVV=zeros(Npoint,l)
# main.m:175
        for i in arange(1,Npoint).reshape(-1):
            input_=Y(i,arange()).T
# main.m:178
            rdm,rda,rvtm,rvta,rvv=rpart_post(input_,nargout=5)
# main.m:179
            RDm[i,arange()]=rdm
# main.m:180
            RDa[i,arange()]=rda
# main.m:181
            RVTm[i,arange()]=rvtm
# main.m:182
            RVTa[i,arange()]=rvta
# main.m:183
            RVV[i,arange()]=rvv
# main.m:184
        RD_mol=RDm + RDa
# main.m:187
        RVT=RVTm + RVTa
# main.m:188
        RD_at=dot(- 2,sum(RD_mol,2))
# main.m:189
        # Here you can save the variables you want to create the dataset
# which after you will use for the regression ...
# ... for example:
#dataset = [x_s, n_i, n_a, rho, v, p, E, RD_mol, RD_at];
# This is the filename which you should use in the regression.py
# when you load the dataset:
# dataset=np.loadtxt("../data/your_dataset_filename.dat")
#save your_dataset_filename.dat dataset -ascii -append
        save('database','Temp','x_s','n_i','n_a','rho','v','p','E','RD_mol','RD_at')
        toc
    
