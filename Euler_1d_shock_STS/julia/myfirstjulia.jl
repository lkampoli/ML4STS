
versioninfo() # version information

using MAT

using SymPy

using JuMP, Ipopt, GLPK

using NLsolve

using IntervalRootFinding, IntervalArithmetic, StaticArrays

include("mytest.jl")

my_module.my_module_test_print()

my_module.addme(10.,2)

function main()
    
    println("Ciao Lollo!")
    
    sw_sp = 1;                                                                                                                          
    sw_o = 1;                                                                                                                           
    xc = [1; 0];                                                                                                                        
                                                                                                                                       
    x_w = 1000;                                                                                                                         
                                                                                                                                      
    c = 2.99e8;                                                                                                                         
    h = 6.6261e-34;                                                                                                                     
    k = 1.3807e-23;                                                                                                                     
    N_a = 6.0221e23;                                                                                                                    
    h_bar = h/(2*pi);                                                                                                                   
    R = 8.3145;     
    
    file = matopen("data_species.mat")
    varnames = names(file)
    vars = matread("data_species.mat")
    #read(file, "varname") # note that this does NOT introduce a variable ``varname`` into scope     
    close(file)
    
    display(vars)
    display(varnames)
           
    # Accessing all the Keys  using 'keys' keyword 
    Keys = keys(vars) 
    println("Keys = ", Keys) 
  
    # Accessing all the Values using 'values' keyword 
    Values = values(vars) 
    println("Values = ", Values) 
    
    om_e = get(vars, "OMEGA", 0)
    om_e = om_e[sw_sp,1] # m^-1
    display(om_e)
    
    om_x_e = get(vars, "OMEGA", 0)
    om_x_e = om_x_e[sw_sp,2] # m^-1
    display(om_x_e)
    
    Be = get(vars, "BE", 0)
    Be = Be[sw_sp] # m^-1
    display(Be)
    
    CA = get(vars, "CArr", 0)
    CA = CA[sw_sp,:] # m^3/s                                                                                                         
    display(CA)
    
    nA = get(vars, "NArr", 0)
    nA = nA[sw_sp,:]                                                                                                                 
    display(nA)                
    
    l = get(vars, "QN", 0)
    l = [sw_sp,sw_o]
    display(l)
    
    om_0 = om_e - om_x_e                                                                                                                
    om_x_e_0 = om_x_e                                                                                                                   
     
    #e_i = en_vibr
    if sw_o == 1                                                                                                                        
        e_i = h*c*(om_0*(0:l-1)'-om_x_e_0*((0:l-1)').^2)                                                                                  
    else                                                                                                                                
        e_i = h*c*om_e*(0:l-1)'                                                                                                           
    end
    display(e_i)
       
    #e_0 = en_vibr_0    
    if sw_o == 1                                                                                                                        
        e_0 = h*c*(0.5*om_e-0.25*om_x_e)                                                                                                 
    else                                                                                                                                
        e_0 = h*c*0.5*om_e                                                                                                               
    end         
    display(e_0)
              
    mu = get(vars, "MU", 0)
    mu = [mu[sw_sp]; 0.5*mu[sw_sp]]*1e-3
    m = mu / N_a
    display(m)
      
    R0_ = get(vars, "R0", 0)
    sigma0 = pi*R0_[sw_sp,1]^2                                                                                                                                                                                                                         
    r0 = [R0_[sw_sp,1]; 0.5*(R0_[sw_sp,1]+R0_[sw_sp,2])]                                             
    display(r0)
                                                      
    EM_ = get(vars, "EM", 0)
    em = [EM_[sw_sp,1]; sqrt(EM_[sw_sp,1]*EM_[sw_sp,2]*R0_[sw_sp,1]^6 * R0_[sw_sp,2]^6)/r0[2]^6]
    display(em)
    
    RE_ = get(vars, "RE", 0)
    re = RE_[sw_sp]
    display(re)
            
    p0 = 0.8*133.322 # Pa            
    T0 = 300 # K
    Tv0 = T0
    M0 = 13.4
    
    n0 = p0/(k*T0)
    
    if xc[1] != 0
        gamma0 = 1.4
    else
        gamma0 = 5/3
    end
            
    rho0_c = m.*xc*n0
                                                                                                                                      
    rho0 = sum(rho0_c)
                                                                                                                                      
    mu_mix = sum(rho0_c./mu)/rho0                                                                                              
                                                                                                                                       
    R_bar = R*mu_mix                                                                                                           
                                                                                                                                       
    a0 = sqrt(gamma0*R_bar*T0)                                                                                                 
                                                                                                                                       
    v0 = M0*a0
           
    #n, v, t = Sym("n", "v", "t") 
    #n, v, t = symbols("n", "v", "t", real=true)
                
    xm = xc[1]                                                                                                                         
    xa = xc[1]                                                                                                                         
                                                                                                                                      
    C1 = sum(m.*xc)*v0^2/(k*T0)                                                                                                        
    C2 = 0.5*C1  
                                                                                                                                          
    #S = [n*v == 1, n*t+n*v^2*C1 == 1+C1, 3.5*xm*t+2.5*xa*t+v^2*C2 == 3.5*xm+2.5*xa+C2]
                                                                                                                                      
    #as = nsolve(S,[n,t,v],[10,100,0.1])
    #N.(as)	
    #X = [N.n;N.t;N.v]                                                                                                 
    #sol = double(X)                                                                                                                    
                                                                                                                                      
    #sol1 = sol([1 3 5])                                                                                                                
    #sol2 = sol([2 4 6])
  
    #if any(sol1-1)                                                                                                                      
    #  Y = sol1                                                                                                                       
    #elseif any(sol2-1)                                                                                                                  
    #  Y = sol2                                                                                                                       
    #end
           
    #function f(z)
    #   n, t, v = z
    #   SVector(n*v-1, n*t+n*v^2*C1-1-C1, 3.5*xm*t+2.5*xa*t+v^2*C2-3.5*xm-2.5*xa-C2)
    #end
    #rts = roots(f, IntervalBox(10,100,0.1))
                        
    #function f!(F, x)
    #    F[1] = x[1]*x[3]-1
    #    F[2] = x[1]*x[2]+x[1]*x[3]^2*C1-1-C1
    #    F[3] = 3.5*xm*x[2]+2.5*xa*x[2]+x[3]^2*C2-3.5*xm-2.5*xa-C2
    #end
    #sol = nlsolve(f!, [10;100;0.1])
    #sol.zero

    #system = Model(solver=IpoptSolver())
    #model = Model(with_optimizer(Ipopt.Optimizer, max_cpu_time=60.0))
    #model = Model(solver=IpoptSolver())
    #model = Model()
    #model = Model(GLPK.Optimizer)
    #model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0));
    #@variable(model, n)
    #@variable(model, v)
    #@variable(model, t)
    #@NLconstraint(model, n*v == 1)
    #@NLconstraint(model, n*t+n*v^2*C1 == 1+C1)
    #@NLconstraint(model, 3.5*xm*t+2.5*xa*t+v^2*C2 == 3.5*xm+2.5*xa+C2)
    #@NLobjective(model, 1 / x)
    #solve(model)
    #optimize!(model)
    #optimize!(model)
    #objective_value(model)
    #@NLobjective(model)
    #optimize!(model)
    #println("n = ", value(n), " t = ", value(t), " v = ", value(v))
    #sol = getvalue.((n,v,t))
    #            
    #sol1 = sol([1 3 5])                                                                                                                
    #sol2 = sol([2 4 6])
    # 
    #if any(sol1-1)                                                                                                                      
    #  Y = sol1                                                                                                                       
    #elseif any(sol2-1)                                                                                                                  
    #  Y = sol2                                                                                                                       
    #end
                    
    n1 = 1 #Y[1]                                                                                                                 
    T1 = 1 #Y[2]                                                                                                                 
    v1 = 1 #Y[3]                                                                                                                 
    
    Zvibr_0 = sum(exp.(-e_i/Tv0/k))
    display(Zvibr_0)
              
    display(l+3)
    #Y0_bar = zeros(l+3)                                                                                                      
    Y0_bar = zeros(50)                                                                                                              
                                                                                                                                       
    #Y0_bar[1:l] = xc[1]*n1/Zvibr_0*exp(-e_i/Tv0/k)                                                                             
    #Y0_bar[l+1] = xc[2]*n1                                                                                                     
    #Y0_bar[l+2] = v1                                                                                                           
    #Y0_bar[l+3] = T1                                                                                                           
                                                                                                                                      
    #Delta = 1/(sqrt(2)*n0*sigma0)                                                                                              
    #xspan = [0, x_w]./Delta
                    
    #opts = odeset('RelTol', 1e-12, 'AbsTol', 1e-12)
    #[X,Y] = ode15s(@rpart, xspan, Y0_bar, opts)     
                
    #x_s = X*Delta*100                                                                                                                                                                                                                                             
    #Temp = Y[:,l+3]*T0                                                                                                             
    #v = Y[:,l+2]*v0                                                                                                                   
    #n_i = Y[:,1:l]*n0                                                                                                                  
    #n_a = Y[:,l+1]*n0                                                                                                                  
    #n_m = sum(n_i,2)                                                                                                                  
    #time_s = X*Delta/v0 # sec                                                                                                          
    #Npoint = length(X)                                                                                                                 
    #Nall = sum(n_i,2)+n_a                                                                                                              
    #ni_n = n_i./repmat(Nall,1,l);     # repeat(Nall,1,l)                                                                                                   
    #nm_n = sum(ni_n,2)                                                                                                                 
    #na_n = n_a./Nall                                                                                                                   
    #rho = m[1]*n_m + m[2]*n_a;                                                                                                          
    #p = Nall*k.*Temp                                                                                                                   
    #e_v = repmat(e_i'+e_0,Npoint,1).*n_i;                                                                                               
    #e_v = sum(e_v,2)                                                                                                                   
    #e_v0 = n0*xc[1]/Zvibr_0*sum(exp(-e_i/Tv0/k).*(e_i+e_0));                                                                            
    #e_f = 0.5*D*n_a*k                                                                                                                  
    #e_f0 = 0.5*D*xc[2]*n0*k                                                                                                            
    #e_tr = 1.5*Nall*k.*Temp                                                                                                            
    #e_tr0 = 1.5*n0*k.*T0                                                                                                               
    #e_rot = n_m*k.*Temp                                                                                                                
    #e_rot0 = n0*xc[1]*k.*T0                                                                                                            
    #E = e_tr+e_rot+e_v+e_f
    #H0 = (E0+p0)./rho0;                                                                                                                 
    #u10 = rho0*v0;                                                                                                                      
    #u20 = rho0*v0^2+p0;                                                                                                                 
    #u30 = H0+v0^2/2;                                                                                                                    
    #u1 = u10-rho.*v;                                                                                                                    
    #u2 = u20-rho.*v.^2-p;                                                                                                               
    #u3 = u30-H-v.^2/2;                                                                                                                  
    #d1 = max(abs(u1)/u10);                                                                                                              
    #d2 = max(abs(u2)/u20);                                                                                                              
    #d3 = max(abs(u3)/u30);                                                                                                              
    #display('Relative error of conservation law of:');                                                                                     
    #display(['mass = ',num2str(d1)]);                                                                                                      
    #display(['momentum = ',num2str(d2)]);                                                                                                  
    #display(['energy = ',num2str(d3)]);
                        
end



main()
