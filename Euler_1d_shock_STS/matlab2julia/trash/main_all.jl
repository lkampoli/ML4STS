using BenchmarkTools
using MAT
using SymPy
using Plots; gr(fmt=:png)
using UnicodePlots
using PyPlot; pygui(true)
using Images
using JLD
using DifferentialEquations
using DiffEqOperators
using DiffEqParamEstim
using DiffEqDevTools
using StaticArrays
using OrdinaryDiffEq
using LinearAlgebra
using ODE
using ODEInterface
using ODEInterfaceDiffEq
using MATLABDiffEq
using LSODA
#using SciPyDiffEq
#using deSolveDiffEq
using ModelingToolkit
using SparsityDetection
using SparseArrays
#using AlgebraicMultigrid
using Sundials
using Test
using Distributed
using ParameterizedFunctions
#
#addprocs()
#@everywhere using DifferentialEquations
#using Unitful
#using PhysicalConstants.CODATA2014
LinearAlgebra.BLAS.set_num_threads(1)

# Switch for mixture species: 1 - N2/N; 2 - O2/O
const sw_sp = 1

# Switch for model oscillator: 1 - anharmonic; 2 - harmonic
const sw_o = 1

# Species molar fractions [xA2 xA] (xc = nc/n)
const xc = [1 0]

# Range of integration
const x_w = 100.

# Physical constants
const c     = 2.99e8;     #SpeedOfLightInVacuum
const h     = 6.6261e-34; #PlanckConstant
const k     = 1.3807e-23; #BoltzmannConstant
const N_a   = 6.0221e23;  #AvogadroConstant
const R     = 8.3145;     #MolarGasConstant
const h_bar = h/(2*pi);   #PlanckConstantOver2pi

# To read a single variable from a MAT file (compressed files are detected and handled automatically):
file  = matopen("data_species.mat")
const OMEGA = read(file, "OMEGA"); # Ω
const BE    = read(file, "BE")   ; # Bₑ
const ED    = read(file, "ED")   ;
const CArr  = read(file, "CArr") ;
const NArr  = read(file, "NArr") ;
const QN    = read(file, "QN")   ;
const MU    = read(file, "MU")   ; # μ
const R0    = read(file, "R0")   ;
const EM    = read(file, "EM")   ;
const RE    = read(file, "RE")   ;
close(file)

const om_e   = OMEGA[sw_sp,1];      println("om_e = ",   om_e,   "\n") # ωₑ
const om_x_e = OMEGA[sw_sp,2];      println("om_x_e = ", om_x_e, "\n") # ωₓₑ
const Be     = BE[sw_sp]     ;      println("Be = ",     Be,     "\n") # Bₑ
const D      = ED[sw_sp]     ;      println("D = ",      D,      "\n")
const CA     = CArr[sw_sp,:] ;      println("CA = ",     CA,     "\n")
const nA     = NArr[sw_sp,:] ;      println("nA = ",     nA,     "\n")
const l      = Int(QN[sw_sp,sw_o]); println("l = ",      l,      "\n")

#include("en_vibr.jl")
function en_vibr()
  om_0     = om_e-om_x_e
  om_x_e_0 = om_x_e
  if sw_o == 1
    e = h*c*(om_0*(0:l-1)-om_x_e_0*((0:l-1)).^2)
  else
    e = h*c*om_e*(0:l-1)
  end
end
const e_i = en_vibr(); println("eᵢ = ", e_i, "\n")

#include("en_vibr_0.jl")
function en_vibr_0()
  if sw_o == 1
    e = h*c*(0.5*om_e-0.25*om_x_e)
  else
    e = h*c*0.5*om_e
  end
end
const e_0 = en_vibr_0(); println("e₀ = ", e_0, "\n")

const mu     = [MU[sw_sp] 0.5*MU[sw_sp]]*1e-3;                                                  println("mu = ", mu, "\n")
const m      = mu / N_a;                                                                        println("m = ", m, "\n")
const sigma0 = pi*R0[sw_sp,1]^2;                                                                println("sigma0 = ", sigma0, "\n")
const r0     = [R0[sw_sp,1] 0.5*(R0[sw_sp,1]+R0[sw_sp,2])];                                     println("r0 = ", r0, "\n")
const em     = [EM[sw_sp,1] sqrt(EM[sw_sp,1]*EM[sw_sp,2]*R0[sw_sp,1]^6*R0[sw_sp,2]^6)/r0[2]^6]; println("em = ", em, "\n")
const re     = RE[sw_sp];                                                                       println("re = ", re, "\n")

# ICs
const p0  = 0.8*133.322; println("p0 = ", p0, "\n")   # p₀
const T0  = 300.;        println("T0 = ", T0, "\n")   # T₀
const Tv0 = T0;          println("Tv0 = ", Tv0, "\n") # Tᵥ₀
const M0  = 13.4;        println("M0 = ", M0, "\n")   # M₀
const n0  = p0/(k*T0);   println("n0 = ", n0, "\n")   # n₀

if xc[1] != 0
  const gamma0 = 1.4;    println("gamma0 = ", gamma0, "\n") # γ₀
else
  const gamma0 = 5/3;    println("gamma0 = ", gamma0, "\n")
end

const rho0_c = m.*xc*n0;              println("rho0_c = ", rho0_c, "\n")
const rho0   = sum(rho0_c);           println("rho0 = ", rho0, "\n") # ρ₀
const mu_mix = sum(rho0_c./mu)/rho0;  println("mu_mix = ", mu_mix, "\n")
const R_bar  = R*mu_mix;              println("R_bar = ", R_bar, "\n")
const a0     = sqrt(gamma0*R_bar*T0); println("a0 = ", a0, "\n") # a₀
const v0     = M0*a0;                 println("v0 = ", v0, "\n") # v₀

#include("in_con.jl")
function in_con()
  n = Sym("n")
  v = Sym("v")
  t = Sym("t")
  n, v, t = symbols("n, v, t", real=true)

  xm = xc[1]
  xa = xc[2]

  C1 = sum(m.*xc)*v0^2/(k*T0)
  C2 = 0.5*C1

  S = [n*v ⩵ 1, n*t+n*v^2*C1 ⩵ (1+C1), 3.5*xm*t+2.5*xa*t+v^2*C2 ⩵ (3.5*xm+2.5*xa+C2)]
  N = SymPy.solve(S, n, v, t)

  sol1 = collect(values(N[2]))
  sol2 = collect(values(N[1]))

  #if any(sol1 != 0)
  #  Y = sol1
  #elseif any(sol2 != 0)
  #  Y = sol2
  #end

  return sol1
end
NN = in_con()
n1 = NN[1]; println("n1 = ", n1, "\n")
v1 = NN[2]; println("v1 = ", v1, "\n")
T1 = NN[3]; println("T1 = ", T1, "\n")

const Zvibr_0 = sum(exp.(-e_i./Tv0./k)); println("Zvibr_0 = ", Zvibr_0, "\n")

Y0_bar      = zeros(l+3)
Y0_bar[1:l] = xc[1]*n1/Zvibr_0*exp.(-e_i./Tv0./k)
Y0_bar[l+1] = xc[2]*n1
Y0_bar[l+2] = v1
Y0_bar[l+3] = T1
println("Y0_bar = ", Y0_bar, "\n", size(Y0_bar), "\n", typeof(Y0_bar), "\n")

const Delta = 1/(sqrt(2)*n0*sigma0); println("Delta = ", Delta, "\n")
xspan       = [0, x_w]./Delta;       println("xspan = ", xspan, "\n", size(xspan), "\n")

#include("kdis.jl")
function kdis(temp)
  kd_eq = CA .* temp.^nA*exp(-D/temp);             #print("kd_eq = ", kd_eq, "\n")
  ZvT   = sum(exp.(-e_i ./ (temp*k)));             #print("ZvT = ",   ZvT,   "\n")

  U = D/6;                                         #print("U = ", U, "\n")
  # U = 3*temp
  # U = Inf

  ZvU = sum(exp.(e_i ./ (U*k)));                   #print("ZvU = ", ZvU, "\n")
  Z   = ZvT / ZvU * exp.(e_i ./ k*(1/temp + 1/U)); #print("Z = ",   Z,   "\n")
  kd  = kd_eq .* Z';                               #print("kd = ",  kd,  "\n")
end

#include("kvt_ssh.jl")
function kvt_ssh(t)
  if sw_o == 1
    om10 = om_e-2*om_x_e;          #println("om10 = ", om10, "\n")
  elseif sw_o == 2
    om10 = om_e;                   #println("om10 = ", om10, "\n")
  end

  Lmax = l-1;                      #println("Lmax = ", Lmax, "\n")
  i    = collect(0:Lmax-1);        #println("i = ", i, "\n", typeof(i), "\n", length(i), "\n")
  om0  = 2*pi*c*om10;              #println("om0 = ", om0, "\n")
  mu   = m[1] .* m ./ (m[1] .+ m); #println("mu = ", mu, "\n")
  a    = 17.5 ./ r0;               #println("a = ", a, "\n")

  Z        = r0.^2 ./ sqrt.(mu) * sqrt(8*pi*k*t);                        #println("Z = ", Z, "\n")
  chi      = ((pi^2 * om0^2 / 2 / k * t^(-1)) .* (mu ./ a.^2)).^(1/3);   #println("chi = ", chi, "\n")
  r        = r0 .* (0.5 .* sqrt.(1 .+ (chi .* t) ./ em) .+ 0.5).^(-1/6); #println("r = ", r, "\n")
  Z_0      = (a .* re).^2 .* exp.(-((3 .* a .* re.^2) ./ (8 .* r)));     #println("Z_0 = ", Z_0, "\n")
  Const    = 1.294 ./ Z_0 .* 4*pi^2*om0/h_bar*sqrt(4*pi/3);              #println("Const = ", Const, "\n")
  r_c_r_02 = (0.5 .* sqrt.(1 .+ (chi.*t) ./ em) .+ 0.5).^(-1/3);         #println("r_c_r_02 = ", r_c_r_02, "\n")
  P10      = Const .* r_c_r_02 .* (1 .+ 1.1 .* em ./ t).^(-1) .* mu ./ a.^2 .* chi.^0.5 .* exp.(-3 .* chi .+ h_bar*om0/(2*k*t) .+ em ./ t)
  k10      = Z .* P10;                                                   #println("k10 = ", k10, "\n")

  if sw_o == 2
    k_down = k10.*(i.+1);                             #println("k_down = ", k_down, "\n")
  elseif sw_o == 1
    aA      = a .* 1e-10;                             #println("aA = ", aA, "\n")
    mu_amu  = m[1].*m ./ (m[1].+m)./1.6605e-27;       #println("mu_amu = ", mu_amu, "\n")
    E1      = (om_e-2*om_x_e)*1.4388e-2;              #println("E1 = ", E1, "\n")
    dE      = om_x_e*1.4388e-2;                       #println("dE = ", dE, "\n")
    # https://julialang.org/blog/2017/01/moredots/
    diffE = zeros(length(i))
    for ii in 1:length(i)
      println("i = ", ii, "\n")
      diffE[ii] = (om_e - 2 * om_x_e * (i[ii] + 1)) * h * c
    end
    println("diffE = ", diffE, "\n", size(diffE), "\n")
    gamma_n = pi ./ a ./ h_bar .* sqrt.(mu./(2*k*t));
    gamma_n = gamma_n .* diffE;                       println("gamma_n = ", gamma_n, "\n", size(gamma_n), "\n")
    gamma_n = gamma_n';                               println("gamma_n = ", gamma_n, "\n", size(gamma_n), "\n")
    gamma_0 = 0.32 ./ aA .* sqrt.(mu_amu./t).*E1;     println("gamma_0 = ", gamma_0, "\n")
    delta   = zeros(2,length(i));
    k_down  = zeros(2,length(i))
    for ii in 1:length(i)
      if gamma_n[1,ii] < 20
        delta[1,ii] = 4/3*gamma_0[1] * dE/E1
        delta[2,ii] = 4/3*gamma_0[1] * dE/E1
      elseif gamma_n[1,ii] >= 20
        delta[1,ii] = 4*(gamma_0[1])^(2/3) * dE/E1
        delta[2,ii] = 4*(gamma_0[1])^(2/3) * dE/E1
      end
    end
   println("delta = ", delta, "\n", size(delta), "\n")
    for ii in 1:length(i)
      k_down[1,ii] = (i[ii]+1) * k10[1] * exp(i[ii] * delta[1,ii]) * exp(-i[ii]*h*c*om_x_e / (k*t))
      k_down[2,ii] = (i[ii]+1) * k10[2] * exp(i[ii] * delta[2,ii]) * exp(-i[ii]*h*c*om_x_e / (k*t))
    end
  end
  println("k_down = ", k_down, "\n", size(k_down), "\n")
end

#include("kvv_ssh.jl")
function kvv_ssh(t)
  mu    = 0.5  * m[1]
  m_osc = 0.25 * m[1]

  if sw_o == 1
    om10 = om_e-2*om_x_e
  elseif sw_o == 2
    om10 = om_e
  end

  Lmax = l-1
  om0  = 2*pi*c*om10

  a = 17.5 / r0[1]
  Z = r0[1]^2 / sqrt(mu) * sqrt(8*pi*k*t)

  Q10 = 0.5^4 / m_osc * a^2 / om0^2 * 4 * k * t
  k10 = Q10 * Z

  kdown = zeros(Lmax,Lmax)
  j_up  = 0:Lmax-1
  if sw_o == 2
    for i_down = 1:Lmax
      kdown[i_down,:] = i_down * (j_up+1) * k10
    end
  println("kdown = ", kdown, "\n", size(kdown), "\n")
  elseif sw_o == 1
    aA     = a * 1e-10
    mu_amu = mu / 1.6605e-27
    dE     = om_x_e * 1.4388e-2
    delta  = 0.427 / aA * sqrt(mu_amu / t) * dE; println("delta = ", delta, "\n")
    for i_down = 1:Lmax
    @. kdown[i_down,:] = i_down * (j_up+1) * k10 .* exp.(-delta .* abs(i_down-1-j_up)) .*
      (1.5 - 0.5 .* exp.(-delta .* abs(i_down-1-j_up))) .* exp.((j_up-i_down+1) * h * c * om_x_e / (k * t))
    end
  end
  println("kdown = ", kdown, "\n", size(kdown), "\n")
end

#include("rpart.jl")
#Defining your ODE function to be in-place updating can have
#performance benefits. What this means is that, instead of writing
#a function which outputs its solution, you write a function which
#updates a vector that is designated to hold the solution. By doing
#this, DifferentialEquations.jl's solver packages are able to reduce the
#amount of array allocations and achieve better performance.
#
#The way we do this is we simply write the output to the 1st input of
#the function. For example, our Lorenz equation problem would be defined
#by the function:
#function rpart!(dy,y,p,t)
function rpart!(u,p,t)

  #u    = @view y[1:l+3]

  #ni_b = @view u[1:l];      #print("ni_b = ", ni_b, "\n")
  #na_b = @view u[l+1];      #print("na_b = ", na_b, "\n")
  #v_b  = @view u[l+2];      #print("v_b = ",  v_b,  "\n")
  #T_b  = @view u[l+3];      #print("T_b = ",  T_b,  "\n")

  #println("p = ", p, "\n")
  #println("t = ", t, "\n")
  #u    = y[1:l+3]; println("u = ", u, "\n", size(u), "\n", typeof(u), "\n")

  ni_b = u[1:l];      #print("ni_b = ", ni_b, "\n")
  na_b = u[l+1];      #print("na_b = ", na_b, "\n")
  v_b  = u[l+2];      #print("v_b = ",  v_b,  "\n")
  T_b  = u[l+3];      #print("T_b = ",  T_b,  "\n")

  #du = @view dy[1:l+3]
  #du = dy[1:l+3]

  nm_b = sum(ni_b);   #print("nm_b = ", nm_b, "\n")
  Lmax = l-1;         #println("Lmax = ", Lmax, "\n")
  xx   = t*Delta;     #println("xx = ", xx, "\n")
  temp = T_b*T0;      #print("T = ", temp, "\n")

  ef_b = 0.5*D/T0;    #println("ef_b = ", ef_b, "\n")

  ei_b = e_i./(k*T0); #println("ei_b = ", ei_b, "\n")
  e0_b = e_0/(k*T0);  #println("e0_b = ", e0_b, "\n")

  sigma   = 2;                      #println("sigma = ", sigma, "\n")
  Theta_r = Be*h*c/k;               #println("Theta_r = ", Theta_r, "\n")
  Z_rot   = temp./(sigma.*Theta_r); #println("Z_rot = ", Z_rot, "\n")

  M  = sum(m); #println("M = ", M, "\n")
  mb = m/M;    #println("mb = ", mb, "\n")

  A = zeros(l+3,l+3)

  for i = 1:l
    A[i,i]   = v_b
    A[i,l+2] = ni_b[i]
  end

  A[l+1,l+1] = v_b
  A[l+1,l+2] = na_b

  for i = 1:l+1
    A[l+2,i] = T_b
  end
  A[l+2,l+2] = M*v0^2/k/T0*(mb[1]*nm_b+mb[2]*na_b)*v_b
  A[l+2,l+3] = nm_b+na_b

  for i = 1:l
    A[l+3,i] = 2.5*T_b+ei_b[i]+e0_b
  end
  A[l+3,l+1] = 1.5*T_b+ef_b
  A[l+3,l+2] = 1/v_b*(3.5*nm_b*T_b+2.5*na_b*T_b+sum((ei_b.+e0_b).*ni_b)+ef_b*na_b)
  A[l+3,l+3] = 2.5*nm_b+1.5*na_b

  AA = A; #println("AA = ", AA, "\n")
  #display(UnicodePlots.spy(AA))
  #spy(sparse(A), ms=5)
  #display(PyPlot.spy(A))
  #display(Plots.spy(A))

  # Equilibrium constant for DR processes
  Kdr = (m[1]*h^2/(m[2]*m[2]*2*pi*k*temp))^(3/2)*Z_rot*exp.(-e_i/(k*temp))*exp(D/temp); println("Kdr = ", Kdr, "\n")

  # Equilibrium constant for VT processes
  Kvt = exp.((e_i[1:end-1]-e_i[2:end])/(k*temp)); println("Kvt = ", Kvt, "\n")

  # Dissociation processes
  kd = zeros(2,l)
  kd = kdis(temp) * Delta*n0/v0;
  println("kd = ", kd, "\n", size(kd), "\n")

  # Recombination processes
  kr = zeros(2,l)
  for iM = 1:2
    kr[iM,:] = kd[iM,:] .* Kdr * n0
  end
  println("kr = ", kr, "\n", size(kr), "\n")

  # VT processes: i+1 -> i
  kvt_down = zeros(2,Lmax)
  kvt_up   = zeros(2,Lmax)
  #kvt_down = 0.0 #kvt_ssh!(temp) * Delta*n0/v0; println("kvt_down = ", kvt_down, "\n")
  #for ip = 1:2
  #  kvt_up[ip,:] = kvt_down[ip,:] .* Kvt
  #end
  #println("kvt_up = ", kvt_up, "\n", size(kvt_up), "\n")

  # VV processes
  kvv_down = zeros(Lmax,Lmax)
  kvv_up   = zeros(Lmax,Lmax)
  #kvv_down = kvv_ssh!(temp) * Delta*n0/v0
  #deps     = e_i[1:end-1]-e_i[2:end]
  #for ip in 1:Lmax
  #@. kvv_up[ip,:] = kvv_down[ip,:] .* exp.((deps[ip].-deps) / (k*temp))
  #end
  #println("kvv_up = ", kvv_up, "\n", size(kvv_up), "\n")

  RD  = zeros(l)
  RVT = zeros(l)
  RVV = zeros(l)

  for i1 = 1:l

    RD[i1] = nm_b*(na_b*na_b*kr[1,i1]-ni_b[i1]*kd[1,i1]) + na_b*(na_b*na_b*kr[2,i1]-ni_b[i1]*kd[2,i1])

      if i1 == 1 # 0<->1

        RVT[i1] = nm_b*(ni_b[i1+1]*kvt_down[1,i1]-ni_b[i1]*kvt_up[1,i1])+
                  na_b*(ni_b[i1+1]*kvt_down[2,i1]-ni_b[i1]*kvt_up[2,i1])

        RVV[i1] = ni_b[i1+1]*sum(ni_b[1:end-1] .* kvv_down[i1,:]) -
                  ni_b[i1]  *sum(ni_b[2:end]   .* kvv_up[i1,:])

      elseif i1 == l # Lmax <-> Lmax-1

        RVT[i1] = nm_b*(ni_b[i1-1]*kvt_up[1,i1-1]-ni_b[i1]*kvt_down[1,i1-1]) +
                  na_b*(ni_b[i1-1]*kvt_up[2,i1-1]-ni_b[i1]*kvt_down[2,i1-1])

        RVV[i1] = ni_b[i1-1]*sum(ni_b[2:end]   .* kvv_up[i1-1,:]) -
                  ni_b[i1]  *sum(ni_b[1:end-1] .* kvv_down[i1-1,:])

      else

        RVT[i1] = nm_b*(ni_b[i1+1] * kvt_down[1,i1] + ni_b[i1-1] * kvt_up[1,i1-1] -
                        ni_b[i1]   * (kvt_up[1,i1]  + kvt_down[1,i1-1])) +
                  na_b*(ni_b[i1+1] * kvt_down[2,i1] + ni_b[i1-1] * kvt_up[2,i1-1] -
                        ni_b[i1]   * (kvt_up[2,i1]  + kvt_down[2,i1-1]))

        RVV[i1] = ni_b[i1+1] * sum(ni_b[1:end-1]  .* kvv_down[i1,:]) +
                  ni_b[i1-1] * sum(ni_b[2:end]    .* kvv_up[i1-1,:]) -
                  ni_b[i1]   * (sum(ni_b[2:end]   .* kvv_up[i1,:]) +
                                sum(ni_b[1:end-1] .* kvv_down[i1-1,:]))
      end
  end

  println("RD = ",  RD,  "\n", size(RD))
  println("RVT = ", RVT, "\n", size(RVT))
  println("RVV = ", RVV, "\n", size(RVV))

  B      = zeros(l+3)
  B[1:l] = RD + RVT + RVV
  B[l+1] = - 2*sum(RD)
  println("B = ", B, "\n", size(B))
  u      = inv(AA)*B
  #println("inv(A) = ", inv(A), "\n", size(inv(A)))
  #println("u = ", u, "\n", size(u))
  # du = u
end

prob = ODEProblem(rpart!, Y0_bar, xspan)
#sol = DifferentialEquations.solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8, save_everystep=false)
sol  = DifferentialEquations.solve(prob)
println("sol: ", size(sol), "\n")

#display(Plots.plot(sol))
#display(PyPlot.plot(sol))
#display(UnicodePlots.plot(sol))
#display(Plots.plot(sol,vars=(0,1)))

#X      = sol.t;                                                println("X = ", X, "\n", size(X), "\n")
#x_s    = X*Delta*100;                                          println("x_s = ", x_s, "\n")
#Temp   = sol[l+3,:]*T0;                                        println("Temp = ", Temp, "\n")
#v      = sol[l+2,:]*v0;                                        println("v = ", v, "\n")
#display(Plots.plot(x_s,Temp, xaxis=:log, yaxis=:log))

#n_i    = sol[1:l,:]*n0;                                        println("n_i = ", n_i, "\n", "Size of n_i = ", size(n_i), "\n")
#n_a    = sol[l+1,:]*n0;                                        println("n_a = ", n_a, "\n", "Size of n_a = ", size(n_a), "\n")
#n_m    = sum(n_i,dims=1);                                      println("n_m = ", n_m, "\n", "Size of n_m = ", size(n_m), "\n")
#time_s = X*Delta/v0;                                           println("time_s = ", time_s, "\n")
#Npoint = length(X);                                            println("Npoint = ", Npoint, "\n")
#Nall   = sum(n_i,dims=1);                                      println("Nall = ", Nall, "\n", size(Nall), "\n")
#Nall   = Nall[1,:]+n_a;                                        println("Nall = ", Nall, "\n", size(Nall), "\n")
#ni_n   = n_i ./ repeat(Nall,1,l);                              println("ni_n = ", ni_n, "\n")
##ni_n  = n_i ./ repeat(Nall,l,1);                              println("ni_n = ", ni_n, "\n")
#nm_n   = sum(ni_n,dims=2);                                     println("nm_n = ", nm_n, "\n")
#na_n   = n_a ./ Nall;                                          println("na_n = ", na_n, "\n")
#rho    = m[1]*n_m + m[2]*n_a;                                  println("rho = ", rho, "\n")
#p      = Nall*k .* Temp;                                       println("p = ", p, "\n")
##e_v   = repeat(e_i+e_0,Npoint,1) .* n_i;
#e_v    = repeat(e_i+e_0,1,Npoint) .* n_i;
#e_v    = sum(e_v,dims=2);                                      println("eᵥ = ", e_v, "\n")
#e_v0   = n0*xc[1]/Zvibr_0*sum(exp.(-e_i./Tv0/k) .* (e_i+e_0)); println("eᵥ₀ = ", e_v0, "\n")
#e_f    = 0.5*D*n_a*k;                                          println("e_f = ", e_f, "\n")
#e_f0   = 0.5*D*xc[2]*n0*k;                                     println("e_f0 = ", e_f0, "\n")
#e_tr   = 1.5*Nall*k .* Temp;                                   println("e_tr = ", e_tr, "\n")
#e_tr0  = 1.5*n0*k .* T0;                                       println("e_tr0 = ", e_tr0, "\n")
#e_rot  = n_m*k .* Temp;                                        println("e_rot = ", e_rot, "\n")
#e_rot0 = n0*xc[1]*k .* T0;                                     println("e_rot0 = ", e_rot0, "\n")
#E      = e_tr+e_rot+e_v+e_f;                                   println("E = ", E, "\n")
#E0     = e_tr0+e_rot0+e_v0+e_f0;                               println("E0 = ", E0, "\n")
#H      = (E+p) ./ rho;                                         println("H = ", H, "\n")
#H0     = (E0+p0) ./ rho0;                                      println("H0 = ", H0, "\n")
#u10    = rho0*v0;                                              println("u10 = ", u10, "\n")
#u20    = rho0*v0^2+p0;                                         println("u20 = ", u20, "\n")
#u30    = H0+v0^2/2;                                            println("u30 = ", u30, "\n")
#u1     = u10-rho .* v;                                         println("u₁ = ", u1, "\n")
#u2     = u20-rho .* v.^2-p;                                    println("u₂ = ", u2, "\n")
#u3     = u30-H-v.^2/2;                                         println("u₃ = ", u3, "\n")
#d1     = max(abs(u1)/u10);
#d2     = max(abs(u2)/u20);
#d3     = max(abs(u3)/u30);
#
#display("Relative error of conservation law of:");
#println("mass = ", d1);
#println("momentum = ", d2);
#println("energy = ", d3);

#include("rpart_post.jl")
#RDm  = zeros(Npoint,l);
#RDa  = zeros(Npoint,l);
#RVTm = zeros(Npoint,l);
#RVTa = zeros(Npoint,l);
#RVV  = zeros(Npoint,l);
#
#for i = 1:Npoint
#  input = Y[i,:]
#  rdm, rda, rvtm, rvta, rvv = rpart_post(input)
#  RDm[i,:]  = rdm
#  RDa[i,:]  = rda
#  RVTm[i,:] = rvtm
#  RVTa[i,:] = rvta
#  RVV[i,:]  = rvv
#end
#
#RD_mol = RDm+RDa
#RVT    = RVTm+RVTa
#RD_at  = -2*sum(RD_mol,2)
