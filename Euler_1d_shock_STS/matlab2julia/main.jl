using Unitful
using BenchmarkTools
using MAT
using SymPy
using Plots; gr() #call the GR backend
using UnicodePlots
using PyPlot; pygui(true)
using Images
using JLD
using DifferentialEquations
using DiffEqParamEstim
using StaticArrays
using ModelingToolkit
using SparsityDetection
using SparseArrays
using LinearAlgebra
using DiffEqOperators
using AlgebraicMultigrid
using Sundials
#using PhysicalConstants.CODATA2014

# Switch for mixture species: 1 - N2/N; 2 - O2/O
sw_sp = 1

# Switch for model oscillator: 1 - anharmonic; 2 - harmonic
sw_o = 1

# Species molar fractions [xA2 xA] (xc = nc/n)
xc = [1 0]

# Range of integration
x_w = 100.

# Physical constants
c     = 2.99e8;     #SpeedOfLightInVacuum
h     = 6.6261e-34; #PlanckConstant
k     = 1.3807e-23; #BoltzmannConstant
N_a   = 6.0221e23;  #AvogadroConstant
R     = 8.3145;     #MolarGasConstant
h_bar = h/(2*pi);   #PlanckConstantOver2pi

# To read a single variable from a MAT file (compressed files are detected and handled automatically):
file  = matopen("data_species.mat")
OMEGA = read(file, "OMEGA"); # Ω
BE    = read(file, "BE")   ; # Bₑ
ED    = read(file, "ED")   ;
CArr  = read(file, "CArr") ;
NArr  = read(file, "NArr") ;
QN    = read(file, "QN")   ;
MU    = read(file, "MU")   ; # μ
R0    = read(file, "R0")   ;
EM    = read(file, "EM")   ;
RE    = read(file, "RE")   ;
close(file)

om_e   = OMEGA[sw_sp,1];      println("om_e = ",   om_e,   "\n") # ωₑ
om_x_e = OMEGA[sw_sp,2];      println("om_x_e = ", om_x_e, "\n") # ωₓₑ
Be     = BE[sw_sp]     ;      println("Be = ",     Be,     "\n") # Bₑ
D      = ED[sw_sp]     ;      println("D = ",      D,      "\n")
CA     = CArr[sw_sp,:] ;      println("CA = ",     CA,     "\n")
nA     = NArr[sw_sp,:] ;      println("nA = ",     nA,     "\n")
l      = Int(QN[sw_sp,sw_o]); println("l = ",      l,      "\n")

include("en_vibr.jl")
e_i = en_vibr(); println("eᵢ = ", e_i, "\n")

include("en_vibr_0.jl")
e_0 = en_vibr_0(); println("e₀ = ", e_0, "\n")

mu     = [MU[sw_sp] 0.5*MU[sw_sp]]*1e-3;                                                  println("mu = ", mu, "\n")
m      = mu / N_a;                                                                        println("m = ", m, "\n")
sigma0 = pi*R0[sw_sp,1]^2;                                                                println("sigma0 = ", sigma0, "\n")
r0     = [R0[sw_sp,1] 0.5*(R0[sw_sp,1]+R0[sw_sp,2])];                                     println("r0 = ", r0, "\n")
em     = [EM[sw_sp,1] sqrt(EM[sw_sp,1]*EM[sw_sp,2]*R0[sw_sp,1]^6*R0[sw_sp,2]^6)/r0[2]^6]; println("em = ", em, "\n")
re     = RE[sw_sp];                                                                       println("re = ", re, "\n")

# ICs
p0  = 0.8*133.322; println("p0 = ", p0, "\n")   # p₀
T0  = 300.;        println("T0 = ", T0, "\n")   # T₀
Tv0 = T0;          println("Tv0 = ", Tv0, "\n") # Tᵥ₀
M0  = 13.4;        println("M0 = ", M0, "\n")   # M₀
n0  = p0/(k*T0);   println("n0 = ", n0, "\n")   # n₀

if xc[1] != 0
  gamma0 = 1.4;    println("gamma0 = ", gamma0, "\n") # γ₀
else
  gamma0 = 5/3;    println("gamma0 = ", gamma0, "\n")
end

rho0_c = m.*xc*n0;              println("rho0_c = ", rho0_c, "\n")
rho0   = sum(rho0_c);           println("rho0 = ", rho0, "\n") # ρ₀
mu_mix = sum(rho0_c./mu)/rho0;  println("mu_mix = ", mu_mix, "\n")
R_bar  = R*mu_mix;              println("R_bar = ", R_bar, "\n")
a0     = sqrt(gamma0*R_bar*T0); println("a0 = ", a0, "\n") # a₀
v0     = M0*a0;                 println("v0 = ", v0, "\n") # v₀

include("in_con.jl")
NN = in_con()
n1 = NN[1]; println("n1 = ", n1, "\n")
v1 = NN[2]; println("v1 = ", v1, "\n")
T1 = NN[3]; println("T1 = ", T1, "\n")

Zvibr_0 = sum(exp.(-e_i./Tv0./k)); println("Zvibr_0 = ", Zvibr_0, "\n")

Y0_bar      = zeros(l+3)
Y0_bar[1:l] = xc[1]*n1/Zvibr_0*exp.(-e_i./Tv0./k)
Y0_bar[l+1] = xc[2]*n1
Y0_bar[l+2] = v1
Y0_bar[l+3] = T1
println("Y0_bar = ", Y0_bar, "\n", size(Y0_bar), "\n")

Delta = 1/(sqrt(2)*n0*sigma0); println("Delta = ", Delta, "\n")
xspan = [0, x_w]./Delta;       println("xspan = ", xspan, "\n")

include("rpart.jl")
include("kdis.jl")
include("kvt_ssh.jl")
include("kvv_ssh.jl")
#prob = ODEProblem(rpart!, Y0_bar, xspan)
prob = ODEProblem(rpart, Y0_bar, xspan)
#sol = DifferentialEquations.solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8, save_everystep=true)
sol  = DifferentialEquations.solve(prob)
#@btime sol  = DifferentialEquations.solve(prob)
#@benchmark sol  = DifferentialEquations.solve(prob)

# The last value of sol is the timestep, and the beginning values are for the component.
#display(Plots.plot(sol))
#display(Plots.plot(sol,vars=(0,1)))
println("sol: ", size(sol), "\n")

X      = sol.t;                                                println("X = ", X, "\n", size(X), "\n")
x_s    = X*Delta*100;                                          println("x_s = ", x_s, "\n")
Temp   = sol[l+3,:]*T0;                                        println("Temp = ", Temp, "\n")
v      = sol[l+2,:]*v0;                                        println("v = ", v, "\n")
n_i    = sol[1:l,:]*n0;                                        println("n_i = ", n_i, "\n", "Size of n_i = ", size(n_i), "\n")
n_a    = sol[l+1,:]*n0;                                        println("n_a = ", n_a, "\n", "Size of n_a = ", size(n_a), "\n")
n_m    = sum(n_i,dims=1);                                      println("n_m = ", n_m, "\n", "Size of n_m = ", size(n_m), "\n")
time_s = X*Delta/v0;                                           println("time_s = ", time_s, "\n")
Npoint = length(X);                                            println("Npoint = ", Npoint, "\n")
Nall   = sum(n_i,dims=1);                                      println("Nall = ", Nall, "\n", size(Nall), "\n")
Nall   = Nall[1,:]+n_a;                                        println("Nall = ", Nall, "\n", size(Nall), "\n")
ni_n   = n_i ./ repmat(Nall,1,l);                              println("ni_n = ", ni_n, "\n")
#ni_n  = n_i ./ repmat(Nall,l,1);                              println("ni_n = ", ni_n, "\n")
#ni_n  = n_i ./ repeat(Nall,1,l);                              println("ni_n = ", ni_n, "\n")
#ni_n  = n_i ./ repeat(Nall,l);                                println("ni_n = ", ni_n, "\n")
nm_n   = sum(ni_n,dims=2);                                     println("nm_n = ", nm_n, "\n")
na_n   = n_a ./ Nall;                                          println("na_n = ", na_n, "\n")
rho    = m[1]*n_m + m[2]*n_a;                                  println("rho = ", rho, "\n")
p      = Nall*k .* Temp;                                       println("p = ", p, "\n")
#e_v   = repeat(e_i+e_0,Npoint,1) .* n_i;
e_v    = repeat(e_i+e_0,Npoint) .* n_i;
e_v    = sum(e_v,dims=2);                                      println("eᵥ = ", e_v, "\n")
e_v0   = n0*xc[1]/Zvibr_0*sum(exp.(-e_i./Tv0/k) .* (e_i+e_0)); println("eᵥ₀ = ", e_v0, "\n")
e_f    = 0.5*D*n_a*k;                                          println("e_f = ", e_f, "\n")
e_f0   = 0.5*D*xc[2]*n0*k;                                     println("e_f0 = ", e_f0, "\n")
e_tr   = 1.5*Nall*k .* Temp;                                   println("e_tr = ", e_tr, "\n")
e_tr0  = 1.5*n0*k .* T0;                                       println("e_tr0 = ", e_tr0, "\n")
e_rot  = n_m*k .* Temp;                                        println("e_rot = ", e_rot, "\n")
e_rot0 = n0*xc[1]*k .* T0;                                     println("e_rot0 = ", e_rot0, "\n")
E      = e_tr+e_rot+e_v+e_f;                                   println("E = ", E, "\n")
E0     = e_tr0+e_rot0+e_v0+e_f0;                               println("E0 = ", E0, "\n")
H      = (E+p) ./ rho;                                         println("H = ", H, "\n")
H0     = (E0+p0) ./ rho0;                                      println("H0 = ", H0, "\n")
u10    = rho0*v0;                                              println("u10 = ", u10, "\n")
u20    = rho0*v0^2+p0;                                         println("u20 = ", u20, "\n")
u30    = H0+v0^2/2;                                            println("u30 = ", u30, "\n")
u1     = u10-rho .* v;                                         println("u₁ = ", u1, "\n")
u2     = u20-rho .* v.^2-p;                                    println("u₂ = ", u2, "\n")
u3     = u30-H-v.^2/2;                                         println("u₃ = ", u3, "\n")
d1     = max(abs(u1)/u10);
d2     = max(abs(u2)/u20);
d3     = max(abs(u3)/u30);

display("Relative error of conservation law of:");
println("mass = ", d1);
println("momentum = ", d2);
println("energy = ", d3);

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
