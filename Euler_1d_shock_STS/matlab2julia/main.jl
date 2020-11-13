
using Unitful
using BenchmarkTools
using MAT
using SymPy;
using Plots
using UnicodePlots
using PyPlot
using Images
using JLD
using DifferentialEquations
using StaticArrays
using ModelingToolkit
using SparsityDetection
using SparseArrays
using LinearAlgebra
#using DiffEqOperators
#using AlgebraicMultigrid
#using Sundials

# Switch for species: 1 - N2/N; 2 - O2/O
sw_sp = 1

# Switch for oscillator: 1 - anharmonic; 2 - harmonic
sw_o = 1

# Species molar fractions [xA2 xA] (xc = nc/n)
xc = [1 0]

# Range of integration
x_w = 100.

# To load the module for physical constants
#using PhysicalConstants.CODATA2014

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

om_e   = OMEGA[sw_sp,1]; # ωₑ
om_x_e = OMEGA[sw_sp,2]; # ωₓₑ
Be     = BE[sw_sp]     ; # Bₑ
D      = ED[sw_sp]     ;
CA     = CArr[sw_sp,:] ;
nA     = NArr[sw_sp,:] ;
l      = Int(QN[sw_sp,sw_o]);

println("om_e = ",   om_e,   "\n")
println("om_x_e = ", om_x_e, "\n")
println("Be = ",     Be,     "\n")
println("D = ",      D,      "\n")
println("CA = ",     CA,     "\n")
println("nA = ",     nA,     "\n")
println("l = ",      l,      "\n")

include("en_vibr.jl")
e_i = en_vibr()

include("en_vibr_0.jl")
e_0 = en_vibr_0()

mu     = [MU[sw_sp] 0.5*MU[sw_sp]]*1e-3;                                                  println("mu = ", mu, "\n")
m      = mu / N_a;                                                                        println("m = ", m, "\n")
sigma0 = pi*R0[sw_sp,1]^2;                                                                println("sigma0 = ", sigma0, "\n")
r0     = [R0[sw_sp,1] 0.5*(R0[sw_sp,1]+R0[sw_sp,2])];                                     println("r0 = ", r0, "\n")
em     = [EM[sw_sp,1] sqrt(EM[sw_sp,1]*EM[sw_sp,2]*R0[sw_sp,1]^6*R0[sw_sp,2]^6)/r0[2]^6]; println("em = ", em, "\n")
re     = RE[sw_sp];                                                                       println("re = ", re, "\n")

# ICs
p0  = 0.8*133.322 # p₀
T0  = 300.        # T₀
Tv0 = T0          # Tᵥ₀
M0  = 13.4        # M₀
n0  = p0/(k*T0)   # n₀

if xc[1] != 0
  gamma0 = 1.4    # γ₀
else
  gamma0 = 5/3
end

rho0_c = m.*xc*n0
rho0   = sum(rho0_c)           # ρ₀
mu_mix = sum(rho0_c./mu)/rho0
R_bar  = R*mu_mix
a0     = sqrt(gamma0*R_bar*T0) # a₀
v0     = M0*a0                 # v₀

include("in_con.jl")
NN = in_con()
T1 = 1 #NN[1]
v1 = 1 #NN[2]
n1 = 1 #NN[3]
println("T1 = ", T1, "\n", "v1 = ", v1, "\n", "n1 = ", n1, "\n")

Zvibr_0 = sum(exp.(-e_i./Tv0./k))

Y0_bar      = zeros(l+3)
Y0_bar[1:l] = xc[1]*n1/Zvibr_0*exp.(-e_i./Tv0./k)
Y0_bar[l+1] = xc[2]*n1
Y0_bar[l+2] = v1
Y0_bar[l+3] = T1
println("Y0_bar = ", Y0_bar, "\n")

Delta = 1/(sqrt(2)*n0*sigma0); println("Delta = ", Delta, "\n")
xspan = [0, x_w]./Delta;       println("xspan = ", xspan, "\n")

include("rpart.jl")
include("kdis.jl")
include("kvt_ssh.jl")
include("kvv_ssh.jl")
prob = ODEProblem(rpart!, Y0_bar, xspan)
#sol = DifferentialEquations.solve(prob, alg_hints=[:stiff], reltol=1e-8, abstol=1e-8, save_everystep=true)
sol  = DifferentialEquations.solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8, save_everystep=true)

X = sol.t

x_s    = X*Delta*100;
Temp   = sol[l+3,:]*T0; println("Temp = ", Temp, "\n")
v      = sol[l+2,:]*v0; println("v = ", v, "\n")
n_i    = sol[1:l,:]*n0; println("n_i = ", n_i, "\n", "Size of n_i = ", size(n_i), "\n")
n_a    = sol[l+1,:]*n0; println("n_a = ", n_a, "\n")
n_m    = sum(n_i,1);    println("n_m = ", n_m, "\n")
time_s = X*Delta/v0;    println("time_s = ", time_s, "\n")
Npoint = length(X);     println("Npoint = ", Npoint, "\n")
Nall   = sum(n_i,2)+n_a;
ni_n   = n_i ./ repmat(Nall,1,l);
nm_n   = sum(ni_n,2);
na_n   = n_a ./ Nall;
rho    = m[1]*n_m + m[2]*n_a;
p      = Nall*k .* Temp;
e_v    = repmat(e_i+e_0,Npoint,1) .* n_i;
e_v    = sum(e_v,2);
e_v0   = n0*xc[1]/Zvibr_0*sum(exp.(-e_i./Tv0/k) .* (e_i+e_0));
e_f    = 0.5*D*n_a*k;
e_f0   = 0.5*D*xc[2]*n0*k;
e_tr   = 1.5*Nall*k .* Temp;
e_tr0  = 1.5*n0*k .* T0;
e_rot  = n_m*k .* Temp;
e_rot0 = n0*xc[1]*k .* T0;
E      = e_tr+e_rot+e_v+e_f;
E0     = e_tr0+e_rot0+e_v0+e_f0;
H      = (E+p) ./ rho;
H0     = (E0+p0) ./ rho0;
u10    = rho0*v0;
u20    = rho0*v0^2+p0;
u30    = H0+v0^2/2;
u1     = u10-rho .* v;
u2     = u20-rho .* v.^2-p;
u3     = u30-H-v.^2/2;
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
