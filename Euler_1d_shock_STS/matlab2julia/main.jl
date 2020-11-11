
#using Unitful
using BenchmarkTools

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

# To load the module to read .mat files
using MAT

# To read a single variable from a MAT file (compressed files are detected and handled automatically):
file  = matopen("data_species.mat")
OMEGA = read(file, "OMEGA");
BE    = read(file, "BE")   ;
ED    = read(file, "ED")   ;
CArr  = read(file, "CArr") ;
NArr  = read(file, "NArr") ;
QN    = read(file, "QN")   ;
MU    = read(file, "MU")   ;
R0    = read(file, "R0")   ;
EM    = read(file, "EM")   ;
RE    = read(file, "RE")   ;
close(file)

om_e   = OMEGA[sw_sp,1];
om_x_e = OMEGA[sw_sp,2];
Be     = BE[sw_sp]     ;
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

mu     = [MU[sw_sp] 0.5*MU[sw_sp]]*1e-3
m      = mu / N_a
sigma0 = pi*R0[sw_sp,1]^2
r0     = [R0[sw_sp,1] 0.5*(R0[sw_sp,1]+R0[sw_sp,2])]
em     = [EM[sw_sp,1] sqrt(EM[sw_sp,1]*EM[sw_sp,2]*R0[sw_sp,1]^6*R0[sw_sp,2]^6)/r0[2]^6]
re     = RE[sw_sp]

# ICs
p0  = 0.8*133.322
T0  = 300.
Tv0 = T0
M0  = 13.4
n0  = p0/(k*T0)

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

#using SymPy
#include("in_con.jl")
#NN = in_con()
n1 = 1 #NN[1]
T1 = 1 #NN[2]
v1 = 1 #NN[3]

Zvibr_0 = sum(exp.(-e_i./Tv0./k))

Y0_bar = zeros(l+3)
Y0_bar[1:l] = xc[1]*n1/Zvibr_0*exp.(-e_i./Tv0./k)
Y0_bar[l+1] = xc[2]*n1
Y0_bar[l+2] = v1
Y0_bar[l+3] = T1

Delta = 1/(sqrt(2)*n0*sigma0)
xspan = [0, x_w]./Delta

#using Plots
#using UnicodePlots
using PyPlot
using Images
using JLD
using DifferentialEquations
using StaticArrays
#using AbstractArray
using ModelingToolkit
using SparsityDetection, SparseArrays
using LinearAlgebra
#using DiffEqOperators
#using AlgebraicMultigrid
#using Sundials
include("rpart.jl")
include("kdis.jl")
include("kvt_ssh.jl")
include("kvv_ssh.jl")
prob = ODEProblem(rpart!, Y0_bar, xspan)
sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)

#x_s = X*Delta*100;
#Temp = Y(:,l+3)*T0;
#v = Y(:,l+2)*v0;
#n_i = Y(:,1:l)*n0;
#n_a = Y(:,l+1)*n0;
#n_m = sum(n_i,2);
#time_s = X*Delta/v0;
#Npoint = length(X);
#Nall = sum(n_i,2)+n_a;
#ni_n = n_i./repmat(Nall,1,l);
#nm_n = sum(ni_n,2);
#na_n = n_a./Nall;
#rho = m(1)*n_m + m(2)*n_a;
#p = Nall*k.*Temp;
#e_v = repmat(e_i'+e_0,Npoint,1).*n_i;
#e_v = sum(e_v,2);
#e_v0 = n0*xc(1)/Zvibr_0*sum(exp(-e_i/Tv0/k).*(e_i+e_0));
#e_f = 0.5*D*n_a*k;
#e_f0 = 0.5*D*xc(2)*n0*k;
#e_tr = 1.5*Nall*k.*Temp;
#e_tr0 = 1.5*n0*k.*T0;
#e_rot = n_m*k.*Temp;
#e_rot0 = n0*xc(1)*k.*T0;
#E = e_tr+e_rot+e_v+e_f;
#E0 = e_tr0+e_rot0+e_v0+e_f0;
#H = (E+p)./rho;
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
#disp('Relative error of conservation law of:');
#disp(['mass = ',num2str(d1)]);
#disp(['momentum = ',num2str(d2)]);
#disp(['energy = ',num2str(d3)]);

#RDm = zeros(Npoint,l);
#RDa = zeros(Npoint,l);
#RVTm = zeros(Npoint,l);
#RVTa = zeros(Npoint,l);
#RVV = zeros(Npoint,l);
#
#for i = 1:Npoint
#    input = Y(i,:)';
#    [rdm, rda, rvtm, rvta, rvv] = rpart_post(input)
#    RDm(i,:) = rdm;
#    RDa(i,:) = rda;
#    RVTm(i,:) = rvtm;
#    RVTa(i,:) = rvta;
#    RVV(i,:) = rvv;
#end
#
#RD_mol = RDm+RDa;
#RVT    = RVTm+RVTa;
#RD_at  = -2*sum(RD_mol,2);
