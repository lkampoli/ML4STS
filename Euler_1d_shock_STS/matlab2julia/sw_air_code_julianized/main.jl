# https://diffeq.sciml.ai/v2.0.0/solvers/ode_solve.html#Stiff-Problems-1
using BenchmarkTools
using MAT
#using CSV
using DataFrames
using SymPy
using Plots; #gr(fmt=:png)
#using UnicodePlots
#using PyPlot; pygui(true)
#using Images
#using JLD
using DifferentialEquations
#using DiffEqOperators
#using DiffEqParamEstim
#using DiffEqDevTools
#using DiffEqSensitivity
#using StaticArrays
#using OrdinaryDiffEq
using LinearAlgebra
using ODE
using ODEInterface
using ODEInterfaceDiffEq
#using MATLABDiffEq
using LSODA
#using SciPyDiffEq
#using deSolveDiffEq
using ModelingToolkit
using SparsityDetection
using SparseArrays
#using AlgebraicMultigrid
using Sundials
#using Test
#using Distributed
#using ParameterizedFunctions
using PolynomialRoots
using Polynomials
using Roots
#addprocs()
#@everywhere using DifferentialEquations
#using Unitful
#using PhysicalConstants.CODATA2014
LinearAlgebra.BLAS.set_num_threads(1)

#using Flux, DiffEqFlux, Zygote

# Switch for model oscillator: 1 - anharmonic; 2 - harmonic
const sw_o   = 1;
# const sw_o = 2;

const sw_arr   = "Park";
# const sw_arr = "Scanlon";

const sw_u   = "D/6k";
# const sw_u = "3T";
# const sw_u = "Inf";

const sw_z   = "Savelev";
# const sw_z = "Starik";
# const sw_z = "Stellar";

const c     = 2.99e8;
const h     = 6.6261e-34;
const k     = 1.3807e-23;
const N_a   = 6.0221e23;
const h_bar = h/(2*pi);
const R     = 8.3145;

# N2-O2-NO, i^-1
const om_e   = [235857 158019 190420];
const om_x_e = [1432 1198 1407.5];
const Be     = [1.998 1.4377 1.6720]*100;

# load("wurster_data_times.mat");
# data_40_60 = sortrows(data_40_60);
# data_22_77 = sortrows(data_22_77);
# data_5_95 = sortrows(data_5_95);
# data_exp = [data_5_95; data_22_77; data_40_60];

if (sw_arr == "Park")

  #const ARR_D_data = readtable("arr_d_park.dat");
  #const ARR_Z_data = readtable("arr_z_park.dat");

  #const D  = ARR_D_data([3,6,9],1)';
  const D  = [113200  113200  113200  113200  113200;
              59500 59500 59500 59500 59500;
              75500 75500 75500 75500 75500];
  println("D = ", D, "\n", size(D), "\n");
  #const CA = ARR_D_data([1,4,7],:);
  const CA = [1.162385214460072e-08 1.162385214460072e-08 1.162385214460072e-08 4.981650919114595e-08 4.981650919114595e-08;
              3.321100612743063e-09 3.321100612743063e-09 3.321100612743063e-09 1.660550306371531e-08 1.660550306371531e-08;
              8.302751531857657e-15 8.302751531857657e-15 1.826605337008685e-13 1.826605337008685e-13 1.826605337008685e-13];
  println("CA = ", CA, "\n", size(CA), "\n");
  #const nA = ARR_D_data([2,5,8],:);
  const nA = [-1.6  -1.6  -1.6  -1.6  -1.6;
              -1.5  -1.5  -1.5  -1.5  -1.5;
              0 0 0 0 0];
  println("nA = ", nA, "\n", size(nA), "\n");

  #const CAZ = ARR_Z_data(:,1);
  #const nAZ = ARR_Z_data(:,2);
  #const EaZ = ARR_Z_data(:,3);
  const CAZ = [1.06274e-12 1.39485e-17]; # [m^3/sec]
  const nAZ = [-1. 0.];
  const EaZ = [5.29756e-19 2.68537e-19]; # [J]

elseif (sw_arr == "Scanlon")

  # NO DATA
  #const ARR_D_data = readtable("arr_d_scanlon.dat");
  #const ARR_Z_data = readtable("arr_z_scanlon.dat");

  #const D  = ARR_D_data([3,6,9],1)'/k; # J->K
  #const CA = ARR_D_data([1,4,7],:); # m^3/sec
  #const nA = ARR_D_data([2,5,8],:);

  #const CAZ = ARR_Z_data(:,1); # m^3/sec
  #const nAZ = ARR_Z_data(:,2);
  #const EaZ = ARR_Z_data(:,3);

else
  disp("Error! Check Arrhenius input.")
  return;
end

const QN = [47 36 39; 33 26 28];
const l = QN[sw_o,:];

include("en_vibr.jl")
include("en_vibr_0.jl")
const en2_i = en_vibr(1);   println("en2_i = ", en2_i, "\n", size(en2_i), "\n")
const en2_0 = en_vibr_0(1); println("en2_0 = ", en2_0, "\n", size(en2_0), "\n")
const eo2_i = en_vibr(2);   println("eo2_i = ", eo2_i, "\n", size(eo2_i), "\n")
const eo2_0 = en_vibr_0(2); println("eo2_0 = ", eo2_0, "\n", size(eo2_0), "\n")
const eno_i = en_vibr(3);   println("eno_i = ", eno_i, "\n", size(eno_i), "\n")
const eno_0 = en_vibr_0(3); println("eno_0 = ", eno_0, "\n", size(eno_0), "\n")

const ef = [0., 0., (0.5*(D[1]+D[2])-D[3]), 0.5*D[1], 0.5*D[2]]*k;

const mu = [28 32 30 14 16]*1e-3; # kg/mol
const m = mu./N_a; # N2-O2-NO-N-O

# N2-O2-NO
const osc_mass = [0.5*m[4], 0.5*m[5], m[4]*m[5]/(m[4]+m[5])];

# mass(atom1)/mass(molecule), mass(atom2)/mass(molecule)
const ram_masses = [0.5 0.5;        # 'N2'
                    0.5 0.5;        # 'O2'
                    0.4668 0.5332]; # 'NO'

const fho_data_n2 = [3.9e10  1   * k  0.9;    # 'N2+N2'
                     3.9e10  6   * k  0.95;   # 'N2+O2'
                     4e10    2   * k  0.75;   # 'N2+NO'
                     4.6e10  1   * k  0.99;   # 'N2+N'
                     7.3e10  500 * k  0.175]; # 'N2+O'

const fho_data_o2 = [4.1e10  150   * k  0.333333; # 'O2+N2'
                     4.3e10  40    * k  0.99;     # 'O2+O2'
                     4.1e10  150   * k  0.333333; # 'O2+NO'
                     7.3e10  10    * k  0.25;     # 'O2+N'
                     2.6e10  17000 * k  0.2];     # 'O2+O'

const fho_data_no = [4.4e10  20 * k  0.9;      # 'NO+N2'
                     6.75e10  1500 * k  0.2;   # 'NO+O2'
                     6.25e10  4500 * k  0.03;  # 'NO+NO'
                     5e10  200 * k  0.3183;    # 'NO+N'
                     7.9e10  16000 * k  0.06]; # 'NO+O'

const FHO_data = cat(fho_data_n2,fho_data_o2,fho_data_no, dims=3);

# VSS: dref [i], omega
const vss_data_n2 = [4.04e-10 0.686;                # 'N2+N2'
                     3.604e-10 0.703;               # 'N2+O2'
                     4.391e-10 0.756;               # 'N2+NO'
                     4.088e-10 0.762;               # 'N2+N'
                     3.2220000000000004e-10 0.702]; # 'N2+O'

const vss_data_o2 = [3.604e-10 0.703;              # 'O2+N2'
                     3.8960000000000003e-10 0.7;   # 'O2+O2'
                     4.054e-10 0.718;              # 'O2+NO'
                     3.7210000000000004e-10 0.757; # 'O2+N'
                     3.734e-10 0.76];              # 'O2+O'

const vss_data_no = [4.391e-10 0.756;              # 'NO+N2'
                     4.054e-10 0.718;              # 'NO+O2'
                     4.2180000000000003e-10 0.737; # 'NO+NO'
                     4.028e-10 0.788;              # 'NO+N'
                     3.693e-10 0.752];             # 'NO+O'

const vss_data = cat(vss_data_n2,vss_data_o2,vss_data_no, dims=3);

# N2-N2, O2-O2, NO-NO, N-N, O-O
const R0 = [3.621e-10 3.458e-10 3.47e-10 3.298e-10 2.75e-10]; # m

# N2-N2, O2-O2, NO-NO, N-N, O-O
const EM = [97.5 107.4 119 71.4 80]; # K

const r0 = 0.5 .* [R0[1].+R0; R0[2].+R0; R0[3].+R0]; println("r0 = ", r0, "\n", size(r0), "\n")

const em = [sqrt.(EM[1]*R0[1]^6 .* EM.*R0.^6)./r0[1,:].^6;
            sqrt.(EM[2]*R0[2]^6 .* EM.*R0.^6)./r0[2,:].^6;
            sqrt.(EM[3]*R0[3]^6 .* EM.*R0.^6)./r0[3,:].^6];

const re = [1.097 1.207 1.151] .* 1e-10; # N2-O2-NO

# Wurster, 1991
const us_95_5_exp  = [3.87; 3.49; 3.15; 2.97];
const us_78_22_exp = [3.85; 3.52; 3.26; 2.99];
const us_60_40_exp = [3.85; 3.47; 3.24; 3.06];

# xN2-xO2-us
const incon = [0.95 0.05 us_95_5_exp[1]];
#const incon = [0.777, 0.223, us_78_22_exp[1]];
#const incon = [0.6, 0.4, us_60_40_exp[1]];

const Torr = 133.322;
const p0 = 2.25*Torr; # Pa
const T0 = 300.;      # K
const Tv0n2 = T0;     # K
const Tv0o2 = T0;     # K
const Tv0no = T0;     # K

const Zv0_n2 = sum(exp.(-en2_i./Tv0n2/k));
const Zv0_o2 = sum(exp.(-eo2_i./Tv0o2/k));
const Zv0_no = sum(exp.(-eno_i./Tv0no/k));

const n0 = p0/(k*T0);

# N2-N2
const sigma0 = pi*R0[1]^2;
const lambda0 = 1/(sqrt(2)*n0*sigma0);
const Delta = lambda0;

# xc = nc/n
const xc = zeros(5);
const xc[1] = incon[1];
const xc[2] = incon[2];
const v0    = incon[3]*1e3;

const gamma0 = 1.4;
const rho0_c = m.*xc*n0;
const rho0 = sum(rho0_c);
const mu0_mix = sum(rho0_c./mu)/rho0;
const R_bar = R*mu0_mix;
const a0 = sqrt(gamma0*R_bar*T0);
const M0 = v0/a0;

include("in_con.jl")
#NN = in_con();
NN = [5.761793697284453e+00 1.735570644383367e-01 2.445965060176968e+01]
n1 = NN[1]; println("n1 = ", n1, "\n")
v1 = NN[2]; println("v1 = ", v1, "\n")
T1 = NN[3]; println("T1 = ", T1, "\n")

Y0_bar = zeros(sum(l)+4);

Y0_bar[1:l[1]] = xc[1]*n1/Zv0_n2*exp.(-en2_i./Tv0n2/k);             # N2
Y0_bar[l[1]+1:l[1]+l[2]] = xc[2]*n1/Zv0_o2*exp.(-eo2_i./Tv0o2/k);   # O2
Y0_bar[l[1]+l[2]+1:sum(l)] = xc[3]*n1/Zv0_no*exp.(-eno_i./Tv0no/k); # NO
Y0_bar[sum(l)+1] = xc[4];                                           # N
Y0_bar[sum(l)+2] = xc[5];                                           # O
Y0_bar[sum(l)+3] = v1;
Y0_bar[sum(l)+4] = T1;

const x_w = 2.; # m
const xspan = [0., x_w]./Delta;

include("k_ex_savelev_st.jl")
include("kdis.jl")
include("kvt_fho.jl")
include("kvv_fho.jl")
include("rpart_fho.jl")
prob = ODEProblem(rpart_fho!, Y0_bar, xspan, 1.)
sol  = DifferentialEquations.solve(prob, radau(), reltol=1e-8, abstol=1e-8, save_everystep=true, progress=true)

#x_s = X*Delta*100;
#Temp = Y(:,sum(l)+4)*T0;
#v = Y(:,sum(l)+3)*v0;
#
#nn2_i = Y(:,1:l(1))*n0;
#no2_i = Y(:,l(1)+1:l(1)+l(2))*n0;
#nno_i = Y(:,l(1)+l(2)+1:sum(l))*n0;
#
## [i^-3]
#nn2 = sum(nn2_i,2);
#no2 = sum(no2_i,2);
#nno = sum(nno_i,2);
#
#nn = Y(:,sum(l)+1)*n0; # N
#no = Y(:,sum(l)+2)*n0; # O
#
#n_a = nn+no;
#n_m = nn2+no2+nno;
#
#time_s = X*Delta/v0;   # sec
#time_mcs = time_s*1e6; # mcsec
#
#Npoint = length(X);
#Nall = n_m+n_a;
#nn2i_n = nn2_i./repmat(Nall,1,l(1));
#no2i_n = no2_i./repmat(Nall,1,l(2));
#nnoi_n = nno_i./repmat(Nall,1,l(3));
#nn2_n = sum(nn2i_n,2);
#no2_n = sum(no2i_n,2);
#nno_n = sum(nnoi_n,2);
#nn_n = nn./Nall;
#no_n = no./Nall;
#
#rho = m(1)*sum(nn2_i,2)+m(2)*sum(no2_i,2)+m(3)*sum(nno_i,2)+m(4)*nn+m(5)*no;
#p = (n_m+n_a)*k.*Temp;
#Tvn2 = en2_i(2)./(k*log(nn2i_n(:,1)./nn2i_n(:,2)));
#Tvo2 = eo2_i(2)./(k*log(no2i_n(:,1)./no2i_n(:,2)));
#Tvno = eno_i(2)./(k*log(nnoi_n(:,1)./nnoi_n(:,2)));
#ev_n2 = repmat(en2_i'+en2_0,Npoint,1).*nn2_i;
#ev_o2 = repmat(eo2_i'+eo2_0,Npoint,1).*no2_i;
#ev_no = repmat(eno_i'+eno_0,Npoint,1).*nno_i;
#e_v = sum(ev_n2,2)+sum(ev_o2,2)+sum(ev_no,2);
#e_f = 0.5*k*(D(1)*nn+D(2)*no)+k*(0.5*(D(1)+D(2))-D(3))*sum(nno_i,2);
#e_tr = 1.5*(n_m+n_a)*k.*Temp;
#e_rot = n_m*k.*Temp;
#H = (3.5*n_m*k.*Temp+2.5*n_a*k.*Temp+e_v+e_f)./rho;
#
#e_v0 = n0*(xc(1)/Zv0_n2*sum(exp(-en2_i/Tv0n2/k).*(en2_i+en2_0))+
#           xc(2)/Zv0_o2*sum(exp(-eo2_i/Tv0o2/k).*(eo2_i+eo2_0))+
#           xc(3)/Zv0_no*sum(exp(-eno_i/Tv0no/k).*(eno_i+eno_0)));
#
#e_f0 = 0.5*k*(D(1)*xc(4)*n0+D(2)*xc(5)*n0)+k*(0.5*(D(1)+D(2))-D(3))*xc(3)*n0;
#
#u10 = rho0*v0;
#u20 = rho0*v0^2+p0;
#u30 = (3.5*(sum(xc(1:3)))*n0*k*T0+2.5*(sum(xc(4:5)))*n0*k*T0+e_v0+e_f0)/rho0+v0^2/2;
#
#u1 = u10-rho.*v;
#u2 = u20-rho.*v.^2-p;
#u3 = u30-H-v.^2/2;
#
#d1 = max(abs(u1)/u10);
#d2 = max(abs(u2)/u20);
#d3 = max(abs(u3)/u30);
#
#tol = 1e-4;
#if (d1>tol)||(d2>tol)||(d3>tol)
#  disp("Big error!");
#  return;
#end
