using BenchmarkTools
using MAT
using SymPy
using Plots;
using DifferentialEquations
using StaticArrays
using OrdinaryDiffEq
using LinearAlgebra
using ODE
using ODEInterface
using ODEInterfaceDiffEq
LinearAlgebra.BLAS.set_num_threads(1)

const sw_sp = 1
const sw_o = 1
const xc = [1 0]
const x_w = 100.
const c     = 2.99e8;
const h     = 6.6261e-34;
const k     = 1.3807e-23;
const N_a   = 6.0221e23;
const R     = 8.3145;
const h_bar = h/(2*pi);

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

include("en_vibr.jl")
const e_i = en_vibr(); println("eᵢ = ", e_i, "\n")

include("en_vibr_0.jl")
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

include("in_con.jl")
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

#kvv_down = Array{Float64}(undef, l-1, l-1)
#kvv_down = zeros(l-1,l-1)

include("kvv_ssh.jl")
@time kvv_down = kvv_ssh!(1500.)
#@time kvv_ssh!(1500.)
