using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux

@parameters x y θ
@variables u[1:2,1:2](..)
@derivatives Dxx''~x
@derivatives Dyy''~y

# matrix PDE
eqs  = @. [(Dxx(u_(x,y,θ)) + Dyy(u_(x,y,θ))) for u_ in u] ~ -sin(pi*x)*sin(pi*y)*[0 1; 0 1]

# Initial and boundary conditions
bcs = [u[1](x,0,θ) ~ x, u[2](x,0,θ) ~ 2, u[3](x,0,θ) ~ 3, u[4](x,0,θ) ~ 4]

@parameters x θ
@variables u(..)
@derivatives Dxxx'''~x
@derivatives Dx'~x

# ODE
eq = Dxxx(u(x,θ)) ~ cos(pi*x)

# Initial and boundary conditions
bcs = [u(0.,θ) ~ 0.0,
       u(1.,θ) ~ cos(pi),
       Dx(u(1.,θ)) ~ 1.0]

# Space and time domains
domains = [x ∈ IntervalDomain(0.0,1.0)]

# Discretization
dx = 0.05

# Neural network
chain = FastChain(FastDense(1,8,Flux.σ),FastDense(8,1))

discretization = PhysicsInformedNN(dx,
                                   chain,
                                   strategy=StochasticTraining(include_frac=0.5))
pde_system = PDESystem(eq,bcs,domains,[x],[u])
prob = discretize(pde_system,discretization)

res = GalacticOptim.solve(prob, ADAM(0.01), progress = false; cb = cb, maxiters=2000)
phi = discretization.phi

analytic_sol_func(x) = (π*x*(-x+(π^2)*(2*x-3)+1)-sin(π*x))/(π^3)

xs = [domain.domain.lower:dx/10:domain.domain.upper for domain in domains][1]
u_real  = [analytic_sol_func(x) for x in xs]
u_predict  = [first(phi(x,res.minimizer)) for x in xs]

x_plot = collect(xs)
plot(x_plot ,u_real,title = "real")
plot!(x_plot ,u_predict,title = "predict")
