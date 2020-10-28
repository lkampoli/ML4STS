
"$VERSION"

using Pkg
Pkg.add("ModelingToolkit")
Pkg.add("DiffEqFlux")
Pkg.add("DiffEqBase")
Pkg.add("Flux")
Pkg.add("Plots")
Pkg.add("Test")
Pkg.add("Optim")
Pkg.add("CUDA")

]add GalacticOptim

]add NeuralPDE

using ModelingToolkit

using NeuralPDE

using DiffEqFlux

using Flux

using Plots

using GalacticOptim

using Test

using Optim

#using CUDA

cb = function (p,l)
    println("Current loss is: $l")
    return false
end


