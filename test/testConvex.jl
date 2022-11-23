#=
testConvex.jl
=======
Uses .ConvexSample to evaluate a affine underestimator function, replicating
the results of calculations from the following article:

[1]: KA Khan et al (2021), https://doi.org/10.1016/j.compchemeng.2021.107413

An alternate method under construction has also been implemented to evaluate
the function with fewer sample points.
=#

include("../src/ConvexSampling.jl")

using .ConvexSampling
using LinearAlgebra
using Plots
using Test

##
println("Running Example 1: A function with 1 variable:\n")

f1(x) = @. (2)*x[1]^2 + (5)*x[1] + 3 + (2)*abs(x[1] - 2)
xL = [-3.0]
xU = [3.0]
println("Function: f1(x) = 2*x[1]^2 + 5*x[2] + 3 + 2*abs(x[1] - 2) \n with bounds x = ",xL," and ",xU,".\n")

_, bcoeff1, ccoeff1 = eval_sampling_underestimator_coeffs(f1, xL, xU)
println("Calculated b and c coefficients are ", bcoeff1, " and ", ccoeff1, " respectively.\n")

yOutput1 = eval_sampling_underestimator(f1, xL, xU, [2.0])
println("At x = ", [2.0], " the affine underestimator outputs ", yOutput1, ".\n")

fL1 = eval_sampling_lower_bound(f1, xL, xU)
println("The lower bound (fL) is = ", fL1, ".\n")

println("A 2D plot of the function, lowerbound, and affine underestimator is provided.")
graph1 = plot_sampling_underestimator(f1, xL, xU, fEvalResolution=100)

##
println("Running Example 2: A function with 2 variables:\n")

f2(x) = dot(x,([65 56; 56 65]),x) + dot([6; 2],x) + 23
xL = [-5.0, -2.0]
xU = [5.0, 3.0]
println("Function: f2(x) = dot(x,(",[65 56; 56 65],"),x) + dot(",[6;2],",x) + ",23," \n with bounds x = ",xL," and ",xU,".\n")

_, bcoeff2, ccoeff2 = eval_sampling_underestimator_coeffs(f2, xL, xU)
println("Calculated b and c coefficients are ", bcoeff2, " and ", ccoeff2, " respectively.\n")

yOutput2 = eval_sampling_underestimator(f2, xL, xU, [2.0, 2.0])
println("At x = ", [2.0, 2.0], " the affine underestimator outputs ", yOutput2, ".\n")

fL2 = eval_sampling_lower_bound(f2, xL,xU)
println("The lower bound (fL) is = ", fL2, ".\n")

graph2 = plot_sampling_underestimator(f2, xL, xU)
println("A 3D plot of the function (multicolor), lowerbound (red/pink), and affine underestimator (wireframe) is provided.")
println("Enter @show <graphname> in the command window to access graphs.\n")

##
println("Running Example 3: A function with 3 variables:\n")
f3(x) = dot(x,[66 36 30; 36 33 36; 30 36 66],x) + dot([6; 2; 6],x) + 23
xL = [-5.0, -2.0, -1.0]
xU = [6.0, 3.0, 7.0]
println("Function: f3(x) = dot(x,(",[66 36 30; 36 33 36; 30 36 66],"),x) + dot(",[6;2;6],",x) + ",23," \n with bounds x = ",xL," and ",xU,".\n")

display(sample_convex_function(f3, xL, xU))

_, bcoeff3, ccoeff3 = eval_sampling_underestimator_coeffs(f3, xL, xU)
println("Calculated b and c coefficients are ", bcoeff3, " and ", ccoeff3, " respectively.\n")

yOutput3 = eval_sampling_underestimator(f3, xL, xU, [1.0, 1.0, 1.0])
println("At x = ", [1.0, 1.0, 1.0], " the affine underestimator outputs ", yOutput3, ".\n")

fL3 = eval_sampling_lower_bound(f3, xL,xU)
println("The lower bound (fL) is = ", fL3, ".\n")
