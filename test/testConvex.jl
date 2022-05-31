#=
testConvex.jl
=======
Uses .ConvexSample to evaluate a affine underestimator function, replicating
the results of calculations from the following article:

[1]: KA Khan et al (2021), https://doi.org/10.1016/j.compchemeng.2021.107413
=#

include("../src/ConvexSample.jl")

using .ConvexSample
using LinearAlgebra
using Plots

println("Running Example 1: A function with 1 variable:\n")

f2(x) = 2*x.^2
#Try f(x) = exp(x), f(x) = 0.5/x^^
xL = [-5.0]
xU = [5.0]

b_coeff1, c_coeff1 = aff_coefficients(f2, 1, xL, xU)
println("Calculated b and c coefficients are ", b_coeff1[1], " and ", c_coeff1, " respectively.\n")

aff_underestimator(f2, 1, xL, xU)
yOutput1 = aff_underestimator_atx(f2, 1, xL, xU, [2.0])
println("At x = ", [2.0], " the affine underestimator outputs ", yOutput1, ".\n")

fL1 = lower_bound(f2, 1, xL, xU)
println("The lower bound (fL) is = ", fL1, ".\n")

# println("A 3D plot of the function, lowerbound, and affine underestimator is provided.")
# graph1 = plotter(f2, 1, xL, xU)
# @show graph1

println("Running Example 2: A function with 2 variables:\n")

A= [7 4; 4 7]
b = [6; 2]
c = 23
f3(x) = dot(x,(A'*A),x) + dot(b,x) + c
xL = [-5.0, -3.0]
xU = [5.0, 3.0]
println("Function: f3(x) = dot(x,(",A'*A,"),x) + dot(",b,",x) + ",c," \n with bounds x = ",xL," and ",xU,".\n")

b_coeff, c_coeff = aff_coefficients(f3, 2, xL, xU)
println("Calculated b and c coefficients are ", b_coeff, " and ", c_coeff, " respectively.\n")

affine_function = aff_underestimator(f3, 2, xL, xU)
println("Affine underestimator format is f(x) = c + dot(b, (x1,x2)-w0). \n")


yOutput = aff_underestimator_atx(f3, 2, xL, xU, [2.0, 2.0])
println("At x = ", [2.0, 2.0], " the affine underestimator outputs ", yOutput, ".\n")

fL = lower_bound(f3,2,xL,xU)
println("The lower bound (fL) is = ", fL, ".\n")

# graph2 = plotter(f3, 2, xL, xU)
# println("A 3D plot of the function (multicolor), lowerbound (black), and affine underestimator (red/pink) is provided.")
# @show graph2
