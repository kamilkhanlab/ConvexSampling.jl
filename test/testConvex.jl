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

##
println("Running Example 1: A function with 1 variable:\n")

f2(x) = @. (2)*x^2 + (5)*x + 3 + (2)*abs(x - 2)
xL = [-3.0]
xU = [3.0]

b_coeff1, c_coeff1 = aff_coefficients(f2, xL, xU)
println("Calculated b and c coefficients are ", b_coeff1[1], " and ", c_coeff1, " respectively.\n")

aff_underestimator(f2, xL, xU)
yOutput1 = aff_underestimator_atx(f2, xL, xU, [2.0])
println("At x = ", [2.0], " the affine underestimator outputs ", yOutput1, ".\n")

fL1 = lower_bound(f2, xL, xU)
println("The lower bound (fL) is = ", fL1, ".\n")

println("A 3D plot of the function, lowerbound, and affine underestimator is provided.")
graph1 = plotter(f2, xL, xU)

##
println("Running Example 2: A function with 2 variables:\n")

f3(x) = dot(x,([65 56; 56 65]),x) + dot([6; 2],x) + 23
xL = [-5.0, -3.0]
xU = [5.0, 3.0]
println("Function: f3(x) = dot(x,(",[65 56; 56 65],"),x) + dot(",[6;2],",x) + ",23," \n with bounds x = ",xL," and ",xU,".\n")

b_coeff2, c_coeff2 = aff_coefficients(f3, xL, xU)
println("Calculated b and c coefficients are ", b_coeff2, " and ", c_coeff2, " respectively.\n")

affine_function2 = aff_underestimator(f3, xL, xU)
println("Affine underestimator format is f(x) = c + dot(b, (x1,x2)-w0). \n")

yOutput2 = aff_underestimator_atx(f3, xL, xU, [2.0, 2.0])
println("At x = ", [2.0, 2.0], " the affine underestimator outputs ", yOutput2, ".\n")

fL2 = lower_bound(f3, xL,xU)
println("The lower bound (fL) is = ", fL2, ".\n")

graph2 = plotter(f3, xL, xU)
println("A 3D plot of the function (multicolor), lowerbound (red/pink), and affine underestimator (wireframe) is provided.")
println("Enter @show <graphname> in the command window to access graphs.\n")

##
println("Running Example 3: A function with 3 variables:\n")
f5(x) = dot(x,[7 4 1; 4 1 4; 1 4 7],x) + dot([6; 2; 6],x) + 23
xL = [-5.0, -2.0, -1.0]
xU = [6.0, 3.0, 7.0]

b_coeff3, c_coeff3 = aff_coefficients(f5, xL, xU)
println("Calculated b and c coefficients are ", b_coeff3, " and ", c_coeff3, " respectively.\n")

affine_function3 = aff_underestimator(f5, xL, xU)
println("Affine underestimator format is f(x) = c + dot(b, (x1,x2)-w0). \n")

yOutput3 = aff_underestimator_atx(f5, xL, xU, [1.0, 1.0, 1.0])
println("At x = ", [1.0, 1.0, 1.0], " the affine underestimator outputs ", yOutput3, ".\n")

fL3 = lower_bound(f5, xL,xU)
println("The lower bound (fL) is = ", fL3, ".\n")
