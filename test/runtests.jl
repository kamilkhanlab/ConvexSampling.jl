# get numbers from a second source - MATLAB?
##
include("../src/ConvexSampling.jl")

using .ConvexSampling
using LinearAlgebra
using Plots
using Test

@info "Testing ConvexSampling"
@info "Function: TODO"

tupleTest(x,y) = all(map((i,j) -> isapprox(i, j; atol=1e-8), x, y))

@testset "Example 1: A function with 1 variable:" begin
    f1(x) = @. (2)*x[1]^2 + (5)*x[1] + 3 + (2)*abs(x[1] - 2)
    xL = [-3.0]
    xU = [3.0]
    @test tupleTest(eval_sampling_underestimator_coeffs(f1, xL, xU),
        ([0.0], [3.0000], 6.8200, [0.0]))
    @test isapprox(eval_sampling_underestimator(f1, xL, xU, [2.0]),
         12.8200, atol = 1e-8)
    @test isapprox(eval_sampling_lower_bound(f1, xL, xU),
         -0.2000, atol = 1e-8)
end

@testset "Example 2: A function with 2 variables:" begin
    f2(x) = dot(x,([65 56; 56 65]),x) + dot([6; 2],x) + 23
    xL = [-5.0, -2.0]
    xU = [5.0, 3.0]
    @test tupleTest(eval_sampling_underestimator_coeffs(f2, xL, xU),
        ([0.0, 0.5], [62.0, 67.0], -162.875, Array{Float64}(undef, 2, 1)))
    @test isapprox(eval_sampling_underestimator(f2, xL, xU, [2.0, 2.0]),
        61.6250, atol = 1e-8)
    @test isapprox(eval_sampling_lower_bound(f2, xL, xU),
        -640.3750, atol = 1e-8)
end

@testset "Example 3: A function with 3 variables:" begin
    f3(x) = dot(x,[66 36 30; 36 33 36; 30 36 66],x) + dot([6; 2; 6],x) + 23
    xL = [-5.0, -2.0, -1.0]
    xU = [6.0, 3.0, 7.0]
    @test tupleTest(eval_sampling_underestimator_coeffs(f3, xL, xU),
        ([0.5, 0.5, 3.0], [288.0000, 287.0000, 468.0000], 553.8750, Array{Float64}(undef, 3, 1)))
    @test isapprox(eval_sampling_underestimator(f3, xL, xU, [1.0, 1.0, 1.0]),
        -94.6250, atol = 1e-8)
    @test isapprox(eval_sampling_lower_bound(f3, xL, xU),
        -3619.6250, atol = 1e-8)
end
