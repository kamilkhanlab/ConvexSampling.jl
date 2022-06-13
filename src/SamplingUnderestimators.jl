#=
module ConvexSample
=====================
A quick implementation of:

- Bounding convex relaxations of process models from below by
tractable black-box sampling, developed in the article:
Song et al. (2021),
https://doi.org/10.1016/j.compchemeng.2021.107413

This implementation applies the formulae in this article to calculate affine
relaxations of a convex function on a box domain.

...

Written by Maha Chaudhry on June 13, 2022
=#
module SamplingUnderestimators

using LinearAlgebra
using Plots

export eval_sampling_underestimator_coeffs,
    construct_sampling_underestimator,
    eval_sampling_underestimator,
    eval_sampling_lower_bound,
    plot_sampling_underestimator

## define affine under-estimator function operations, given a convex function f
## defined on a box domain with a Vector input

#default step size (alpha)
#small alphas generates tighter relaxations, but the smaller it gets, the larger
#the source of error introduced into calculation
const DEFAULT_ALPHA = 0.1

# compute 2n+1 sampled values required to construct affine relaxation:
#   (w0,y0) = midpoint of the box domain
#   (wi,y0) = 2n points along defined step lengths (α)
function sample_convex_function(
        f::Function, #functions must accept Vector{Float64} inputs and output scalar Float64
        xL::Vector{Float64},
        xU::Vector{Float64};
        alpha::Vector{Float64} = fill(DEFAULT_ALPHA, length(xL)) #sets default value for α
    )
    n = length(xL) #function dimension
    if length(xU) != n
        throw(DomainError("function dimension: length of xL and xU must be equal"))
    end

    w0 = 0.5*(xL + xU)
    y0 = f(w0)
    if typeof(y0) != Float64
        throw(DomainError("function dimension: function output must be scalar Float64"))
    end

    wStep = @. 0.5*alpha*(xU - xL)
    yPlus = [f(wPlus) for wPlus in eachcol(w0 .+ diagm(wStep))]
    yMinus = [f(wMinus) for wMinus in eachcol(w0 .- diagm(wStep))]

    return w0, y0, wStep, yPlus, yMinus
end #function

# compute coefficients for affine underestimator function where:
# f(x) = c + dot(b, x - w0)
#   coefficient b = centered simplex gradient of f at w0 sampled
#                   along coordinate vectors
#   coefficient c = resembles standard difference approximation of
#                   second-order partial derivatives
function eval_sampling_underestimator_coeffs(
        f::Function,
        xL::Vector{Float64},
        xU::Vector{Float64};
        alpha::Vector{Float64} = fill(DEFAULT_ALPHA, length(xL))
    )
    n = length(xL)
    w0, y0, wStep, yPlus, yMinus = sample_convex_function(f, xL, xU; alpha)

    b = zeros(n,1)
    for (i, bi) in enumerate(b)
        if (xL[i] < xU[i]) || (xL[i] == xU[i])
            b[i] = ((yPlus[i] - yMinus[i])/maximum(abs.(2.0.*diagm(wStep)[i,:])))
        end #if
    end #for

    #coefficient c can be tightened in special cases where f is univariate
    #dependent on the defined step length:
    c = y0[1]
    if n > 1
        for i in range(1,n)
            c -= 0.5*((yPlus[i]+yMinus[i]-2.0*y0)/alpha[i])
        end #for
    elseif n == 1 && alpha != [1.0]
        c = 2.0*c - 0.5*(yPlus[1]+yMinus[1])
    end #if
    return w0, b, c
end #function

# define affine underestimator function using calculated b, c coefficients:
function construct_sampling_underestimator(
        f::Function,
        xL::Vector{Float64},
        xU::Vector{Float64};
        alpha::Vector{Float64} = fill(DEFAULT_ALPHA, length(xL))
    )
    n = length(xL)
    w0, b, c = eval_sampling_underestimator_coeffs(f, xL, xU; alpha)
    return x -> c + dot(b, x - w0)
end #function

# compute affine underestimator y-value using:
#  (1) computed affine underestimator function
#  (2) x-input value
function eval_sampling_underestimator(
    f::Function,
    xL::Vector{Float64},
    xU::Vector{Float64},
    xIn::Vector{Float64}; #define x-input value
    alpha::Vector{Float64} = fill(DEFAULT_ALPHA, length(xL))
)
    affinefunc = construct_sampling_underestimator(f,xL,xU;alpha)
    return affinefunc(xIn)
end #function

# compute:
#  fL = guaranteed constant scalar lower bound of f on X
function eval_sampling_lower_bound(
        f::Function,
        xL::Vector{Float64},
        xU::Vector{Float64};
        alpha::Vector{Float64} = fill(DEFAULT_ALPHA, length(xL))
    )
    n = length(xL)
    w0, y0, wStep, yPlus, yMinus = sample_convex_function(f, xL, xU; alpha)

    #coefficient c can be tightened in special cases where f is univariate:
    if n > 1
        fL = y0
        for i in range(1,n)
            fL -= (max(yPlus[i], yMinus[i])-y0)/alpha[i]
        end #for
    elseif n == 1
        fL = (@. min(2.0*y0-yPlus, 2.0*y0-yMinus,
            (1.0/alpha)*yMinus-((1.0-alpha)/alpha)*y0,
            (1.0/alpha)*yPlus-((1.0-alpha)/alpha)*y0))[1]
    end #if
    return fL
end #function

# plot:
#  function f on plane (R^n) within box domain
#  lower bound fL on plane (R^n) within box domain
#  affine underestimator on plane within box domain
#  sampled points = (w0,y0) and (wi, yi)
function plot_sampling_underestimator(
    f::Function,
    xL::Vector{Float64},
    xU::Vector{Float64};
    alpha::Vector{Float64} = fill(DEFAULT_ALPHA, length(xL)),
    plot3DStyle::Vector = [surface!, wireframe!, surface], #Set plot style
    fEvalResolution::Int64 = 10, #Set # of function evaluations as points^n
)
    if !all(xU .> xL)
        throw(DomainError("function dimension: individual components of xU must be greater than individual components of xL"))
    end

    n = length(xL)
    #set function definition to speed up computational time:
    affine = construct_sampling_underestimator(f, xL, xU; alpha)
    #calculate scalar values:
    w0, y0, wStep, yPlus, yMinus = sample_convex_function(f, xL, xU; alpha)
    fL = eval_sampling_lower_bound(f, xL, xU; alpha)

    if n == 1
        #sampled points on univariate functions are collinear, so range of points
        #is also univariate:
        xMesh = range(xL[1], xU[1], fEvalResolution)
        yMeshF = zeros(fEvalResolution,1) #to collect function evaluations
        yMeshAffine = zeros(fEvalResolution,1) #to collect affine underestimator evaluations
        for (i, xi) in enumerate(xMesh)
            yMeshF[i] = f(xi)
            yMeshAffine[i] = affine([xi])
        end #for

        #to plot along 2 dimensions:
        plot(xMesh, yMeshF, label = "Function", xlabel = "x axis", ylabel = "y axis")
        plot!(xMesh, yMeshAffine, label = "Affine underestimator")
        plot!(xMesh, fill!(yMeshF,fL), label = "Lower bound")
        scatter!([w0; w0 + wStep; w0 - wStep], [y0; yPlus; yMinus], label = "Sampled points")

    elseif n == 2
        #for higher dimension functions, a meshgrid of points is required
        #as function and affine accuracy may differ, each require individual meshgrids
        x1range = range(xL[1], xU[1], fEvalResolution)
        x2range = range(xL[2], xU[2], fEvalResolution)
        yMeshF = zeros(length(x1range),length(x2range)) #to collect function evaluations
        yMeshAffine = zeros(length(x1range),length(x2range)) #to collect affine underestimator evaluations
        for (i, x1) in enumerate(x1range)
            for (j, x2) in enumerate(x2range)
                yMeshF[i,j] = f([x1, x2])
                yMeshAffine[i,j] = affine([x1, x2])
            end #for
        end #for

        #to plot along 3 dimensions:
        plot3DStyle[3](x1range, x2range, fill(fL, length(x1range), length(x2range)),
            label = "Lower bound", c=:PRGn_3)
        plot3DStyle[2](x1range, x2range, yMeshAffine,
            label = "Affine underestimator", c=:grays)
        colorBar = true
        if plot3DStyle[1] == wireframe!
            colorBar = false
        end #if
        plot3DStyle[1](x1range, x2range, yMeshF, colorbar=colorBar,
            title="From top to bottom: (1) Original function,
            (2) Affine underestimator, and (3) Lower bound",titlefontsize=10,
            xlabel = "x₁ axis", ylabel = "x₂ axis", zlabel = "y axis", label = "Function", c=:dense)
        wPlus = w0 .+ diagm(wStep)
        wMinus= w0 .- diagm(wStep)
        scatter!([w0[1]; wPlus[1,:]; wMinus[1,:]], [w0[2]; wPlus[2,:]; wMinus[2,:]], [y0; yPlus; yMinus],
            c=:purple, legend=false)

    else
        throw(DomainError("function dimension: must be 1 or 2"))
    end #if
end #function

end #module
