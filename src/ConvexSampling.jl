#=
module SamplingUnderestimators
=====================
A quick implementation of:

- Bounding convex relaxations of process models from below by
tractable black-box sampling, developed in the article:
Song et al. (2021),
https://doi.org/10.1016/j.compchemeng.2021.107413

This implementation applies the formulae in this article to calculate affine
relaxations of a convex function on a box domain on a box domain using
    2n+1 function evaluations. An alternate method under construction
has also been implemented, which uses n+2 evaluations.

...

Written by Maha Chaudhry on July 4, 2022
Edited by Kamil Khan
=#
module ConvexSampling

using LinearAlgebra
using Plots

export SamplingType, SAMPLE_COMPASS_STAR, SAMPLE_SIMPLEX_STAR,
    eval_sampling_underestimator_coeffs,
    construct_sampling_underestimator,
    eval_sampling_underestimator,
    eval_sampling_lower_bound,
    plot_sampling_underestimator

## define affine under-estimator function operations, given a convex function f
## defined on a box domain with a Vector input

# implemented sampling procedures
#   In comments, n=domain of provided convex function.
"""
    SamplingType(x::Int64)

Specify the sampling strategy and the number of evaluations of f. Of the type `enum`.

# Possible Inputs
- `SAMPLE_COMPASS_STAR`: (default), uses `(2n+1)` function evaluations in a
                        compass-star stencil, where `n` is domain dimension of `f`.
- `SAMPLE_SIMPLEX_STAR`: uses (n+2) evaluations; this is experimental,
                        and does not currently utilize lambda or epsilon.

"""
@enum SamplingType begin
    SAMPLE_COMPASS_STAR # sample (2n+1) points, as by Song et al. (2021)
    SAMPLE_SIMPLEX_STAR # sample (n+2) points, in an experimental procedure
end

# default dimensionless stepsize.
#   Smaller is generally better, but pushing alpha too small
#   will leave us vulnerable to numerical error in subtraction
const DEFAULT_ALPHA = 0.1

# sample convex function, to later generate affine underestimators.
#   Notation is as in Song et al. (2021).
function sample_convex_function(
    f::Function, # must be convex and of the form f(x::Vector{Float64})::Float64
    xL::Vector{Float64},
    xU::Vector{Float64};
    samplingPolicy::SamplingType = SAMPLE_COMPASS_STAR,
    alpha::Vector{Float64} = fill(DEFAULT_ALPHA, length(xL)), # dimensionless stepsize
    lambda::Vector{Float64} = zeros(length(xL)), # dimensionless offset of sampling stencil
    epsilon::Float64 = 0.0 # absolute error in evaluating f
)
    # verify consistency of inputs
    if length(xU) != length(xL)
        throw(DomainError("xL and xU", "must have the same dimension"))
    end #if
    if !(xL <= xU)
        throw(DomainError("xL and xU", "must have xL[i]<=xU[i] for each i"))
    end #if
    if !all(0.0 .< alpha .<= (1.0 .- lambda))
        throw(DomainError(:alpha, "each component must be between 0.0 and (1.0-lambda)"))
    end #if
    if !all(-1.0 .< lambda .< 1.0)
        throw(DomainError(:lambda, "each component must be between -1.0 and 1.0"))
    end #if

    # sample midpoint of stencil
    w0 = @. 0.5*(xL + xU) + 0.5*lambda*(xU - xL)
    y0 = f(w0)
    if !(y0 isa Float64)
        throw(DomainError(:f, "function output must be scalar Float64"))
    end #if

    # sample other points in stencil
    wStep = @. 0.5*alpha*(xU - xL)
    wStepDiag = Diagonal(wStep)
    yPlus = [f(wPlus) for wPlus in eachcol(w0 .+ wStepDiag)]
    if samplingPolicy == SAMPLE_COMPASS_STAR
        yMinus = [f(wMinus) for wMinus in eachcol(w0 .- wStepDiag)]
    elseif samplingPolicy == SAMPLE_SIMPLEX_STAR
        yMinus = [f(w0 - wStep)]
    else
        throw(DomainError(:samplingPolicy, "unsupported sampling method"))
    end # if
    return w0, y0, wStep, yPlus, yMinus
end #function

# compute coefficients c, b, w0 so that:
#   fAffine(x) = c + dot(b, x - w0)
#   is an affine underestimator of f in the interval [xL, xU]
#
# The output sR is used by the experimental (n+2)-sampling method.
"""
    eval_sampling_underestimator_coeffs(f::Function, xL::T, xU::T;
        samplingPolicy::SamplingType = SAMPLE_COMPASS_STAR, alpha::T,
        lambda::T, epsilon::Float64 = 0.0)

    where T is either Vector{Float64} or Float64

Evaluate the coefficients `c`, `b`, `w0` for which the affine function
` fAffine(x) = c + dot(b, x - w0) ` is guaranteed to underestimate `f` on `[xL, xU]`.

# Arguments
- `f:Function`: must be convex and of the form f(x::Vector{Float64})::Float64
- `xL::T`: coordinates for lower bound of box domain on which `f` is defined
- `xU::T`: coordinates for upper bound of box domain on which `f` is defined

# Keywords
TODO: need to find a way to include all key argument detail
- `samplingPolicy::SamplingType`: an `enum` specifying sampling strategy.
        See [`SamplingType`](@ref SamplingType) for more details.
- `lambda::T`: an offset of location of domain midpoint, `w0`.
        All components must be between `(-1.0, 1.0)`.
- `alpha::T`: dimensionless step length of each sampled point from stencil centre `w0`.
- `epsilon::Float64`: an absolute error bound for evaluations of `f`.

# Notes
Additional output `sR` is only used by experimental method `SAMPLE_SIMPLEX_STAR`.

# Examples
To construct the underestimator function as its constant coefficients for function
`f` on box domain `xL[i] <= x[i] <= xU[i]`:
```Julia
w0, b, c = eval_sampling_underestimator_coeffs(f, xL, xU)
```
in which case `fAffine(x) == c + dot(b, x - w0)` for all `x` inputs.

"""
function eval_sampling_underestimator_coeffs(
    f::Function,  # must be convex and of the form f(x::Vector{Float64})::Float64
    xL::Vector{Float64},
    xU::Vector{Float64};
    samplingPolicy::SamplingType = SAMPLE_COMPASS_STAR,
    alpha::Vector{Float64} = fill(DEFAULT_ALPHA, length(xL)), # dimensionless stepsize
    lambda::Vector{Float64} = zeros(length(xL)), # dimensionless offset of sampling stencil
    epsilon::Float64 = 0.0 # absolute error in evaluating f
)
    n = length(xL)
    w0, y0, wStep, yPlus, yMinus = sample_convex_function(f, xL, xU;
                                                          samplingPolicy, alpha, lambda, epsilon)

    if n == 1 || samplingPolicy == SAMPLE_COMPASS_STAR
        b = zeros(n)
        for (i, xLI, xUI, yPlusI, yMinusI, wStepI) in zip(eachindex(b), xL, xU, yPlus, yMinus, wStep)
            if xLI < xUI
                b[i] = (yPlusI - yMinusI)/abs(2.0*wStepI)
            end
        end #for

        #coefficient c can be tightened in special cases where f is univariate
        #dependent on the defined step length:
        c = y0[1]
        if n > 1
            c -= epsilon
            c -= sum((1.0 .+ abs.(lambda)).*(yPlus .+ yMinus
                                             .- 2.0*y0 .+ 4.0*epsilon)./(2.0*alpha); init=0.0)
        elseif n == 1 && alpha != [1.0]
            c = 2.0*c - 0.5*(yPlus[1] + yMinus[1])
        end #if
        sR = similar(b)

        #alternate calculation for b and c vectors assuming n+2 sampled points:
    elseif samplingPolicy == SAMPLE_SIMPLEX_STAR
        sU = @. 2.0*(yPlus - y0)/abs(2.0*wStep)
        sL = zeros(n)
        for (i, wStepI) in zip(eachindex(sL), wStep)
            yjSum = sum(y0 - yPlusJ for (j, yPlusJ) in enumerate(yPlus) if j != i; init=0.0)
            sL[i] = 2.0*(y0 - yMinus[1] + yjSum)/abs(2.0*wStepI)
        end #for
        b = 0.5.*(sL + sU)

        #coefficient c calculated as affineFunc(w0):
        sR = 0.5.*(sU - sL)
        c = y0 - 0.5.*dot(sR, xU - xL)

    else
        throw(DomainError(:samplingPolicy, "unsupported sampling method"))
    end #if

    return w0, b, c, sR
end #function

# compute coefficients using scalar inputs for univariate functions:
function eval_sampling_underestimator_coeffs(
    f::Function, # must be convex and of the form f(x::Float64)::Float64
    xL::Float64,
    xU::Float64;
    samplingPolicy::SamplingType = SAMPLE_COMPASS_STAR,
    alpha::Float64 = DEFAULT_ALPHA,
    lambda::Float64 = 0.0,
    epsilon::Float64 = 0.0
)
    fMultiVar(x) = f(x[1])
    w0Vec, bVec, c, sR =
        eval_sampling_underestimator_coeffs(fMultiVar, [xL], [xU];
                                            samplingPolicy, alpha = [alpha], lambda = [lambda], epsilon)
    return w0Vec[1], b[1], c, sR[1]
end #function

# define affine underestimator function using calculated b, c coefficients:
"""
    construct_sampling_underestimator(f::Function, xL::T, xU::T;
        samplingPolicy::SamplingType = SAMPLE_COMPASS_STAR, alpha::T,
        lambda::T, epsilon::Float64 = 0.0)

    where T is either Vector{Float64} or Float64

Return affine underestimator function of the format `fAffine(x) = c + dot(b, x - w0)`
by sampling function `f` at `2n+1` domain points where `n` is the function dimension.

See [`eval_sampling_underestimator_coeffs`](@ref eval_sampling_underestimator_coeffs) for more details on function inputs.

# Example
To construct the underestimator function for the function `f` on box domain
`xL[i] <= x[i] <= xU[i]` for all `x` inputs:

```jldoctest
julia> 2+2
5
```

fAffine(x) = construct_sampling_underestimator(f, xL, xU)
"""
function construct_sampling_underestimator(
    f::Function,  # must be convex and of the form f(x::Vector{Float64})::Float64
    xL::Vector{Float64},
    xU::Vector{Float64};
    samplingPolicy::SamplingType = SAMPLE_COMPASS_STAR,
    alpha::Vector{Float64} = fill(DEFAULT_ALPHA, length(xL)), # dimensionless stepsize
    lambda::Vector{Float64} = zeros(length(xL)), # dimensionless offset of sampling stencil
    epsilon::Float64 = 0.0 # absolute error in evaluating f
)
    w0, b, c, _ = eval_sampling_underestimator_coeffs(f, xL, xU;
                                                      samplingPolicy, alpha, lambda, epsilon)
    return x -> c + dot(b, x - w0)
end #function

# define affine underestimator function using scalar inputs for univariate functions:
function construct_sampling_underestimator(
    f::Function, # must be convex and of the form f(x::Float64)::Float64
    xL::Float64,
    xU::Float64;
    samplingPolicy::SamplingType = SAMPLE_COMPASS_STAR,
    alpha::Float64 = DEFAULT_ALPHA,
    lambda::Float64 = 0.0,
    epsilon::Float64 = 0.0
)
    fMultiVar(x) = f(x[1])
    return construct_sampling_underestimator(fMultiVar, [xL], [xU];
                                      samplingPolicy, alpha = [alpha], lambda = [lambda], epsilon)
end #function

# compute affine underestimator y-value using:
#  (1) computed affine underestimator function
#  (2) x-input value
"""
    eval_sampling_underestimator(f::Function, xL::T, xU::T, xIn::T;
    samplingPolicy::SamplingType = SAMPLE_COMPASS_STAR, alpha::T,
    lambda::T, epsilon::Float64 = 0.0)

    where T is either Vector{Float64} or Float64

Evaluate underestimator `fAffine` constructed by `construct_sampling_underestimator`
at a domain point `xIn`. That is, `yOut = fAffine(xIn)`.

See [`eval_sampling_underestimator_coeffs`](@ref eval_sampling_underestimator_coeffs) for more details on function inputs.

"""
function eval_sampling_underestimator(
    f::Function,
    xL::Vector{Float64},
    xU::Vector{Float64},
    xIn::Vector{Float64}; #define x-input value
    kwargs...
)
    fAffine = construct_sampling_underestimator(f, xL, xU; kwargs...)
    return fAffine(xIn)
end #function

# affine underestimator y-value using scalar inputs for univariate functions:
function eval_sampling_underestimator(
    f::Function,
    xL::Float64,
    xU::Float64,
    xIn::Float64;
    samplingPolicy::SamplingType = SAMPLE_COMPASS_STAR,
    alpha::Float64 = DEFAULT_ALPHA,
    lambda::Float64 = 0.0,
    epsilon::Float64 = 0.0
)
    fMultiVar(x) = f(x[1])
    return eval_sampling_underestimator(fMultiVar, [xL], [xU], [xIn];
                                 samplingPolicy, alpha = [alpha], lambda = [lambda], epsilon)
end #function

# compute:
#  fL = scalar lower bound of f on the interval [xL, xU]
"""
    eval_sampling_lower_bound(f::Function, xL::T, xU::T;
    samplingPolicy::SamplingType = SAMPLE_COMPASS_STAR, alpha::T,
    lambda::T, epsilon::Float64 = 0.0)

    where T is either Vector{Float64} or Float64

Compute the scalar lower bound of `f` on the interval `[xL, xU]`,
so that `f(x) >= fL` for each `x` in the box.

See [`eval_sampling_underestimator_coeffs`](@ref eval_sampling_underestimator_coeffs) for more details on function inputs.

# Example
```Julia
fL = eval_sampling_lower_bound(f, xL, xU)
```

"""
function eval_sampling_lower_bound(
    f::Function,
    xL::Vector{Float64},
    xU::Vector{Float64};
    samplingPolicy::SamplingType = SAMPLE_COMPASS_STAR,
    alpha::Vector{Float64} = fill(DEFAULT_ALPHA, length(xL)),
    lambda::Vector{Float64} = zeros(length(xL)),
    epsilon::Float64 = 0.0
)
    n = length(xL)

    #coefficient c can be tightened in special cases where f is univariate:
    if n == 1 && epsilon == 0.0 && lambda == zeros(n)
        w0, y0, wStep, yPlus, yMinus = sample_convex_function(f, xL, xU;
                                                              samplingPolicy, alpha, lambda, epsilon)
        fL = (@. min(2.0*y0-yPlus,
                     2.0*y0-yMinus,
                     (1.0/alpha)*yMinus-((1.0-alpha)/alpha)*y0,
                     (1.0/alpha)*yPlus-((1.0-alpha)/alpha)*y0))[1]
    elseif samplingPolicy == SAMPLE_SIMPLEX_STAR
        w0, b, c, sR = eval_sampling_underestimator_coeffs(f, xL, xU;
                                                           samplingPolicy, alpha, lambda, epsilon)
        fL = y0 - 0.5*dot(abs.(b), abs.(xU - xL)) - 0.5*dot(sR, xU - xL)
    elseif samplingPolicy == SAMPLE_COMPASS_STAR
        w0, y0, wStep, yPlus, yMinus = sample_convex_function(f, xL, xU;
                                                              samplingPolicy, alpha, lambda, epsilon)
        fL = y0 - epsilon
        fL -= sum((1.0 .+ abs.(lambda)).*(max.(yPlus, yMinus) .- y0 .+ 2.0*epsilon)./alpha; init=0.0)
    else
        throw(DomainError(:samplingPolicy, "unsupported sampling method"))
    end #if
    return fL
end #function

# compute lower bound using scalar inputs for univariate functions:
function eval_sampling_lower_bound(
    f::Function,
    xL::Float64,
    xU::Float64;
    samplingPolicy::SamplingType = SAMPLE_COMPASS_STAR,
    alpha::Float64 = DEFAULT_ALPHA,
    lambda::Float64 = 0.0,
    epsilon::Float64 = 0.0
)
    fMultiVar(x) = f(x[1])
    return eval_sampling_lower_bound(fMultiVar, [xL], [xU];
                              samplingPolicy, alpha = [alpha], lambda = [lambda], epsilon)
end #function

# plot:
#  function f on plane (R^n) within box domain
#  lower bound fL on plane (R^n) within box domain
#  affine underestimator on plane within box domain
#  sampled points = (w0, y0) and (wi, yi)
"""
    plot_sampling_underestimator(f::Function, xL::T, xU::T;
    samplingPolicy::SamplingType = SAMPLE_COMPASS_STAR, alpha::T,
    lambda::T, epsilon::Float64 = 0.0,
    plot3DStyle::Vector = [surface!, wireframe!, surface],
    fEvalResolution::Int64 = 10)

    where T is either Vector{Float64} or Float64

Plot (1) function, `f`, (2) affine underestimator, `fAffine`, and (3) lower bound `fL`
on the box domain `[xL, xU]`.

# Additional Keywords
- `plot3DStyle::Vector`: sets the plot style (ex. wireframe, surface, etc.)
        of each individual plot component in the set order:
        (1) lower bound, (2) affine under-estimator, (3) convex function.
- `fEvalResolution::Int64`: number of mesh rows per domain dimension in the resulting plot.

# Notes
`f` must be a function of either 1 or 2 variables and must take a `Vector{Float64}` input.
The produced graph may be stored to a variable and later retrieved with @show.

See [`eval_sampling_underestimator_coeffs`](@ref eval_sampling_underestimator_coeffs) for more details on function inputs.

"""
function plot_sampling_underestimator(
    f::Function,
    xL::Vector{Float64},
    xU::Vector{Float64};
    samplingPolicy::SamplingType = SAMPLE_COMPASS_STAR,
    alpha::Vector{Float64} = fill(DEFAULT_ALPHA, length(xL)),
    lambda::Vector{Float64} = zeros(length(xL)),
    epsilon::Float64 = 0.0,
    plot3DStyle::Vector = [surface!, wireframe!, surface], #Set plot style
    fEvalResolution::Int64 = 10 #Set # of function evaluations as points^n
)
    if !(xU > xL)
        throw(DomainError("xL and xU", "for plotting, we must have xU[i] > xL[i] for each i"))
    end

    n = length(xL)
    #set function definition to speed up computational time:
    affineFunc = construct_sampling_underestimator(f, xL, xU;
                                                   samplingPolicy, alpha, lambda, epsilon)
    #calculate scalar values:
    w0, y0, wStep, yPlus, yMinus = sample_convex_function(f, xL, xU;
                                                          samplingPolicy, alpha, lambda, epsilon)
    fL = eval_sampling_lower_bound(f, xL, xU;
                                   samplingPolicy, alpha, lambda, epsilon)

    if n == 1
        #sampled points on univariate functions are collinear, so range of points
        #is also univariate:
        xMesh = range(xL[1], xU[1], fEvalResolution)
        yMeshF = zeros(fEvalResolution,1) #to collect function evaluations
        yMeshAffine = zeros(fEvalResolution,1) #to collect affine underestimator evaluations
        for (i, xI) in enumerate(xMesh)
            yMeshF[i] = f(xI)
            yMeshAffine[i] = affineFunc([xI])
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
                yMeshAffine[i,j] = affineFunc([x1, x2])
            end #for
        end #for

        #to plot along 3 dimensions:
        plot3DStyle[3](x1range, x2range,
                       fill(fL, length(x1range), length(x2range)),
                       label = "Lower bound", c=:PRGn_3)
        plot3DStyle[2](x1range, x2range, yMeshAffine,
                       label = "Affine underestimator", c=:grays)
        colorBar = true
        if plot3DStyle[1] == wireframe!
            colorBar = false
        end #if
        plot3DStyle[1](x1range, x2range, yMeshF, colorbar=colorBar,
                       title="From top to bottom: (1) Original function,
                (2) Affine underestimator, and (3) Lower bound",
                       titlefontsize=10, xlabel = "x₁", ylabel = "x₂",
                       zlabel = "y", label = "Function", c=:dense)
        wPlus = w0 .+ diagm(wStep)
        if samplingPolicy == SAMPLE_COMPASS_STAR
            wMinus= w0 .- diagm(wStep)
        elseif samplingPolicy == SAMPLE_SIMPLEX_STAR
            wMinus = w0 - wStep
        end #if
        scatter!([w0[1]; wPlus[1,:]; wMinus[1,:]],
                 [w0[2]; wPlus[2,:]; wMinus[2,:]],
                 [y0; yPlus; yMinus],
                 c=:purple, legend=false)
    else
        throw(DomainError(:f, "domain dimension must be 1 or 2"))
    end #if
end #function

# plot using scalar inputs for univariate functions:
function plot_sampling_underestimator(
    f::Function,
    xL::Float64,
    xU::Float64;
    samplingPolicy::SamplingType = SAMPLE_COMPASS_STAR,
    alpha::Float64 = DEFAULT_ALPHA,
    lambda::Float64 = 0.0,
    epsilon::Float64 = 0.0,
    plot3DStyle::Vector = [surface!, wireframe!, surface],
    fEvalResolution::Int64 = 10,
)
    fMultiVar(x) = f(x[1])
    plot_sampling_underestimator(fMultiVar, [xL], [xU];
                                 samplingPolicy, alpha = [alpha], lambda = [lambda], epsilon,
                                 plot3DStyle, fEvalResolution)
end #function

end #module
