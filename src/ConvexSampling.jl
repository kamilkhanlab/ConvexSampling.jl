#=
module SamplingUnderestimators
=====================
Implements the sampling-based underestimators of convex functions proposed in:

- Bounding convex relaxations of process models from below by
tractable black-box sampling, developed in the article:
Song et al. (2021),
https://doi.org/10.1016/j.compchemeng.2021.107413
- Felix Bonhoff's master thesis, RWTH Aachen (2023)

Notation is as in the article by Song et al.
=#
module ConvexSampling

using LinearAlgebra
using Plots

export SampledData

export sample_convex_function,
    evaluate_underestimator_coeffs,
    construct_underestimator,
    evaluate_underestimator,
    evaluate_lower_bound,
    plot_sampling_underestimator

## define affine under-estimator function operations, given a convex function f
## defined on a box domain with a Vector input

# default dimensionless stepsize.
#   Smaller is generally better, but pushing alpha too small
#   will leave us vulnerable to numerical error in subtraction
const DEFAULT_ALPHA = 0.1

"""
A `SampledData{T}` object holds sampled values of a supplied convex function, along with associated metadata. The parameter `T` is the function's input type, and may take the following values:

- `T = Float64` is for univariate functions only (with domain dimension 1), and
- `T = Vector{Float64}` is for multivariate functions only (with domain dimension >= 2).

The value of `T` affects the formulas used by other methods in this package, since univariate convex functions have access to tighter sampling-based underestimators.

# Fields of SampledData{T}

- `f:Function`: must be convex and with the signature `f(x::T) -> Float64`; this implementation treats `f` as a black box and cannot verify convexity
- `xL::T`: coordinates for lower bound of box domain on which `f` is defined
- `xU::T`: coordinates for upper bound of box domain on which `f` is defined
- `stencilShape::Symbol`: specifies shape of stencil on which `f` is sampled.
- `alpha::T`: dimensionless step length of each sampled point from stencil centre `w0`.
    Each component `alpha[i]` must satisfy `0.0 < alpha[i] <= 1.0 - lambda[i]``,
    and is set to `0.1` by default. If the step length is too small, then subtraction
    operations in finite difference formulas might cause unacceptable numerical error.
- `lambda::T`: scaled offset of the center of the sampling stencil, relative to the center of the box domain of `f`. All components of `lambda` must be between
    `(-1.0, 1.0)`, and are `0.0` by default.
- `epsilon::Float64`: an absolute error bound for evaluations of `f`. We presume that
    each numerical evaluation of `f(x)` is within `epsilon` of the true value.
    Set to `0.0` by default.
- `w0::T`: center of sampling stencil in the domain of `f`
- `wStep::T`: contains a perturbation distance in each coordinate direction, to generate the remaining points in the sampling stencil from `w0`
- `y0::Float64 = f(w0)`
- `yPlus` and `yMinus`: contains values of `f` at points in sampling stencil other than `w0`.
"""
struct SampledData{T}
    xL::T
    xU::T
    stencilShape::Symbol
    alpha::T
    lambda::T
    epsilon::T
    w0::T
    wStep::T
    y0::Float64
    yPlus::Vector{Float64}
    yMinus::Vector{Float64}
end

# sample convex function, to later generate affine underestimators.
#   Notation is as in Song et al. (2021).
function sample_convex_function(
    f::Function,
    xL::Float64,
    xU::Float64;
    alpha::Float64 = DEFAULT_ALPHA,
    lambda::Float64 = 0.0,
    epsilon::Float64 = 0.0
)
    # verify consistency of inputs
    (xU > xL) ||
        throw(DomainError(:xU, "must satisfy xL < xU"))
    
    (-1.0 < lambda < 1.0) ||
        throw(DomainError(:lambda, "must satisfy -1.0 < lambda < 1.0"))
    
    (0.0 < alpha <= (1.0 - lambda)) ||
        throw(DomainError(:alpha, "must satisfy 0.0 < alpha <= (1.0 - lambda)"))

    # sample midpoint of stencil
    w0 = 0.5*(xL + xU + lambda*(xU - xL))
    y0 = f(w0)
    
    (y0 isa Float64) ||
        throw(DomainError(:f, "function's output must be Float64"))

    # sample other points in stencil
    wStep = 0.5*alpha*(xU - xL)
    yPlus = [f(w0 + wStep)]
    yMinus = [f(w0 - wStep)]

    # pack samples into a SampledData object
    return SampledData{Float64}(
        xL, xU, :compass, alpha, lambda, epsilon, w0, wStep, y0, yPlus, yMinus
    )
end

function sample_convex_function(
    f::Function,
    xL::Vector{Float64},
    xU::Vector{Float64};
    stencilShape::Symbol = :compass,
    alpha::Vector{Float64} = fill(DEFAULT_ALPHA, length(xL)),
    lambda::Vector{Float64} = zeros(length(xL)),
    epsilon::Float64 = 0.0
)
    # verify consistency of inputs
    (length(xL) >= 2) ||
        throw(DomainError(:xL, "a provided function of one variable must have the signature f(x::Float64) -> Float64."))
    
    (length(xL) == length(xU) == length(alpha) == length(lambda)) ||
        throw(DomainError("xL, xU, alpha, lambda", "must all have the same number of components"))
    
    all(xL .<= xU) ||
        throw(DomainError(:xU, "must satisfy xL[i] <= xU[i] for each i"))

    all(-1.0 .< lambda .< 1.0) ||
        throw(DomainError(:lambda, "must satisfy -1.0 < lambda[i] < 1.0 for each i"))

    all(0.0 .< alpha .<= (1.0 .- lambda)) ||
        throw(DomainError(:xU, "must satisfy 0.0 < alpha[i] <= (1.0 - lambda[i]) for each i"))

    # sample midpoint of stencil
    w0 = @. 0.5*(xL + xU) + 0.5*lambda*(xU - xL)
    y0 = f(w0)
    
    (y0 isa Float64) ||
        throw(DomainError(:f, "provided function must produce Float64 outputs"))

    # sample other points in stencil
    wStep = @. 0.5*alpha*(xU - xL)
    yPlus = fill(y0, length(w0))
    if stencilShape == :compass
        yMinus = copy(yPlus)
    elseif stencilShape == :simplex
        yMinus = [f(w0 - wStep)]
    else
        throw(DomainError(:stencilShape, "unsupported stencil shape"))
    end
    wNew = copy(w0)
    for i in eachindex(yPlus)
        if wStep[i] != 0.0
            wNew[i] = w0[i] + wStep[i]
            yPlus[i] = f(wNew)
            if stencilShape == :compass
                wNew[i] = w0[i] - wStep[i]
                yMinus[i] = f(wNew)
            end
            wNew[i] = w0[i]
        end
    end

    # pack samples into a SampledData object
    return SampledData{Vector{Float64}}(
        xL, xU, stencilShape, alpha, lambda, epsilon, w0, wStep, y0, yPlus, yMinus
    )
end

"""
    (w0, b, c) = evaluate_underestimator_coeffs(data::SampledData)

Evaluates coefficients of an affine underestimator of a sampled convex function `f` on a specified interval domain `[xL, xU]`, based only on the sampled data generated by [sample_convex_function](@ref). 

Using the evaluated coefficients, if we define `fAffine(x) = c + dot(b, x - w0)`, then
`fAffine(x) <= f(x)` whenever `all(xL .<= x .<= xU)`. `b` and `w0` have the same type as `x`.

# Example (outdated)

To construct the underestimator function for the function `f` on box domain
`xL[i] <= x[i] <= xU[i]` for all `x` inputs:

```Julia
A = [25.0 24.0; 24.0 25.0]
b = [2.0; 3.0]
c = 15.0
f(x) = dot(x, A, x) + dot(b, x) + c
xL = [-1.0, -2.0]
xU = [3.0, 5.0]
evaluate_underestimator_coeffs(f, xL, xU)

# output

(w0, b, c) = ([1.0, 1.5], [123.99999999999991, 126.00000000000006], 134.125)
```
"""
function evaluate_underestimator_coeffs(data::SampledData{Float64})
    # unpack
    (xL, xU, alpha, lambda, epsilon, w0, wStep, y0, yPlus, yMinus) =
        (data.xL, data.xU, data.alpha, data.lambda, data.epsilon,
         data.w0, data.wStep, data.y0, data.yPlus, data.yMinus)

    # relabel for convenience
    yPos = yPlus[1]
    yNeg = yMinus[1]

    # evaluate coefficients; we already know w0
    b = (yPos - yNeg)/(2.0*wStep)
    c = 2.0*y0 - 0.5*(yPos + yNeg) - epsilon*(3.0 + (1.0 + abs(lambda))/alpha)
    # TODO: perhaps this c can be increased

    return (w0, b, c)
end

function evaluate_underestimator_coeffs(data::SampledData{Vector{Float64}})
    # unpack
    (xL, xU, stencilShape, alpha, lambda, epsilon, w0, wStep, y0, yPlus, yMinus) =
        (data.xL, data.xU, data.stencilShape, data.alpha, data.lambda, data.epsilon,
         data.w0, data.wStep, data.y0, data.yPlus, data.yMinus)
    
    # evaluate coefficients; we already know w0. Formulas depend on stencil choice.
    n = length(xL)
    
    if stencilShape == :compass
        b = zeros(n)
        c = y0 - epsilon
        for i in eachindex(b)
            if wStep[i] > 0.0
                b[i] = (yPlus[i] - yMinus[i])/abs(2.0*wStep[i])
                c -= (1.0 + abs(lambda[i]))*(yPlus[i] + yMinus[i] - 2.0*y0 + 4.0*epsilon)
            end
        end

    elseif stencilShape == :simplex
        yNeg = yMinus[1]
        ySum = sum(@. y0 - yPlus; init=0.0)
        
        sU = zeros(n)
        sL = copy(sU)
        for i in eachindex(sU)
            if wStep[i] > 0.0
                sU[i] = (yPlus[i] - y0)/wStep[i]
                sL[i] = (yPlus[i] - yNeg + ySum)/wStep[i]
            end
        end
        
        b = 0.5*(sU + sL)

        sR = 0.5*(sU - sL)
        c = y0 - epsilon - 0.5*sum(@. (1.0 + abs(lambda))*sR*(xU - xL); init=0.0)
    else
        throw(DomainError(:stencilShape, "unsupported stencil shape"))
    end
    
    return (w0, b, c)
end

"""
    construct_underestimator(data::SampledData) -> Function

Same as [`evaluate_underestimator_coeffs`](@ref), but instead returns the underestimator as the function:
```julia
fAffine(x) = c + dot(b, x - w0)
```
With `f` denoting the originally sampled function, it then holds that `fAffine(x) <= f(x)` 
whenever `all(xL .<= x .<= xU)`.

# Example (outdated)
To construct the underestimator function for the function `f` on box domain
`xL[i] <= x[i] <= xU[i]` for all `x` inputs:

```Julia
A = [25.0 24.0; 24.0 25.0]
b = [2.0; 3.0]
c = 15.0
f(x) = dot(x, A, x) + dot(b, x) + c
xL = [-1.0, -2.0]
xU = [3.0, 5.0]
fAffine(x) = construct_underestimator(f, xL, xU)
```
"""
function construct_underestimator(data::SampledData{T}) where T
    (w0, b, c) = evaluate_underestimator_coeffs(data)
    return x::T -> c + dot(b, x - w0)
end

"""
    evaluate_underestimator(data::SampledData, xIn) -> Float64

Same as [`construct_underestimator`](@ref), but returns the value `fAffine(xIn)` instead of the underestimating function `fAffine`. Useful when `fAffine` will be evaluated only at one domain point.

# Example (outdated)

```Julia
A = [25.0 24.0; 24.0 25.0]
b = [2.0; 3.0]
c = 15.0
f(x) = dot(x, A, x) + dot(b, x) + c
xL = [-1.0, -2.0]
xU = [3.0, 5.0]
evaluate_underestimator(f, xL, xU, [2.0, 2.0])

# output

321.12499999999994
```
"""
function evaluate_underestimator(data::SampledData{T}, xIn::T) where T
    (w0, b, c) = evaluate_underestimator_coeffs(data)
    return c + dot(b, xIn - w0)
end

"""
    fL = evaluate_lower_bound(data::SampledData)

Compute a lower bound on a convex function `f` sampled by [sample_convex_function](@ref),
so that `f(x) >= fL` whenenver `all(xL .<= x .<= xU)`.

# Example (outdated)

```Julia
A = [25.0 24.0; 24.0 25.0]
b = [2.0; 3.0]
c = 15.0
f(x) = dot(x, A, x) + dot(b, x) + c
xL = [-1.0, -2.0]
xU = [3.0, 5.0]
eval_sampling_lower_bound(f, xL, xU)

# output

-554.875
```
"""
function evaluate_lower_bound(data::SampledData{Float64})
    # TODO: account for non-default choices of lambda or epsilon
    
    # unpack
    (xL, xU, alpha, lambda, epsilon, w0, wStep, y0, yPlus, yMinus) =
        (data.xL, data.xU, data.alpha, data.lambda, data.epsilon,
         data.w0, data.wStep, data.y0, data.yPlus, data.yMinus)
    
    # relabel for convenience
    yPos = yPlus[1]
    yNeg = yMinus[1]

    # evaluate lower bound candidates, then choose the lowest of these
    fLA = 2.0*y0 - yPos
    fLB = 2.0*y0 - yNeg
    fLC = y0 + (yNeg - y0)*(w0 - xL)/wStep
    fLD = y0 + (yPos - y0)*(xU - w0)/wStep

    return min(fLA, fLB, fLC, fLD)
end

function evaluate_lower_bound(data::SampledData{Vector{Float64}})
    # unpack
    (xL, xU, stencilShape, alpha, lambda, epsilon, w0, wStep, y0, yPlus, yMinus) =
        (data.xL, data.xU, data.stencilShape, data.alpha, data.lambda, data.epsilon,
         data.w0, data.wStep, data.y0, data.yPlus, data.yMinus)
    
    # initialize at center of sampling stencil
    fL = y0 - epsilon

    # incorporate info from remaining samples in stencil
    if stencilShape == :compass
        for (i, alphaI) in enumerate(alpha)
            if alphaI > 0.0
                fL -= (1.0 + lambda[i])*(max(yPlus[i], yMinus[i]) - y0 + 2.0*epsilon)/alphaI
            end
        end
        
    elseif stencilShape == :simplex
        yNeg = yMinus[1]
        ySum = sum(@. y0 - yPlus; init=0.0)
        n = length(xL)

        for (i, alphaI) in enumerate(alpha)
            if alphaI > 0.0
                fL -= (1.0 + lambda[i])*
                    (max(yPlus[i], yNeg + y0 - yPlus[i] - ySum) - y0 + 2.0*epsilon*n)/alphaI
            end
        end

    else
        throw(DomainError(:stencilShape, "unsupported stencil shape"))
    end
    
    return fL
end

# TODO: update these plotting functions
"""
    plot_sampling_underestimator(args...; kwargs...)

Plot (1) function, `f`, (2) affine underestimator, `fAffine`, and (3) lower bound `fL`
on the box domain `[xL, xU]`. 

See [`eval_sampling_underestimator_coeffs`](@ref eval_sampling_underestimator_coeffs) for more details on function inputs.

# Additional Keywords
- `plot3DStyle::Vector`: sets the plot style (ex. wireframe, surface, etc.)
    of each individual plot component in the set order:
    (1) lower bound, (2) affine under-estimator, (3) convex function.
    Default value: [surface!, wireframe!, surface]
- `fEvalResolution::Int64`: number of mesh rows per domain dimension in the resulting plot.
    Default value: `10`

# Notes
- `f` must be a function of either 1 or 2 variables and must take a `Vector{Float64}` input.
- The produced graph may be stored to a variable and later retrieved with @show.

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
    if any(xL .>= xU
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
