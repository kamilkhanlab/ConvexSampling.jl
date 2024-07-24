"""
Module for implementing the sampling-based underestimators of convex functions proposed in:

- Song et al., Bounding convex relaxations of process models from below by
tractable black-box sampling, (2021), doi:10.1016/j.compchemeng.2021.107413
- Felix Bonhoff's master thesis, RWTH Aachen (2023)

These underestimators are tractable to construct and evaluate, assuming that evaluation of the original convex function has a computational cost of `O(1)`. Notation is as in the article by Song et al.
"""
module ConvexSampling

using LinearAlgebra
using Plots

export SampledData

export sample_convex_function,
    evaluate_underestimator_coeffs,
    construct_underestimator,
    evaluate_underestimator,
    evaluate_lower_bound,
    plot_underestimator

"""
Default dimensionless stepsize, for input to [`sample_convex_function`](@ref).
"""
const DEFAULT_ALPHA = 0.1

"""
A `SampledData{T}` object holds sampled values of a supplied convex function, along with associated metadata. Constructed with [`sample_convex_function`](@ref). The parameter `T` is the convex function's input type, and may take the following values:

- `T = Float64` is for univariate functions only (with domain dimension 1), and
- `T = Vector{Float64}` is for multivariate functions only (with domain dimension >= 2).

The value of `T` affects the formulas used by other methods in this package, since univariate convex functions have access to tighter sampling-based underestimators.

# Fields of `SampledData{T}`

- `xL::T`: as provided to [`sampled_convex_function`](@ref)
- `xU::T`: as provided to [`sampled_convex_function`](@ref)
- `stencil::Symbol`: as provided to [`sampled_convex_function`](@ref)
- `alpha::T`: as provided to [`sampled_convex_function`](@ref)
- `lambda::T`: as provided to [`sampled_convex_function`](@ref)
- `epsilon::Float64`: as provided to [`sampled_convex_function`](@ref)
- `iSet`: a collection that holds indices `i` for which `xU[i] > xL[i]`
- `w0::T`: center of the sampling stencil, in the domain of `f`
- `wStep::T`: contains a perturbation distance in each coordinate direction, to generate the remaining points in the sampling stencil from `w0`
- `y0::Float64 = f(w0)`
- `yPlus` and `yMinus`: contain values of `f` at points in sampling stencil other than `w0`.
"""
struct SampledData{T}
    xL::T
    xU::T
    stencil::Symbol
    alpha::T
    lambda::T
    epsilon::Float64
    iSet
    w0::T
    wStep::T
    y0::Float64
    yPlus::Vector{Float64}
    yMinus::Vector{Float64}
end

# typedefs for convenience
const UnivariateData = SampledData{Float64}
const MultivariateData = SampledData{Vector{Float64}}

"""
    data::SampledData = sample_convex_function(f, xL, xU; 
                            stencil, alpha, lambda, epsilon)

Given a convex function `f` of `n` variables in a box domain `[xL, xU]`, 
sample `f` `O(n)` times and store these samples as a [`SampledData`](@ref) object. 
This information can then be used e.g. by the following tractable methods:

- [`evaluate_underestimator_coeffs`](@ref): to compute coefficients for an affine underestimator of `f`, 
- [`construct_underestimator`](@ref): to construct this affine underestimator as a `Function`,
- [`evaluate_lower_bound`](@ref): to compute a constant lower bound of `f`.

# Inputs

In the following input descriptions, you must use `T = Float64` if `n == 1`, and `T = Vector{Float64}` if `n >= 2`. Notation is generally as by Song et al. (doi:10.1016/j.compchemeng.2021.107413).

- `f::Function`: The convex function to be sampled and underestimated, with a signature of `f(x::T) -> Float64`. This implementation cannot verify convexity; if `f` is actually nonconvex, then the corresponding calculation results will be meaningless.
- `xL::T` and `xU::T`: opposite corners of the box domain of `f`. A point `x::T` is considered to be in this domain if 
`all(xL .<= x .<= xU)`.

## Optional keyword arguments

- `stencil`: The shape of the stencil on which `f`'s domain is sampled; used only when `n>=2`. Permitted values:
    - `:compass` (default): samples `(2n+1)` points in a compass-star arrangement, as by Song et al. (doi:10.1016/j.compchemeng.2021.107413)
    - `:simplex`: samples `(n+2)` points in a simplex-star arrangement, as in Bonhoff's master's thesis (RWTH Aachen, 2023). Yields cheaper but weaker relaxations than `:compass`.
- `alpha::T`: dimensionless step length of each sampled point from stencil centre `w0`.
    Each component `alpha[i]` must satisfy `0.0 < alpha[i] <= 1.0 - lambda[i]`,
    and is set to `0.1` by default. If the step length is too small, then subtraction
    operations in finite difference formulas might cause unacceptable numerical error.
- `lambda::T`: scaled offset of the center of the sampling stencil, relative to the center of the box domain of `f`. All components of `lambda` must be between
    `(-1.0, 1.0)`, and are `0.0` by default.
- `epsilon::Float64`: an absolute error bound for evaluations of `f`. We presume that
    each numerical evaluation of `f(x)` is within `epsilon` of the true value.
    Set to `0.0` by default.
"""
function sample_convex_function(
    f::Function,
    xL::Float64,
    xU::Float64;
    stencil::Symbol = :compass,
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
    return UnivariateData(
        xL, xU, stencil, alpha, lambda, epsilon, [1], w0, wStep, y0, yPlus, yMinus
    )
end

function sample_convex_function(
    f::Function,
    xL::Vector{Float64},
    xU::Vector{Float64};
    stencil::Symbol = :compass,
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

    # set of indices i for which xU[i] > xL[i]
    iSet = Iterators.filter(i -> (xU[i] > xL[i]), eachindex(xL))
    
    # sample midpoint of stencil
    w0 = @. 0.5*(xL + xU) + 0.5*lambda*(xU - xL)
    y0 = f(w0)
    
    (y0 isa Float64) ||
        throw(DomainError(:f, "provided function must produce Float64 outputs"))

    # sample other points in stencil
    wStep = @. 0.5*alpha*(xU - xL)
    yPlus = fill(y0, length(w0))
    if stencil == :compass
        yMinus = copy(yPlus)
    elseif stencil == :simplex
        yMinus = [f(w0 - wStep)]
    else
        throw(DomainError(:stencil, "unsupported stencil shape"))
    end
    wNew = copy(w0)
    for i in iSet
        wNew[i] = w0[i] + wStep[i]
        yPlus[i] = f(wNew)
        if stencil == :compass
            wNew[i] = w0[i] - wStep[i]
            yMinus[i] = f(wNew)
        end
        wNew[i] = w0[i]
    end

    # pack samples into a SampledData object
    return MultivariateData(
        xL, xU, stencil, alpha, lambda, epsilon, iSet, w0, wStep, y0, yPlus, yMinus
    )
end

"""
    (w0, b, c) = evaluate_underestimator_coeffs(data::SampledData)

Evaluates coefficients of an affine underestimator of a sampled convex function `f` on a specified interval domain `[xL, xU]`, based only on the sampled data generated by [`sample_convex_function`](@ref). 

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
function evaluate_underestimator_coeffs(data::UnivariateData)
    # unpack
    (alpha, lambda, epsilon, w0, wStep, y0, yPlus, yMinus) =
        (data.alpha, data.lambda, data.epsilon,
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

function evaluate_underestimator_coeffs(data::MultivariateData)
    # unpack
    (xL, xU, stencil, alpha, lambda, epsilon, iSet, w0, wStep, y0, yPlus, yMinus) =
        (data.xL, data.xU, data.stencil, data.alpha, data.lambda, data.epsilon,
         data.iSet, data.w0, data.wStep, data.y0, data.yPlus, data.yMinus)
    
    # evaluate coefficients; we already know w0. Formulas depend on stencil choice.
    n = length(xL)
    
    if stencil == :compass
        b = zeros(n)
        c = y0 - epsilon
        for i in iSet
            b[i] = (yPlus[i] - yMinus[i])/(2.0*wStep[i])
            c -= (1.0 + abs(lambda[i]))*(yPlus[i] + yMinus[i] - 2.0*y0 + 4.0*epsilon)/(2.0*alpha[i])
        end

    elseif stencil == :simplex
        yNeg = yMinus[1]
        ySum = sum(@. y0 - yPlus; init=0.0)
        
        sU = zeros(n)
        sL = copy(sU)
        sR = copy(sU)
        for i in iSet
            sU[i] = (yPlus[i] - y0)/wStep[i]
            sL[i] = (yPlus[i] - yNeg + ySum)/wStep[i]
            sR[i] = (yNeg - y0 - ySum + 4.0*epsilon*n)/(2.0*wStep[i])
        end
        
        b = 0.5*(sU + sL)

        c = y0 - epsilon - 0.5*sum(@. (1.0 + abs(lambda))*sR*(xU - xL); init=0.0)
        
    else
        throw(DomainError(:stencil, "unsupported stencil shape"))
    end
    
    return (w0, b, c)
end

"""
    fAffine = construct_underestimator(data::SampledData)

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
    yOut = evaluate_underestimator(data::SampledData, xIn)

Same as [`evaluate_underestimator_coeffs`](@ref), but instead returns the underestimator value `yOut = c + dot(b, xIn - w0)`. Useful when the constructed underestimator will be evaluated only at one domain point.

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

Compute a lower bound on a convex function `f` sampled by [`sample_convex_function`](@ref),
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
function evaluate_lower_bound(data::UnivariateData)
    # unpack
    (alpha, lambda, epsilon, y0, yPlus, yMinus) =
        (data.alpha, data.lambda, data.epsilon, data.y0, data.yPlus, data.yMinus)
    
    # relabel for convenience
    yPos = yPlus[1]
    yNeg = yMinus[1]

    # evaluate lower bound candidates, then choose the lowest of these
    fLCandidates = Vector{Float64}(undef, 4)
    
    fLCandidates[1] = 2.0*y0 - yPos - 3.0*epsilon
    
    fLCandidates[2] = 2.0*y0 - yNeg - 3.0*epsilon
    
    fLCandidates[3] = y0 + ((1.0 - lambda)/alpha)*(yPos - y0) -
        epsilon*(1.0 - lambda + abs(alpha + lambda - 1.0))/alpha
    
    fLCandidates[4] = y0 + ((1.0 + lambda)/alpha)*(yNeg - y0) -
        epsilon*(1.0 + lambda + abs(alpha - lambda - 1.0))/alpha

    return minimum(fLCandidates)
end

function evaluate_lower_bound(data::MultivariateData)
    # unpack
    (xL, xU, stencil, alpha, lambda, epsilon, iSet, w0, wStep, y0, yPlus, yMinus) =
        (data.xL, data.xU, data.stencil, data.alpha, data.lambda, data.epsilon,
         data.iSet, data.w0, data.wStep, data.y0, data.yPlus, data.yMinus)
    
    # initialize at center of sampling stencil
    fL = y0 - epsilon

    # incorporate info from remaining samples in stencil
    if stencil == :compass
        for i in iSet
            fL -= (1.0 + lambda[i])*(max(yPlus[i], yMinus[i]) - y0 + 2.0*epsilon)/alpha[i]
        end
        
    elseif stencil == :simplex
        yNeg = yMinus[1]
        ySum = sum(@. y0 - yPlus; init=0.0)
        n = length(xL)

        for i in iSet
            fL -= (1.0 + lambda[i])*
                    (max(yPlus[i], yNeg + y0 - yPlus[i] - ySum) - y0 + 2.0*epsilon*n)/alpha[i]
        end

    else
        throw(DomainError(:stencil, "unsupported stencil shape"))
    end
    
    return fL
end

# TODO: update these plotting functions
"""
    plot_underestimator(data::SampledData, f; kwargs...)

Plot (1) function, `f`, (2) affine underestimator, `fAffine`, and (3) lower bound `fL`
on the box domain `[xL, xU]`. 

See [`eval_sampling_underestimator_coeffs`](@ref) for more details on function inputs.

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
function plot_underestimator(
    data::UnivariateData,
    f::Function;
    nMeshPoints::Int = 10
)

    # unpack
    (xL, xU, w0, wStep, y0, yPlus, yMinus) =
        (data.xL, data.xU, data.w0, data.wStep, data.y0, data.yPlus, data.yMinus)

    if any(xL .>= xU)
        throw(DomainError("xL and xU", "for plotting, we must have xU[i] > xL[i] for each i"))
    end

    # evaluate f and its underestimators
    fAffine = construct_underestimator(data)
    fL = evaluate_lower_bound(data)
    
    xMesh = range(xL, xU, length=nMeshPoints)
    yMeshF = f.(xMesh)
    yMeshAffine = fAffine.(xMesh)

    # build plot
    plot(xMesh, yMeshF, label = "f", xlabel = "x", ylabel = "y")
    plot!(xMesh, yMeshAffine, label = "affine relaxation")
    plot!(xMesh, fill(fL, length(xMesh)), label = "lower bound")
    scatter!([w0; w0 + wStep; w0 - wStep], [y0; yPlus[1]; yMinus[1]], label = "sampled points")
end

function plot_underestimator(
    data::MultivariateData,
    f::Function;
    nMeshPoints::Int = 10,
    plotStyle::Vector = [surface!, wireframe!, surface]
)
    # unpack
    (xL, xU, stencil, w0, wStep, y0, yPlus, yMinus) =
        (data.xL, data.xU, data.stencil, data.w0, data.wStep, data.y0, data.yPlus, data.yMinus)

    # additional restrictions for plotting
    (length(xL) == 2) ||
        throw(DomainError(:f, "only functions of 1 or 2 variables can be plotted"))
    
    all(xL .< xU) ||
        throw(DomainError(:xU, "must be strictly .> xL for plotting"))

    # evaluate f and its underestimators
    fAffine = construct_underestimator(data)
    fL = evaluate_lower_bound(data)
    
    x1Mesh = range(xL[1], xU[1], length=nMeshPoints)
    x2Mesh = range(xL[2], xU[2], length=nMeshPoints)
    yMeshF = [f([x1, x2]) for x1 in x1Mesh, x2 in x2Mesh]
    yMeshAffine = [fAffine([x1, x2]) for x1 in x1Mesh, x2 in x2Mesh]

    # build plot
    plotStyle[3](x1Mesh, x2Mesh,
                   fill(fL, length(x1Mesh), length(x2Mesh)),
                   label = "lower bound", c=:PRGn_3)
    
    plotStyle[2](x1Mesh, x2Mesh, yMeshAffine,
                   label = "affine relaxation", c=:grays)
    
    if plotStyle[1] == wireframe!
        colorBar = false
    else
        colorBar = true
    end
    plotStyle[1](x1Mesh, x2Mesh, yMeshF, colorbar=colorBar,
                   title="From top to bottom: (1) original function,
                    (2) affine underestimator, and (3) lower bound",
                   titlefontsize=10, xlabel = "x₁", ylabel = "x₂",
                   zlabel = "y", label = "f", c=:dense)

    wPlus = w0 .+ diagm(wStep)
    if stencil == :compass
        wMinus = w0 .- diagm(wStep)
    elseif stencil == :simplex
        wMinus = w0 - wStep
    end
    scatter!([w0[1]; wPlus[1,:]; wMinus[1,:]],
             [w0[2]; wPlus[2,:]; wMinus[2,:]],
             [y0; yPlus; yMinus],
             c=:purple, legend=false)
end

end #module
