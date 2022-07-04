# convex-sampling

The `SamplingUnderestimators` module in [SamplingUnderestimators.jl](src/SamplingUnderestimators.jl) provides an implementation in Julia to:

- compute affine under-estimators for convex functions using (2n+1) black box function evaluations where n is the number of function variables
- compute affine underestimators for convex functions using an experimental (n+2) black box function evaluations method
- generate a graphical representation of the convex function and under-estimator when n = 1 or 2

This implementation depends on the external package Plots.jl. Tested in Julia v.1.7.

## Method Decscription

The [(2n+1) method](https://doi.org/10.1016/j.compchemeng.2021.107413) uses black-box sampling to evaluate the function at (2n+1) points within the defined box domain where n represents the number of function variables.

Vectors `xL` and `xU` define the upper and lower limits of the box domain. Visually, when considering two or three dimensional domains, this can be described as the lower left point and the upper right point of the box respectively. For one dimensional domains, it can be described as the left most and right most points.

Step length, `alpha`, defines the dimensionless length away from the midpoint of the box domain `w0`. It is used to locate the x-values of the the (2n) sampled points. Note that as the step length approaches `0.0`, the affine relaxations and lower bound `fL` become tighter. However, decreasing it too much may introduce significant numerical error due to the loss of precision during subtraction.

The structure of the affine underestimator is defined as: `f(x) = c + dot(b, x - w0)`. The `b` coefficient is the standard centered finite difference approximation of gradient `∇(f(w0))`. The `c` coefficient resembles the standard difference approximation of a second order partial derivative for functions of 2+ variables.

For univariate functions, a special condition is evaluated for coefficient calculation, relying on the collinear sampling of points.

## Exported functions

The following functions are exported by `SamplingUnderestimators`. In each case, `f::Function` may be provided as an anonymous function or defined beforehand. `f` must be convex and accept `Vector{Float64}` inputs and produce scalar `Float64` outputs. If `f` is univariate, inputs for exported functions may be scalar `Float{64}` instead of `Vector{Float64}` as indicated below.

- `(w0, b, c) = eval_sampling_underestimator_coeffs(f::Function, xL::Vector{Float64}, xU::Vector{Float64})`:
  - evaluates the `w0`, `b`, and `c` coefficients as `Vector{Float64}`, `Vector{Float64}`, and scalar `Float64` respectively.
  - evaluates additional variable `sR` as Vector{Float64} when considering (n+2) sampled points.

- `fUnderestimator = construct_sampling_underestimator(f::Function, xL::Vector{Float64}, xU::Vector{Float64})`
  - returns a function with structure `x -> c + dot(b, x - w0)` representing the affine under-estimator.

- `yOutput = eval_sampling_underestimator(f::Function, xL::Vector{Float64}, xU::Vector{Float64}, xIn::Vector{Float64})`
  - evaluates the affine under-estimator at a given `Vector{Float64}` x input value (`xIn`) as a `Float64` output.

-  `fL = eval_sampling_lower_bound(f::Function, xL::Vector{Float64}, xU::Vector{Float64})`:
    -   computes a `Float64` lower bound `fL::Float64` for which f(x) >= fL for each x in the box `[xL, xU]`.

-  `plot_sampling_underestimator(f::Function, xL::Vector{Float64}, xU::Vector{Float64}; plot3DStyle::Vector = [surface!, wireframe!, surface], fEvalResolution::Int64 = 10)`
    -  generates a plot of the convex function, affine under-estimator, and lower bound planes/lines within the given box domain for functions of n = 1 or 2.
    - key argument `plot3DStyle` sets the plot style (ex. wireframe, surface, etc.) of each individual plot component in the set order: (1) lower bound, (2) affine under-estimator, (3) convex function.
    - key argument `fEvalResolution` represents the plot accuracy where the number of function evaluations used for plotting is `fEvalResolution` to the exponent `n`.
    - enter `@show` along with the graph variable name in the command window to access stored graphs (see example below).

### Key Arguments

All exported functions include the listed optional key word arguments:
- `SamplingPolicy::SamplingType`:
  - An enum datatype that determines the number of sample points calculated. The default is set to `SAMPLE_COMPASS_STAR`, which uses (2n+1) function evaluations. `SAMPLE_SIMPLEX_STAR` may be entered to access the (n+2) evaluation method. Note the (n+2) method does not currently utilize `lambda` or `epsilon`.
- `alpha::Vector{Float64}`:
  - The dimensionless step length away from the central point `w0`. A default value is set to an array of constants `DEFAULT_ALPHA = 0.1`. All components of alpha must be between `(0.0, 1.0 - lambda]`.
- `lambda::Vector{Float64}`:
  - An offset for the location of `w0` to employ sampled sets where `w0` is not the midpoint of said set. A default value is set to a constant `0.0`. All components of `lambda` must be between `(-1.0, 1.0)`.
- `epsilon::Float64`:
  - Error tolerance accounting for absolute error in function evaluations; that is, within plus or minus epsilon of said evaluation. A default value is set to a constant `0.0`.

## Example

The usage of `SamplingUnderestimators` is demonstrated by script [testConvex.jl](test/testConvex.jl).
Consider the following convex function of two variables within a box domain of `xL = [-5.0, -3.0]` and `xU = [5.0, 3.0]`:

```Julia
f(x) = 〈x, [65 56; 56 65], x〉 + 〈[6; 2], x〉 + 23

```
Using the `SamplingUnderestimators` module (after commands `include(“ConvexSampling.jl”)` and `using .SamplingUnderestimators`), we can evaluate the affine under-estimator at an input value of `x = [2.0, 2.0]`.
- By defining f beforehand:
```Julia
yOutput = eval_sampling_underestimator(f, [-5.0, -3.0], [5.0, 3.0], [2.0, 2.0])
```

- By defining f as an anonymous function:
```Julia
yOutput = eval_sampling_underestimator([-5.0, -3.0], [5.0, 3.0], [2.0, 2.0]) do x
    dot(x,[65 56; 56 65], x) + dot([6;2], x) + 23
end
```

- By using the `construct_sampling_underestimator` function to construct an under-estimator function of the form `f(x) = c + dot(b, x – w0)` and manually computing the y-value:
```Julia
fUnderestimator() = construct_sampling_underestimator(f, [-5.0, -3.0], [5.0, 3.0])
yOutput = fUnderestimator([2.0, 2.0])
```
Contructing the underestimator is worthwhile if you plan on evaluating the under-estimator at more than one x-value.

The function plane can also be plotted with the affine underestimator plane within the given box boundary:
 ```Julia
graph = plot_sampling_underestimator(f::Function, xL::Vector{Float64}, xU::Vector{Float64})
@show graph
 ```

![ConvexSampleplot](https://user-images.githubusercontent.com/104848815/173203263-26bdc553-c1b5-496a-913f-eeb0553461d7.png)

Note that if the plot_sampling_underestimator function is entered directly into the command window, the `@show` command is not required.

# References

- Yingkai Song, Huiyi Cao, Chiral Mehta, and Kamil A. Khan, [Bounding convex relaxations of process models from below by tractable black-box sampling]( https://doi.org/10.1016/j.compchemeng.2021.107413), _Computers & Chemical Engineering_, 153:107413, 2021, DOI: 10.1016/j.compchemeng.2021.107413
