# convex-sampling

The `SamplingUnderestimators` module in [SamplingUnderestimators.jl](src/SamplingUnderestimators.jl) provides an implementation in Julia to:

- compute affine under-estimators for convex functions using (2n+1) black box function evaluations
- generate a graphical representation of the convex function and under-estimator

This implementation depends on external package Plots.jl. Tested in Julia v.1.7.

## Exported functions

The following functions are exported by `SamplingUnderestimators`. In each case, `f` may be provided as an anonymous function or defined beforehand. `f` must accept Vector{Float64} inputs and produce scalar Float64 outputs.

- `(w0, b, c) = eval_sampling_underestimator_coeffs(f::Function, xL::Vector{Float64}, xU::Vector{Float64})`:
  - evaluates the `w0`, `b`, and `c` coefficients as Vector{Float64}, Matrix{Float64}, and scalar Float64 respectively where  `b` is the standard centered finite difference approximation of gradient `∇(f(w0))`, `c` resembles the standard difference approximation of a second order partial derivative for functions of 2+ variables, and `w0` is the midpoint of the given box domain.
  - evaluates a special condition for coefficient calculation for univariate functions, relying on the collinear sampling of points, based on the proof from Larson et al (2021).

- `fUnderestimator = construct_sampling_underestimator(f::Function, xL::Vector{Float64}, xU::Vector{Float64})`
  - returns an anonymous function with structure `x -> c + dot(b, x - w0)` representing the affine under-estimator based on the calculated `w0`, `b`, `c` coefficients.

- `yOutput = eval_sampling_underestimator(f::Function, xL::Vector{Float64}, xU::Vector{Float64}, xIn::Vector{Float64})`
  - computes the y value as a Float64 of the affine under-estimator given a Float64 x input (`xIn`) based on the calculated `w0`, `b`, `c` coefficients.

-  `fL = eval_sampling_lower_bound(f::Function, xL::Vector{Float64}, xU::Vector{Float64})`
  - computes a Float64 scalar guaranteed constant lower bound of f on X.
  - evaluates a special condition for coefficient calculation for univariate functions, relying on the collinear sampling of points, based on the proof from Larson et al (2021).

-  `plot_sampling_underestimator(f::Function, xL::Vector{Float64}, xU::Vector{Float64}; plot3DStyle::Vector = [surface!, wireframe!, surface], fEvalResolution::Int64 = 10)`
  - generates a plot of the convex function, affine under-estimator, and lower bound planes/lines within the given box domain.
  - key argument plot3DStyle sets the plot style (ex. wireframe, surface, etc.) of each individual plot component in the set order: (1) lower bound, (2) affine under-estimator, (3) convex function.
  - key argument fEvalResolution sets the plot accuracy through the number of function evaluations calculated by the function.
  - enter `@show <graphname>` in the command window to access stored graphs.

### Key Arguments

All exported functions include the listed optional key word arguments:
- `alpha::Vector{Float64}`: The step length between successive sample points. A default value is set to an array of constants `DEFAULT_ALPHA = 0.1`. Note that as the step length approaches 0, the affine relaxations and lower bound `fL` become tighter. However, decreasing it too much may introduce significant numerical error.
- `lambda::Float64`: An offset for the location of `w0`, which accounts for sampled sets where `w0` is not the midpoint of said set. A default value is set to a constant `0.0`.
- `epsilon::Float64`: Error tolerance to account for possible error in function evaluations. A default value is set to a constant `0.0`.

## Example

The usage of `SamplingUnderestimators` is demonstrated by script testConvex.jl.
Consider the following convex function of two variables within a box domain of `xL = [-5.0, -3.0]` and `xU = [5.0, 3.0]`:

```Julia
f(x) = 〈x, [65 56; 56 65], x〉 + 〈[6; 2], x〉 + 23

```
Using the `SamplingUnderestimators` module (through commands `include(“ConvexSampling.jl”)` and `using .SamplingUnderestimators`), we can compute `yOutput` for the affine under-estimator at an input value of `x = [2.0, 2.0]`.
- By defining f beforehand:
```Julia
yOutput = eval_sampling_underestimator(f, [-5.0, -3.0], [5.0, 3.0], [2.0, 2.0])
```

- By defining f as an anonymous function:
```Julia
yOutput = eval_sampling_underestimator(x -> dot(x,[65 56; 56 65], x) + dot([6;2], x) + 23, [-5.0, -3.0], [5.0, 3.0], [2.0, 2.0])
```

- By using the `construct_sampling_underestimator` function to construct an anonymous under-estimator function of the form `f(x) = c + 〈b, x – w0〉` and manually computing the y-value:
```Julia
fUnderestimator() = construct_sampling_underestimator(f, [-5.0, -3.0], [5.0, 3.0])
yOutput = fUnderestimator([2.0, 2.0])
```

The function plane can also be plotted with the affine underestimator plane within the given box boundary:
 ```Julia
graph = plot_sampling_underestimator(f::Function, xL::Vector{Float64}, xU::Vector{Float64})
@show graph
 ```

![ConvexSampleplot](https://user-images.githubusercontent.com/104848815/173203263-26bdc553-c1b5-496a-913f-eeb0553461d7.png)

# References

- Yingkai Song, Huiyi Cao, Chiral Mehta, and Kamil A. Khan, [Bounding convex relaxations of process models from below by tractable black-box sampling]( https://doi.org/10.1016/j.compchemeng.2021.107413), _Computers & Chemical Engineering_, 153:107413, 2021, DOI: 10.1016/j.compchemeng.2021.107413
-Larson, Jeffrey, Sven Leyffer, Prashant Palkar, and Stefan M. Wild, [A method for convex black-box integer global optimization]( https://link.springer.com/article/10.1007/s10898-020-00978-w), _Journal of Global Optimization_, 80(2):439-77, 2021, DOI: 10.1007/s10898-020-00978-w
