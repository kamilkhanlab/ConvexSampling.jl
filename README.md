# convex-sampling

The `ConvexSampling` module in ConvexSampling.jl provides an implementation in Julia to: 

- compute affine under-estimators for convex functions using (2n+1) black box function evaluations
- generate a graphical representation of the convex function and under-estimator

This implementation doesn't depend on any packages external to Julia. Tested in Julia v.1.7.

### Exported functions

The following functions are exported by `ConvexSampling`. In each case, `f` may be provided as an anonymous function:

- `(b, c) = aff_coefficients(f::Function, n::Int64, xL::Vector{Float64}, xU::Vector{Float64}; a::Vector{Float64} = vec(fill!(zeros(1,n),0.1)))`:
  - evaluates the `b` and `c` coefficients as Matrix{Float64} and scalar Float64 respectively where  `b` is the standard centered finite difference approximation of gradient `∇(f(w0))` and `c` resembles the standard difference approximation of a second order partial derivative for functions of 2+ variables.  A special condition for coefficient calculation is set for univariate functions, relying on the collinear sampling of points, based on the proof from Larson et al (2021).

- `(x -> c + dot(b, x - w0)) = function aff_underestimator(f::Function, n::Int64, xL::Vector{Float64}, xU::Vector{Float64}; a::Vector{Float64} = vec(fill!(zeros(1,n),0.1)))`
  - returns an anonymous function representing the affine under-estimator based on the calculated `b` and `c` coefficients and the midpoint of the given box domain. 

- `yOutput = aff_underestimator_atx(f::Function, n::Int64, xL::Vector{Float64}, xU::Vector{Float64}, xInput::Vector{Float64}; a::Vector{Float64} = vec(fill!(zeros(1,n),0.1)))`
  - computes the y value of the affine under-estimator given an x input based on the calculated `b` and `c` coefficients and the midpoint of the given box domain. 

-  `fL = lower_bound(f::Function, n::Int64, xL::Vector{Float64}, xU::Vector{Float64}; a::Vector{Float64} = vec(fill!(zeros(1,n),0.1)))`
   - computes a scalar guaranteed constant lower bound of f on X. A special condition for lower    bound calculation is set for univariate functions, relying on the collinear sampling of points, based on the proof from Larson et al (2021).

-  `plotter(f::Function, n::Int64, xL::Vector{Float64}, xU::Vector{Float64}; a::Vector{Float64} = vec(fill!(zeros(1,n),0.1)))`
   - generates a plot of the convex function, affine under-estimator, and lower bound planes/lines within the given box domain.

All exported functions include an optional key word argument for step length: `a`. A default value is set to `0.1`. Note that as the step length approaches 0, the affine relaxations and lower bound `fL` become tighter. However, decreasing it too much may introduce significant numerical error.

### Example

The usage of `ConvexSampling` is demonstrated by script testConvex.jl. 

Consider the following convex function of two variables within a box domain of `xL = [-5.0, -3.0]` and `xU = [5.0, 3.0]`:

```Julia
f(x) = 〈x, [65 56; 56 65], x〉 + 〈[6; 2], x〉 + 23
```

Using the `ConvexSampling` module (after `include(“ConvexSampling.jl”)` and `using .ConvexSamping`), we can compute `yOutput` for the affine under-estimator at an input value of `x = [2.0, 2.0]`.

- By defining f beforehand:
  ```Julia
  yOutput = aff_underestimator_atx(f, 2, [-5.0, -3.0], [5.0, 3.0], [2.0, 2.0])
  ```
- By defining f as an anonymous function: 
  ```Julia
  yOutput = aff_underestimator_atx(x -> dot(x,[65 56; 56 65], x) + dot([6;2], x) + 23, 2, [-5.0, -3.0], [5.0, 3.0], [2.0, 2.0])
  ```
- By using the `aff_underestimator` function to constructs an anonymous under-estimator function of the form `f(x) = c + 〈b, x – w0〉` and manually compute the y-value:
  ```Julia
  affine_function() = aff_underestimator(f, 2, [-5.0, -3.0], [5.0, 3.0])
  yOutput = affine_function([2.0, 2.0])
  ```


## References

- Yingkai Song, Huiyi Cao, Chiral Mehta, and Kamil A. Khan, [Bounding convex relaxations of process models from below by tractable black-box sampling]( https://doi.org/10.1016/j.compchemeng.2021.107413), _Computers & Chemical Engineering_, 153:107413, 2021, DOI: 10.1016/j.compchemeng.2021.107413

- Larson, Jeffrey, Sven Leyffer, Prashant Palkar, and Stefan M. Wild, [A method for convex black-box integer global optimization]( https://link.springer.com/article/10.1007/s10898-020-00978-w), _ Journal of Global Optimization_, 80(2):439-77, 2021, DOI: 10.1007/s10898-020-00978-w
