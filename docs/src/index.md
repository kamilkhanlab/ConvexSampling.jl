# **ConvexSampling Documentation**

## Installation and Usage

ConvexSampling is not currently a registered Julia package. From the Julia REPL, type ] to access the Pkg REPL mode and then run the following command:

```Julia
add https://github.com/kamilkhanlab/convex-sampling
```

Then, to use the package:

```Julia
using ConvexSampling
```

## Example

The usage of `SamplingUnderestimators` is demonstrated by script [testConvex.jl](test/testConvex.jl).
Consider the following convex quadratic function of two variables within a box domain of `xL = [-5.0, -3.0]` and `xU = [5.0, 3.0]`:

```Julia
f(x) = dot(x, [65.0 56.0; 56.0 65.0], x) + dot([6.0, 2.0], x) + 23.0

```
Using the `SamplingUnderestimators` module (after commands `include(“ConvexSampling.jl”)` and `using .SamplingUnderestimators`), we can evaluate its affine underestimator at an input value of `x = [2.0, 2.0]`.
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

- By constructing the underestimator as a function, and then evaluating it:
  ```Julia
  fAffine() = construct_sampling_underestimator(f, [-5.0, -3.0], [5.0, 3.0])
  yOutput = fAffine([2.0, 2.0])
  ```
Contructing the underestimator is worthwhile if you plan on evaluating it at more than one `x`-value.

The function `f` may be plotted with its sampling-based underestimator `fAffine` and lower bound `fL`:

```Julia
graph = plot_sampling_underestimator(f, xL, xU)
@show graph
```

![ConvexSampleplot](https://user-images.githubusercontent.com/104848815/173203263-26bdc553-c1b5-496a-913f-eeb0553461d7.png)

Note that if the `plot_sampling_underestimator` function is entered directly into the command window, the `@show` command is not required.

## Authors

- Maha Chaudhry, Department of Chemical Engineering, McMaster University
- Kamil Khan, Department of Chemical Engineering, McMaster University

## References

- Yingkai Song, Huiyi Cao, Chiral Mehta, and Kamil A. Khan, [Bounding convex relaxations of process models from below by tractable black-box sampling]( https://doi.org/10.1016/j.compchemeng.2021.107413), _Computers & Chemical Engineering_, 153:107413, 2021, DOI: 10.1016/j.compchemeng.2021.107413
