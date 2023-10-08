# Method Outline

Suppose we have a convex function `f` of `n` variables, defined on a box domain `X = [xL, xU]`. Our [new underestimating approach](https://doi.org/10.1016/j.compchemeng.2021.107413) samples `f` at `2n+1` domain points: the midpoint of `X`, and perturbations of this midpoint in each positive/negative coordinate direction. These sampled values are then tractably assembled using new finite difference formulas to yield guaranteed affine underestimators and guaranteed lower bounds for `f` on `X`. These underestimators are guaranteed by convex analysis theory; roughly, the sampled information is sufficient to infer a compact polyhedral set that encloses all subgradients of `f` at the midpoint of `X`. Using this information, we can deduce the "worst-case" convex functions that are consistent with the sampled data.

As in our paper, this implementation also allows for absolute error in evaluating `f`, and for off-center sampling stencils. When `n=1`, we additionally exploit the fact that each domain point is collinear with all three sampled points.

An experimental new procedure is also implemented, requiring `n+2` samples instead of `2n+1` samples.
