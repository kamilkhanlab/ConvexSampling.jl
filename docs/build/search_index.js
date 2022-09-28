var documenterSearchIndex = {"docs":
[{"location":"methodOutline.html#Method-Outline","page":"Method Outline","title":"Method Outline","text":"","category":"section"},{"location":"methodOutline.html","page":"Method Outline","title":"Method Outline","text":"Suppose we have a convex function f of n variables, defined on a box domain X = [xL, xU]. Our new underestimating approach samples f at 2n+1 domain points: the midpoint of X, and perturbations of this midpoint in each positive/negative coordinate direction. These sampled values are then tractably assembled using new finite difference formulas to yield guaranteed affine underestimators and guaranteed lower bounds for f on X. These underestimators are guaranteed by convex analysis theory; roughly, the sampled information is sufficient to infer a compact polyhedral set that encloses all subgradients of f at the midpoint of X. Using this information, we can deduce the \"worst-case\" convex functions that are consistent with the sampled data.","category":"page"},{"location":"methodOutline.html","page":"Method Outline","title":"Method Outline","text":"As in our paper, this implementation also allows for absolute error in evaluating f, and for off-center sampling stencils. When n=1, we additionally exploit the fact that each domain point is collinear with all three sampled points.","category":"page"},{"location":"methodOutline.html","page":"Method Outline","title":"Method Outline","text":"An experimental new procedure is also implemented, requiring n+2 samples instead of 2n+1 samples.","category":"page"},{"location":"functions.html#Exported-Types","page":"Exported Functions","title":"Exported Types","text":"","category":"section"},{"location":"functions.html","page":"Exported Functions","title":"Exported Functions","text":"SamplingType","category":"page"},{"location":"functions.html#ConvexSampling.SamplingType","page":"Exported Functions","title":"ConvexSampling.SamplingType","text":"An enum that specifies the sampling strategy and the number of evaluations of f.\n\nPossible Values\n\nSAMPLE_COMPASS_STAR: (default value), uses (2n+1) function evaluations in a                       compass-star stencil, where n is domain dimension of f.\nSAMPLE_SIMPLEX_STAR: uses (n+2) evaluations; this is experimental,                       and does not currently utilize lambda or epsilon.\n\n\n\n\n\n","category":"type"},{"location":"functions.html#Exported-Functions","page":"Exported Functions","title":"Exported Functions","text":"","category":"section"},{"location":"functions.html","page":"Exported Functions","title":"Exported Functions","text":"eval_sampling_underestimator_coeffs\r\nconstruct_sampling_underestimator\r\neval_sampling_underestimator\r\neval_sampling_lower_bound\r\nplot_sampling_underestimator","category":"page"},{"location":"functions.html#ConvexSampling.eval_sampling_underestimator_coeffs","page":"Exported Functions","title":"ConvexSampling.eval_sampling_underestimator_coeffs","text":"eval_sampling_underestimator_coeffs(args...; kwargs...)\n\nArguments\n\nf:Function: must be convex and of the form f(x::T)::Float64;   otherwise implementation treats f as a black box and cannot verify convexity\nxL::T: coordinates for lower bound of box domain on which f is defined\nxU::T: coordinates for upper bound of box domain on which f is defined\n\nwhere T is either Vector{Float64} or Float64\n\nKeywords\n\nsamplingPolicy::SamplingType: an enum specifying sampling strategy.   See SamplingType for more details.\nlambda::T: an offset of location of w0 to employ sampling stencils where w0   is not the domain midpoint. All components of lambda must be between   (-1.0, 1.0), and are 0.0 by default.\nalpha::T: dimensionless step length of each sampled point from stencil centre w0.   Each component alpha[i] must satisfy 0.0 < alpha[i] <= 1.0 - lambda[i],   and is set to0.1` by default. If the step length is too small, then subtraction   operations in our finite difference formulas might cause unacceptable numerical error.\nepsilon::Float64: an absolute error bound for evaluations of f. We presume that   each numerical evaluation of f(x) is within epsilon of the true value.   Set to 0.0 by default.\n\nwhere T is either Vector{Float64} or Float64\n\nNotes\n\nAdditional output sR is only used by experimental method SAMPLE_SIMPLEX_STAR.\n\nExample\n\nTo construct the underestimator function for the function f on box domain xL[i] <= x[i] <= xU[i] for all x inputs:\n\nA = [25.0 24.0; 24.0 25.0]\nb = [2.0; 3.0]\nc = 15.0\nf(x) = dot(x, A, x) + dot(b, x) + c\nxL = [-1.0, -2.0]\nxU = [3.0, 5.0]\neval_sampling_underestimator_coeffs(f, xL, xU)\n\n# output\n\n(w0, b, c) = ([1.0, 1.5], [123.99999999999991, 126.00000000000006], 134.125)\n\n\n\n\n\n","category":"function"},{"location":"functions.html#ConvexSampling.construct_sampling_underestimator","page":"Exported Functions","title":"ConvexSampling.construct_sampling_underestimator","text":"construct_sampling_underestimator(args...; kwargs...)\n\nReturn affine underestimator function of the format fAffine(x) = c + dot(b, x - w0) by sampling function f at 2n+1 domain points where n is the function dimension.\n\nSee eval_sampling_underestimator_coeffs for more details on function inputs.\n\nExample\n\nTo construct the underestimator function for the function f on box domain xL[i] <= x[i] <= xU[i] for all x inputs:\n\nA = [25.0 24.0; 24.0 25.0]\nb = [2.0; 3.0]\nc = 15.0\nf(x) = dot(x, A, x) + dot(b, x) + c\nxL = [-1.0, -2.0]\nxU = [3.0, 5.0]\nfAffine(x) = construct_sampling_underestimator(f, xL, xU)\n\n\n\n\n\n","category":"function"},{"location":"functions.html#ConvexSampling.eval_sampling_underestimator","page":"Exported Functions","title":"ConvexSampling.eval_sampling_underestimator","text":"eval_sampling_underestimator(args...; kwargs...)\n\nEvaluate underestimator fAffine constructed by construct_sampling_underestimator at a domain point xIn. That is, yOut = fAffine(xIn).\n\nSee eval_sampling_underestimator_coeffs for more details on function inputs.\n\nExample\n\nA = [25.0 24.0; 24.0 25.0]\nb = [2.0; 3.0]\nc = 15.0\nf(x) = dot(x, A, x) + dot(b, x) + c\nxL = [-1.0, -2.0]\nxU = [3.0, 5.0]\neval_sampling_underestimator(f, xL, xU, [2.0, 2.0])\n\n# output\n\n321.12499999999994\n\n\n\n\n\n","category":"function"},{"location":"functions.html#ConvexSampling.eval_sampling_lower_bound","page":"Exported Functions","title":"ConvexSampling.eval_sampling_lower_bound","text":"eval_sampling_lower_bound(args...; kwargs...)\n\nCompute the scalar lower bound of f on the interval [xL, xU], so that f(x) >= fL for each x in the box.\n\nSee eval_sampling_underestimator_coeffs for more details on function inputs.\n\nExample\n\nA = [25.0 24.0; 24.0 25.0]\nb = [2.0; 3.0]\nc = 15.0\nf(x) = dot(x, A, x) + dot(b, x) + c\nxL = [-1.0, -2.0]\nxU = [3.0, 5.0]\neval_sampling_lower_bound(f, xL, xU)\n\n# output\n\n-554.875\n\n\n\n\n\n","category":"function"},{"location":"functions.html#ConvexSampling.plot_sampling_underestimator","page":"Exported Functions","title":"ConvexSampling.plot_sampling_underestimator","text":"plot_sampling_underestimator(args...; kwargs...)\n\nPlot (1) function, f, (2) affine underestimator, fAffine, and (3) lower bound fL on the box domain [xL, xU].\n\nSee eval_sampling_underestimator_coeffs for more details on function inputs.\n\nAdditional Keywords\n\nplot3DStyle::Vector: sets the plot style (ex. wireframe, surface, etc.)   of each individual plot component in the set order:   (1) lower bound, (2) affine under-estimator, (3) convex function.   Default value: [surface!, wireframe!, surface]\nfEvalResolution::Int64: number of mesh rows per domain dimension in the resulting plot.   Default value: 10\n\nNotes\n\nf must be a function of either 1 or 2 variables and must take a Vector{Float64} input.\nThe produced graph may be stored to a variable and later retrieved with @show.\n\n\n\n\n\n","category":"function"},{"location":"index.html#**ConvexSampling-Documentation**","page":"Introduction","title":"ConvexSampling Documentation","text":"","category":"section"},{"location":"index.html#Installation-and-Usage","page":"Introduction","title":"Installation and Usage","text":"","category":"section"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"ConvexSampling is not currently a registered Julia package. From the Julia REPL, type ] to access the Pkg REPL mode and then run the following command:","category":"page"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"add https://github.com/kamilkhanlab/convex-sampling","category":"page"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"Then, to use the package:","category":"page"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"using ConvexSampling","category":"page"},{"location":"index.html#Example","page":"Introduction","title":"Example","text":"","category":"section"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"The usage of SamplingUnderestimators is demonstrated by script testConvex.jl. Consider the following convex quadratic function of two variables within a box domain of xL = [-5.0, -3.0] and xU = [5.0, 3.0]:","category":"page"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"f(x) = dot(x, [65.0 56.0; 56.0 65.0], x) + dot([6.0, 2.0], x) + 23.0\r\n","category":"page"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"Using the SamplingUnderestimators module (after commands include(“ConvexSampling.jl”) and using .SamplingUnderestimators), we can evaluate its affine underestimator at an input value of x = [2.0, 2.0].","category":"page"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"By defining f beforehand:\nyOutput = eval_sampling_underestimator(f, [-5.0, -3.0], [5.0, 3.0], [2.0, 2.0])","category":"page"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"By defining f as an anonymous function:\nyOutput = eval_sampling_underestimator([-5.0, -3.0], [5.0, 3.0], [2.0, 2.0]) do x\r\n    dot(x,[65 56; 56 65], x) + dot([6;2], x) + 23\r\nend","category":"page"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"By constructing the underestimator as a function, and then evaluating it:\nfAffine() = construct_sampling_underestimator(f, [-5.0, -3.0], [5.0, 3.0])\r\nyOutput = fAffine([2.0, 2.0])","category":"page"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"Contructing the underestimator is worthwhile if you plan on evaluating it at more than one x-value.","category":"page"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"The function f may be plotted with its sampling-based underestimator fAffine and lower bound fL:","category":"page"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"graph = plot_sampling_underestimator(f, xL, xU)\r\n@show graph","category":"page"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"(Image: ConvexSampleplot)","category":"page"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"Note that if the plot_sampling_underestimator function is entered directly into the command window, the @show command is not required.","category":"page"},{"location":"index.html#Authors","page":"Introduction","title":"Authors","text":"","category":"section"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"Maha Chaudhry, Department of Chemical Engineering, McMaster University\nKamil Khan, Department of Chemical Engineering, McMaster University","category":"page"},{"location":"index.html#References","page":"Introduction","title":"References","text":"","category":"section"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"Yingkai Song, Huiyi Cao, Chiral Mehta, and Kamil A. Khan, Bounding convex relaxations of process models from below by tractable black-box sampling, Computers & Chemical Engineering, 153:107413, 2021, DOI: 10.1016/j.compchemeng.2021.107413","category":"page"}]
}