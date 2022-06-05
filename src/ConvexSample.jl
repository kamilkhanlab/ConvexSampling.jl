#=
module ConvexSample
=====================
A quick implementation of:

- bounding convex relaxations of process models from below by
tractable black-box sampling, developed in the article:
KA Khan et al (2021),https://doi.org/10.1016/j.compchemeng.2021.107413

This implementation applies theorem 1 and 2 to calculate affine relaxations of
a convex function on a box domain.

...

Written by Maha Chaudhry on May 25, 2022
=#
module ConvexSample

using LinearAlgebra
using Plots

export aff_coefficients,
    aff_underestimator,
    aff_underestimator_atx,
    lower_bound,
    plotter

## define affine under-estimator function operations, given a convex function f
## defined on a box domain with a Vector input

# compute 2n+1 sampled values required to construct affine relaxation:
#   (w0,y0) = midpoint of the box domain
#   (wi,y0) = 2n points along defined step lengths (α)
function sampled_points(
        f::Function,
        xL::Vector{Float64},
        xU::Vector{Float64};
        alpha::Vector{Float64} = fill(0.1, length(xL)) #sets default value for α
    )
    n = length(xL) #function dimension
    w0 = 0.5*(xL + xU)
    y0 = f(w0)

    #define matrix of unit coordinator vectors in R^n:
    e = Matrix(I, n, n)

    #structure of wi and yi set to i = [+1, -1, +2, -2, ...]
    yi = zeros(2*n,1)
    wi = zeros(2*n,n)
    for i in 2*range(1,n)
        wpos = w0 + ((1)*0.5*alpha[i÷2]*(xU[i÷2] - xL[i÷2])).*e[i÷2,:]
        wneg = w0 + ((-1)*0.5*alpha[i÷2]*(xU[i÷2] - xL[i÷2])).*e[i÷2,:]
        wi[i-1:i,1:n] = [wpos'; wneg']
        yi[i-1:i] = [f(wpos);f(wneg)]
    end

    return w0, y0, wi, yi
end

# compute coefficients for affine underestimator function where:
# f(x) = c + dot(b, x - w0)
#   coefficient b = centered simplex gradient of f at w0 sampled
#                   along coordinate vectors
#   coefficient c = resembles standard difference approximation of
#                   second-order partial derivatives
function aff_coefficients(
        f::Function,
        xL::Vector{Float64},
        xU::Vector{Float64};
        alpha::Vector{Float64} = fill(0.1, length(xL))
    )
    n = length(xL)
    w0, y0, wi, yi = sampled_points(f, xL, xU; alpha)

    b = zeros(n,1)
    if all(xL .< xU)
        for i in range(1,n)
            b[i] = (yi[2*i-1] - yi[2*i])/maximum(abs.(wi[2*i-1,:]-wi[2*i,:]))
        end
    end

    #coefficient c can be tightened in special cases where f is univariate
    #dependent on the defined step length:
    c = y0[1]
    if n > 1
        for i in 2*range(1,n)
            c = c - 0.5*((yi[i-1]+yi[i]-2y0)/alpha[i÷2])
        end
    elseif n == 1 && alpha != 1
        c = 2*c - 0.5*(yi[1]+yi[2])
    end
    return b, c
end

# define affine underestimator function using calculated b, c coefficients:
function aff_underestimator(
        f::Function,
        xL::Vector{Float64},
        xU::Vector{Float64};
        alpha::Vector{Float64} = fill(0.1, length(xL))
    )
    n = length(xL)
    b, c = aff_coefficients(f, xL, xU; alpha)
    w0 = 0.5*(xL + xU)
    #structure of function is f(x) = c + dot(b, x - w0)
    func(x) = c + dot(b, x - w0)
    return func
end

# compute affine underestimator y-value using:
#  (1) computed affine underestimator function
#  (2) x-input value
function aff_underestimator_atx(
    f::Function,
    xL::Vector{Float64},
    xU::Vector{Float64},
    xInput::Vector{Float64}; #define x-input value
    alpha::Vector{Float64} = fill(0.1, length(xL))
)
    affinefunc = aff_underestimator(f, xL, xU; alpha)
    yOutput = affinefunc(xInput)
    return yOutput
end

# compute:
#  fL = guaranteed constant scalar lower bound of f on X
function lower_bound(
        f::Function,
        xL::Vector{Float64},
        xU::Vector{Float64};
        alpha::Vector{Float64} = fill(0.1, length(xL))
    )
    n = length(xL)
    w0, y0, wi, yi = sampled_points(f, xL, xU; alpha)

    #coefficient c can be tightened in special cases where f is univariate:
    if n > 1
        fL = y0
        for i in 2*range(1,n)
            fL = fL - (max(yi[i-1], yi[i])-y0)/alpha[i÷2]
        end
    elseif n == 1
        fL = min(2*y0[1]-yi[1], 2*y0[1]-yi[2], (1/alpha[1])*yi[2]-((1-alpha[1])/alpha[1])*y0[1], (1/alpha[1])*yi[1]-((1-alpha[1])/alpha[1])*y0[1])
    end
    return fL
end

# plot:
#  function f on plane (R^n) within box domain
#  lower bound fL on plane (R^n) within box domain
#  affine underestimator on plane within box domain
#  sampled points = (w0,y0) and (wi, yi)
function plotter(
    f::Function,
    xL::Vector{Float64},
    xU::Vector{Float64};
    alpha::Vector{Float64} = fill(0.1, length(xL)),
    style::Vector = [surface!, wireframe!, surface], #Set plot style
    functionaccuracy::Int64 = 10, #Set number of function evaluations as points^2
    affineaccuracy::Int64 = 10 #Set number of affine evaluations as points^2
)
    n = length(xL)
    #set function definition to speed up computational time:
    affine = aff_underestimator(f, xL, xU; alpha)
    #calculate scalar values:
    w0, y0, wi, yi = sampled_points(f, xL, xU; alpha)
    fL = lower_bound(f, xL, xU; alpha)

    if n == 1
        #sampled points on univariate functions are collinear, so range of points
        #is also univariate:
        xCoord = range(xL[1], xU[1], 100)
        funcyCoord = zeros(100,1) #to collect function evaluations
        affyCoord = zeros(100,1) #to collect affine underestimator evaluations
        for i in 1:length(xCoord)
            funcyCoord[i] = f(xCoord[i])
            affyCoord[i] = affine([xCoord[i]])
        end

        #to plot along 2 dimensions:
        plot(xCoord, funcyCoord, label = "Function", xlabel = "x axis", ylabel = "y axis")
        plot!(xCoord, affyCoord, label = "Affine underestimator")
        plot!(xCoord,fill!(funcyCoord,fL), label = "Lower bound")
        scatter!([vcat(w0, wi)],[vcat(y0, yi)],label = "Sampled points")

    elseif n == 2
        #for higher dimension functions, a meshgrid of points is required
        #as function and affine accuracy may differ, each require individual meshgrids
        x1frange = range(xL[1], xU[1], functionaccuracy)
        x2frange = range(xL[2], xU[2], functionaccuracy)
        funcyCoord = zeros(length(x1frange),length(x2frange))
        for x1 in 1:length(x1frange)
            for x2 in 1:length(x2frange)
                funcyCoord[x1,x2] = f([x1frange[x1],x2frange[x2]])
            end
        end

        x1arange = range(xL[1], xU[1], affineaccuracy)
        x2arange = range(xL[2], xU[2], affineaccuracy)
        affyCoord = zeros(length(x1arange),length(x2arange))
        for x1 in 1:length(x1arange)
            for x2 in 1:length(x2arange)
                affyCoord[x1,x2] = affine([x1arange[x1],x2arange[x2]])
            end
        end

        #to plot along 3 dimensions:
        style[3](x1frange, x2frange, fill(fL, length(x1frange), length(x2frange)), label = "Lower bound", c=:PRGn_4)
        style[2](x1arange, x2arange, affyCoord, label = "Affine underestimator", c=:grays)
        colorBar = true
        if style[1] == wireframe!
            colorBar = false
        end
        style[1](x1frange, x2frange, funcyCoord, colorbar=colorBar, title="From top to bottom: (1) Original function,
        (2) Affine underestimator, and (3) Lower bound",titlefontsize=10,
        xlabel = "x₁ axis", ylabel = "x₂ axis", zlabel = "y axis", label = "Function", c=:dense)
        scatter!([vcat(w0[1], wi[:,1])], [vcat(w0[2], wi[:,2])], [vcat(y0, yi)],legend=false)
    end
end

end
