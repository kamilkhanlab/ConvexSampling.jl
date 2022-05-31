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
        n::Int64, #function dimension
        xL::Vector{Float64},
        xU::Vector{Float64};
        a::Vector{Float64} = vec(fill!(zeros(1,n),0.1)) #sets default value for α
    )
    w0 = 0.5*(xL + xU)
    y0 = f(w0)

    #define unit coordinator vector in R^n:
    if n == 2
        e = [[1, 0], [0, 1]]
    elseif n == 1
        e = [[1], [1]]
    end

    wi = zeros(1,n)
    yi = zeros(1,1)
    for i in range(1,n)
        for j in [1, -1]
            wtemp = w0 + (j*0.5*a[i]*(xU[i] - xL[i])).*e[i]
            wi = vcat(wi, wtemp')
            yi = vcat(yi, f(wtemp))
        end
    end
    wi = wi[2:end,:]
    yi = yi[2:end]

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
        n::Int64,
        xL::Vector{Float64},
        xU::Vector{Float64};
        a::Vector{Float64} = vec(fill!(zeros(1,n),0.1))
    )
    w0, y0, wi, yi = sampled_points(f, n, xL, xU; a)

    b = zeros(n,1)
    if all(xL .< xU)
        for i in range(1,n)
            b[i] = (yi[2*i-1] - yi[2*i])/maximum(abs.(wi[2*i-1,:]-wi[2*i,:]))
        end
    end

    #coefficient c can be tightened in special cases where f is univariate
    #dependent on the defined step length:
    if n == 2
        c = y0 - 0.5*((yi[1]+yi[2]-2y0)/a[1]) - 0.5*((yi[3]+yi[4]-2y0)/a[2])
    elseif n == 1 && a == 1
        c = y0
    elseif n == 1
        c = 2*y0[1] - 0.5*(yi[1]+yi[2])
    end

    return b, c
end

# define affine underestimator function using calculated b, c coefficients:
function aff_underestimator(
        f::Function,
        n::Int64,
        xL::Vector{Float64},
        xU::Vector{Float64};
        a::Vector{Float64} = vec(fill!(zeros(1,n),0.1))
    )
    b, c = aff_coefficients(f,n,xL,xU;a)
    w0 = 0.5*(xL+xU)

    return x -> c + dot(b, x - w0)
end

# compute affine underestimator y-value using:
#  (1) computed affine underestimator function
#  (2) x-input value
function aff_underestimator_atx(
        f::Function,
        n::Int64,
        xL::Vector{Float64},
        xU::Vector{Float64},
        xInput::Vector{Float64}; #define x-input value
        a::Vector{Float64} = vec(fill!(zeros(1,n),0.1))
    )
    affinefunc = aff_underestimator(f,n,xL,xU;a)
    yOutput = affinefunc(xInput)

    return yOutput
end

# compute:
#  fL = guaranteed constant scalar lower bound of f on X
function lower_bound(
        f::Function,
        n::Int64,
        xL::Vector{Float64},
        xU::Vector{Float64};
        a::Vector{Float64} = vec(fill!(zeros(1,n),0.1))
    )
    w0, y0, wi, yi = sampled_points(f, n, xL, xU; a)

    #coefficient c can be tightened in special cases where f is univariate:
    if n == 2
        fL = y0 - (max(yi[1], yi[2])-y0)/a[1] - (max(yi[3], yi[4])-y0)/a[2]
    elseif n == 1
        fL = min(2*y0[1]-yi[1], 2*y0[1]-yi[2], (1/a[1])*yi[2]-((1-a[1])/a[1])*y0[1], (1/a[1])*yi[1]-((1-a[1])/a[1])*y0[1])
    end

    return fL
end

# plot:
#  function f on plane (R^n) within box domain
#  lower bound fL on plane (R^n) within box domain
#  affine underestimator on plane within box domain
function plotter(
        f::Function,
        n::Int64,
        xL::Vector{Float64},
        xU::Vector{Float64};
        a::Vector{Float64} = vec(fill!(zeros(1,n),0.1))
    )
    #set function definition to speed up computational time:
    affine = aff_underestimator(f, n, xL, xU;a)

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

        fL = lower_bound(f, n, xL, xU; a)

        #to plot along 2 dimensions:
        plot(xCoord, funcyCoord, label = "Function", xlabel = "x axis", ylabel = "y axis")
        plot!(xCoord, affyCoord, label = "Affine underestimator")
        plot!(xCoord,fill!(funcyCoord,fL), label = "Lower bound")

    elseif n == 2
        #for higher dimension functions, a meshgrid of points is required:
        xCoord = [0 0]
        funcyCoord = [0] #to collect function evaluations
        affyCoord = [0] #to collect affine underestimator evaluations
        for x1 in range(xL[1], xU[1], 10)
            for x2 in range(xL[2], xU[2], 10)
                xCoord = vcat(xCoord, [x1 x2])
                funcyCoord = vcat(funcyCoord, f([x1, x2]))
                affyCoord = vcat(affyCoord, affine([x1, x2]))
            end
        end
        xCoord = xCoord[2:end,:]
        funcyCoord = funcyCoord[2:end]
        affyCoord = affyCoord[2:end]

        fL = lower_bound(f, n, xL, xU; a)

        #to plot along 3 dimensions:
        surface(xCoord[:,1],xCoord[:,2],fill!(xCoord[:,1],fL), label = "Lower bound", c=:reds)
        wireframe!(range(xL[1], xU[1], 10), range(xL[2], xU[2], 10), affyCoord)
        surface!(xCoord[:, 1], xCoord[:,2], funcyCoord, label = "Function", xlabel = "x axis", ylabel = "y axis", zlabel = "z axis", c=:matter)
    end
end

end
