# __ Basic utilities
using CSV, DataFrames, Dates
using LinearAlgebra, Statistics, StatsBase
using Random

# __ General helper functions
""" rx(x, a) = round `x` sigdigits `a` """
rx(x, a) = round(x; sigdigits=a)

""" round `x` with 2 sigdigits """
ro2(x) = round(x; sigdigits=4)

""" round `x` with 4 sigdigits """
r4(x) = round(x; sigdigits=4)

""" round `x` with 5 sigdigits """
r5(x) = round(x; sigdigits=5)

# __ Dates related function
# full_date() = Dates.format(now(), "yyyy-mm-dd-HH-MM-SS")
# time_now() = Dates.format(now(), "HH-MM-SS")
# date_now() = Dates.format(now(), "yyyy-mm-dd") # same as now()

# __ Data related functions
"""
    transformation_asinh(x)
Transformation for electricity prices and data
"""
function transformation_asinh(x)
    a = median(x)
    b = 1.4826f0 * mad(x)
    y = asinh.((x .- a) ./ b) .|> Float32
    return y, a, b
end

"""
    inv_transformation_sinh(y, a, b)
Inverse to transformation for electricity prices and data
"""
function inv_transformation_sinh(y, a, b)
    return(sinh.(y) .* b) .+ a
end

"""
    even_log_scale_generator(a = 0, b = 100, splits = 10)
Even spaced log scale of parameters between two values
"""
function even_log_scale_generator(a = 0, b = 100, splits = 10)
    # zb = log(b)                    # initial axis length scaled to log
    az = LinRange(a, log(b), splits)  # dividing the scaled axis in splits
    # a = exp(a)                     # going back to unscaled values
    return exp.(az)
end

"""
    even_log_scale(a=0.005f0, b=0.1f0, splits=10)
Even log scale range of parameters
"""
even_log_scale(a=0.005f0, b=0.1f0, splits=10)
function even_log_scale(a=0.005f0, b=0.1f0, splits=10)
    return exp.(LinRange(log(a), log(b), splits))
end


# __ Estimation related functions
"""
    prediction_intervals_AB(ci, yhat, empq, yt; grains=400)
Find estimated quantiles, `Q(α)`

- given `ci` α-levels, original empirical qunatiles, fit of distribution
    - time-series up to t-1, or before OOS
    - Chebyshev Interpolation
    - having interpolated full quantiles
- yhat is estimate of given empirical quantile
- granularize the space between the quantiles and estimates,
- then find closes quantile value according to probability I want (0.05, 0.95)
"""
function prediction_intervals_AB(ci, yhat, empq, yt; grains=400)

    # _ Working with full time series, should have been according to train data min/max
    a = minimum(yt)
    b = maximum(yt)
    # _ slightly add/sub: quantiles -> [0, τ₁, ... τⱼ, 1]
    #       works with all (positive and negative) time series
    empq1 = [-(a/10)*sign(a) + a; empq; b + sign(b)*(b/10)]
    yhat = [0.0; yhat; 1.0]
    range_x = collect(LinRange((empq1[1] - 0.01), (empq1[end] + 0.01), grains))
    # _ Interpolate CDF, Fritsch and Carlson (1980)
    int_cdf = [interpol_points(empq1, yhat, i) for i in range_x] |> vec
    # _ Retrive quantiles for given ci (α probabilities)
    preds_interpol = Float64[]
    for ci_i in ci
        pos_i = findfirst(x -> x >= ci_i, int_cdf)
        # problem of pos_i-1 if pos_i=1
        pos_i == 1 ? pos_i += 1 : nothing
        #
        xa, xb = range_x[pos_i-1], range_x[pos_i] # quantiles we want are Y
        ya, yb = int_cdf[pos_i-1], int_cdf[pos_i] # cdf 0.01:0.99 is X
        append!(preds_interpol, linear_interpol(ci_i, ya, xa, yb, xb))
    end

    return preds_interpol |> permutedims .|> Float32
end

"""
    linear_interpol(x, xa, ya, xb, yb)
Basic linear interpolation given two points, finding the one between
"""
function linear_interpol(x, xa, ya, xb, yb)
    y = ya + (yb - ya) * ((x - xa) / (xb - xa))
    return y
end


"""
    interpol_points(rk, Frk, x)
Interopolate cdf : F(x)
- Anatolyev and Barunik (2019) algorithm of Fritsch and Carlson (1980).

# Example:
interpol_points(emp_quantiles, cdf_hat, x)
int_cdf = [interpol_points(empq_tr, h_prob, i) for i in range_x] # full cdf over x more granular
"""
function interpol_points(rk, Frk, x)

    n = length(rk)
    drk = rk[2:n] .- rk[1:n-1]
    dFrk = Frk[2:n] .- Frk[1:n-1]
    Dk = dFrk ./ drk
    ms = [Dk[1]; (Dk[1:n-2] .+ Dk[2:n-1]) ./ 2; Dk[n-1]]
    alpha_s = ms[1:n-1] ./ Dk[1:n-1]
    beta_s = ms[2:n] ./ Dk[1:n-1]
    cir = (alpha_s .^ 2) + (beta_s .^ 2)
    ex = cir .> 9
    tau = 3.0 ./ sqrt.(cir) .* ex .+ 1 .- ex
    ms = ms .* [tau; 1] .* [1; tau]

    if x < minimum(rk)
        x = minimum(rk)
    elseif x > maximum(rk)
        x = maximum(rk) .- 0.0001
    else
        x = x
    end

    if findfirst(rk .> x) == 0
        ind_u = 2
    elseif findfirst(rk .> x) == 1
        ind_u = 2
    else
        ind_u = findfirst(rk .> x)
    end
    x_u = rk[ind_u]
    x_l = rk[ind_u-1]
    h = x_u - x_l
    t = (x - x_l) ./ h
    fx = Frk[ind_u-1] * (2 * t^3 - 3 * t^2 + 1) .+ h * ms[ind_u-1] * (t^3 - 2 * t^2 + t) .+ Frk[ind_u] * (-2 * t^3 + 3 * t^2) .+ h * ms[ind_u] * (t^3 - t^2)

    return fx
end
