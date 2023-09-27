using HypothesisTests
using Statistics

# __ Losses and tests time series
"""
    pq_loss(yhat, yt, q)
Pinball/Quantile loss
"""
function pq_loss(yhat, yt, q)
    if yhat >= yt
        return (1 - q) * (yhat - yt)
    else
        return (q) * (yt - yhat)
    end
end

"""
    one_forecast_pinballs(fo1, yt, ciq = collect(1:99) ./ 100)
Do pinball and CRPS for one forecast set

    Returns a matrix of pinball losses for α and time (t,h),
        and CRPS for time (t,h) aggregated over αs
"""
function one_forecast_pinballs(fo1, yt, ciq = collect(1:99) ./ 100)
    pins = []
    for (i, q_i) in enumerate(ciq)
        push!(pins, pq_loss.(fo1[:,i], yt, q_i))
    end
    pinballs_mat = reduce(hcat, pins)
    crps_vec = sum(pinballs_mat; dims=2) ./ length(ciq)
    return pinballs_mat, crps_vec
end

# __ coverage but not used, check them, FIXME:
function tau_cov(q_l_tau, q_u_tau, y)
    return mean((q_l_tau .< y) .* (y .< q_u_tau))
end

function interval_score(l_tau, u_tau, y, alpha)
    avg_sum = u_tau .- l_tau
    low_part = (2/alpha) .* (l_tau .- y) .* (y .<= l_tau)
    upp_part = (2/alpha) .* (y .- u_tau) .* (y .> u_tau)
    return mean(avg_sum .+ low_part .+ upp_part)
end
