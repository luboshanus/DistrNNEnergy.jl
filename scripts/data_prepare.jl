using DrWatson
@quickactivate

using CSV, DataFrames, Dates
using Distributions, Statistics, StatsBase

# __ Electricity data of Marcjasz, G., M. Narajewski, R. Weron, and F. Ziel (2023). Distributional neural networks for electricity price forecasting. Energy Economics 125, 106843.
# Available at Energy Economics journal website.
d0 = CSV.read(datadir("../../Shared_data/Electricity/DDNN_DE.csv"), DataFrame);
first(d0)
# DataFrameRow
#  Row │ Column1              Price    Load_DA_Forecast  Renewables_DA_Forecast  EUA      API2_Coal  TTF_Gas  Brent_oil
#      │ String31             Float64  Float64           Float64                 Float64  Float64    Float64  Float64
# ─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#    1 │ 2015-01-01 00:00:00    25.02           41841.9                 8364.61     7.27      54.67   21.296      47.38


d0.Column1 = DateTime.(replace.(d0.Column1, " " => "T"));
println("Size: ", size(d0), " Days ", size(d0,1) ÷ 24)

# _ Create datasets, loop by days
days_x = []
x = []
y = []
for d in axes(d0, 1)[7*24:24:end-24]
    dayi = []
    push!(days_x, Date.(d0[d,:Column1]))
    # append!(dayi, d0[(d-1*24+1):(d),:Column1]) # date and hours check
    append!(dayi, d0[(d-1*24+1):(d),:Price])
    append!(dayi, d0[(d-2*24+1):(d-1*24),:Price])
    append!(dayi, d0[(d-3*24+1):(d-2*24),:Price])
    append!(dayi, d0[(d-7*24+1):(d-6*24),:Price])
    append!(dayi, d0[d+1:d+24, 3])
    append!(dayi, d0[(d-1*24+1):(d), 3])
    append!(dayi, d0[(d-7*24+1):(d-6*24), 3])
    append!(dayi, d0[d+1:d+24, 4])
    append!(dayi, d0[(d-1*24+1):(d), 4])
    append!(dayi, d0[(d-3*24+1), 4])
    append!(dayi, d0[(d-3*24+1), 5])
    append!(dayi, d0[(d-3*24+1), 6])
    append!(dayi, d0[(d-3*24+1), 7])
    append!(dayi, dayofweek(d0[d, 1]) .== 1:7)
    push!(x, dayi)
    push!(y, d0[(d+1):(d+24), :Price])
end

# _ DataFrames final with :date
xtrain = [DataFrame("date" => Date.(days_x)) DataFrame(permutedims(reduce(hcat, x)) .* 1.0, :auto)]
ytarget = [DataFrame("date" => days_x .+ Day(1)) DataFrame(permutedims(reduce(hcat, y)), "h".*string.(collect(0:23)))]

describe(xtrain)
describe(ytarget)

# CSV.write(datadir("exp_raw", "x-de-15-20.csv"), xtrain)
# CSV.write(datadir("exp_raw", "y-de-15-20.csv"), ytarget)
