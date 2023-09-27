# __ Simple execution for one day and one hour
# _ Intra-day Electricity prices distributions
# _ See readme or notebooks, this is the script to execute step by step
# _ runs on one core (possibly multithreaded) and it takes time

using DrWatson
@quickactivate "DistrNNEnergy"

using CSV, CSVFiles, DataFrames, Dates, LinearAlgebra, Random, Statistics
using Flux, MLUtils, ProgressMeter, Hyperopt, BSON, FFTW

# __ Start with id for saving
task_id = rand(101:399) + minute(now()) + second(now())

# __ Include functions
include(srcdir("utils_data.jl"));
include(srcdir("utils_args.jl"));
include(srcdir("utils_train.jl"));
include(srcdir("utils_eval.jl"));

# __ Data load on workers/processes
x_orig = CSV.read(datadir("exp_raw", "x-de-15-20.csv"), DataFrame);
y_orig = CSV.read(datadir("exp_raw", "y-de-15-20.csv"), DataFrame);

first(x_orig)

# __ Take only data without datetime column
xm = x_orig[!,2:end] |> Matrix .|> Float32
ym = y_orig[!,2:end] |> Matrix .|> Float32
# __ Data time column
dy = y_orig[!,1]

# __ Define days to work on
T = size(ym,1)
Tt = T - 360 - 180*1
oos_days = collect(Tt+1:T)

# __ One day one hour, first day of 00S and its first hour
day_t = oos_days[1]
h_hour = 1

# __ Fixed parameters shared through estimation
# _ original
# shared_pars = (epochs=350, hidden_layers=2, kfolds=7, λm=1.5f0, progbar=false, hpo_size=60, ensembles=8, shuffle_train=true, early_stopping=15, net_output=Flux.identity, js=31, alphas=Float32.(LinRange(0.01,0.99,31)), num_tr_batches=12*4+6)

# _ faster (not precise, just to see if the code is working)
shared_pars = (epochs=150, hidden_layers=2, kfolds=2, λm=1.5f0, progbar=false, hpo_size=5, ensembles=4, shuffle_train=true, early_stopping=15, net_output=Flux.identity, js=31, alphas=Float32.(LinRange(0.01,0.99,31)), num_tr_batches=12*4+6)

# __ If hyperoptimization, true
@time out_hpo = day_i_run(day_t, h_hour, xm, ym, true; shared_pars...);
println(out_hpo)
println("Minimum: ", minimum(out_hpo), "   Best pars: ", out_hpo.minimizer)

# _ Best parameters from the hyperoptimization
best_pars = (out_hpo.params .=> out_hpo.minimizer)

# _ OR we can provide best parameters by ourselves
# best_pars = (:batch_size => 32, :act_fun => tanh, :act_fun2 => NNlib.softplus, :nodes => 64, :nodes2 => 64, :ϕ => 0.4f0, :ϕ2 => 0.15f0, :η => 0.001f0, :λ => 0.005f0)

# __ Doing OOS day estimation of ensembles, hyperopt=false
# 28.486081 seconds
@time out_d_h = day_i_run(day_t, h_hour, xm, ym, false; shared_pars..., best_pars...);

# predicted quantiles ensemble
out_d_h[1]
# predicted probabilities of top N/2 ensembles
out_d_h[2]
# validation losses of all ensembles
out_d_h[3]

# true value of price
ym[day_t,h_hour]

# _ Evaluation using CRPS, pinball loss:
pq_mat, crps_vec = one_forecast_pinballs(out_d_h[1], ym[day_t,h_hour], collect(1:99) ./ 100)
pq_mat
crps_vec

# __ Plot results
include(srcdir("utils_plots.jl"))

# _ probabilities
plot(shared_pars.alphas, out_d_h[2], title="CDFs vs Probability levels", l=1, m=2, msw=0, xlabel="Probabilities (α)", ylabel="Probability predictions", label="")

# _ quantile function
begin
    plt_qts = plot(out_d_h[1]', title="Quantile forecasts", l=1, m=2, msw=0, xlabel="Probabilities (α)", ylabel="Price", label="")
    plt_qts = hline!([ym[day_t,h_hour]], label="True price")
end

# _ Plot pinball loss for day_t,h_hour
plt_pq = plot(collect(1:99) ./ 100, pq_mat', ylabel="Pinball loss", xlabel="Probability (α)")
