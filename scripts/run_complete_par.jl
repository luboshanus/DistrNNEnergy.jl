# __ Parallel execution DistrNNEnergy
# _ Intra-day Electricity prices distributions
using DrWatson
@quickactivate "DistrNNEnergy"

using CSV, CSVFiles, DataFrames, Dates, LinearAlgebra, Random, Statistics
using Flux, MLUtils, ProgressMeter, Hyperopt, BSON, FFTW

using Distributed
number_of_procs = 60
addprocs(number_of_procs)

# @everywhere import Pkg
# @everywhere Pkg.instantiate()

@everywhere using DrWatson
@everywhere @quickactivate "DistrNNEnergy"

@everywhere using CSV, CSVFiles, DataFrames, Dates, LinearAlgebra, Random, Statistics
@everywhere using Flux, MLUtils, ProgressMeter, Hyperopt, FFTW

# Random.seed!(111);
@everywhere task_id = rand(101:399) + minute(now()) + second(now())

@everywhere include(srcdir("utils_data.jl"))
@everywhere include(srcdir("utils_args.jl"))
@everywhere include(srcdir("utils_train.jl"))
@everywhere include(srcdir("utils_eval.jl"))

println(" Number of workers: ", nprocs())
println("\n________________________________")
println("\n  DistrNNEnergy learning")
println("    Packages loaded.")
flush(stdout)

# __ Data load on workers/processes
@everywhere x_orig = CSV.read(datadir("exp_raw", "x-de-15-20.csv"), DataFrame)
@everywhere y_orig = CSV.read(datadir("exp_raw", "y-de-15-20.csv"), DataFrame)
# _ Take only data without datetime column
@everywhere xm = x_orig[!,2:end] |> Matrix .|> Float32
@everywhere ym = y_orig[!,2:end] |> Matrix .|> Float32
# _ Data time column
@everywhere dy = y_orig[!,1]

# __ Define days to work on
T = size(ym,1)
Tt = T - 554 - 182
oos_days = collect(Tt+1:T)

# __ First day of OOS, its train and validation is used for HPO
day_do = oos_days[1]

# __ SHARED PARAMETERS INFO
println("    Loss is logitbinarycrossentropy")
println("    DATA WITH NOISE augmentation = 0.03 ")
println("  ID: $(task_id), File: \n\t", @__FILE__, "   at: ", gethostname())
println("  OOS is 1.5 years (because of QRA we evaluate 360 days), and full train 4.5 years")
# ________________________________
# __ Fixed parameters shared through estimation
shared_pars = (epochs=1000, hidden_layers=2, kfolds=2, λm=1.5f0, progbar=false, hpo_size=number_of_procs, ensembles=8, shuffle_train=true, early_stopping=15, net_output=Flux.identity, js=31, alphas=Float32.(LinRange(0.01,0.99,31)), num_tr_batches=12*30*4)
println("\n  Shared parameters: ", shared_pars, "\n")
flush(stdout)

# __ DO RESULTS
ciq = collect(1:99) ./ 100 .|> Float32
ciq_labels = "q_".*replace(string.(ciq), "."=>"_")
js_labels = "js_".*replace(string.(shared_pars.alphas .|> ro2), "."=>"_")
# __
oos_date_price_full = y_orig[oos_days,:]
price_oos_full = permutedims(Matrix(oos_date_price_full[!,2:end]))[:] .|> Float32
hours_oos_full = permutedims(DateTime.(oos_date_price_full[!,1]) .+ repeat(Hour.(collect(0:23)'), size(oos_date_price_full,1),1))[:]
# __ PREPARED resutls dataframe with zeros, to fill in quantiles
oos_res_hours = [DataFrame(:datetime => hours_oos_full) DataFrame(:price => price_oos_full) DataFrame(zeros(Float32, size(price_oos_full,1), 99), ciq_labels)]
oos_res_probs_hours = [DataFrame(:datetime => hours_oos_full) DataFrame(:price => price_oos_full) DataFrame(zeros(Float32, size(price_oos_full,1), shared_pars.js), js_labels)]
oos_res_losses = []
oos_res_probs = []
ins_hpo = Dict()

# __ COMPUTING LOOP OF HOURS
for h_hour in 1:24
    println(" 🚙 Doing Hour ", h_hour, " __ \t", now())
    out_hpo = nothing
    out_oo = nothing
    out_dict = nothing
    # __ run Hyperopt get best pars
    flush(stdout)
    out_hpo = day_i_run(day_do, h_hour, xm, ym, true; shared_pars...);
    GC.gc()

    # __ run Ensembles get quantiles and losses
    flush(stdout)
    best_pars = (out_hpo.params .=> out_hpo.minimizer)
    println("    > Best pars: ", best_pars)
    println("    > Min loss HPO: ", out_hpo.minimum)
    println("   PMAP rolling days, start: ", now())
    flush(stdout)
    # __ Parallel map of OOS day on processor, for each day and hour, # of ensembles with different initialization
    @time out_oo = pmap(day_t -> day_i_run(day_t, h_hour, xm, ym, false; shared_pars..., best_pars...), oos_days);

    # __ Fill in by hours
    println("   Fillling results into array, hour ", h_hour, "  ", now())
    flush(stdout)
    # __ Saving into data FRAME
    edit_h = h_hour:24:size(oos_res_hours,1)
    h_predict_oos = reduce(vcat, first.(out_oo))
    oos_res_hours[edit_h[1:size(h_predict_oos,1)], 3:end] .= h_predict_oos
    # _ Probabilities
    h_predict_oos_probs = reduce(vcat, permutedims.(mean.(last.(first.(out_oo,2)))))
    oos_res_probs_hours[edit_h[1:size(h_predict_oos_probs,1)], 3:end] .= h_predict_oos_probs
    # __ Saving files
    task_url_fold = "run-$(task_id)"
    wsave(datadir("exp_pro", task_url_fold, "$(task_url_fold)_qt.csv"), oos_res_hours)
    wsave(datadir("exp_pro", task_url_fold, "$(task_url_fold)_pr.csv"), oos_res_probs_hours)
    # __ Saving Hyperoptimization, Ensebles and Losses
    push!(oos_res_losses, last.(out_oo))
    push!(ins_hpo, "h".*string(h_hour) => best_pars)
    out_dict = Dict(:hpos => ins_hpo, :losses => oos_res_losses, :time_done => now(), :shared_pars => shared_pars, :task_id => task_id, :task_url => task_url_fold)
    wsave(datadir("exp_pro", task_url_fold, "$(task_url_fold)_info.bson"), out_dict)
    GC.gc()
    println("  >> Saved hour $(h_hour). Task ID: ", task_id, "\n")
end

println("  >> Saved and done. Task ID: ", task_id)
