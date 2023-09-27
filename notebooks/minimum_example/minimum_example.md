# Minimum example DistrNNEnergy.jl

This markdown provides a readable but minimum example using the approach from the paper *"Learning Probability Distributions of Intraday Electricity Prices"*, Barunik and Hanus (2023).

The same instructions can be found in the script `scripts/simple_run.jl`. And full replication of the learning of DistrNN reported in the paper is in the file `scripts/run_complete_par.jl`. Please bear in mind that the full run takes more than 24 hours when using 60 cpu cores.

## Learning Probability Distributions of Intra-day Electricity Prices


```julia
using DrWatson
@quickactivate "DistrNNEnergy"
```

Load all files with code


```julia
# __ Include functions
include(srcdir("utils_data.jl"));
include(srcdir("utils_args.jl"));
include(srcdir("utils_train.jl"));
include(srcdir("utils_eval.jl"));
```

## Load prepared data

The two `.csv` files are prepared before the estimation. It uses the code provided `data_prepapre.jl` in `/scripts`.


```julia
# __ Data load on workers/processes
x_orig = CSV.read(datadir("exp_raw", "x-de-15-20.csv"), DataFrame);
y_orig = CSV.read(datadir("exp_raw", "y-de-15-20.csv"), DataFrame);
```


```julia
x_orig[1:2, 1:10]
```




<div><div style = "float: left;"><span>2×10 DataFrame</span></div><div style = "clear: both;"></div></div><div class = "data-frame" style = "overflow-x: scroll;"><table class = "data-frame" style = "margin-bottom: 6px;"><thead><tr class = "header"><th class = "rowNumber" style = "font-weight: bold; text-align: right;">Row</th><th style = "text-align: left;">date</th><th style = "text-align: left;">x1</th><th style = "text-align: left;">x2</th><th style = "text-align: left;">x3</th><th style = "text-align: left;">x4</th><th style = "text-align: left;">x5</th><th style = "text-align: left;">x6</th><th style = "text-align: left;">x7</th><th style = "text-align: left;">x8</th><th style = "text-align: left;">x9</th></tr><tr class = "subheader headerLastRow"><th class = "rowNumber" style = "font-weight: bold; text-align: right;"></th><th title = "Date" style = "text-align: left;">Date</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th><th title = "Float64" style = "text-align: left;">Float64</th></tr></thead><tbody><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">1</td><td style = "text-align: left;">2015-01-07</td><td style = "text-align: right;">24.97</td><td style = "text-align: right;">24.65</td><td style = "text-align: right;">24.68</td><td style = "text-align: right;">25.06</td><td style = "text-align: right;">26.1</td><td style = "text-align: right;">26.95</td><td style = "text-align: right;">31.43</td><td style = "text-align: right;">45.98</td><td style = "text-align: right;">47.91</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">2</td><td style = "text-align: left;">2015-01-08</td><td style = "text-align: right;">21.92</td><td style = "text-align: right;">19.35</td><td style = "text-align: right;">16.11</td><td style = "text-align: right;">14.93</td><td style = "text-align: right;">15.0</td><td style = "text-align: right;">19.04</td><td style = "text-align: right;">26.17</td><td style = "text-align: right;">38.34</td><td style = "text-align: right;">37.7</td></tr></tbody></table></div>




```julia
# __ Take only data without datetime column
xm = x_orig[!,2:end] |> Matrix .|> Float32
ym = y_orig[!,2:end] |> Matrix .|> Float32
# __ Data time column
dy = y_orig[!,1];
```


```julia
# __ Define days to work on
T = size(ym,1)
Tt = T - 360 - 180*1
oos_days = collect(Tt+1:T);
println("First OOS obs: ", oos_days[1], ", Last OOS obs: ", oos_days[end], "  days")
```

    First OOS obs: 1646, Last OOS obs: 2185  days



```julia
# __ One day one hour of OOS
day_t = oos_days[1]
h_hour = 12;
```

### Hyperoptimization run


```julia
# __ Fixed parameters shared through estimation
# _ original
# shared_pars = (epochs=350, hidden_layers=2, kfolds=7, λm=1.5f0, progbar=false, hpo_size=60, ensembles=8, shuffle_train=true, early_stopping=15, net_output=Flux.identity, js=31, alphas=Float32.(LinRange(0.01,0.99,31)), num_tr_batches=12*4+6)

# _ faster (not precise, just to see if the code is working)
shared_pars = (
    epochs=350, hidden_layers=2, kfolds=2, λm=1.5f0, progbar=false, hpo_size=5, ensembles=4, 
    shuffle_train=true, early_stopping=15, net_output=Flux.identity, 
    js=21, alphas=Float32.(LinRange(0.01,0.99,21)), num_tr_batches=12*4+6,
    );
```

Run hyperoptimization:


```julia
# __ If hyperoptimization, true
# _ For two hidden layers we do not need \phi_2 to be optimised because it is not employed. If number of layers is bigger, it will be optimized.
@time out_hpo = day_i_run(day_t, h_hour, xm, ym, true; shared_pars...);
```

      ▓  HPO Starts   now 2023-09-27T14:37:42.931
    1 	(η = 0.003f0, λ = 1.0f-5, ϕ = 0.0f0, ϕ2 = 0.25f0, nodes = 80, nodes2 = 56, batch_size = 32, act_fun = NNlib.relu, act_fun2 = tanh)
    2 	(η = 0.0012819f0, λ = 0.001f0, ϕ = 1.0f0, ϕ2 = 0.75f0, nodes = 104, nodes2 = 32, batch_size = 32, act_fun = tanh, act_fun2 = NNlib.relu)
    3 	(η = 0.00054772f0, λ = 0.01f0, ϕ = 0.25f0, ϕ2 = 0.5f0, nodes = 32, nodes2 = 80, batch_size = 32, act_fun = NNlib.σ, act_fun2 = NNlib.σ)
    4 	(η = 0.0001f0, λ = 1.0f-6, ϕ = 0.5f0, ϕ2 = 1.0f0, nodes = 56, nodes2 = 104, batch_size = 32, act_fun = NNlib.softplus, act_fun2 = NNlib.softplus)
    5 	(η = 0.00023403f0, λ = 0.0001f0, ϕ = 0.75f0, ϕ2 = 0.0f0, nodes = 128, nodes2 = 128, batch_size = 32, act_fun = NNlib.relu, act_fun2 = NNlib.relu)
      ▓  HPO Finished now 2023-09-27T14:42:14.524
    279.328373 seconds (296.94 M allocations: 65.807 GiB, 5.62% gc time, 20.81% compilation time)


Note: To save time, one can run it in parallel, the hyperoptimisation function is ready for multiple core, one just need to load all data and variables on number of workers that those can work with it. See `scripts/run_complete_par.jl`.

#### Select best parameters set


```julia
#println(out_hpo)
println("Minimum: ", minimum(out_hpo), "   Best pars: ", out_hpo.minimizer)
```

    Minimum: 0.17415866   Best pars: (32, NNlib.relu, tanh, 80, 56, 0.0f0, 0.25f0, 0.003f0, 1.0f-5)



```julia
# _ Best parameters from the hyperoptimization
best_pars = (out_hpo.params .=> out_hpo.minimizer)
```




    (:batch_size => 32, :act_fun => NNlib.relu, :act_fun2 => tanh, :nodes => 80, :nodes2 => 56, :ϕ => 0.0f0, :ϕ2 => 0.25f0, :η => 0.003f0, :λ => 1.0f-5)



### OOS for one day and one hour, simple run

This and above function is run in parallel loop in the script over `hours=1:24` and days of OOS `oos_days`.


```julia
# __ Doing OOS day estimation of ensembles, hyperopt=false
@time out_d_h = day_i_run(day_t, h_hour, xm, ym, false; shared_pars..., best_pars..., verbose=true);
```

    Doing Ensembles
     80.146380 seconds (79.95 M allocations: 29.476 GiB, 9.55% gc time, 3.56% compilation time)


Predictions:


```julia
# predicted quantiles ensemble
out_d_h[1]
```




    1×99 Matrix{Float32}:
     40.2176  42.7865  44.5437  45.3454  …  68.5778  70.7899  75.0124  89.5392




```julia
# predicted probabilities of top N/2 ensembles
out_d_h[2]
```




    2-element Vector{Any}:
     Float32[0.00022274487; 0.00031156492; … ; 0.92451644; 0.9853919;;]
     Float32[0.00017681543; 0.0003141111; … ; 0.88653064; 0.9884841;;]




```julia
# validation losses of all ensembles
out_d_h[3]
```




<div><div style = "float: left;"><span>4×2 DataFrame</span></div><div style = "clear: both;"></div></div><div class = "data-frame" style = "overflow-x: scroll;"><table class = "data-frame" style = "margin-bottom: 6px;"><thead><tr class = "header"><th class = "rowNumber" style = "font-weight: bold; text-align: right;">Row</th><th style = "text-align: left;">id</th><th style = "text-align: left;">loss</th></tr><tr class = "subheader headerLastRow"><th class = "rowNumber" style = "font-weight: bold; text-align: right;"></th><th title = "Any" style = "text-align: left;">Any</th><th title = "Any" style = "text-align: left;">Any</th></tr></thead><tbody><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">1</td><td style = "text-align: left;">1</td><td style = "text-align: left;">0.150924</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">2</td><td style = "text-align: left;">2</td><td style = "text-align: left;">0.176748</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">3</td><td style = "text-align: left;">3</td><td style = "text-align: left;">0.160596</td></tr><tr><td class = "rowNumber" style = "font-weight: bold; text-align: right;">4</td><td style = "text-align: left;">4</td><td style = "text-align: left;">0.168905</td></tr></tbody></table></div>



True values:


```julia
ym[day_t,h_hour]
```




    53.99f0



Evaluation using CRPS, pinball loss:


```julia
pq_mat, crps_vec = one_forecast_pinballs(out_d_h[1], ym[day_t,h_hour], collect(1:99) ./ 100)
```




    ([0.13772418975830078 0.2240697479248047 … 0.42044853210449257 0.35549243927001983], [0.6366741631247781;;])




```julia
pq_mat
```




    1×99 Matrix{Float64}:
     0.137724  0.22407  0.283389  0.345785  …  0.503997  0.420449  0.355492




```julia
pq_mat |> mean
```




    0.6366741631247781




```julia
crps_vec
```




    1×1 Matrix{Float64}:
     0.6366741631247781



### Plots of results


```julia
# __ Plot results
include(srcdir("utils_plots.jl"));
```


```julia
# _ probabilities
plt_prbs = plot(1:shared_pars.js, out_d_h[2], title="CDFs vs Probability levels", l=1, m=2, msw=0, xlabel="j=1:p", ylabel="Probability predictions", label="")
```




    
![svg](output_35_0.svg)
    




```julia
# _ quantile function
plt_qts = plot(out_d_h[1]', title="Quantile forecasts", l=1, m=2, msw=0, xlabel="Probabilities (α)", ylabel="Price", label="")
plt_qts = hline!([ym[day_t,h_hour]], label="True price")
```




    
![svg](output_36_0.svg)
    



Plot pinball loss for day_t,h_hour


```julia
plt_pq = plot(collect(1:99) ./ 100, pq_mat', ylabel="Pinball loss", xlabel="Probability (α)")
```




    
![svg](output_38_0.svg)
    




```julia

```
