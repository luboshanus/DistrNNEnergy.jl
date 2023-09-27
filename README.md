# DistrNNEnergy.jl

This repository supports the paper Baruník & Hanus (2023) [arxiv].
We provide functions and full exercise code to obtain the results. The results might not be exactly the same numerically, since it depends on the random seed and initializations of neural network parameters.

The process how to activate and use this project code is below. The code is build in `Julia v1.8.5`.

How to work with the code in julia [below](#how-to-install-and-use-this-package).

## Learning Distribution of Intra-day Electricity Prices


Open a file you want to run, or just run bash commands to execute whole code from terminal. 

The complete script saves results into `.csv` and `.bson` files into folder `data/exp_pro`.

### Simple estimation (one day and one hour)

We provide a working minimum example at [notebooks/minimum_example/minimum_example.md](./notebooks/minimum_example/minimum_example.md)

One can train the network for one day and one hour using `run_simple.jl`.
The setup in the simple file is as follows. The lines below are basically disaggregated the code from file with figures and matrices.

### Full run

RUN in terminal/bash:

```bash
# _ full file to run, takes 24-36 hours on 60 cores:
julia scripts/run_complete_par.jl > data/temp_"`date +%FT%H%M`".txt

# _ One can use nohup to run the code in background (nohup and `&`):
nohup julia scripts/run_complete_par.jl > data/temp_"`date +%FT%H%M`".txt &
```

## How to install and use this package

*DrWatson manual*:
This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/) to make a reproducible scientific project named
> DistrNNEnergy

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:

```julia
using DrWatson
@quickactivate "DistrNNEnergy"
```

which auto-activate the project and enable local path handling from DrWatson.

## TODO

- Polish code
- Upload LEAR QRA in Julia, now we ask to use `epftoolbox` written in python.
  - Jesus Lago, Grzegorz Marcjasz, Bart De Schutter, Rafał Weron. “Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark”. Applied Energy 2021; 293:116983.

### Notes

Please, be aware that during the execution ensembles estimation for individual hours, hence days in the loop, there might occur a memory overflow. Although, tt does not happen often, it may happen. If so, just start the loop over hours `1:24` at hour when it happened, like `14:24`.
