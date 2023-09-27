# DistrNNEnergy.jl

This repository supports the paper Baruník & Hanus (2023) [arxiv].
We provide functions and full exercise code to obtain the results. The results might not be exactly the same numerically, since it depends on the random seed and initializations of neural network parameters.

The process how to activate and use this project code is below. The code is build in `Julia v1.8.5`.

How to work with the code in julia below.

## How to install and use this package

### One approach to use the project

This is a simple approach how to instantiate the project and install its dependencies. This is also required before one opens the notebook example.

   1. Open a Julia console in the folder of this repository.
   2. Activate the project's environment
   ```julia
   using Pkg
   Pkg.activate(".")
   ```
   3. And instantiate the project, which will install all necessary packages and their correct versions.
   ```julia
   Pkg.instantiate()
   ```
   4. Open the script `scripts/run_simple_day_hour.jl` in your favourite editor and copy+paste or execute the code in the Julia console

**Jupyter Notebooks/Lab**: Be aware that if you want to use julia in Jupyter Notebooks or Jupyter Lab, use first need to add IJulia package to your global julia environment. In a Julia console run this:
   ```julia
   using Pkg
   Pkg.add("IJulia")
   ```
   This install the julia kernel and then you may start your Jupyter Notebook/Lab.
   
### DrWatson manual (alternative approach to the repository)

This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/) to make a reproducible scientific project.

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently. (We provide the data in `data/exp_raw`.)
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

### Other alternative

It is just to open this project in VSCode or Atom editors and it will locate your julia console at the folder once use execute/run a line of julia code from one of the scripts.

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

#### Notes

Please, be aware that during the execution ensembles estimation for individual hours, hence days in the loop, there might occur a memory overflow. Although, tt does not happen often, it may happen. If so, just start the loop over hours `1:24` at hour when it happened, like `14:24`.

## TODO

- Polish code
- Upload LEAR QRA in Julia, now we ask to use `epftoolbox` written in python.
  - Jesus Lago, Grzegorz Marcjasz, Bart De Schutter, Rafał Weron. “Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark”. Applied Energy 2021; 293:116983.
