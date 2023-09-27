# Arguments kwdef
import Flux
import Flux: tanh, relu
import Flux: ADAMW
# import Optimisers: ADAMW

# __ Create parameters set
# using Parameters
# @with_kw mutable struct Args

@Base.kwdef mutable struct Args
    seed::Int = 123             # Random seed
    # _ Data setup
    js::Int = 31                # Number of cut-offs
    alphas = LinRange{Float32,Int32}(0.01, 0.99, js)
    horz::Int = 24              # Forecasting horizon (1:h)
    seqlen::Int = 10            # Sequence length to use as input
    seqshift::Int = 10          # Shift between sequences (see utils.jl)
    oos_span::Int = 720         # Number of OOS time steps
    lags::Int = 48              # Number of t in lags for LSTM/recurrence, same as seqlen
    num_tr_batches = 12         # How much of OOS to use in training/validation part
    split_ratio::Float32 = 0.8  # Percentage of data in the train set
    batch_size::Int = 32        # on what size loss is calculated and averaged
    shuffle_train = true        # shuffle minibatches in learning,
    shuffle_valid = false       # maybe good even for seasonal data, generealizes model and its training

    # _ Recurrent net parameters
    dev = Flux.cpu              # Device: cpu or gpu
    λ::Float32 = 0.00001        # Weights decay
    η::Float32 = 0.003          # Learning rate
    ϕ::Float32 = 0.05           # Dropout p
    ϕ2::Float32 = 0.05          # Dropout p, second and next layers
    λm::Float32 = 1.0f0         # Monotonicity lambda
    opt = Flux.ADAMW            # Optimizer
    nodes = 64                  # Number of hidden nodes
    nodes2 = 64                 # Number of hidden nodes, second and next layers
    hidden_layers::Int = 2      # Number of hidden layers
    layer_type = "LSTM"         # Type of layer, should be one of LSTM, GRU, RNN
    act_fun = Flux.relu         # Activation function
    act_fun2 = Flux.relu        # Activation function, second and next layers
    net_output = Flux.sigmoid   # Output of the network

    # _ Training parameters
    epochs::Int = 150           # Number of epochs
    ensembles::Int = 10         # N of ensembles
    hpo_size::Int = 20          # N of hpo searches
    kfolds::Int = 3             # N of CV folds
    div_eta::Int = 50           # divide eta in learning, every K epoch
    patience::Int = 20          # number of epochs before breaking, TODO: could be batch_size * 2
    early_stopping::Int = 20    # sequence of epochs of not improved validation
    verbose::Bool = false       # Whether we manually log the results during training or not
    report::Int = 10            # After # epoch we log
    progbar::Bool = false       # if progress bar
    tblogger = false            # log training with tensorboard
    savepath = "exp_pro/runs/"      # results path
end
