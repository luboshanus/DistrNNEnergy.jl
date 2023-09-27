using DrWatson
using CSV, DataFrames, Dates
using Flux, MLUtils, Hyperopt, FFTW
using Statistics, LinearAlgebra, StatsBase

""" Monotonicity failure if difference is positive """
monofail(prob) = Float32.(relu.(prob[1:end-1, :] .- prob[2:end, :]))

""" Checks how many cdf fails monotonicity """
checkmono(prob) = floor(Int32, sum(sign.(monofail(prob))))

""" Error of not monotonious CDF """
monotonicity_error(y_pred) = sum(relu.(y_pred[1:end-1, :] .- y_pred[2:end, :]))

""" Count monotonicity crossings of CDF """
monotonicity_count(y_pred) = sum((y_pred[1:end-1, :] .- y_pred[2:end, :]) .> 0)

"""
    day_i_run(di, h, xfull, yfull, hyperoptim=false; kws...)
Train and does Hyperoptimization or Forecasts OOS one day `di` and one hour `h`

- hyperoptim=false # does OOS
- hyperoptim=true # hyperoptimization in parallel
"""
function day_i_run(di, h, xfull, yfull, hyperoptim=false; kws...)

    # __ IT DOES run one day fit
    # Approach of daily forecast is similar to epftoolbox and Marcjasz et al. (2023)
    # - Jesus Lago, Grzegorz Marcjasz, Bart De Schutter, Rafał Weron. “Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark”. Applied Energy 2021; 293:116983.
    # - Marcjasz, G., M. Narajewski, R. Weron, and F. Ziel (2023). Distributional neural networks for electricity price forecasting. Energy Economics 125, 106843.

    # __ Xfull and Yfull are shifted by one day in .CSV files
    # __ first definitions
    # __ then computations, below

    #""" Instantiate model """
    function inst_model(in_dim, out_dim, args)
        lays = []
        if args.hidden_layers > 1
            for hi in 2:args.hidden_layers
                if hi==args.hidden_layers
                    push!(lays, Flux.Dense(args.nodes => args.nodes2, args.act_fun2))
                    # push!(lays, Flux.Dropout(args.ϕ2))
                else
                    push!(lays, Flux.Dense(args.nodes2 => args.nodes2, args.act_fun2))
                    push!(lays, Flux.BatchNorm(Int(args.nodes2)))
                    push!(lays, Flux.Dropout(args.ϕ2))
                end
            end
            push!(lays, Flux.Dense(args.nodes2, out_dim, args.net_output)) # output layer
        else
            push!(lays, Flux.Dense(args.nodes, out_dim, args.net_output)) # output layer
        end
        network = Flux.Chain(
            Flux.Dense(in_dim => args.nodes, args.act_fun),
            Flux.BatchNorm(Int(args.nodes)),
            Flux.Dropout(args.ϕ),
            lays...,
        )
        return network
    end

    # loss(x, y; model) = Flux.binarycrossentropy(model(x), y)
    function loss(x, y; model = model, λm = 1.5f0)
        ŷ = model(x)
        mono_pen = monotonicity_error(sigmoid.(ŷ))
        # return Flux.binarycrossentropy(ŷ, y) + λm * mono_pen # less stable
        return Flux.logitbinarycrossentropy(ŷ, y) + λm * mono_pen
    end

    #""" Optimiser for Flux >0.13
    function create_optimiser(η, λ)
        if λ > 0
            return Flux.ADAMW(η, (9f-1, 9.99f-1), λ)
        else
            return Flux.Adam(η)
        end
    end
    # """ augment data with nois without last feature """ # 0.03f0
    augment_n_1(x) = x .+ ([repeat([0.03f0], (size(x,1)-1)); 0.0f0] .* randn.(Float32))

    #""" Train function for Cross-validation and  Hyper-optimizaiton """
    function train_kfolds(xtrain, ytrain; kws...)

        # _ Load arguments = specific changes to each run, as HPO for instance
        args = Args(; kws...)

        # __ We can shuffle all data, train + validation
        # _ partitioning it into training- and test-set.
        if args.shuffle_train
            Xs, Ys = shuffleobs((Float32.(xtrain), Float32.(ytrain)))
        else
            Xs, Ys = Float32.(xtrain), Float32.(ytrain)
        end

        # __
        # cv_data is for train and validation
        cv_data = (Xs, Ys)

        # _ empty arrays to save data from validation
        cv_tr_loss = []
        cv_vl_loss = []
        models = []

        # __ Start progress bar
        if args.progbar
            pg = Progress(args.epochs * args.kfolds, barlen=30)
        end

        # _ Next we partition the data using a k-fold scheme.
        # _ Training, k-folds function does it proportionally
        for (train_data, val_data) in kfolds(cv_data; k=args.kfolds)
            # __
            # Same runs on different with epochs / kfolds
            # __ Reset and Instantiate model
            model = nothing
            opt_state = nothing
            model = inst_model(size(train_data[1],1), size(train_data[2],1), args)
            opt_state = Flux.setup(create_optimiser(args.η, args.λ), model)

            # _ We apply a lazy transform for data augmentation
            # train_data = mapobs(xy -> (xy[1] .+ 0.05f0 .* randn.(Float32), xy[2]),  train_data) # 0.1f0
            train_data = mapobs(xy -> (augment_n_1(xy[1]), xy[2]),  train_data) # N(0,0.03f0)

            # __ Saving arrays
            tr_losses = []
            vl_losses = []
            best_vl_loss = Inf
            best_model = deepcopy(model)
            monofails = Int32[]
            early_count = 0
            mono_once = false
            # _ Loop of training
            for epoch = 1:args.epochs
                Flux.trainmode!(model)
                mini_losses = []
                # Iterate over the data using mini-batches of batch_size observations each
                # ... for each fold and epoch
                for (x, y) in eachobs(train_data, batchsize=args.batch_size)
                    # ... train supervised model on minibatches here
                    val, grads = Flux.withgradient(model) do m
                        # loss(x, y; model=m)
                        loss(x, y; model=m, λm=args.λm)
                    end
                    push!(mini_losses, val)
                    Flux.update!(opt_state, model, grads[1])
                end
                # _ Validation with val_data
                push!(tr_losses, mean(mini_losses))
                Flux.testmode!(model)
                push!(vl_losses, loss(val_data[1], val_data[2]; model=model, λm=args.λm))
                append!(monofails, checkmono(sigmoid.(model(val_data[1]))))

                if best_vl_loss > vl_losses[end]
                    # _ until we reach monotonious model, we save it, then only those monotonious
                    if !mono_once
                        best_vl_loss = vl_losses[end]
                        best_model = deepcopy(model)
                    else
                        if monofails[end] == 0
                            best_vl_loss = vl_losses[end]
                            best_model = deepcopy(model)
                        end
                    end
                elseif (epoch > args.early_stopping) # _ Early-stopping, number of consecutive epochs
                    early_count = sum(diff(vl_losses[end-args.early_stopping:end]) .>= 0)
                    if early_count == args.early_stopping
                        args.verbose ? println("  Breaking patience! E: $epoch") : nothing
                        break;
                    end
                end

                # _ Progress bar show
                if args.progbar
                    sleep(0.01)
                    ProgressMeter.next!(pg; showvalues=[(:E, [epoch, args.epochs, early_count[end], monofails[end]]), (:L, [r4(tr_losses[end]), r4(vl_losses[end]), r4(best_vl_loss)])])
                end
            end
            push!(cv_vl_loss, best_vl_loss)
            push!(models, best_model)
        end
        return cv_vl_loss, models
    end

    #""" Train function with given best set and do OOS prediction ""
    function train_oos(xtrain, ytrain; kws...)

        # __ Load arguments (one model)
        args = Args(; kws...)

        # We can shuffle
        # partitioning it into training- and test-set.
        if args.shuffle_train
            Xs, Ys = shuffleobs((Float32.(xtrain), Float32.(ytrain)))
        else
            Xs, Ys = Float32.(xtrain), Float32.(ytrain)
        end

        # __ Split data for train and validation 80/20
        train_data, val_data = splitobs((Xs, Ys); at=args.split_ratio)

        # __ Empty arrays
        cv_tr_loss = []
        cv_vl_loss = []
        models = []

        # _ start progress bar
        if args.progbar
            pg = Progress(args.epochs * 1, barlen=30)
        end

        # __ Training
        # _ Init model
        model = nothing
        opt_state = nothing
        model = inst_model(size(xtrain,1), args.js, args)
        opt_state = Flux.setup(create_optimiser(args.η, args.λ), model)

        # _ We apply a lazy transform for data augmentation
        train_data = mapobs(xy -> (augment_n_1(xy[1]), xy[2]),  train_data) # N(0,0.03f0)

        # _ Saving arrays
        tr_losses = []
        vl_losses = []
        best_vl_loss = Inf
        best_model = deepcopy(model)
        monofails = Int32[]
        early_count = 0
        mono_once = false
        # _ Epochs training
        for epoch = 1:args.epochs
            Flux.trainmode!(model)
            mini_losses = []
            # _
            for (x, y) in DataLoader(train_data, batchsize=args.batch_size)
                # _ Train supervised model on minibatches here
                val, grads = Flux.withgradient(model) do m
                    loss(x, y; model=m, λm=args.λm)
                end
                push!(mini_losses, val)
                Flux.update!(opt_state, model, grads[1])
            end
            # _ Validation with val_data
            push!(tr_losses, mean(mini_losses))
            Flux.testmode!(model)
            push!(vl_losses, loss(val_data[1], val_data[2]; model=model, λm=args.λm))
            append!(monofails, checkmono(sigmoid.(model(val_data[1]))))

            if best_vl_loss > vl_losses[end]
                # _ until we reach monotonious model, we save it, then only those monotonious
                if !mono_once
                    best_vl_loss = vl_losses[end]
                    best_model = deepcopy(model)
                else
                    if monofails[end] == 0 # _ Early-stopping, number of consecutive epochs
                        best_vl_loss = vl_losses[end]
                        best_model = deepcopy(model)
                    end
                end
            elseif (epoch > args.early_stopping)
                early_count = sum(diff(vl_losses[end-args.early_stopping:end]) .>= 0)
                if early_count == args.early_stopping
                    args.verbose ? println("  Breaking patience! E: $epoch") : nothing
                    break;
                end
            end

            # _ Progress bar show
            if args.progbar
                sleep(0.01)
                ProgressMeter.next!(pg; showvalues=[(:E, [epoch, args.epochs, early_count[end], monofails[end]]), (:L, [r4(tr_losses[end]), r4(vl_losses[end]), r4(best_vl_loss)])])
            end
        end
        push!(cv_vl_loss, best_vl_loss)
        # push!(models, best_model)

        return best_model, best_vl_loss
    end

    # cv_vl, mx = train_kfolds(xtrain, yind; kws...)

    """
    par_hpo_with_cv(x_df, y_full, hpo_size=5; shared_pars, verbose=true, start_oos=DateTime("2017-10-01T00:00:00"))

    Parallel Hyperoptimization of simple train, given Args
    """
    function par_hpo_kfolds(x_data, y_data, hpo_size=5, verbose=false; kws...)
        # _ data prep
        hpo_size < 2 ? hpo_size = 2 : nothing
        println("  ▓  HPO Starts   now ", now())
        # _ HPO
        ho = @phyperopt for i=hpo_size, sampler=CLHSampler(dims=[Hyperopt.Categorical(1), Hyperopt.Categorical(4), Hyperopt.Categorical(4), Hyperopt.Continuous(), Hyperopt.Continuous(), Hyperopt.Continuous(), Hyperopt.Continuous(), Hyperopt.Continuous(), Hyperopt.Continuous()]), #Hyperopt.Continuous(),
                    batch_size = [32],
                    act_fun = [tanh, softplus, relu, sigmoid], #
                    act_fun2 = [tanh, softplus, relu, sigmoid], #
                    nodes = floor.(Int, LinRange(32,128,hpo_size)),
                    nodes2 = floor.(Int, LinRange(32,128,hpo_size)),
                    # nodes = [64,128,256],
                    # nodes2 = [64,128,256],
                    ϕ = r5.(LinRange{Float32,Int32}(0.0, 1.0, hpo_size)),
                    ϕ2 = r5.(LinRange{Float32,Int32}(0.0, 1.0, hpo_size)),
                    η = r5.(even_log_scale(0.0001f0, 0.003f0, hpo_size)),
                    λ = r5.(even_log_scale(0.000001f0, 0.01f0, hpo_size))
            verbose ? println(i, " \t", (η=η, λ=λ, ϕ=ϕ, ϕ2=ϕ2, nodes=nodes, nodes2=nodes2, batch_size=batch_size, act_fun=act_fun, act_fun2=act_fun2)) : nothing
            cost = mean(train_kfolds(x_data, y_data; η=η, λ=λ, ϕ=ϕ, ϕ2=ϕ2, nodes=nodes, nodes2=nodes2, batch_size=batch_size, act_fun=act_fun, act_fun2=act_fun2, kws...)[1])
        end
        println("  ▓  HPO Finished now ", now())
        return ho
    end

    # __ Fourier features, NOT used now
    function make_fourier_features(x)
        fft_signal = []
        for i in collect(3:2:11) # number of fouriers, [1] = freq[=0]
            tran_x = fft(x)
            spc_abs = abs2.(tran_x)
            cut_freq = sort(spc_abs[2:end], rev=true)[i]
            # sig_freq = findall(spc_abs .>= cut_freq)
            not_sig_freq = findall(spc_abs .< cut_freq)
            # plot(spc_abs.|>log)
            tran_x[not_sig_freq] .= 0.0
            # tran_x[[1]] .= 0.0
            push!(fft_signal, real.(ifft(tran_x)))
        end
        return fft_signal
    end

    # __ Arguments used to setup the run
    argsx = Args(; kws...)
    cal_window = argsx.num_tr_batches * 30 # days of training and validation
    oos_span = 1 # Not this: args.oos_span,

    # __ Standardize data
    # xfull = mapslices(x -> first(transformation_asinh(x)), xfull; dims=1)
    # _ Not standardize dummies (last column of data): days 1,...,7
    xfull[:,1:end-1] = mapslices(x -> first(transformation_asinh(x)), xfull[:,1:end-1]; dims=1)

    # __ Add fourier smoothed data, not used
    # fourier_feat = make_fourier_features(xfull[:,h])
    # xfull = [xfull reduce(hcat, fourier_feat)]

    # __ Data preparation for train and validation window used for forecasting day `di`
    xtrain = xfull[di-cal_window:di-1,:] |> permutedims .|> Float32
    ytrain = yfull[di-cal_window:di-1,:]
    # _ OOS day
    xoos = xfull[di,:] |> x -> reshape(x, :, 1) .|> Float32
    yoos = yfull[di,:] |> x -> reshape(x, :, 1) # row of 24 hours

    # __ Prepare target variable y target
    # _ hour dependent data, for HOUR = h
    # _ Winsorise data extremes, 0.1% of data
    yh, ya, yb = transformation_asinh(collect(winsor(ytrain[:, h], prop=0.001)))
    # _ transform without winsorisaton
    # yh, ya, yb = transformation_asinh(ytrain[:,h])

    # _ Prepare Y indicator for train and validation
    empq_h = quantile(yh, argsx.alphas) # Hour specific quantiles_α_j
    yind = reduce(hcat, [yh[t] .< empq_h for t in axes(yh,1)])

    # __ Choose whether we do Hyperoptimization or OOS forecasts
    # _ this uses the two `train` functions defined above, and parallel hyper-optimization
    if hyperoptim
        argsx.verbose ? println("Doing Hyperoptimization") : nothing
        # _ hyperoptimization
        # _ cv folds
        # _ this is in the hyperopt: cv_vl, mx = train_kfolds(xtrain, yind; kws...)
        hpo = par_hpo_kfolds(xtrain, yind, argsx.hpo_size, true; kws...)
        # returns HPO pars
        return hpo
    else
        argsx.verbose ? println("Doing Ensembles") : nothing
        # _ do ensembles of OOS day
        ens_probs = []
        ens_vl_losses = []
        # _ train a number of ensembles
        for i in 1:argsx.ensembles
            mx, b_loss = train_oos(xtrain, yind; kws...)
            push!(ens_probs, sigmoid.(mx(xoos)))
            push!(ens_vl_losses, b_loss)
        end
        # _ Sort ensembles given validation loss and take 1/2 of top ones
        ens_losses = DataFrame([collect(1:size(ens_probs,1)) ens_vl_losses], [:id, :loss])
        nE = size(ens_losses,1) ÷ 2 # half of ensembles
        # nE = size(ens_losses,1) - 1 # N-1 ensembles = all without the worst
        take_ens = Int.(sort(ens_losses, :loss)[1:nE, 1])
        # _ Interpolate and Inverse CDF to get collection of Quantiles, Q(α)
        ciq = collect(1:99) ./ 100
        qt = map(yo -> prediction_intervals_AB(ciq, sort(yo[:,1]), empq_h, yh; grains=400), ens_probs[take_ens])
        qt_destd = map(qi -> inv_transformation_sinh(qi, ya, yb), qt)
        qt_ciq = mean(qt_destd)
        # _ Inverse of Probs will be saved into DataFrame, rows by DAY and by HOUR and cols by Quantiles
        return qt_ciq, ens_probs[take_ens], ens_losses
    end
end
