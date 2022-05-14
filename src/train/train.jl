"""
    single_loss(nnps, AEVs, IDs, true_energy)

Calculate a single data point's contribution to the overall loss function.
"""
function single_loss(nnps, AEVs::Matrix{Float32}, IDs::Vector{Int},
                        true_energy::Float32) :: Float32
    energy_fit = nnp_energy(nnps, AEVs, IDs)
    return energy_fit - true_energy
end
function d_single_loss_squared(nnps, ps, AEVs::Matrix{Float32}, IDs::Vector{Int},
                            true_energy::Float32) :: Flux.Zygote.Grads
    energy_fit, back = Flux.Zygote.pullback(() -> nnp_energy(nnps, AEVs, IDs), ps)
    # d_single_loss = Flux.gradient(() -> nnp_energy(nnps, AEVs, IDs), ps)
    d_single_loss = back(1.f0)
    temp = 2.f0 * (energy_fit - true_energy)
    return temp .* d_single_loss
end


"""
    total_loss(nnps, batch)

Calculates the loss function of the batch, given a vector of Neural Network Potentials
"""
function total_loss_parallel(nnps, batch) :: Float32
    loss = Threads.Atomic{Float64}(0.0)  # first do it in 64 bits reduce numerical errors
    Threads.@threads for i in batch
        temp::Float64 = single_loss(nnps, i...)^2
        Threads.atomic_add!(loss, temp)
    end
    return loss.value / length(batch)
end
function total_loss(nnps, batch) :: Float32
    loss::Float64 = 0
    for i in batch
        loss += single_loss(nnps, i...)^2
    end
    return loss / length(batch)
end


"""
    ani_total_loss(nnps, τ, batch)

Calculates the loss function of the batch, given a vector of Neural Network Potentials.
From the ANI-1 paper
"""
function ani_total_loss(nnps, τ::Float32, batch) :: Float32
    cumsum::Float32 = 0.
    for i in batch  # iterate through each training data point in the batch
        cumsum += single_loss(nnps, i...)^2
    end
    return τ * exp(1/τ * cumsum)
end


"""
    max_norm_reg(nnps)

My attempt to enforce a pseudo-max-norm-regularization on the loss function.

Makes derivative == `scaling` if norm of layer weights is larger than `cutoff`
"""
function max_norm_reg(nnps) :: Float32
    cutoff::Float32 = 3
    scaling::Float32 = 10000
    cumsum::Float32 = 0
    # cumsum = Threads.Atomic{Float32}(0.f0)
    # Threads.@threads for nnp in nnps
    for nnp in nnps
        for layer in nnp
            x = norm(layer.weight)
            temp = scaling * relu(x - cutoff)
            cumsum += temp
            # Threads.atomic_add!(cumsum, temp)
        end
    end
    return cumsum
    # return cumsum.value
end


"""
    norm(v)

L2-norm of a vector
"""
function norm(v) :: Float32
    return sqrt(sum( v.^2 ))
end


"""
    sum(Gs::Vector{Flux.Zygote.Grads})

Calculates the sum of a vector of Flux.Zygote.Grads types
"""
function sum!(Gs::Vector{Flux.Zygote.Grads}) :: Flux.Zygote.Grads
    result = Gs[1]
    for i in 2:length(Gs)
        result .+= Gs[i]
    end
    return result
end



"""
    train_one_epoch!(nnps_vector, loss_function, ps, dataset, opt)

Trains the given vector of Neural Network Potentials through the entire dataset on a given loss
function.
`ps` is implied to be equivalent to `Flux.params(nnps_vector)`
"""
function train_one_epoch!(loss_function, ps, dataset, opt) :: Nothing
    # counter = 0
    for batch in dataset
        gs = gradient( () -> loss_function(batch), ps )
        #=
        grad_cumsum = 0
        for (_, value) in gs.grads
            grad_cumsum += sum(value.^2)
        end
        grad_cumsum = sqrt(grad_cumsum)
        counter += 1
        @show counter, grad_cumsum
        =#
        Flux.update!(opt, ps, gs)
    end

    return nothing
end
function train_one_epoch_parallel!(nnps, ps, dataset, opt) :: Nothing
    for batch in dataset

        # parallel grad calculation
        nDataPoints = length(batch)
        partial_loss_grads = Vector{Flux.Zygote.Grads}(undef, nDataPoints)
        Threads.@threads for i in 1:nDataPoints
            datapoint = batch[i]
            partial_loss_grads[i] = d_single_loss_squared(nnps, ps, datapoint...)
        end

        # add all partial grads
        gs = sum!(partial_loss_grads) .+ gradient(() -> max_norm_reg(nnps), ps)

        Flux.update!(opt, ps, gs)

    end

    return nothing
end


function train_full!(nnps, train_loss_func, val_loss_func, ps, opt, train_batches,
                      train_dataset, val_dataset,
                      best_trained_model_filename::String,
                      best_val_model_filename::String,
                      log_filename::String)
    start_time = time()
    val_regression_counter::Int = 0  # training stops when it reaches max_counter
    max_counter::Int = 50
    new_val_loss = val_loss_func(val_dataset)
    min_val_loss = new_val_loss  # minimum validation loss observed so far
    # train_loss_value = train_loss_func(train_dataset)
    # min_train_loss = train_loss_value
    epoch = 0  # epoch number counter

    # log header and initial state
    train_logger(log_filename, "Epoch", "Validation loss", "Time elasped (sec)";
                    mode="w")
    train_logger(log_filename, epoch, new_val_loss, time() - start_time)

    while val_regression_counter < max_counter
        epoch += 1
        # @time train_one_epoch!(train_loss_func, ps, train_batches, opt)
        @time train_one_epoch_parallel!(nnps, ps, train_batches, opt)
        new_val_loss = val_loss_func(val_dataset)
        # train_loss_value = train_loss_func(train_dataset)

        # check and update counter
        if new_val_loss < min_val_loss  # model hit a new best!
            val_regression_counter = 0  # reset counter
            min_val_loss = new_val_loss
            save_nnps(best_val_model_filename, nnps)  # save new best model just in case
            println("New best validation model at epoch $epoch")
        else  # model is not the best
            val_regression_counter += 1
        end
        # if train_loss_value < min_train_loss
            # min_train_loss = train_loss_value
            # save_nnps(best_trained_model_filename, nnps)
            # println("New best trained model at epoch $epoch")
        # end

        # log current state
        train_logger(log_filename, epoch, new_val_loss, time() - start_time)
    end
end


function check_data(data::Tuple{Vector{Matrix{Float32}}, Vector{Vector{Int}}, Vector{Float32}}
                    ) :: Int
    N::Int = length(last(data))
    @assert length(first(data)) == N
    @assert length(data[2]) == N

    return N
end


"""
    train_nnps!(nnps_vector; training_set, validation_set, batch_size=100, η=0.001,
                    best_trained_save_file="bestTrainedNNPs.bson",
                    best_val_save_file="bestValNNPs.bson",
                    log_file="training_log.csv")

Trains the given vector of atomic NNPs to the true energies.

The training and validation sets must be provided in a tuple containing:
    * Vector of AEVs            :: Vector{Matrix{Float32}}
    * Vector of symbol IDs/tags :: Vector{Vector{Int}}
    * Vector of true energies   :: Vector{Float32}
"""
function train_nnps!(nnps_vector;
            training_set::Tuple{Vector{Matrix{Float32}},Vector{Vector{Int}},Vector{Float32}},
            validation_set::Tuple{Vector{Matrix{Float32}},Vector{Vector{Int}},Vector{Float32}},
            batch_size::Int=100, η=0.001,
            best_trained_save_file::String="bestTrainedNNPs.bson",
            best_val_save_file::String="bestValNNPs.bson",
            log_file::String = "training_log.csv") :: Nothing
    check_data(validation_set)
    N = check_data(training_set)  # number of data points in training set

    # τ::Float32 = batch_size * 10000  # parameter in ANI-1 loss function

    opt = ADAM(η, (0.9, 0.999), 1e-8)
    ps = Flux.params(nnps_vector)
    validation_loss(batch) = total_loss_parallel(nnps_vector, batch)
    training_loss(batch) = total_loss(nnps_vector, batch) + max_norm_reg(nnps_vector)

    val_dataset = collect(zip(validation_set...))
    train_dataset = collect(zip(training_set...))
    train_batches = Flux.DataLoader( train_dataset, batchsize=batch_size, shuffle=true, 
                                        partial=true )

    train_full!(nnps_vector, training_loss, validation_loss, ps, opt, train_batches,
                train_dataset, val_dataset, best_trained_save_file, best_val_save_file,
                log_file)

    return nothing
end

"""
    train_logger(log_filename, varargs...)

Logs various training info into a log file in CSV format.
"""
function train_logger(log_filename::String, varargs...; mode::String="a")
    open(log_filename, mode) do io
        for arg in varargs
            write(io, "$arg, ")
        end
        write(io, "\n")
    end
end