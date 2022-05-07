using Random

"""
    single_loss(nnps, AEVs, IDs, true_energy)

Calculate a single data point's contribution to the overall loss function.
"""
function single_loss(nnps, AEVs::Matrix{Float32}, IDs::Vector{Int},
                        true_energy::Float32) :: Float32
    energy_fit = nnp_energy(nnps, AEVs, IDs)
    return energy_fit - true_energy
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

function norm(v) :: Float32
    return sqrt(sum( v.^2 ))
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


function train_full!(nnps, train_loss_func, val_loss_func, ps, opt, train_batches,
                      train_dataset, val_dataset, model_filename::String, log_filename::String)
    start_time = time()
    val_regression_counter::Int = 0  # training stops when it reaches max_counter
    max_counter::Int = 100
    old_val_loss = val_loss_func(val_dataset)
    min_val_loss = old_val_loss  # minimum validation loss observed so far
    train_loss_value = val_loss_func(train_dataset)
    epoch = 0  # epoch number counter

    # log header and initial state
    train_logger(log_filename, "Epoch", "Train loss", "Validation loss", "Time elasped (sec)";
                    mode="w")
    train_logger(log_filename, epoch, train_loss_value, old_val_loss, time() - start_time)

    while val_regression_counter < max_counter
        epoch += 1
        train_one_epoch!(train_loss_func, ps, train_batches, opt)
        new_val_loss = val_loss_func(val_dataset)
        train_loss_value = val_loss_func(train_dataset)

        # check and update counter
        if new_val_loss > old_val_loss  # model got worse
            val_regression_counter += 1
        elseif new_val_loss < min_val_loss  # model hit a new best!
            val_regression_counter = 0  # reset counter
            min_val_loss = new_val_loss
            save_nnps(model_filename, nnps)  # save new best model just in case
        end

        old_val_loss = new_val_loss

        # log current state
        train_logger(log_filename, epoch, train_loss_value, old_val_loss, time() - start_time)
    end
end


"""
    train_nnps!(nnps_vector, AEVs, symbo_IDs, energies)

Trains the given vector of atomic NNPs to the true energies
"""
function train_nnps!(nnps_vector, AEVs::Vector{Matrix{Float32}},
                        symbol_IDs::Vector{Vector{Int}}, energies::Vector{Float32};
                        batch_size::Int=100, η=0.001, validation_proportion=0.1,
                        model_save_file::String="benchmarkNNPs.bson",
                        log_file::String = "training_log.csv") :: Nothing
    N = length(energies)  # number of data points
    @assert length(symbol_IDs) == N
    @assert length(AEVs) == N

    # τ::Float32 = batch_size * 10000  # parameter in ANI-1 loss function

    opt = ADAM(η, (0.9, 0.999), 1e-8)
    ps = Flux.params(nnps_vector)
    loss(batch) = total_loss_parallel(nnps_vector, batch)  # for validation
    training_loss(batch) = total_loss(nnps_vector, batch) + max_norm_reg(nnps_vector)

    data = collect(zip(AEVs, symbol_IDs, energies))
    shuffle!(data)  # randomly shuffle data
    val_num::Int = floor(N * validation_proportion)
    val_dataset = data[1:val_num]  # validation dataset
    train_dataset = data[val_num+1:end]
    train_batches = Flux.DataLoader( train_dataset, batchsize=batch_size, shuffle=true, 
                                        partial=true )
    data = nothing  # clear memory

    train_full!(nnps_vector, training_loss, loss, ps, opt, train_batches, train_dataset,
                val_dataset, model_save_file, log_file)
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