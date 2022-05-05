using Random

"""
    single_loss(nnps, AEVs, IDs, true_energy)

Calculate a single data point's contribution to the overall loss function.
"""
function single_loss(nnps, AEVs::Matrix{Float32}, IDs::Vector{Int},
                        true_energy::Float32) :: Float32
    energy_fit = sum([ nnps[IDs[i]](AEVs[:, i])[1] for i in 1:length(IDs) ])
    return (energy_fit - true_energy)^2
end


"""
    total_loss(nnps, c, batch)

Calculates the loss function of the batch, given a vector of Neural Network Potentials
"""
function total_loss(nnps, c::Float32, batch) :: Float32
    cumsum::Float32 = 0.
    for i in batch  # iterate through each training data point in the batch
        cumsum += single_loss(nnps, i...)
    end
    loss = c * cumsum
    return loss
end


"""
    ani_total_loss(nnps, τ, batch)

Calculates the loss function of the batch, given a vector of Neural Network Potentials.
From the ANI-1 paper
"""
function ani_total_loss(nnps, τ::Float32, batch) :: Float32
    cumsum::Float32 = 0.
    for i in batch  # iterate through each training data point in the batch
        cumsum += single_loss(nnps, i...)
    end
    return τ * exp(1/τ * cumsum)
end


"""
    train_one_epoch!(nnps_vector, loss_function, ps, dataset, opt)

Trains the given vector of Neural Network Potentials through the entire dataset on a given loss
function.
`ps` is implied to be equivalent to `Flux.params(nnps_vector)`
"""
function train_one_epoch!(loss_function, ps, dataset, opt) :: Nothing
    for batch in dataset
        gs = gradient( () -> loss_function(batch), ps )
        Flux.update!(opt, ps, gs)
    end

    return nothing
end


"""
    train_nnps!(nnps_vector, AEVs, symbo_IDs, energies)

Trains the given vector of atomic NNPs to the true energies
"""
function train_nnps!(nnps_vector, AEVs::Vector{Matrix{Float32}},
                        symbol_IDs::Vector{Vector{Int}}, energies::Vector{Float32};
                        batch_size::Int=100, η=0.001, validation_proportion=0.1) :: Nothing
    N = length(energies)  # number of data points
    @assert length(symbol_IDs) == N
    @assert length(AEVs) == N

    # τ::Float32 = batch_size * 10000  # parameter in ANI-1 loss function
    c::Float32 = 1/batch_size  # parameter in loss function

    opt = ADAM(η, (0.9, 0.999), 1e-8)
    ps = Flux.params(nnps_vector)
    loss(batch) = total_loss(nnps_vector, c, batch)

    data = collect(zip(AEVs, symbol_IDs, energies))
    shuffle!(data)  # randomly shuffle data
    val_num::Int = floor(N * validation_proportion)
    val_dataset = data[1:val_num]  # validation dataset
    temp = data[val_num+1:end]
    train_dataset = Flux.DataLoader( data[val_num+1:end], batchsize=batch_size, shuffle=true, 
                                        partial=true )
    data = nothing  # clear memory

    # for now, train just 1 epoch
    val_loss_before = loss(val_dataset)
    train_loss_before = loss(temp)
    start = time()
    train_one_epoch!(loss, ps, train_dataset, opt)
    finish = time()
    val_loss_after = loss(val_dataset)
    train_loss_after = loss(temp)
    println("$(finish - start), $(val_loss_after - val_loss_before), $(train_loss_after - train_loss_before)")
    
    return nothing
end