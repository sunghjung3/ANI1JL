"""
    atomic_nnp(params)

Creates and initializes a new atomic neural network potential model
"""
function atomic_nnp(params::Params)
    layers = []
    for i in 1:length(params.activation)
        in = params.architecture[i]
        out = params.architecture[i+1]
        activation = getfield( Flux, Symbol(params.activation[i]) )
        bias_allowed = params.biases[i]

        # initialization (biases are initialized to 0 by default in creation of Dense layer)
        in_inv = 1 / in
        initialization = Flux.truncated_normal(mean=0, std=in_inv/3, lo=-in_inv, hi=in_inv)

        layer = Dense( in => out, activation; bias=bias_allowed, init=initialization )
        push!(layers, layer)
    end

    return Chain(layers...)
end


"""
    atomic_nnps(params)

Returns a vector of new atomic neural network potentials.
Model in index `i` of the vector corresponds to element with tag `i`
"""
function atomic_nnps(params::Params)
    return [atomic_nnp(params) for i in 1:length(params.elements)]
end


"""
    nnp_energy(nnps, AEVs, IDs)

Evaluates the neural network potential energy given the AEVs
"""
function nnp_energy(nnps, AEVs::Matrix{Float32}, IDs::Vector{Int}) :: Float32
    E = sum([ nnps[IDs[i]](AEVs[:, i])[1] for i in 1:length(IDs) ])
    return E
end


#==============================================================================================#
#============================== Saving & Loading Models from file =============================#

using BSON: @save, @load

function save_nnps(filename::String, nnps) :: Int
    @save filename nnps
    return 0
end


function load_nnps(filename::String)
    @load filename nnps
    return nnps
end