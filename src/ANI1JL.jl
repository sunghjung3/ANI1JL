# main file that brings everything together

module ANI1JL


using Flux, NNlib


include("parameters.jl")
export set_subAEV!, set_elements!, set_architecture!, set_activation!, set_biases!, set_params


include("train/geometry.jl")
export distance_matrix

include("train/fingerprints.jl")
export compute_AEVs!, coordinates_to_AEVs, save_AEVs, concat_AEVs, load_AEVs

include("train/nnp_model.jl")
export atomic_nnps, nnp_energy, save_nnps, load_nnps

include("train/train.jl")
export train_nnps!, train_one_epoch!, total_loss, single_loss, max_norm_reg, total_loss_parallel


include("../tools/fileio.jl")
export fileio  # the module

include("../tools/pyanitools.jl")
export pyanitools  # the module


end


#= Old
function train(symbols::Vector{Vector{String}},              # required
               coordinates::Vector{Matrix{Float32}},         # required
               energies::Vector{Float32};                    # required
               parameterFile = nothing                       # optional (keyword)
               )
    # get settings from user
    if isnothing(parameterFile)  # find default setting file
        pathwin = split(pwd(), "\\")
        pathnux = split(pwd(), "/")
        path = (length(pathwin) > length(pathnux)) ? pathwin : pathnux  # depending on system
        basepos = last(findall(x -> x == "ANI1JL", path))  # location of lowest "ANI1JL" dir
        path = path[1:basepos]  # only the path up to the lowest ANI1JL directory
        parameterFile = join(path, "/") * "/src/training/default.par"
        println(parameterFile)
    end
    params = parse_params(parameterFile)
    println(params)
    
    default_params = set_params()
    println(default_params)

    # make AEVs for all data points
    # AEVs_list = coordinates_to_AEVs(params, symbols, coordinates)
    # save_AEVs("ani_gdb_s01_AEVs.bson", params, AEVs_list)

    # params, AEVs = load_AEVs(raw"C:\Users\sungh\code\ANI1JL\test\AEVs\ani_gdb_s01_AEVs.bson", raw"C:\Users\sungh\code\ANI1JL\test\AEVs\ani_gdb_s02_AEVs.bson")
    # @show length(AEVs)

    return "success"
end
=#
