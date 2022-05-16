# main file that brings everything together

module ANI1JL


using Flux, NNlib


include("parameters.jl")
export set_subAEV!, set_elements!, set_architecture!, set_activation!, set_biases!, set_params,
        check_consistency


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
