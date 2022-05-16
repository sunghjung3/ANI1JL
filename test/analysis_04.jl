# Test training script using part of ANI-1's existing dataset

#==============================================================================================#
using ANI1JL
using PyCall
using Random
using JLD
# using BSON
using Flux, NNlib
using Plots


#==============================================================================================#
#=================================== Start of analysis script =================================#



# set parameters for AEVs and NN
# params = set_params("params.par")
# params = set_params()
# println(params)

# load NNPs
nnps_filename = "NNPs_01-04_0.1.bson"
nnps = load_nnps(nnps_filename)

# results
true_energies = Float32[]
nnps_energies = Float32[]

filenames = ["AEVs/training_set_01-04.jld",
             "AEVs/validation_set_01-04.jld",
             "AEVs/test_set_01-04.jld"]

# Make AEVS
for file in filenames
    @show file
    params, AEVs_matrices, symbol_id_list, energies = load_AEVs(file)

    println("finished loading file")

    # make AEVs for all data points
    # AEVs_list, symbol_id_list = coordinates_to_AEVs(params, symbols, coordinates)
    # println("finished making AEVs")

    # Get NNP energies
    energies_fit = nnp_energy.([nnps], AEVs_matrices, symbol_id_list)
    println("finished calculting nnps energies")

    # RMSE calculation

    rmse = sqrt(sum( (energies - energies_fit).^2 ) / length(energies))
    @show rmse
    # append to results vectors
    append!(true_energies, energies)
    append!(nnps_energies, energies_fit)
end

# save to file
# results_filename = "results/results_08.jld"
# save(results_filename, "true_energies", true_energies, "nnps_energies", nnps_energies)



#==============================================================================================#
# Analze results

# RMSE calculation


# plotting
gr()
p = plot(true_energies, nnps_energies, seriestype = :scatter, title = "Bigger training")
xlabel!("DFT Energy (kcal/mol)")
ylabel!("NNP Energy (kcal/mol)")
savefig(p, "bigger_training.png")





println("Analysis script finished")
