# Test training script using part of ANI-1's existing dataset

#==============================================================================================#
using ANI1JL
using PyCall
using Random
using JLD
using Flux, NNlib
using Plots

#==============================================================================================#

unzip(a) = map(x -> getfield.(a, x), fieldnames(eltype(a)))


#==============================================================================================#
#=================================== Start of training script =================================#



# set parameters for AEVs and NN
params = set_params("test/params.par")
println(params)

# load NNPs
nnps_filename = "test/trained_models/01-04/NNPs_01-04_0.1.bson"
nnps = load_nnps(nnps_filename)

# results
true_energies = Float32[]
nnps_energies = Float32[]

# Make AEVS
for i in 0:19
    file = "data_08_$(i).jld"
    @show file
    filedict = load(file)
    symbols = filedict["symbols"]
    coordinates = filedict["coordinates"]
    energies = filedict["energies"]
    println("finished loading file")

    # make AEVs for all data points
    AEVs_list, symbol_id_list = coordinates_to_AEVs(params, symbols, coordinates)
    println("finished making AEVs")

    # Get NNP energies
    energies_fit::Vector{Float64} = nnp_energy.([nnps], AEVs_list, symbol_id_list)
    println("finished nnp_energy calculation")

    # RMSE calculation
    rmse = sqrt(sum( (energies - energies_fit).^2 ) / length(true_energies))
    println("RMSE = $rmse")

    # plotting
    gr()
    p = plot(energies, energies_fit, seriestype = :scatter, title = "data_08 on NNPs_04 $i (RMSE = $(rmse))")
    xlabel!("DFT Energy (kcal/mol)")
    ylabel!("NNP Energy (kcal/mol)")
    savefig(p, "plot_08_on_04_$i.png")


    # append to results vectors
    append!(true_energies, energies)
    append!(nnps_energies, energies_fit)
end

# plotting
gr()
p = plot(true_energies, nnps_energies, seriestype = :scatter, title = "data_08 on NNPs_04")
xlabel!("DFT Energy (kcal/mol)")
ylabel!("NNP Energy (kcal/mol)")
savefig(p, "plot_08_on_04.png")




# save to file
results_filename = "results/results_08.jld"
save(results_filename, "true_energies", true_energies, "nnps_energies", nnps_energies)



#==============================================================================================#
# Analze results







println("Training script finished")
