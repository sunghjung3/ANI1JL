# Test training script using part of ANI-1's existing dataset

#==============================================================================================#
using ANI1JL
using PyCall
using Random
using BSON: @save, @load
using Flux, NNlib

#==============================================================================================#
# Helper functions for this script


function import_ani_data(filename::String)
    adl = pyanitools.anidataloader(filename)

    # data vectors to return
    symbols = Vector{Vector{String}}()
    coordinates = Vector{Matrix{Float32}}()
    energies = Vector{Float32}()

    # extract all data
    for data in adl
        P = data["path"]
        X = data["coordinates"]
        E = data["energies"]
        S = data["species"]
        sm = data["smiles"]

        # add to return vectors
        append!(energies, E)
        for i = 1:length(E)  # for all the energies
            push!(symbols, S)  # all the same species
            push!(coordinates, X[i, :, :]')  # transpose to make each (x, y, z) a column
        end
    end

    adl.cleanup()

    return symbols, coordinates, energies
end


unzip(a) = map(x -> getfield.(a, x), fieldnames(eltype(a)))


#==============================================================================================#
#=================================== Start of training script =================================#

#=
Required training data (let's say there are N datapoints):
- symbols:
    vector of length N, each element being a vector of chemical symbols
- coordinates:
    vector of length N, each element being a matrix of atomic coordinates (same order as
    the corresponding element in the "symbols" vector)
        * Each atom's (x, y, z) as a COLUMN in the matrix
- energies:
    vector of length N, each element being the true energy of the corresponding system in
    the "coordinates" vector

Workflow:
    * Take symbols, coordinates, and energies as input
    * should change all datatypes to 32 bit
    * Calculate distance and angle matrices from coordinates and symbols
    * Compute atomic environment vectors (AEV) from distance and angle matrices
        * to vectorize training, create unnecessary AEVs too ?
    * Pre-process data (mean = 0, std = 1)
    * initalizing weights: "He initialization for RELU"
        * https://stats.stackexchange.com/questions/229885/whats-the-recommended-weight-initialization-strategy-when-using-the-elu-activat
        * but the ANI paper initialized from random normal distribution between the ranges (-1/d, 1/d), where d is the # of inputs into the node
    * pre-process results in each layer? (batch normalization)
    * Train and return the model (multiple times, each with smaller learning rate)
        * max norm regularization
        * first training: vectorized (0 inputs); subsequent trainings with no 0 inputs ?
=#


#==============================================================================================#
# Make AEVS

#=
# load data
data_filenames = ["test/data/ani_gdb_s01.h5",
                  "test/data/ani_gdb_s02.h5"]#=,
                  "test/data/ani_gdb_s03.h5",
                  "test/data/ani_gdb_s04.h5",
                  "test/data/ani_gdb_s05.h5",
                  "test/data/ani_gdb_s06.h5"
                ]=#

symbols = Vector{Vector{String}}()
coordinates = Vector{Matrix{Float32}}()
energies = Vector{Float32}()
for file in data_filenames
    local_symbols, local_coordinates, local_energies = import_ani_data(file)
    append!(symbols, local_symbols)
    append!(coordinates, local_coordinates)
    append!(energies, local_energies)
end

# set parameters for AEVs and NN (using default here, but can also specify path to a .par file)
# params = set_params("test/params.par")
params = set_params()  # default parameters
println(params)

# make AEVs for all data points
AEVs_list, symbol_id_list = coordinates_to_AEVs(params, symbols, coordinates)
println("finished making AEVs")

# Randomly shuffle dataset
dataset = collect(zip(AEVs_list, symbol_id_list, energies))
shuffle!(dataset)
AEVs_list, symbol_id_list, energies = unzip(dataset)
println("finished random shuffle")

# Creats training, validation, and test sets
validation_proportion = 0.08  # proportion of data to use as validation set
test_set_proportion = 0.02  # proportion of data to use as test set
N = Int( length(energies) )
N_val = Int( floor(N * validation_proportion) )
N_test = Int( floor(N * test_set_proportion) )
validation_set = (AEVs_list[1:N_val], symbol_id_list[1:N_val], energies[1:N_val])
test_set = (AEVs_list[(N_val+1):(N_val+N_test)], symbol_id_list[(N_val+1):(N_val+N_test)],
                energies[(N_val+1):(N_val+N_test)])
training_set = (AEVs_list[(N_val+N_test+1):end], symbol_id_list[(N_val+N_test+1):end],
                energies[(N_val+N_test+1):end])
println("finished making training, validation, and test sets")

# save to file
save_AEVs("test/AEVs/training_set_01-02.bson", params, training_set...)
save_AEVs("test/AEVs/validation_set_01-02.bson", params, validation_set...)
save_AEVs("test/AEVs/test_set_01-02.bson", params, test_set...)
println("finished saving datasets")
=#

#==============================================================================================#
# Train

training_set_filename = "AEVs/training_set_01-02.bson"
validation_set_filename = "AEVs/validation_set_01-02.bson"

# load datasets
params, AEVs, symbol_ids, energies = load_AEVs(training_set_filename)  # training
training_set = (AEVs, symbol_ids, energies)
println(params)
params_val, AEVs_val, symbol_ids_val, energies_val = load_AEVs(validation_set_filename)  # val
validation_set = (AEVs_val, symbol_ids_val, energies_val)
if check_consistency(params, params_val) >= 2
    println("Exiting due to inconsistent params between training and validations sets")
    exit(code=1)
end


# Load neural network
NN_Potential = atomic_nnps(params)

# Training settings
batch_size = 68
η = 0.1

# Train network (package automatically saves checkpoints)
train_nnps!(NN_Potential; training_set=training_set, validation_set=validation_set,
                batch_size=batch_size, η=η)

# Save final model
save_nnps("finished_model.bson", NN_Potential)

#==============================================================================================#
# Analze results

#=
training_set_filename = "AEVs/training_set_01-02.bson"
validation_set_filename = "AEVs/validation_set_01-02.bson"
test_set_filename = "test/AEVs/test_set_01-02.bson"
nnps_filename = ""

# load datasets
params, AEVs, symbol_ids, energies = load_AEVs(training_set_filename)  # training
training_set = (AEVs, symbol_ids, energies)
println(params)
params_val, AEVs_val, symbol_ids_val, energies_val = load_AEVs(validation_set_filename)  # val
validation_set = (AEVs_val, symbol_ids_val, energies_val)
if check_consistency(params, params_val) >= 2
    println("Exiting due to inconsistent params between training and validations sets")
    exit(code=1)
end
params_test, AEVs_test, symbol_ids_test, energies_test = load_AEVs(test_set_filename)  # test
test_set = (AEVs_test, symbol_ids_test, energies_test)
if check_consistency(params, params_val) >= 2
    println("Exiting due to inconsistent params between training and test sets")
    exit(code=1)
end

# load NNPs
NN_Potential = load_nnps(nnps_filename)

# Get NNP energy for training set
training_nnps_energies = Vector{Float32}(undef, length(energies))
training_energy_fit = nnp_energy()

# Get NNP energy for validation set
validation_nnps_energies = Vector{Float32}(undef, length(energies_val))

# Get NNP energy for test set
test_nnps_energies = Vector{Float32}(under, length(energies_test))
=#


println("Training script finished")