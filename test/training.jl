# Test training script using part of ANI-1's existing dataset

#==============================================================================================#
using ANI1JL
using PyCall

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



#=
# load data
data_file_name = raw"C:\Users\sungh\code\ANI1JL\test\data\ani_gdb_s01.h5"
symbols, coordinates, energies = import_ani_data(data_file_name)

# set parameters for AEVs and NN (using default here, but can also specify path to a .par file)
params = set_params()
println(params)

# make AEVs for all data points and save to file
AEVs_list = coordinates_to_AEVs(params, symbols, coordinates)
save_AEVs("ani_gdb_s01_AEVs.bson", params, AEVs_list)
=#

# load existing AEV files
AEVs_filenames = [raw"C:\Users\sungh\code\ANI1JL\test\AEVs\ani_gdb_s01_AEVs.bson",
                  raw"C:\Users\sungh\code\ANI1JL\test\AEVs\ani_gdb_s02_AEVs.bson"
                ]
    # @show length(AEVs)]
params, AEVs = load_AEVs(AEVs_filenames...);


println("Training script finished")