# main file that brings everything together

module ANI1JL

using DelimitedFiles  # to write csv files for debugging

include("parameters.jl")

include("training/geometry.jl")
include("training/fingerprints.jl")

include("../tools/fileio.jl")
using .fileio


"""
    train(symbols, coordinates, energies; parameterFile = nothing)

Trains the force field with given data.

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

Other required arguments:
(None at the moment)

Optional arguments:
- parameterFile:
    File containing settings and parameters to use for training
        * uses default settings if not provided 

Workflow:
    * Take symbols, coordinates, and energies as input
    * should change all datatypes to 32 bit
    * Calculate distance and angle matrices from coordinates and symbols
    * Compute atomic environment vectors (AEV) from distance and angle matrices
        * to vectorize training, create unnecessary AEVs too
    * Pre-process data (mean = 0, std = 1)
    * initalizing weights: "He initialization for RELU"
        * https://stats.stackexchange.com/questions/229885/whats-the-recommended-weight-initialization-strategy-when-using-the-elu-activat
        * but the ANI paper initialized from random normal distribution between the ranges (-1/d, 1/d), where d is the # of inputs into the node
    * pre-process results in each layer? (batch normalization)
    * Train and return the model (multiple times, each with smaller learning rate)
        * max norm regularization
        * first training: vectorized (0 inputs); subsequent trainings with no 0 inputs
"""
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


    # just for debugging
    random_index = rand(1:length(symbols))
    @show random_index
    symbols_single = [symbols[random_index]]
    coordinates_single = [coordinates[random_index]]

    # make AEVs for all data points
    AEVs_list = coordinates_to_AEVs(params, symbols_single, coordinates_single)
    AEVs = AEVs_list[1]
    writedlm("AEVs.csv", AEVs, ',') 
    @show symbols_single[1]
    writedlm("coordinates.csv", coordinates_single[1], ',')
    write_xyz("coordinates.xyz", symbols_single[1], coordinates_single[1])

    return "success"
end


end