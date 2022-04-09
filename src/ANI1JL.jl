# main file that brings everything together

module ANI1JL


include("training/fingerprints.jl")
include("training/geometry.jl")
include("parameters.jl")


"""
    train(symbols::Vector{Vector{String}},
               coordinates::Vector{Matrix{Float32}},
               energies::Vector{Float32};
               parameterFile = nothing
               )

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
    if isnothing(parameterFile)  # find default setting file
        pathwin = split(pwd(), "\\")
        pathnux = split(pwd(), "/")
        path = (length(pathwin) > length(pathnux)) ? pathwin : pathnux  # depending on system
        basepos = last(findall(x -> x == "ANI1JL", path))  # location of lowest "ANI1JL" dir
        path = path[1:basepos]  # only the path up to the lowest ANI1JL directory
        parameterFile = join(path, "/") * "/src/training/default.par"
        println(parameterFile)
    end
    params = parameters.parse_params(parameterFile)
    println(params)
    println(length(symbols[1]))
    println(size(coordinates[1]))
    println(length(energies))

    # convert coordinate into distances and angles
    N = length(energies)  # number of data points
    distanceV = Vector{Matrix{Float32}}(undef, N)  # vector of distance matrices
    angleV =  Vector{Array{Float32, 3}}(undef, N)  # vector of angle arrays (3D)
    for i = 1:N  # convert each coordinate to distances and angles for all data points
        distanceV[i] = geometry.coordinates_to_distances(coordinates[i])
        angleV[i] = geometry.coordinates_to_angles(coordinates[i])
    end
    coordinates = nothing  # free memory

    return "success"
end


end