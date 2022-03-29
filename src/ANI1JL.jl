# main file that brings everything together

module ANI1JL


include("training/fingerprints.jl")
include("parameters.jl")


function train(symbols::Vector{Vector{String}},              # required
               coordinates::Vector{Matrix{Float64}},         # required
               energies::Vector{Float64};                    # required
               parameterFile = nothing                       # optional (keyword)
               )
    """
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
        * Calculate distance and angle matrices from coordinates and symbols
        * Compute atomic environment vectors (AEV) from distance and angle matrices
        * Train and return the model
    """
    if isnothing(parameterFile)  # find default setting file
        pathwin = split(pwd(), "\\")
        pathnux = split(pwd(), "/")
        path = (length(pathwin) > length(pathnux)) ? pathwin : pathnux  # depending on system
        basepos = last(findall(x -> x == "ANI1JL", path))  # location of lowest "ANI1JL" dir
        @show path
        @show basepos
        path = path[1:basepos] # only the path up to the lowest ANI1JL directory
        @show path
        parameterFile = join(path, "/") * "/src/training/default.par"
        println(parameterFile)
    end
    params = parameters.parse_params(parameterFile)
    println(params)
    return "success"
end


end