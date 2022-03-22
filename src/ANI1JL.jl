# main file that brings everything together
include("training/fingerprints.jl")

module ANI1JL


function train(symbols::Vector{Vector{String}},        # required
               coordinates::Vector{Matrix{Float64}},   # required
               energies::Vector{Float64};              # required
               settingsFile=nothing                    # optional (keyword)
               )
    """
    Trains the force field with given data.                                                    ! optional parameters specifying what parameters to use (make it have default value)

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
    - 

    Workflow:
        * Take symbols, coordinates, and energies as input
        * Calculate distance and angle matrices from coordinates and symbols
        * Compute atomic environment vectors (AEV) from distance and angle matrices
        * Train and return the model
    """

    return "success"
end


end