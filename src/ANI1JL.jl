# main file that brings everything together
include("fingerprints.jl")

module ANI1JL


function train(symbols::Vector{Vector{String}},
               coordinates::Vector{Matrix{Float64}},
               energies::Vector{Float64})
    """
    Trains the force field with given data.                                                    ! optional parameters specifying what parameters to use (make it have default value)

    Workflow:
        * Take symbols, coordinates, and energies as input
        * Calculate distance and angle matrices from coordinates and symbols
        * Compute atomic environment vectors (AEV) from distance and angle matrices
        * Train and return the model
    """

    return "success"
end


end