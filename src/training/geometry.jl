"""
    geometry

Methods to deal with geometries (e.g. distance, coordinate, angle, ...)
"""
module geometry


"""
    coordinates_to_distances(coordinates::Matrix{Float32}) :: Matrix{Float32}

Converts a matrix of coordinates (3 x N) to a distance matrix (N x N).
Diagonal elemetns of the distance matrix should be 0.0
"""
function coordinates_to_distances(coordinates::Matrix{Float32}) :: Matrix{Float32}

end


"""
    coordinates_to_angles(coordinates::Matrix{Float32}) :: Array{Float32, 3}

Converts a matrix of coordinates (3 x N) to an angle array (N x N x N).
Each slice o
function coordinates_to_angles(coordinates::Matrix{Float32}) :: Array{Float32, 3}

end

end