"""
    distance_matrix(coordinates::Matrix{Float32}) :: Matrix{Float32}

Converts a matrix of coordinates (3 x N) to a distance matrix (N x N).
Diagonal elemetns of the distance matrix should be 0.0.

In old versions, this function was also called `coordinates_to_distances()`
"""
function distance_matrix(coordinates::Matrix{Float32}) :: Matrix{Float32}
    N = size(coordinates)[2]

    return_matrix = zeros(Float32, N, N)
    for i in 1:(N-1)
        col = coordinates[:, i]
        diff = coordinates[:, (i+1):end] .- col
        d = sqrt.(sum(diff.^2, dims=1))
        return_matrix[(i+1):end, i] = d
        return_matrix[i, (i+1):end] = d
    end
    return return_matrix
end


"""
    coordinates_to_angles(coordinates::Matrix{Float32}) :: Array{Float32, 3}

Converts a matrix of coordinates (3 x N) to an angle array (N x N x N).\
Also takes a pre-calculated matrix of distances for speed
"""
function coordinates_to_angles(coordinates::Matrix{Float32}, distances::Matrix{Float32}
                               ) :: Array{Float32, 3}
    N = size(coordinates)[2]

    angle_array = zeros(Float32, N, N, N)
    for k in 1:N
        
    end
end



#==============================================================================================#
#====================================== Old functions =========================================#

function coordinates_to_distances2(coordinates::Matrix{Float32}) :: Matrix{Float32}
    N = size(coordinates)[2]

    distance_matrix = zeros(Float32, N, N)
    for i in 1:(N-1)
        col = coordinates[:, i]
        diff = coordinates[:, (i+1):end] .- col
        d = sqrt.(sum(diff.^2, dims=1))
        distance_matrix[(i+1):end, i] = d
    end
    distance_matrix += transpose(distance_matrix)
    return distance_matrix
end


function coordinates_to_distances3(coordinates::Matrix{Float32}) :: Matrix{Float32}
    N = size(coordinates)[2]

    distance_matrix = Matrix{Float32}(undef, N, N)
    for i in 1:N
        col = coordinates[:, i]
        diff = coordinates .- col
        d = sqrt.(sum(diff.^2, dims=1))
        distance_matrix[:, i] = d
    end
    return distance_matrix
end

function coordinates_to_distances4(coordinates::Matrix{Float32}) :: Matrix{Float32}
    N = size(coordinates)[2]

    distance_matrix = zeros(Float32, N, N)
    for i in 1:N
        for j in (i+1):N
            diff = coordinates[:, j] - coordinates[:, i]
            d = sqrt(sum(diff.^2))
            distance_matrix[i, j] = d
            distance_matrix[j, i] = d
        end
    end
    return distance_matrix
end

function coordinates_to_distances5(coordinates::Matrix{Float32}) :: Matrix{Float32}
    N = size(coordinates)[2]

    distance_matrix = zeros(Float32, N, N)
    for i in 1:N
        for j in (i+1):N
            diff = coordinates[:, j] - coordinates[:, i]
            d = sqrt(sum(diff.^2))
            distance_matrix[i, j] = d
        end
    end
    distance_matrix += transpose(distance_matrix)
    return distance_matrix
end
function coordinates_to_distances11(coordinates::Matrix{Float32}) :: Matrix{Float32}
    N = size(coordinates)[2]

    distance_matrix = zeros(Float32, N, N)
    for i in 1:(N-1)
        distance_matrix[(i+1):end, i] = sqrt.(sum( (coordinates[:, (i+1):end] .- coordinates[:, i]) .^2, dims=1))
        distance_matrix[i, (i+1):end] = distance_matrix[(i+1):end, i]
    end
    return distance_matrix
end

function coordinates_to_distances22(coordinates::Matrix{Float32}) :: Matrix{Float32}
    N = size(coordinates)[2]

    distance_matrix = zeros(Float32, N, N)
    for i in 1:(N-1)
        distance_matrix[(i+1):end, i] = sqrt.(sum( (coordinates[:, (i+1):end] .- coordinates[:, i]) .^2, dims=1))
    end
    return distance_matrix + transpose(distance_matrix)
end

function repeat(f, c, N)
    for i in 1:N
        f(c)
    end
end

"""
    dot_products(coordinates)

Given a 3 x N matrix of N atomic Cartesian coordinates, returns a 3D array of dot products of
all pair-wise vectors
Element (i, j, k) of the returned array (refer to as array "M") can be interpreted as:
                    M[i, j, k] = (X[i] - X[k])̇ (X[j] - X[k]),
    where X[m] = Cartesian coordinate of atom m = [x_m, y_m, z_m], and ̇  is the dot product
"""
function dot_products(coordinates::Matrix{Float32}) :: Array{Float32, 3}
    N = size(coordinates)[2]  # dims = 3 x N
    return_array = zeros(Float32, N, N, N)  # dims = N x N x N
    for k in 1:N
        col = coordinates[:, k]  # center atom (dims = 3 x 1)

        # vector from atom k to all other atoms
        diff_vecs = coordinates[:, 1:end .!= k] .- col  # dims = 3 x (N-1)

        dots = Matrix{Float32}(undef, (N-1), (N-1))
        for j in 1:(N-1)  # each column in diff_vecs
            diff_vec_T = transpose(diff_vecs[:, j])  # dims = 1 x 3
            temp = diff_vec_T * diff_vecs[:, j:end]  # dims = 1 x (N-j)
            dots[j:end, j] = temp
            dots[j, j:end] = temp
            ### NOTE: sqrt is not included yet
        end
        return_array[1:end .!= k, 1:end .!= k, k] = dots
    end
    return return_array
end

function dot_products_dumb(coordinates::Matrix{Float32}) :: Array{Float32, 3}
    N = size(coordinates)[2]  # dims = 3 x N
    return_array = Array{Float32, 3}(undef, N, N, N)  # dims = N x N x N
    for k in 1:N
        col = coordinates[:, k]  # center atom (dims = 3 x 1)

        # vector from atom k to all other atoms with index > k
        diff_vecs = coordinates .- col  # dims = 3 x N
        dots = transpose(diff_vecs) * diff_vecs
        return_array[:, :, k] = dots
        ### NOTE: sqrt is not included here yet
    end
    return return_array
end
