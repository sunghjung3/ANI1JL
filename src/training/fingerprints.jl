floats_or_ints = Union{Float32, Int32}

#==============================================================================================#
#======================================= Low Level ============================================#

"""
    BPParameters

Struct to hold descriptor parameters
"""
struct BPParameters
    R_s::floats_or_ints  # shifting distance, as used in the radial element of AEV
    η::floats_or_ints  # width parameter in the exponential term
    θ_s::floats_or_ints  # shifting angle, as used in the angular element of AEV
    ζ::floats_or_ints  # angular element width parameter
end


"""
    f_C(R_ij, R_c)

Piecewise cutoff function defined by:
                 _
                |  0.5 * cos(π R_ij/R_c) + 0.5      for R_ij ≤ R_c
    f_C(R_ij) = |
                |_             0.0                  for R_ij > R_c
"""
function f_C(R_ij::floats_or_ints, R_c::floats_or_ints) :: Float32
    R_ij > R_c && return 0f0
    return 0.5f0 * cos(pi * R_ij / R_c) + 0.5f0
end


"""
    G_R_singleTerm(R_ij, p, f_Rij)

Single term of the radial element G_R of the atomic environment vector:
    G_R_i = exp(-η (R_ij - R_s)^2 ) * f_C(R_ij)

f_C(R_ij) is passed in as f_Rij
"""
function G_R_singleTerm(R_ij::floats_or_ints, p::BPParameters, f_Rij::floats_or_ints) :: Float32
    diff = R_ij - p.R_s
    return exp(-p.η * diff * diff) * f_Rij
end


"""
    G_A_singleTerm(θ_ijk, R_ij, R_ik, p)

Single term of the angular element G_A of the atomic environment vector:
    G_A_i = (1 + cos(θ_ijk- θ_s))^ζ * exp(-η([R_ij + R_ik]/2 - R_s)^2) * f_C(R_ij) f_C(R_ik)

f_C(R_ij) and f_C(R_ik) is passed in as f_Rij and f_Rik
"""
function G_A_singleTerm(θ_ijk::floats_or_ints, R_ij::floats_or_ints, R_ik::floats_or_ints,
                        p::BPParameters, f_Rij::floats_or_ints, f_Rik::floats_or_ints
                        ) :: Float32

    diff = (R_ij + R_ik) / 2 - p.R_s
    return (1 + cos(θ_ijk - p.θ_s))^p.ζ * exp(-p.η * diff * diff) * f_Rij * f_Rik
end


# Helper functions for indexing the AEVs matrix
get_radial_index(j::Int, n::Int, m_R::Int) = m_R * (j-1) + n
partial_angular_index(j::Int, k::Int, N::Int, tri::Dict{Int, Int}) = (j-1) * N - tri[j-1] + k
# get_angular_index(j::Int, k::Int, m_A::Int, ao::Int, N::Int, tri::Dict{Int, Int}, n::Int) =
                                            # ao + m_A * partial_angular_index(j, k, N, tri) + n
get_angular_index(j::Int, k::Int, m_A::Int, N::Int, tri::Dict{Int, Int}, n::Int) =
                                                m_A * partial_angular_index(j, k, N, tri) + n

#==============================================================================================#
#======================================= Mid Level ============================================#

"""
    groups_to_BPParameters(radial_groups)

Converts vector of radial parameter tuples (η, R_s) to BPParameters datatype
"""
function groups_to_BPParameters(radial_groups::Vector{NTuple{2, Float32}},
                                ) :: Vector{BPParameters}
    N = length(radial_groups)
    return_V = Vector{BPParameters}(undef, N)
    for i in 1:N
        group = radial_groups[i]
        return_V[i] = BPParameters(group[2], group[1], 0f0, 0f0)
    end
    return return_V
end


"""
    groups_to_BPParameters(angular_groups)

Converts vector of angular parameter tuples (ζ, θ_s, η, R_s) to BPParameters datatype
"""
function groups_to_BPParameters(angular_groups::Vector{NTuple{4, Float32}},
                                 ) :: Vector{BPParameters}
    N = length(angular_groups)
    return_V = Vector{BPParameters}(undef, N)
    for i in 1:N
        group = angular_groups[i]
        return_V[i] = BPParameters(group[4], group[3], group[2], group[1])
    end
    return return_V
end


function compute_radial_subAEVs(symbols_id::Vector{Int}, distance_matrix::Matrix{Float32},
                             params_list::Vector{BPParameters}, R_cut::Float32)

end

function compute_angular_subAEVs(symbols_id::Vector{Int}, distance_matrix::Matrix{Float32},
                              params_list::Vector{BPParameters}, R_cut::Float32)

end

#==============================================================================================#
#======================================= High Level ===========================================#

"""
    compute_AEVs!(symbols, coordinates, params)

Constructs AEV for each of the N atom in the structure, given a 3 x N matrix of coordinates.
Stores the result in the `i`th entry of the first argument
"""
function compute_AEVs!(store_vector::Vector{Matrix{Float32}}, i::Int, params::Params,
                       symbols::Vector{String}, coordinates::Matrix{Float32},
                       radial_params::Vector{BPParameters}, angular_params::Vector{BPParameters}
                       , m_R::Int, m_A::Int, tri::Dict{Int, Int}, N::Int) :: Nothing

    # convert each symbol in the structure to its coresponding tag
    symbols_id = [params.elements2tags[symbol] for symbol in symbols]

    # calculate distance matrix and f_C(R_ij)
    D = distance_matrix(coordinates)
    f_C_radial = f_C.(D, params.R_cut_radial)
    f_C_angular = f_C.(D, params.R_cut_angular)

    # compute its AEV
    M = length(symbols)
    # get radial radial subAEVs
    # get angular subAEVs
    # store_vector[i] = vcat(radial subAEVs, angular subAEVs)

    return nothing
end


"""
    coordinates_to_AEVs(params, symbols_list, coordinates_list)

Return a list of AEVs for all data points in the provided symbols and coordinates lists
"""
function coordinates_to_AEVs(params::Params, symbols_list::Vector{Vector{String}},
                        coordinates_list::Vector{Matrix{Float32}}) :: Vector{Matrix{Float32}}
    N = length(symbols_list)

    # make vector of all BP parameters
    radial_subAEV_params = groups_to_BPParameters(params.radial)
    angular_subAEV_params = groups_to_BPParameters(params.angular)
    m_R = length(radial_subAEV_params)
    m_A = length(angular_subAEV_params)

    # Other preparations needed:

    # triangle number needed for indexing angular subAEV
    tri = Dict( i => sum(0:i) for i in 0:(N-1) )
    # angular_offset = N * length(radial_subAEV_params)

    return_list = Vector{Matrix{Float32}}(undef, N)
    for i in 1:N  # for all data points
        compute_AEVs!(return_list, i, params, symbols_list[i], coordinates_list[1],
                        radial_subAEV_params, angular_subAEV_params, m_R, m_A, tri, N)
    end

    return return_list
end