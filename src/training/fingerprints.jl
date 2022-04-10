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
    R_ij > R_c && return 0.0
    return 0.5 * cos(pi * R_ij / R_c) + 0.5
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


#==============================================================================================#
#======================================= High Level ===========================================#

"""
    make_AEVs(symbols, coordinates, params)

Constructs AEV for each of the N atom in the structure, given a 3 x N matrix of coordinates
"""
function make_AEVs(params::Params, symbols::Vector{String}, coordinates::Matrix{Float32},
                   ) :: Vector{Vector{Float32}}

    # make vector of all BP parameters
    radial_subAEV_params = groups_to_BPParameters(params.radial)
    angular_subAEV_params = groups_to_BPParameters(params.angular)

    # convert each symbol in the structure to its coresponding tag

    # calculate distance matrix
    D = distance_matrix(coordinates)

    # for each atom in the structure, compute its AEV
    N = length(symbols)
    AEVs = Vector{Vector{Float32}}(undef, N)


    return AEVs
end