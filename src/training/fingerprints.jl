floats_or_ints = Union{Float32, Int32}

#==============================================================================================#
#======================================= Low Level ============================================#

"""
    BPParameters

Struct to hold descriptor parameters
"""
struct BPParameters
    R_c::floats_or_ints  # cutoff radius, as used in the cutoff function
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
    G_R_singleTerm(R_ij: p)

Single term of the radial element G_R of the atomic environment vector:
    G_R_i = exp(-η (R_ij - R_s)^2 ) * f_C(R_ij)
"""
function G_R_singleTerm(R_ij::floats_or_ints, p::BPParameters) :: Float32
    diff = R_ij - p.R_s
    return exp(-p.η * diff * diff) * f_C(R_ij, p.R_c)
end


"""
    G_A_singleTerm(θ_ijk, R_ij, R_ik, p)

Single term of the angular element G_A of the atomic environment vector:
    G_A_i = (1 + cos(θ_ijk- θ_s))^ζ * exp(-η([R_ij + R_ik]/2 - R_s)^2) * f_C(R_ij) f_C(R_ik)
"""
function G_A_singleTerm(θ_ijk::floats_or_ints, R_ij::floats_or_ints, R_ik::floats_or_ints,
                        p::BPParameters) :: Float32

    diff = (R_ij + R_ik) / 2 - p.R_s
    f = f_C(R_ij, p.R_c) * f_C(R_ik, p.R_c)
    return (1 + cos(θ_ijk - p.θ_s))^p.ζ * exp(-p.η * diff * diff) * f
end

#==============================================================================================#
#======================================= Mid Level ============================================#


#==============================================================================================#
#======================================= High Level ===========================================#

"""
    groups_to_BPParameters(radial_groups)

Converts vector of radial parameter tuples (η, R_s) and R_cut to BPParameters datatype
"""
function groups_to_BPParameters(radial_groups::Vector{NTuple{2, Float32}},
                                R_cut::Float32) :: Vector{BPParameters}
    N = length(radial_groups)
    return_V = Vector{BPParameters}(undef, N)
    for i in 1:N
        group = radial_groups[i]
        return_V[i] = BPParameters(R_cut, group[2], group[1], 0f0, 0f0)
    end
    return return_V
end


"""
    groups_to_BPParameters(angular_groups)

Converts vector of angular parameter tuples (ζ, θ_s, η, R_s) and R_cut to BPParameters datatype
"""
function groups_to_BPParameters(angular_groups::Vector{NTuple{4, Float32}},
                                 R_cut::Float32) :: Vector{BPParameters}
    N = length(angular_groups)
    return_V = Vector{BPParameters}(undef, N)
    for i in 1:N
        group = angular_groups[i]
        return_V[i] = BPParameters(R_cut, group[4], group[3],
                                   group[2], group[1])
    end
    return return_V
end


"""
    make_AEVs(symbols, coordinates, params)

Constructs AEV for each of the N atom in the structure, given a 3 x N matrix of coordinates
"""
function make_AEVs(symbols::Vector{String}, coordinates::Matrix{Float32},
                   symbols_to_tags::Dict{String, Int32}) :: Vector{Vector{Float32}}
    distance_matrix = sqrt.( transpose(coordinates) * coordinates )

end