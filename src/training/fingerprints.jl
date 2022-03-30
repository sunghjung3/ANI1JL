"""
    fingerprints

Methods to compute Behler-Parrinello Descriptors
"""
module fingerprints


floats_or_ints = Union{Float64, Int64}


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
    f_C(R_ij::floats_or_ints, R_c::floats_or_ints) :: Float64

Piecewise cutoff function defined by:
                 _
                |  0.5 * cos(π R_ij/R_c) + 0.5      for R_ij ≤ R_c
    f_C(R_ij) = |
                |_             0.0                  for R_ij > R_c
"""
function f_C(R_ij::floats_or_ints, R_c::floats_or_ints) :: Float64
    R_ij > R_c && return 0.0
    return 0.5 * cos(pi * R_ij / R_c) + 0.5
end


"""
    G_R_singleTerm(R_ij::floats_or_ints, p::BPParameters) :: Float64

Single term of the radial element G_R of the atomic environment vector:
    G_R_i = exp(-η (R_ij - R_s)^2 ) * f_C(R_ij)
"""
function G_R_singleTerm(R_ij::floats_or_ints, p::BPParameters) :: Float64
    diff = R_ij - p.R_s
    return exp(-p.η * diff * diff) * f_C(R_ij, p.R_c)
end


"""
    G_A_singleTerm(θ_ijk::floats_or_ints, R_ij::floats_or_ints, R_ik::floats_or_ints,
                        p::BPParameters) :: Float64

Single term of the angular element G_A of the atomic environment vector:
    G_A_i = (1 + cos(θ_ijk- θ_s))^ζ * exp(-η([R_ij + R_ik]/2 - R_s)^2) * f_C(R_ij) f_C(R_ik)
"""
function G_A_singleTerm(θ_ijk::floats_or_ints, R_ij::floats_or_ints, R_ik::floats_or_ints,
                        p::BPParameters) :: Float64

    diff = (R_ij + R_ik) / 2 - p.R_s
    f = f_C(R_ij, p.R_c) * f_C(R_ik, p.R_c)
    return (1 + cos(θ_ijk - p.θ_s))^p.ζ * exp(-p.η * diff * diff) * f
end


end