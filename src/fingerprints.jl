"""
Methods to compute fingerprints
"""

module fingerprints


floats_or_ints = Union{Float64, Int64}


function f_C(R_ij::floats_or_ints, R_c::floats_or_ints) :: Float64
    """
    Piecewise cutoff function

                     _
                    |  0.5 * cos(π R_ij/R_c) + 0.5      for R_ij ≤ R_c
        f_C(R_ij) = |
                    |_             0.0                  for R_ij > R_c
    """
    R_ij > R_c && return 0.0
    return 0.5 * cos(pi * R_ij / R_c) + 0.5
end


function G_R_singleTerm(R_ij::floats_or_ints, R_s::floats_or_ints, R_c::floats_or_ints,
                        η::floats_or_ints) :: Float64
    """
    Single term of the radial element G_R of the atomic environment vector

        G_R_i = exp(-η (R_ij - R_s)^2 ) * f_C(R_ij)
    """
    diff = R_ij - R_s
    return exp(-η * diff * diff) * f_C(R_ij, R_c)
end


function G_A_singleTerm(θ_ijk::floats_or_ints, θ_s::floats_or_ints, ζ::floats_or_ints,
                        R_ij::floats_or_ints, R_ik::floats_or_ints, R_s::floats_or_ints,
                        R_c::floats_or_ints, η::floats_or_ints) :: Float64
    """
    Single term of the angular element G_A of the atomic environment vector

        G_A_i = (1 + cos(θ_ijk- θ_s))^ζ * exp(-η([R_ij + R_ik]/2 - R_s)^2) * f_C(R_ij) f_C(R_ik)
    """
    diff = (R_ij + R_ik) / 2 - R_s
    return (1 + cos(θ_ijk - θ_s))^ζ * exp(-η * diff * diff) * f_C(R_ij, R_c) * f_C(R_ik, R_c)
end


end