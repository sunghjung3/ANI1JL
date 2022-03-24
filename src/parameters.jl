"""
Methods to parse and store model parameters
"""

module parameters


struct Params
    """
    Struct to hold all parameters passsed in by the user through the .par file
    """
    elements::Vector{String}  # ex: ["C", "N", "O", "H"]
    R_cut_radial::Float64
    R_cut_angular::Float64
    radial::Vector{Tuple{Float64, Float64}}  # ex: [(eta1, R_s1), (eta2, R_s2), ...]
    angular::Vector{Tuple{Float64, Float64, Float64, Float64}}  # ex: [(zeta1, theta_s1, eta1,
                                                                #       R_s1), ...]
    architecture::Vector{Int64}  # ex: [128, 128, 64]
    biases::Vector{String}  # ex: ["y", "y", "y", "n"]
    activation::Vector{String}  # ex: ["gelu", "gelu", "gelu", "gelu"]
end


function parse(par_file::String) :: Params

end


end