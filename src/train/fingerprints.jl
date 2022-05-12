#==============================================================================================#
#======================================= Low Level ============================================#

"""
    BPParameters

Struct to hold descriptor parameters
"""
struct BPParameters
    R_s::Float32  # shifting distance, as used in the radial element of AEV
    η::Float32  # width parameter in the exponential term
    θ_s::Float32  # shifting angle, as used in the angular element of AEV
    ζ::Float32  # angular element width parameter
    G_A_coeff::Float32  # == 2^(1-ζ)

    # constructor
    BPParameters(a1, a2, a3, a4) = new( a1, a2, a3, a4, 2^(1-a4) )
end


"""
    f_C(R_ij, R_c)

Piecewise cutoff function defined by:
                 _
                |  0.5 * cos(π R_ij/R_c) + 0.5      for R_ij ≤ R_c
    f_C(R_ij) = |
                |_             0.0                  for R_ij > R_c
"""
function f_C(R_ij::Float32, R_c::Float32) :: Float32
    R_ij > R_c && return 0f0
    return 0.5f0 * cos(pi * R_ij / R_c) + 0.5f0
end


"""
    G_R_singleTerm(R_ij, p, f_Rij)

Single term of the radial element G_R of the atomic environment vector:
    G_R_i = exp(-η (R_ij - R_s)^2 ) * f_C(R_ij)

f_C(R_ij) is passed in as `f_Rij`
"""
function G_R_singleTerm(R_ij::Float32, p::BPParameters, f_Rij::Float32) :: Float32
    diff = R_ij - p.R_s
    return exp(-p.η * diff * diff) * f_Rij
end


"""
    G_A_singleTerm(θ_ijk, R_ij, R_ik, p)

Single term of the angular element G_A of the atomic environment vector:
    G_A_i = (1 + cos(θ_ijk- θ_s))^ζ * exp(-η([R_ij + R_ik]/2 - R_s)^2) * f_C(R_ij) f_C(R_ik)

f_C(R_ij) and f_C(R_ik) is passed in as `f_Rij` and `f_Rik`.
"""
function G_A_singleTerm(θ_ijk::Float32, R_ij::Float32, R_ik::Float32, p::BPParameters,
                        f_Rij::Float32, f_Rik::Float32) :: Float32

    diff = (R_ij + R_ik) / 2 - p.R_s
    return p.G_A_coeff * (1 + cos(θ_ijk - p.θ_s))^p.ζ * exp(-p.η * diff * diff) * f_Rij * f_Rik
end


# Helper functions for indexing the AEVs matrix
get_radial_index(j::Int, n::Int, m_R::Int) = m_R * (j-1) + n
partial_angular_index(j::Int, k::Int, N::Int, tri::Dict{Int, Int}) = (j-1) * N - tri[j-1] + k
get_angular_index(j::Int, k::Int, m_A::Int, ao::Int, N::Int, tri::Dict{Int, Int}, n::Int) =
                                        ao + m_A * (partial_angular_index(j, k, N, tri) - 1) + n
# get_angular_index(j::Int, k::Int, m_A::Int, N::Int, tri::Dict{Int, Int}, n::Int) =
                                                # m_A * partial_angular_index(j, k, N, tri) + n

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


"""
    compute_subAEVs(AEV_matrix, radial_params, atomIDs, distances, f_C_Rij, m_R, M)

Computes radial subAEVs for all atoms for all radial parameter permutations.

Argument descriptions:

    `AEV_Matrix`:
        Matrix of AEVs for all atoms in the structure. Each atom's AEV is stored as a column.
        The function modifies this matrix (size: (`m_R`*`N` + `m_A`*`N`*(`N`+1)/2) x `M`)
            `m_R` and `M` are other arguments to this function. See below.
            For description of `N`, see `compute_AEVs()` function
    `radial_params`:        
        Vector of all permutations of BP descriptor parameters for radial subAEV (length `m_R`)
    `atomIDs`:
        Vector of integer representation (aka "tags" or "IDs") for all `M` atoms in the
        structure
    `distances`:
        Distance matrix (`M` x `M`) between all pairs of atoms
    `f_C_Rij`:
        Return value from f_C(`distances`). Passed in as an argument for performance
    `m_R`:
        Length of `radial_params`
    `M`:
        Number of atoms in the structure
"""
function compute_subAEVs!(AEV_matrix::Matrix{Float32}, radial_params::Vector{BPParameters},
                            atomIDs::Vector{Int}, distances::Matrix{Float32},
                            f_C_Rij::Matrix{Float32}, m_R::Int, M::Int) :: Nothing
    for atom1 in 1:(M-1)
        atom1_id = atomIDs[atom1]  # integer representation of atom1's element symbol
        for atom2 in (atom1+1):M
            if f_C_Rij[atom1, atom2] > 0.0
                atom2_id = atomIDs[atom2]  # integer representation of atom2's element symbol

                for (n, param) in enumerate(radial_params)
                    temp = G_R_singleTerm(distances[atom1, atom2], param, f_C_Rij[atom1, atom2])

                    # update atom 1 descriptors
                    row_index = get_radial_index(atom2_id, n, m_R)
                    AEV_matrix[row_index, atom1] += temp
                    # update atom 2 descriptors
                    row_index = get_radial_index(atom1_id, n, m_R)
                    AEV_matrix[row_index, atom2] += temp

                end  # params loop

            end  # distance condition
        end  # atom2 loop
    end  # atom1 loop

    return nothing
end   # function


"""
    compute_subAEVs!(AEV_Matrix, angular_params, atomIDs, distances, f_C_Rij, coordinates, m_A,
                            tri, N, M, ao)

Computes angular subAEVs for all atoms for all angular parameter permutations.

Argument descriptions:

    `AEV_Matrix`:
        Matrix of AEVs for all atoms in the structure. Each atom's AEV is stored as a column.
        The function modifies this matrix (size: (`m_R`*`N` + `m_A`*`N`*(`N`+1)/2) x `M`)
            `M` is another argument to this function. See below.
            For description of `N` and `m_R`, see `compute_AEVs()` function
    `angular_params`:        
        Vector of all permutations of BP descriptor parameters for angular subAEV (length `m_A`)
    `atomIDs`:
        Vector of integer representation (aka "tags" or "IDs") for all `M` atoms in the
        structure
    `distances`:
        Distance matrix (`M` x `M`) between all pairs of atoms
    `f_C_Rij`:
        Return value from f_C(`distances`). Passed in as an argument for performance
    `coordinates`:
        Matrix of Cartesian coordinates of all atoms (size: 3 x `M`)
    `m_A`:
        Length of `angular_params`
    `tri`:
        Dictionary used to map an integer to its triangular number, which is defined as:
            tri(n) = Σ(i, start: i=0, end: i=n)
        Needed to index angular subAEV in the AEVs matrix
    `N`:
        Number of different elements present in the dataset
            ex: if the dataset contains only C, N, O, H, and F, then `N` = 5
    `M`:
        Number of atoms in the structure
    `ao`:
        "Angular offset" for indexing angular subAEV in the AEVs matrix.
        Should be equal to `N*m_R`

"""
function compute_subAEVs!(AEV_Matrix::Matrix{Float32}, angular_params::Vector{BPParameters},
                            atomIDs::Vector{Int}, distances::Matrix{Float32},
                            f_C_Rij::Matrix{Float32}, coordinates::Matrix{Float32}, m_A::Int,
                            tri::Dict{Int, Int}, N::Int, M::Int, ao::Int) :: Nothing
    for centerAtom in 1:M
        # centerAtom_id = atomIDs[centerAtom]
        for neighborAtom1 in 1:(M-1)
            if f_C_Rij[centerAtom, neighborAtom1] > 0.0 && neighborAtom1 != centerAtom
                neighborAtom1_id = atomIDs[neighborAtom1]
                v_c1 = coordinates[:, neighborAtom1] - coordinates[:, centerAtom]
                d_c1 = distances[centerAtom, neighborAtom1]
                f_C_c1 = f_C_Rij[centerAtom, neighborAtom1]
                for neighborAtom2 in (neighborAtom1+1):M
                    if f_C_Rij[centerAtom, neighborAtom2] > 0.0 && neighborAtom2 != centerAtom
                        neighborAtom2_id = atomIDs[neighborAtom2]
                        v_c2 = coordinates[:, neighborAtom2] - coordinates[:, centerAtom]
                        d_c2 = distances[centerAtom, neighborAtom2]
                        f_C_c2 = f_C_Rij[centerAtom, neighborAtom2]

                        # angle calculation (clamp cos(θ) between -1 and 1 just in case)
                        cos_θ = v_c1'v_c2 / ( d_c1 * d_c2 )
                        cos_θ = ( abs(cos_θ) >= 1.f0 ) ? sign(cos_θ) : cos_θ
                        θ = acos( cos_θ )

                        for (n, param) in enumerate(angular_params)
                            row_index = get_angular_index(neighborAtom1_id, neighborAtom2_id,
                                                            m_A, ao, N, tri, n)
                            AEV_Matrix[row_index, centerAtom] +=
                                    G_A_singleTerm(θ, d_c1, d_c2, param, f_C_c1, f_C_c2)
                        end  # params loop

                    end  # self and distance condition: center and neighbor2
                end  # neighbor_atom2
            end  # self and distance condition: center and neighbor1
        end  # neighbor_atom1 loop
    end  # center_atom loop
    return nothing
end

#==============================================================================================#
#======================================= High Level ===========================================#

"""
    compute_AEVs!(symbols, coordinates, params)

Constructs AEV for each of the N atom in the structure, given a 3 x N matrix of coordinates.
Stores the result in the `i`th entry of the first argument.

Argument descriptions:

    `store_vector`:
        Vector containing AEVs for all data points. The function modifies this vector
    `i`:
        The i-th AEVs to compute (i-th index in `store_vector`)
    `params`:
        `Params` object created from user's .par file
    `symbols_id`:
        Ordered list of element symbols for all atoms in the structure (length `M`), given in 
        integer representation (aka "tag" or "id")
    `coordinates`:
        3 x M matrix of Cartesian coordinates of atoms in the structure
    `radial_params`:
        Vector of all permutations of BP descriptor parameters for radial subAEV (length `m_R`)
    `angular_params`:
        Vector of all permutations of BP descriptor parameters for angular subAEV (length `m_A`)
    `m_R`:
        Length of `radial_params`
    `m_A`:
        Length of `angular_params`
    `tri`:
        Dictionary used to map an integer to its triangular number, which is defined as:
            tri(n) = Σ(i, start: i=0, end: i=n)
        Needed to index angular subAEV in the AEVs matrix
    `N`:
        Number of different elements present in the dataset
            ex: if the dataset contains only C, N, O, H, and F, then `N` = 5
    `ao`:
        "Angular offset" for indexing angular subAEV in the AEVs matrix.
        Should be equal to `N*m_R`
"""
function compute_AEVs!(store_vector::Vector{Matrix{Float32}}, i::Int, params::Params,
                       symbols_id::Vector{Int}, coordinates::Matrix{Float32},
                       radial_params::Vector{BPParameters}, angular_params::Vector{BPParameters}
                       , m_R::Int, m_A::Int, tri::Dict{Int, Int}, N::Int, ao::Int) :: Nothing

    # calculate distance matrix and f_C(R_ij)
    D = distance_matrix(coordinates)
    f_C_radial = f_C.(D, params.R_cut_radial)
    f_C_angular = f_C.(D, params.R_cut_angular)

    # compute its AEV
    M = length(symbols_id)
    store_vector[i] = zeros(Float32, params.architecture[1], M)
    compute_subAEVs!(store_vector[i], radial_params, symbols_id, D, f_C_radial, m_R, M)
    compute_subAEVs!(store_vector[i], angular_params, symbols_id, D, f_C_angular, coordinates,
                        m_A, tri, N, M, ao)

    return nothing
end


"""
    coordinates_to_AEVs(params, symbols_list, coordinates_list)

Return a list of AEVs for all data points in the provided symbols and coordinates lists
"""
function coordinates_to_AEVs(params::Params, symbols_list::Vector{Vector{String}},
                        coordinates_list::Vector{Matrix{Float32}})
    # make vector of all BP parameters
    radial_subAEV_params = groups_to_BPParameters(params.radial)
    angular_subAEV_params = groups_to_BPParameters(params.angular)
    m_R = length(radial_subAEV_params)
    m_A = length(angular_subAEV_params)

    # Other preparations needed:
    N = length(params.elements)
    tri = Dict( i => sum(0:i) for i in 0:(N-1) ) # triangle numbers (for angular subAEV index)
    angular_offset = N * m_R

    L = length(symbols_list)  # number of data points
    AEVs_list = Vector{Matrix{Float32}}(undef, L)  # store AEVs
    id_list = Vector{Vector{Int}}(undef, L)  # store symbols ids
    for i in 1:L  # for all data points
        id_list[i] = [ params.elements2tags[symbol] for symbol in symbols_list[i] ]
        compute_AEVs!(AEVs_list, i, params, id_list[i], coordinates_list[i],
                        radial_subAEV_params, angular_subAEV_params, m_R, m_A, tri, N,
                        angular_offset)
    end

    return AEVs_list, id_list
end


#==============================================================================================#
#================================ Saving & Loading AEVs from file =============================#

using BSON: @save, @load

"""
    save_AEVs(filename, params, vector_of_AEVs_matrices)

Save AEVs and parameters that were used to compute the AEVs using the BSON.jl package.
Can be a single AEVs matrix or a vector of multiple AEVs matrices
"""
function save_AEVs(filename::String, params::Params,
                    AEVs_matrices::Union{Vector{Matrix{Float32}}, Matrix{Float32}},
                    symbol_id_list::Union{Vector{Vector{Int}}, Vector{Int}},
                    energies::Union{Vector{Float32}, Float32}) :: Int
    @save filename params AEVs_matrices symbol_id_list energies
    return 0  # successful
end


#"""
#    load_AEVs(filename)
#
#Load AEVs and parameters from saved binary JSON file
#"""
#function load_AEVs(filename::String)
#    @load filename params AEVs_matrices
#    return params, AEVs_matrices
#end


# if I have time, add functions to save as CSV, convert .bson to .csv, and load from CSV


"""
    concat_AEVs(AEVs...)

Combines multiple vectors of AEVs matrices into a single vector.
RISKY TO USE EXTERNALLY (assumes that params for all the AEVs are identical)
"""
function concat_AEVs(AEVs::Vararg{Vector{Matrix{Float32}}})
    return vcat(AEVs...)
end


"""
    load_AEVs(filenames)

Loads AEVs from existing .bson file(s) and combines them into one vector.
Also checks for consistency of params in the AEV files in the argument vector.
"""
function load_AEVs(filenames::Vararg{String})
    ret_params = Params()
    ret_vec_AEV = Matrix{Float32}[]
    ret_vec_ids = Vector{Int}[]
    ret_vec_energies = Float32[]
    first_file = true
    for filename in filenames
        @load filename params AEVs_matrices symbol_id_list energies  # load file

        # compare params with first file's params
        if first_file
            ret_params = params
            first_file = false
        else  # not the first file: check consistency of params
            s = check_consistency(ret_params, params)
            if s == 1
                println("WARNING: minor inconsistency found in params: $(filenames[1]), $filename")
            elseif s == 2
                throw(error("Major inconsistency found in params: $(filenames[1]), $filename"))
            elseif s == 3
                throw(error("Fatal inconsistency found in params: $(filenames[1]), $filename"))
            end
        end 

        # append to the final AEVs vector
        if typeof(AEVs_matrices) == Matrix{Float32}
            push!(ret_vec_AEV, AEVs_matrices)
            push!(ret_vec_ids, symbol_id_list)
            push!(ret_vec_energies, energies)
        else  # should be of type Vector{Matrix{Float32}}
            append!(ret_vec_AEV, AEVs_matrices)
            append!(ret_vec_ids, symbol_id_list)
            append!(ret_vec_energies, energies)
        end
    end  # file loop

    return ret_params, ret_vec_AEV, ret_vec_ids, ret_vec_energies
end  # function