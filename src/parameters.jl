"""
Methods to parse and store model parameters
"""

module parameters


Base.@kwdef mutable struct Params
    """
    Struct to hold all parameters passsed in by the user through the .par file
    """
    # ex: ["C", "N", "O", "H"]
    elements::Vector{String} = String[]
    # ex: 4.6 (Angstroms)
    R_cut_radial::Float64 = 0.0
    # ex: 4.0 (Angstroms)
    R_cut_angular::Float64 = 0.0
    # ex: [(eta1, R_s1), (eta2, R_s2), ...]
    radial::Vector{NTuple{2, Float64}} = NTuple{2, Float64}[]
    # ex: [(zeta1, theta_s1, eta1, R_s1), ...]
    angular::Vector{NTuple{4, Float64}} = NTuple{4, Float64}[]
    architecture::Vector{Int64} = Int64[]  # ex: [768, 128, 128, 64, 1] (include input & output)
    biases::Vector{String} = String[]  # ex: ["y", "y", "y", "n"]
    activation::Vector{String} = String[]  # ex: ["gelu", "gelu", "gelu", "gelu"]
end

# overload @show
Base.show(io::IO, p::Params) =
    print(io, """
              Elements     : $(p.elements)
              R_cut        : $(p.R_cut_radial) (Radial), $(p.R_cut_angular) (Angular)
              Radial pairs : $(length(p.radial))
              $(p.radial)
              Angular pairs: $(length(p.angular))
              $(p.angular)
              Architecture : $(p.architecture)
              Biases       : $(p.biases)
              Activation   : $(p.activation)
              """)
              # add input and final layers; something is still wrong in R_cut...


#==============================================================================================#
# Helper functions for parse_params()

function read_single_line(lines::Vector{String}, i::Int64)
    """
    Return a vector of split words of a single line
    """
    return split(lines[i])
end

function read_par_file(filename::String) :: Vector{String}

    f = open(filename, "r")
    lines = readlines(f)
    close(f)

    filter!(x -> !isempty(x), lines)  # get rid of empty lines
    filter!(x -> (x[1] != '#'), lines)  # get rid of comment lines

    return lines
end

function set_from_groups!(params::Params, numGroups::Int64, lines::Vector{String},
                          prevIndex::Int64, fieldname::String)
    """
    Read from explicitly provided parameter groups and set params field
    """
    field = Symbol(fieldname)
    for i = (prevIndex + 1):(prevIndex + numGroups)
        line = read_single_line(lines, i)
        tline = Tuple(map(x -> parse(Float64, x), line))  # convert into float Tuple
        push!( getproperty(params, field) , tline )  # append to the appropriate vector field
    end
end

function set_from_lists!(params::Params, numParameters::Int64, lines::Vector{String},
                         prevIndex::Int64, fieldname::String)
    """
    Read from lists of individual parameters and set params field by distributing over all
    """
    field = Symbol(fieldname)
    values = Vector{Float64}[]
    for i = (prevIndex + 1):(prevIndex + numParameters)
        line = read_single_line(lines, i)
        tline = map(x -> parse(Float64, x), line)
        push!(values, tline)
    end
    temp = vec( collect(Base.product(values...)) )  # a 1D vector of all groups
    setproperty!(params, field, temp)
end

function set_subAEV!(params::Params, keyword::String, num_elements::Int64,
                     lines::Vector{String}, counter::Int64) :: Int64
    try  # number of pairs explicitly provided
        numGroups = parse(Int64, read_single_line(lines, counter)[2])
        set_from_groups!(params, numGroups, lines, counter, keyword)
        counter += numGroups
    catch  # number of pairs not provided
        #num_to_read = 4  # ζ, θ_s, η, and R_s
        set_from_lists!(params, num_elements, lines, counter, keyword)
        counter += num_elements
    end

    # read R_cut
    counter += 1
    line = read_single_line(lines, counter)
    R_cut_field = Symbol("R_cut_" * keyword)
    R_cut = parse(Float64, line[1])
    setproperty!(params, R_cut_field, R_cut)

    # return updated counter
    return counter
end

function set_elements!(params::Params, d::Union{Vector{SubString{String}}, Vector{String}})
    params.elements = d
end

function set_architecture!(params::Params, d::Union{Vector{SubString{String}}, Vector{String}})
    params.architecture = map(x -> parse(Int64, x), d)
end

function set_biases!(params::Params, d::Union{Vector{SubString{String}}, Vector{String}})
    params.biases = d
end

function set_activation!(params::Params, d::Union{Vector{SubString{String}}, Vector{String}})
    params.activation = d
end

#==============================================================================================#
function parse_params(par_file::String) :: Params
    """
    Parse .par file
    """
    flines = read_par_file(par_file)

    # read parameters
    params = Params()
    L = length(flines)
    counter = 1
    while counter <= L
        line = read_single_line(flines, counter)
        keyword = lowercase(line[1])

        if keyword == "elements"
            set_elements!(params, line[2:end])
        elseif keyword == "radial"
            # 2 is because of: η and R_s
            counter = set_subAEV!(params, keyword, 2, flines, counter)
        elseif keyword == "angular"
            # 4 is because of: ζ, θ_s, η, and R_s
            counter = set_subAEV!(params, keyword, 4, flines, counter)
        elseif keyword == "architecture"
            set_architecture!(params, line[2:end])
        elseif keyword == "biases"
            set_biases!(params, line[2:end])
        elseif keyword == "activation"
            set_activation!(params, line[2:end])
        else
            if isempty(keyword)
                throw(error("Empty keyword in parameter file."))
            else
                throw(error("Unknown keyword in parameter file: $keyword"))
            end
        end

        counter += 1
    end

    # sanity checks: length of bias and activation vectors are one more than architecture length
    try
        @assert length(params.architecture) == length(params.biases) - 1
        @assert length(params.biases) == length(params.activation)
    catch
        throw(error("Inconsistent hyperparameters input"))
    end

    # insert input & output layer information into params
    N = length(params.elements)
    m_R = length(params.radial)
    m_A = length(params.angular)
    len_radial_subAEV = m_R * N
    len_angular_subAEV = m_A * N * (N + 1) / 2
    insert!(params.architecture, 1, len_radial_subAEV + len_angular_subAEV)  # input layer
    push!(params.architecture, 1)  # output layer

    return params
end


end