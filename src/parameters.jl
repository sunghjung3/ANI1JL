"""
    Params
Struct to hold all parameters passsed in by the user through the .par file
"""
Base.@kwdef mutable struct Params
    # ex: ["C", "N", "O", "H"]
    elements::Vector{String} = ["C", "N", "O", "H"]

    # mapping between elements (String representation) to tags (Int representation) and reverse
    elements2tags::Dict{String, Int} = Dict("C" => 1, "N" => 2, "H" => 4, "O" => 3)
    tags2elements::Dict{Int, String} = Dict(1 => "C", 2 => "N", 4 => "H", 3 => "O")

    # ex: 4.6 (Angstroms)
    R_cut_radial::Float32 = 4.6
    # ex: 4.0 (Angstroms)
    R_cut_angular::Float32 = 4.0
    # ex: [(eta1, R_s1), (eta2, R_s2), ...]
    radial::Vector{NTuple{2, Float32}} =
            Tuple{Float32, Float32}[(4.0, 0.5), (4.0, 1.0), (4.0, 1.55), (4.0, 2.1),
                                    (4.0, 2.65), (4.0, 3.2), (4.0, 3.75), (4.0, 4.3)]
    # ex: [(zeta1, theta_s1, eta1, R_s1), ...]
    angular::Vector{NTuple{4, Float32}} =
            NTuple{4, Float32}[(8.0, -1.571, 4.0, 0.5), (8.0, 0.0, 4.0, 0.5),
                               (8.0, 1.571, 4.0, 0.5), (8.0, 3.141, 4.0, 0.5),
                               (8.0, -1.571, 4.0, 1.1), (8.0, 0.0, 4.0, 1.1),
                               (8.0, 1.571, 4.0, 1.1), (8.0, 3.141, 4.0, 1.1),
                               (8.0, -1.571, 4.0, 1.7), (8.0, 0.0, 4.0, 1.7),
                               (8.0, 1.571, 4.0, 1.7), (8.0, 3.141, 4.0, 1.7),
                               (8.0, -1.571, 4.0, 2.3), (8.0, 0.0, 4.0, 2.3),
                               (8.0, 1.571, 4.0, 2.3), (8.0, 3.141, 4.0, 2.3),
                               (8.0, -1.571, 4.0, 2.9), (8.0, 0.0, 4.0, 2.9), 
                               (8.0, 1.571, 4.0, 2.9), (8.0, 3.141, 4.0, 2.9),
                               (8.0, -1.571, 4.0, 3.5), (8.0, 0.0, 4.0, 3.5),
                               (8.0, 1.571, 4.0, 3.5), (8.0, 3.141, 4.0, 3.5)]
    architecture::Vector{Int} = [272, 64, 1]  # include input & output
    biases::Vector{String} = ["y", "y"] # allow biases between each layer?
    activation::Vector{String} = ["gelu", "gelu"]  # activation functions
end

# overload @show
Base.show(io::IO, p::Params) =
    print(io, """
              Elements       : $(p.elements)
              Tags           : $(p.elements2tags)
              R_cut          : $(p.R_cut_radial) (Radial), $(p.R_cut_angular) (Angular)
              Radial groups  : $(length(p.radial))
              $(p.radial)
              Angular groups : $(length(p.angular))
              $(p.angular)
              Architecture   : $(p.architecture)
              Biases         : $(p.biases)
              Activation     : $(p.activation)
              """)
              # add input and final layers; something is still wrong in R_cut...


"""
    check_consistency(params1, params2)

Compares fields of 2 params variables and returns consistency status.
    status == 0 : identical
    status == 1 : minor inconsistency
    status == 2 : major inconsistency
    status == 3 : fatal inconsistency
"""
function check_consistency(params1::Params, params2::Params) :: Int
    if params1.elements != params2.elements
        return 3
    elseif params1.elements2tags != params2.elements2tags
        return 3
    elseif params1.R_cut_radial != params2.R_cut_radial
        return 2
    elseif params1.R_cut_angular != params2.R_cut_angular
        return 2
    elseif params1.radial != params2.radial  # this vector is ordered
        return 2
    elseif params1.angular != params2.angular  # this vector is ordered
        return 2
    elseif params1.architecture != params2.architecture
        return 1
    elseif params1.biases != params2.biases
        return 1
    elseif params1.activation != params2.activation
        return 1
    end
    return 0
end

#==============================================================================================#
# Helper functions for parse_params()

"""
    read_single_line(lines, i)

Return a vector of split words of a single line
"""
function read_single_line(lines::Vector{String},
                          i::Union{Int32, Int64}) :: Vector{SubString{String}}
    return split(lines[i])
end


"""
    read_par_file(filename)

Read and filter given parameter file and returns vector of each line
"""
function read_par_file(filename::String) :: Vector{String}
    f = open(filename, "r")
    lines = readlines(f)
    close(f)

    filter!(x -> !isempty(x), lines)  # get rid of empty lines
    filter!(x -> (x[1] != '#'), lines)  # get rid of comment lines

    return lines
end


"""
    set_from_groups!(params, numGroups, lines, prevIndex, fieldname)

Read from explicitly provided parameter groups and set params field
"""
function set_from_groups!(params::Params, numGroups::Union{Int32, Int64}, lines::Vector{String},
                          prevIndex::Union{Int32, Int64}, fieldname::String)
    field = Symbol(fieldname)
    setproperty!(params, field, [])
    for i = (prevIndex + 1):(prevIndex + numGroups)
        line = read_single_line(lines, i)
        tline = Tuple(map(x -> parse(Float32, x), line))  # convert into float Tuple
        push!( getproperty(params, field) , tline )  # append to the appropriate vector field
    end
end


"""
    set_from_lists!(params, numParameters, lines, prevIndex, fieldname)

Read from lists of individual parameters and set params field by distributing over all
"""
function set_from_lists!(params::Params, numParameters::Union{Int32, Int64}, lines::Vector{String},
                         prevIndex::Union{Int32, Int64}, fieldname::String)
    field = Symbol(fieldname)
    values = Vector{Float32}[]
    for i = (prevIndex + 1):(prevIndex + numParameters)
        line = read_single_line(lines, i)
        tline = map(x -> parse(Float32, x), line)
        push!(values, tline)
    end
    temp = vec( collect(Base.product(values...)) )  # a 1D vector of all groups
    setproperty!(params, field, temp)
end


"""
    set_subAEV!(params, keyword, num_elements, lines, counter::Int)

Sets radial or angular field of Params object (as well as their R_cut fields).
Returns updated counter
"""
function set_subAEV!(params::Params, keyword::String, num_elements::Union{Int32, Int64},
                     lines::Vector{String}, counter::Union{Int32, Int64}) :: Union{Int32, Int64}
    try  # number of pairs explicitly provided
        numGroups = parse(Int32, read_single_line(lines, counter)[2])
        set_from_groups!(params, numGroups, lines, counter, keyword)
        counter += numGroups
    catch BoundsError # number of pairs not provided
        #num_to_read = 4  # ζ, θ_s, η, and R_s
        set_from_lists!(params, num_elements, lines, counter, keyword)
        counter += num_elements
    end

    # read R_cut
    counter += 1
    line = read_single_line(lines, counter)
    R_cut_field = Symbol("R_cut_" * keyword)
    R_cut = parse(Float32, line[1])
    setproperty!(params, R_cut_field, R_cut)

    # return updated counter
    return counter
end

function set_elements!(params::Params, d::Union{Vector{SubString{String}}, Vector{String}})
    params.elements = unique(d)
    params.tags2elements = Dict{Int, String}(collect( enumerate(params.elements) ))
    params.elements2tags = Dict(element=> tag for (tag, element) in params.tags2elements)
end

function set_architecture!(params::Params, d::Union{Vector{SubString{String}}, Vector{String}})
    params.architecture = map(x -> parse(Int, x), d)
end

function set_biases!(params::Params, d::Union{Vector{SubString{String}}, Vector{String}})
    params.biases = d
end

function set_activation!(params::Params, d::Union{Vector{SubString{String}}, Vector{String}})
    params.activation = d
end

#==============================================================================================#

"""
    parse_params(par_file::String) :: Params

Parse given .par file, and returns Params object
"""
function parse_params(par_file::String) :: Params
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


"""
    set_params(par_file)

Returns Params object created from reading par_file. Just calls parse_params()
"""
function set_params(par_file::String)
    return parse_params(par_file)
end


"""
    set_params()

Returns Params object with default parameters.
"""
function set_params()
    return Params()
end