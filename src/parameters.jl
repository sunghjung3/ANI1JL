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
    radial::Vector{Tuple{Float64, Float64}} = Tuple{Float64, Float64}[]
    # ex: [(zeta1, theta_s1, eta1, R_s1), ...]
    angular::Vector{Tuple{Float64, Float64, Float64, Float64}} = Tuple{Float64, Float64,
                                                                       Float64, Float64}[]
    architecture::Vector{Int64} = Int64[]  # ex: [128, 128, 64]
    biases::Vector{String} = String[]  # ex: ["y", "y", "y", "n"]
    activation::Vector{String} = String[]  # ex: ["gelu", "gelu", "gelu", "gelu"]
end

# overload @show
Base.show(io::IO, p::Params) =
    print(io, """
              Elements     : $(p.elements)
              R_cut        : $(p.R_cut_radial) (Radial), $(p.R_cut_angular) (Angular)
              Radial pairs : $(p.radial)
              Angular pairs: $(p.angular)
              Architecture : $(p.architecture)
              Biases       : $(p.biases)
              Activation   : $(p.activation)
              """)
              # add input and final layers; something is still wrong in R_cut...

function multiline_parse(params::Params, flines::Vector{String})  :: Int64
    """
    Helper for parse_params() to parse multi-line inputs
    """

end

function parse_params(par_file::String) :: Params
    """
    Parse .par file
    """
    f = open(par_file, "r")
    flines = readlines(f)
    close(f)
    filter!(x -> !isempty(x), flines)  # get rid of empty lines
    filter!(x -> (x[1] != '#'), flines)  # get rid of comment lines

    # read parameters
    params = Params()
    L = length(flines)
    counter = 1
    while counter <= L
        line = split(flines[counter])  # single line
        keyword = lowercase(line[1])

        if keyword == "elements"
            params.elements = line[2:end]
        elseif keyword == "radial"
            if isempty(line[2])  # explicit pairs not listed

            else  # explicit pairs listed
                numPairs = parse(Int64, line[2])
                for i = (counter + 1):(counter + numPairs)
                    line = split(flines[i])

                    # convert the pair into a tuple
                    push!(params.radial, Tuple(map(x -> parse(Float64, x), line)) )
                end
                counter += numPairs

                # read R_cut
                line = split(flines[counter])
                params.R_cut_radial = parse(Float64, line[1])
                counter += 1
            end
        elseif keyword == "angular"
            try
                numPairs = parse(Int64, line[2])  # number of pairs explicitly provided
                for i = (counter + 1):(counter + numPairs)
                    line = split(flines[i])

                    # convert the pair into a tuple
                    push!(params.angular, Tuple(map(x -> parse(Float64, x), line)) )
                end
                counter += numPairs

                # read R_cut
                line = split(flines[counter])
                params.R_cut_angular = parse(Float64, line[1])
                counter += 1
            catch e  # number of pairs not provided
                counter += 1
                line = split(flines[counter])
                ζ_list = map(x -> parse(Float64, x), line)
                counter += 1
                line = split(flines[counter])
                θs_list = map(x -> parse(Float64, x), line)
                counter += 1
                line = split(flines[counter])
                η_list = map(x -> parse(Float64, x), line)
                counter += 1
                line = split(flines[counter])
                Rs_list = map(x -> parse(Float64, x), line)

                for ζ in ζ_list
                    for θs in θs_list
                        for η in η_list
                            for Rs in Rs_list
                                push!(params.angular, (ζ, θs, η, Rs))
                            end
                        end
                    end
                end
                counter += 1

                # read R_cut
                line = split(flines[counter])
                params.R_cut_angular = parse(Float64, line[1])
                counter += 1

            end

        elseif keyword == "architecture"
            params.architecture = line[2:end]
        elseif keyword == "biases"
            params.biases = line[2:end]
        elseif keyword == "activation"
            params.activation = line[2:end]
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

    println(params)
    return params
end


end