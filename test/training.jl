# Test training script using part of ANI-1's existing dataset

#==============================================================================================#
using PyCall

data_file_name = "C:\\Users\\sungh\\code\\ANI-1_release\\ani_gdb_s01.h5"

include("../src/ANI1JL.jl")

#==============================================================================================#
function import_ani_data(filename::String)
    include("../tools/pyanitools.jl")
    adl = py"anidataloader"(filename)

    # data vectors to return
    symbols = Vector{Vector{String}}()
    coordinates = Vector{Matrix{Float64}}()
    energies = Vector{Float64}()

    # extract all data
    for data in adl
        P = data["path"]
        X = data["coordinates"]
        E = data["energies"]
        S = data["species"]
        sm = data["smiles"]

        # add to return vectors
        append!(energies, E)
        for i = 1:length(E)  # for all the energies
            push!(symbols, S)  # all the same species
            push!(coordinates, X[i, :, :])
        end
    end

    adl.cleanup()

    return symbols, coordinates, energies
end

#==============================================================================================#
function main(data_file_name::String)
    symbols, coordinates, energies = import_ani_data(data_file_name)
    model = ANI1JL.train(symbols, coordinates, energies)
    println(model)
end


main(data_file_name)