# Test training script using part of ANI-1's existing dataset

#==============================================================================================#
using ANI1JL
using PyCall

#==============================================================================================#
function import_ani_data(filename::String)
    adl = pyanitools.anidataloader(filename)

    # data vectors to return
    symbols = Vector{Vector{String}}()
    coordinates = Vector{Matrix{Float32}}()
    energies = Vector{Float32}()

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
            push!(coordinates, X[i, :, :]')  # transpose to make each (x, y, z) a column
        end
    end

    adl.cleanup()

    return symbols, coordinates, energies
end

#==============================================================================================#
function main()
    data_file_name = raw"C:\Users\sungh\code\ANI1JL\test\data\ani_gdb_s01.h5"
    symbols, coordinates, energies = import_ani_data(data_file_name)
    # model = ANI1JL.train(symbols, coordinates, energies)


    # println(model)
end


main()