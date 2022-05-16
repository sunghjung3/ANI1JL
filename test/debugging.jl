using ANI1JL
using JLD
@time eval

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

#=
# load data
data_filenames = ["test/data/ani_gdb_s01.h5",
                  "test/data/ani_gdb_s02.h5",
                  "test/data/ani_gdb_s03.h5",
                  "test/data/ani_gdb_s04.h5",
                  "test/data/ani_gdb_s05.h5",
                  "test/data/ani_gdb_s06.h5"
                ]

symbols = Vector{Vector{String}}()
coordinates = Vector{Matrix{Float32}}()
energies = Vector{Float32}()
for file in data_filenames
    @show file
    local_symbols, local_coordinates, local_energies = import_ani_data(file)
    append!(symbols, local_symbols)
    append!(coordinates, local_coordinates)
    append!(energies, local_energies)
end
=#

data_filename = "test/data/ani_gdb_s08.h5"
symbols, coordinates, energies = import_ani_data(data_filename)
nChunks = 20
N = length(energies)

println("Saving $(N) datapoints in $nChunks chunks...")
div_whole = Int(floor(N / nChunks))
div_remainder = N % nChunks
for n in 11:nChunks-1
    save_filename = "data_08_$(n).jld"
    lower_bound = div_whole * n + min(div_remainder, n) + 1
    upper_bound = lower_bound + div_whole + (n < div_remainder) - 1
    @show n, lower_bound, upper_bound
    @time save(save_filename, "symbols", symbols[lower_bound:upper_bound], "coordinates",
            coordinates[lower_bound:upper_bound], "energies", energies[lower_bound: upper_bound])
end
# save("data_08-08.jld", "symbols", symbols, "coordinates", coordinates, "energies", energies)
println("Done")