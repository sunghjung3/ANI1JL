"""
    fileio

Helper methods for file io
"""
module fileio

export write_xyz

"""
    write_xyz()
"""
function write_xyz(filename::String, symbols::Vector{String},
                    coord::Union{Matrix{Float32}, Matrix{Float64}})
    dims = size(coord)
    N = length(symbols)
    @assert dims[2] == N  # number of columns equal to number of atoms
    
    if dims[1] == 1  # if 1D, add 0 on all y and z coordinates
        coord = vcat(coord, zeros(2, N))
    elseif dims[1] == 2  # if 2D, add 0 on all z coordinates
        coord = vcat(coord, zeros(1, N))
    elseif dims[1] != 3  # if not 3D, throw error
        throw(error(DimensionMismatch("coordinates must be 1, 2, or 3 dimensional")))
    end

    open(filename, "w") do io
        write(io, string(N)*"\n\n")  # number of atoms
        for (atom_num, atom_sym) in enumerate(symbols)
            write(io, atom_sym)
            for x in coord[:, atom_num]
                write(io, "\t"*string(x))
            end  # for x
            write(io, "\n")
        end  # for col
    end  # open
end


end  # module