using PyCall

data_file_name = "C:\\Users\\sungh\\code\\ANI-1_release\\ani_gdb_s02.h5"

#==============================================================================================#
function import_ani_data(filename::String)
    include("../tools/pyanitools.jl")
    adl = py"anidataloader"(filename)

    for data in adl
        @info typeof(data)
    end
end

#==============================================================================================#
function main(data_file_name::String)
    import_ani_data(data_file_name)
end


main(data_file_name)