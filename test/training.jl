# Test training script using part of ANI-1's existing dataset

#==============================================================================================#
using ANI1JL
using PyCall
using Random
using BSON: @save, @load
using Flux, NNlib
using Plots

#==============================================================================================#
# Helper functions for this script


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


unzip(a) = map(x -> getfield.(a, x), fieldnames(eltype(a)))


#==============================================================================================#
#=================================== Start of training script =================================#

#=
Required training data (let's say there are N datapoints):
- symbols:
    vector of length N, each element being a vector of chemical symbols
- coordinates:
    vector of length N, each element being a matrix of atomic coordinates (same order as
    the corresponding element in the "symbols" vector)
        * Each atom's (x, y, z) as a COLUMN in the matrix
- energies:
    vector of length N, each element being the true energy of the corresponding system in
    the "coordinates" vector

Workflow:
    * Take symbols, coordinates, and energies as input
    * should change all datatypes to 32 bit
    * Calculate distance and angle matrices from coordinates and symbols
    * Compute atomic environment vectors (AEV) from distance and angle matrices
        * to vectorize training, create unnecessary AEVs too ?
    * Pre-process data (mean = 0, std = 1)
    * initalizing weights: "He initialization for RELU"
        * https://stats.stackexchange.com/questions/229885/whats-the-recommended-weight-initialization-strategy-when-using-the-elu-activat
        * but the ANI paper initialized from random normal distribution between the ranges (-1/d, 1/d), where d is the # of inputs into the node
    * pre-process results in each layer? (batch normalization)
    * Train and return the model (multiple times, each with smaller learning rate)
        * max norm regularization
        * first training: vectorized (0 inputs); subsequent trainings with no 0 inputs ?
=#


#==============================================================================================#
# Make AEVS

# load data
data_filenames = ["data/data_01-06.jld"]

symbols = Vector{Vector{String}}()
coordinates = Vector{Matrix{Float32}}()
energies = Vector{Float32}()
for file in data_filenames
    local_symbols, local_coordinates, local_energies = import_ani_data(file)
    append!(symbols, local_symbols)
    append!(coordinates, local_coordinates)
    append!(energies, local_energies)
end

# set parameters for AEVs and NN (using default here, but can also specify path to a .par file)
params = set_params("params.par")
# params = set_params()  # default parameters
println(params)

# make AEVs for all data points
AEVs_list, symbol_id_list = coordinates_to_AEVs(params, symbols, coordinates)
println("finished making AEVs")

# Randomly shuffle dataset
dataset = collect(zip(AEVs_list, symbol_id_list, energies))
shuffle!(dataset)
AEVs_list, symbol_id_list, energies = unzip(dataset)
println("finished random shuffle")

# Creats training, validation, and test sets
validation_proportion = 0.08  # proportion of data to use as validation set
test_set_proportion = 0.02  # proportion of data to use as test set
N = Int( length(energies) )
N_val = Int( floor(N * validation_proportion) )
N_test = Int( floor(N * test_set_proportion) )
validation_set = (AEVs_list[1:N_val], symbol_id_list[1:N_val], energies[1:N_val])
test_set = (AEVs_list[(N_val+1):(N_val+N_test)], symbol_id_list[(N_val+1):(N_val+N_test)],
                energies[(N_val+1):(N_val+N_test)])
training_set = (AEVs_list[(N_val+N_test+1):end], symbol_id_list[(N_val+N_test+1):end],
                energies[(N_val+N_test+1):end])
println("finished making training, validation, and test sets")

# save to file
save_AEVs("AEVs/training_set_01-06.jld", params, training_set...)
save_AEVs("AEVs/validation_set_01-06.jld", params, validation_set...)
save_AEVs("AEVs/test_set_01-06.jld", params, test_set...)
println("finished saving datasets")

#==============================================================================================#
# Train

#training_set_filename = "AEVs/training_set_01-06.jld"
#validation_set_filename = "AEVs/validation_set_01-06.jld"
#
## load datasets
#params, AEVs, symbol_ids, energies = load_AEVs(training_set_filename)  # training
#training_set = (AEVs, symbol_ids, energies)
#println(params)
#params_val, AEVs_val, symbol_ids_val, energies_val = load_AEVs(validation_set_filename)  # val
#validation_set = (AEVs_val, symbol_ids_val, energies_val)
#if check_consistency(params, params_val) >= 2
#    println("Exiting due to inconsistent params between training and validations sets")
#    exit(code=1)
#end

#= 0 =#
id=0

# Load neural network
NN_Potential = atomic_nnps(params)

# Training settings
batch_size = 68
η = 0.1

model_filename = "NNPs_$(id).bson"
log_file = "training_log_$(id).csv"

# Train network (package automatically saves checkpoints)
train_nnps!(NN_Potential; training_set=training_set, validation_set=validation_set,
                batch_size=batch_size, η=η, best_val_save_file=model_filename,
                log_file=log_file)

#= 0.1 =#
id=0.1

# Load neural network
NN_Potential =
    try
        load_nnps(model_filename)
    catch
        save_nnps(model_filename, NN_Potential)
        NN_Potential
    end

# Training settings
batch_size = 68
η = 0.01

old_model_filename = model_filename
model_filename = "NNPs_$(id).bson"
log_file = "training_log_$(id).csv"

# Train network (package automatically saves checkpoints)
train_nnps!(NN_Potential; training_set=training_set, validation_set=validation_set,
                batch_size=batch_size, η=η, best_val_save_file=model_filename,
                log_file=log_file)


#= 0.2 =#
id=0.2

# Load neural network
NN_Potential =
    try
        load_nnps(model_filename)
    catch
        temp_model = load_nnps(old_model_filename)
        save_nnps(model_filename, temp_model)
        temp_model  # set to NN_Potential
    end

# Training settings
batch_size = 68
η = 0.001

old_model_filename = model_filename
model_filename = "NNPs_$(id).bson"
log_file = "training_log_$(id).csv"

# Train network (package automatically saves checkpoints)
train_nnps!(NN_Potential; training_set=training_set, validation_set=validation_set,
                batch_size=batch_size, η=η, best_val_save_file=model_filename,
                log_file=log_file)

#= 0.3 =#
id=0.3

# Load neural network
NN_Potential =
    try
        load_nnps(model_filename)
    catch
        temp_model = load_nnps(old_model_filename)
        save_nnps(model_filename, temp_model)
        temp_model  # set to NN_Potential
    end

# Training settings
batch_size = 68
η = 0.0001

old_model_filename = model_filename
model_filename = "NNPs_$(id).bson"
log_file = "training_log_$(id).csv"

# Train network (package automatically saves checkpoints)
train_nnps!(NN_Potential; training_set=training_set, validation_set=validation_set,
                batch_size=batch_size, η=η, best_val_save_file=model_filename,
                log_file=log_file)

#= 0.4 =#
id=0.4

# Load neural network
NN_Potential =
    try
        load_nnps(model_filename)
    catch
        temp_model = load_nnps(old_model_filename)
        save_nnps(model_filename, temp_model)
        temp_model  # set to NN_Potential
    end

# Training settings
batch_size = 68
η = 0.00001

old_model_filename = model_filename
model_filename = "NNPs_$(id).bson"
log_file = "training_log_$(id).csv"

# Train network (package automatically saves checkpoints)
train_nnps!(NN_Potential; training_set=training_set, validation_set=validation_set,
                batch_size=batch_size, η=η, best_val_save_file=model_filename,
                log_file=log_file)


#= 0.5 =#
id=0.5

# Load neural network
NN_Potential =
    try
        load_nnps(model_filename)
    catch
        temp_model = load_nnps(old_model_filename)
        save_nnps(model_filename, temp_model)
        temp_model  # set to NN_Potential
    end

# Training settings
batch_size = 68
η = 0.000001

old_model_filename = model_filename
model_filename = "NNPs_$(id).bson"
log_file = "training_log_$(id).csv"

# Train network (package automatically saves checkpoints)
train_nnps!(NN_Potential; training_set=training_set, validation_set=validation_set,
                batch_size=batch_size, η=η, best_val_save_file=model_filename,
                log_file=log_file)

#==============================================================================================#
# Analze results

#=
training_set_filename = "AEVs/training_set_01-06.jld"
validation_set_filename = "AEVs/validation_set_01-06.jld"
test_set_filename = "AEVs/test_set_01-06.jld"
nnps_filename = ""

# load datasets
params, AEVs, symbol_ids, energies = load_AEVs(training_set_filename)  # training
training_set = (AEVs, symbol_ids, energies)
println(params)
params_val, AEVs_val, symbol_ids_val, energies_val = load_AEVs(validation_set_filename)  # val
validation_set = (AEVs_val, symbol_ids_val, energies_val)
if check_consistency(params, params_val) >= 2
    println("Exiting due to inconsistent params between training and validations sets")
    exit(code=1)
end
params_test, AEVs_test, symbol_ids_test, energies_test = load_AEVs(test_set_filename)  # test
test_set = (AEVs_test, symbol_ids_test, energies_test)
if check_consistency(params, params_val) >= 2
    println("Exiting due to inconsistent params between training and test sets")
    exit(code=1)
end

# load NNPs
nnps = load_nnps(nnps_filename)

# Get NNP energies for the 3 datasets
training_nnps_energies = nnp_energy.([nnps], AEVs, symbol_ids)
validation_nnps_energies = nnp_energy.([nnps], AEVs_val, symbol_ids_val)
test_nnps_energies = nnp_energy.([nnps], AEVs_test, symbol_ids_test)

# write to CSV
#function write_to_csv(filename::String, true_energies, nnps_energies)
#    N = length(true_energies)
#    @assert N == length(nnps_energies)
#    open(filename, "w") do io
#        for i in 1:N
#            true_energy = true_energies[i]
#            nnps_energy = nnps_energies[i]
#            write(io, "$true_energy, $nnps_energy\n")
#        end  # for
#        write(io, "\n")
#    end  # open file
#end  # function

# Plotting results

X = [energies, energies_val, energies_test]
Y = [training_nnps_energies, validation_nnps_energies, test_nnps_energies]
legend = ["Training Set" "Validation Set" "Test Set"]

gr()
p = plot(X, Y, seriestype = :scatter, title = "True vs Fit Energies (RMSE = #)")
xlabel!("DFT Energy (kcal/mol)")
ylabel!("NNP Energy (kcal/mol)")
display(p)
savefig(p, "combined_plot")

trp = plot(X[1], Y[1], seriestype = :scatter, title = "Training Set")
xlabel!("DFT Energy (kcal/mol)")
ylabel!("NNP Energy (kcal/mol)")
display(trp)
savefig(trp, "training_set_plot")

vp = plot(X[2], Y[2], seriestype = :scatter, title = "Validation Set")
xlabel!("DFT Energy (kcal/mol)")
ylabel!("NNP Energy (kcal/mol)")
display(vp)
savefig(vp, "validation_plot")

tep = plot(X[3], Y[3], seriestype = :scatter, title = "Test Set")
xlabel!("DFT Energy (kcal/mol)")
ylabel!("NNP Energy (kcal/mol)")
display(tep)
savefig(tep, "test_plot")

=#

println("Training script finished")