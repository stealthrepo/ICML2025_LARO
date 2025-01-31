### This code solves the entire problem in a monolithic Structure 
### Sanity chcek if the CCG results are valid 
### The same code is used to solve the master problem with selected uncertainty for exact solution


using DataStructures
using JuMP
using Random
using Gurobi
using JSON
using CSV
using DataFrames

folder ="source"
file_name = "instance_generation.jl"
include(joinpath(parent_dir, folder, file_name))

file_name = "RO_functions.jl"
include(joinpath(parent_dir, folder, file_name))
############################################################################################################################################################################################
file_name = "nn_to_MILP.jl"
include(joinpath(parent_dir, folder, file_name))

############################################################################################################################################################################################
include("source\\Master Solve.jl")
include("decomposition_Knapsack_iterative.jl")
############################################################################################################################################################################################

###########################################################################################################################################################################################################################################
### Initiate the data with I items

num_items = 20

num_train_uncern = 50

folder_1 = "post train NN data"
folder_2 = "$(num_items) instance $(num_train_uncern) scenarios improved"   # add improved to the folder name if the improved version of the neural network is used

storage_folder = "results NN 2SRO"
############################################################################################################################################################################################################################################
### get the scaling factors for the problem instance, target and uncertainities that were used before training the neural network

file_name = "min_max_scalers_inst_$(num_items).json"
## Join the path and open the csv file
path = joinpath(parent_dir,folder_1, folder_2, file_name)
min_max_scalers = JSON.parsefile(path)
uncertainty_max = min_max_scalers["scaler_uncern.data_max_"]
uncertainty_min = min_max_scalers["scaler_uncern.data_min_"]
target_max = min_max_scalers["scaler_target.data_max_"]
target_min = min_max_scalers["scaler_target.data_min_"]
instance_min = min_max_scalers["scaler_instance.data_min_"]
instance_max = min_max_scalers["scaler_instance.data_max_"]

#############################################################################################################################################################################################################################################
### Get the weights and bias matrices for the trained neural network
file_name = "model_weight_$(num_items)_instance_$(num_train_uncern)_scen.json"
## Join the path and open the csv file
path = joinpath(parent_dir,folder_1, folder_2, file_name)
weight, bias = json_to_weights(path)


# #############################################################################################################################################################################################################################################

# file_name = "post_NN_results_$(num_items)_instance_$(num_train_uncern)_scen.csv"
# path = joinpath(parent_dir, folder_1, folder_2, file_name)
# df = CSV.read(path, DataFrame)
# input_uncern_column_names = sort([col for col in names(df) if contains(lowercase(col), "uncern")], by=custom_sort)       ## Sort the uncertainity columns custom sort since the column name is a string which contains numeric values

# scaled_global_uncertainty_matrix = unique(Matrix(select(df, input_uncern_column_names)), dims=1)                            ## Get the unique values of the uncertainity set, all values here are scaled between 0 and 1
# unscaled_global_uncertainty_matrix = scale_inverse(scaled_global_uncertainty_matrix', uncertainty_min, uncertainty_max)'     ## Scale the uncertainity set back to the original scale
#############################################################################################################################################################################################################################################
#############################################################################################################################################################################################################################################
## Get the seed list for the problem instance

file_name = "benchmarking_seed_list.txt"
path = joinpath(parent_dir, folder_1, "benchmarking_uncertainties", file_name)
seed_list = readlines(path)

#############################################################################################################################################################################################################################################
function uncertainty_matrix(num_items, gamma)
    file_name = "num_items_$(num_items)_benchmarking_uncertainty_matrix_gamma_$(gamma).csv"
    path = joinpath(parent_dir, folder_1, "benchmarking_uncertainties", file_name)
    benchmark_uncertainty_dict = CSV.read(path, DataFrame)
    return Matrix(benchmark_uncertainty_dict)
end
#############################################################################################################################################################################################################################################

file_name = "benchmarking_seed_list.txt"
path = joinpath(parent_dir, folder_1, "benchmarking_uncertainties", file_name)
seed_list = readlines(path)

#############################################################################################################################################################################################################################################

# model_exact_env = Gurobi.Env()
# seed = 3
# Random.seed!(seed)
# # Create the data instance
# instance = knapsack_instances(seed, num_items; train=false)

# f = instance["f"]
# p_bar = instance["p_bar"]
# t = instance["t"]
# p_hat = instance["p_hat"]
# C = instance["C"]
# w = instance["w"]
# gamma = instance["budget_uncertainity_parameter"]

# global_uncertainty_matrix = uncertainty_matrix(num_items, gamma)

# global_uncertainty_dict = OrderedDict()

# for i in 1:size(global_uncertainty_matrix, 1)
#     global_uncertainty_dict[i] = global_uncertainty_matrix[i, :]
# end

# full_exact_results = master_stage_exact(model_exact_env, num_items, global_uncertainty_dict, f, p_bar, t, p_hat, C, w)


#############################################################################################################################################################################################################################################

#Storage for results
results = OrderedDict{Int, Any}()
time_dict = OrderedDict{Int, Any}()


model_exact_env = Gurobi.Env()

for seed in seed_list
    seed = parse(Int, seed)
    println("Running for seed: ", seed)
    start_time = Dates.now()

    Random.seed!(seed)
    # Create the data instance
    instance = knapsack_instances(seed, num_items; train=false)
    f = instance["f"]
    p_bar = instance["p_bar"]
    t = instance["t"]
    p_hat = instance["p_hat"]
    C = instance["C"]
    w = instance["w"]
    gamma = instance["budget_uncertainity_parameter"]

    global_uncertainty_matrix = uncertainty_matrix(num_items, gamma)
    global_uncertainty_dict = OrderedDict()

    for i in 1:size(global_uncertainty_matrix, 1)
        global_uncertainty_dict[i] = global_uncertainty_matrix[i, :]
    end

    full_exact_results = master_stage_exact(model_exact_env, num_items, global_uncertainty_dict, f, p_bar, t, p_hat, C, w)

    end_time = Dates.now()
    time_dict[seed] = end_time - start_time
    results[seed] = full_exact_results
end

results_obj =  string("results_exact",num_items ,".json")

store_path = joinpath(parent_dir, "Three Stage RO", storage_folder, results_obj)

open(store_path, "w") do io
    JSON.print(io, results)
end

results_time =  string("times_exact",num_items ,".json")
store_path = joinpath(parent_dir, "Three Stage RO", storage_folder, results_time)
# Save the time list to a JSON file
open(store_path, "w") do io
    JSON.print(io, time_dict)
end

#### Exactly solving 30 instance is not feasible (computationally)










