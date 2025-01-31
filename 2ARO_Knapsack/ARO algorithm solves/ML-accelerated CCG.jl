using DataStructures
using JuMP
using Random
using Gurobi
using JSON
using CSV
using DataFrames
using JLD2

parent_dir = pwd()
folder ="source"
file_name = "instance_generation.jl"
include(joinpath(parent_dir, folder, file_name))

file_name = "RO_functions.jl"
include(joinpath(parent_dir, folder, file_name))
############################################################################################################################################################################################
file_name = "nn_to_MILP.jl"
include(joinpath(parent_dir, folder, file_name))

############################################################################################################################################################################################
include(joinpath(parent_dir, folder, "Master Solve.jl"))

include("Master_solve_RMP_Neural_phase 2.jl")
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
#############################################################################################################################################################################################################################################
## load the ML model
ML_folder = "Decision Tree exact model"
file_name = "RF_exact_$(num_items).jld2"
path = joinpath(parent_dir, ML_folder, file_name)

#@load path model
println("model_loaded")

# #############################################################################################################################################################################################################################################

# seed = 5

# num_master_initial_uncertainties = 5
# #num_items = 10
# Random.seed!(seed)
# # Create the data instancead
# instance = knapsack_instances(seed, num_items; train=false)

# f = instance["f"]
# p_bar = instance["p_bar"]
# t = instance["t"]
# p_hat = instance["p_hat"]
# C = instance["C"]
# w = instance["w"]
# gamma = instance["budget_uncertainity_parameter"]

# instance_vec = vcat(f, p_bar, t, p_hat, C, w, gamma...)
# global_uncertainty_matrix = uncertainty_matrix(num_items, gamma)

# model_env_ms = Gurobi.Env()
# model_env_adv = Gurobi.Env()
# #model_exact_env = Gurobi.Env()
# model_env_ms_exact = Gurobi.Env()

# full_NN_results = fullRO_master_decomposition_NN_adv_NN(model_env_ms, model_env_adv,model_env_ms_exact, num_items, global_uncertainty_matrix, 
#                                                         num_master_initial_uncertainties, seed, f, p_bar, t, p_hat, C, w, gamma, 
#                                                         weight, bias, 
#                                                         instance_min, instance_max, 
#                                                         target_min, target_max, 
#                                                         uncertainty_min, uncertainty_max;
#                                                         improved_master = true, ML_model_forward_master = model)


#Storage for results
results = OrderedDict{Int, Dict}()
time_dict = OrderedDict{Int, Any}()

num_master_initial_uncertainties = 3

model_env_ms = Gurobi.Env()
model_env_adv = Gurobi.Env()
model_env_ms_exact = Gurobi.Env()

for seed in 1:250
    #seed = parse(Int, seed)
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

    full_NN_results = fullRO_master_decomposition_NN_adv_NN(model_env_ms, model_env_adv,model_env_ms_exact, num_items, global_uncertainty_matrix, 
                                                            num_master_initial_uncertainties, seed, f, p_bar, t, p_hat, C, w, gamma, 
                                                            weight, bias, 
                                                            instance_min, instance_max, 
                                                            target_min, target_max, 
                                                            uncertainty_min, uncertainty_max;
                                                            improved_master = false, ML_model_forward_master = nothing)

    end_time = Dates.now()
    time_dict[seed] = end_time - start_time
    results[seed] = full_NN_results
end

results_obj =  string("results_2SRO_decompose_NN",num_items ,".json")

store_path = joinpath(parent_dir, "Three Stage RO", storage_folder, results_obj)

open(store_path, "w") do io
    JSON.print(io, results)
end

results_time =  string("times_2SRO_decompose_NN",num_items ,".json")
store_path = joinpath(parent_dir, "Three Stage RO", storage_folder, results_time)
# Save the time list to a JSON file
open(store_path, "w") do io
    JSON.print(io, time_dict)
end
