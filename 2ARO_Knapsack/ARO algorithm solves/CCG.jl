using DataStructures
using JuMP
using Random
using Gurobi
using JSON
using CSV
using DataFrames

parent_dir = "C:\\Users\\dube.rohit\\OneDrive - Texas A&M University\\ROoptjulia"
folder ="source"
file_name = "instance_generation.jl"
include(joinpath(parent_dir, folder, file_name))

file_name = "RO_functions.jl"
include(joinpath(parent_dir, folder, file_name))
############################################################################################################################################################################################
file_name = "nn_to_MILP.jl"
include(joinpath(parent_dir, folder, file_name))

############################################################################################################################################################################################
include("Master Solve.jl")
include("decomposition_Knapsack_iterative.jl")
############################################################################################################################################################################################

###########################################################################################################################################################################################################################################
### Initiate the data with I items

num_items = 30
### Number of uncertainities to be selected in each iteration
num_train_uncern = 100

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

function full_RO_CCG_exact(master_stage_model_env, adv_stage_model_env, global_uncertainty_matrix,
                            num_master_initial_uncertainties, num_items, f, p_bar, t, p_hat, C, w, gamma)

    solution_dict = OrderedDict()
    lower_bound = -1E10
    upper_bound = 1E10

    current_iter_count = 0
    last_gap = Inf
    gap_improvement_threshold = 1E-4

    Random.seed!(seed)
    master_indices = rand(1:size(global_uncertainty_matrix, 1), num_master_initial_uncertainties)

    master_uncertainty_set = OrderedDict()

    for (i, index) in enumerate(master_indices)
        master_uncertainty_set[i] = global_uncertainty_matrix[index, :]
    end

    global_uncertainty_dict = OrderedDict()

    for i in 1:size(global_uncertainty_matrix, 1)
        global_uncertainty_dict[i] = global_uncertainty_matrix[i, :]
    end

    results_list = []
    while true
        result_dict = OrderedDict()
        # Solve the master problem
        Master_solution = master_stage_exact(master_stage_model_env, num_items, master_uncertainty_set, f, p_bar, t, p_hat, C, w)
        lower_bound = Master_solution["objective_value"]
        Master_selected_uncertainty = Master_solution["selected_uncertainty"]
        x_MP = Master_solution["X"] 
        
        Adversarial_solution = max_enumeration_min_exact(adv_stage_model_env, x_MP, global_uncertainty_dict, num_items, f, p_bar, t, p_hat, C, w)
        upper_bound = Adversarial_solution["objective_value_master_second_stage"]
        Adversarial_selected_uncertainty = Adversarial_solution["selected_uncertainty"]

        # Update the uncertainty set
        current_iter_count += 1
        
        gap = (upper_bound - lower_bound) / abs(lower_bound)
        println("Lower bound: ", lower_bound, " Upper bound: ", upper_bound, " Iteration: ", current_iter_count)
        
        result_dict["lower_bound"] = lower_bound
        result_dict["upper_bound"] = upper_bound
        #result_dict["Master_solution"] = Master_solution
        #result_dict["Adversarial_solution"] = Adversarial_solution
        result_dict["gap"] = gap
        result_dict["current_iter_count"] = current_iter_count
        result_dict["Master_selected_uncertainty"] = Master_selected_uncertainty
        result_dict["Adversarial_selected_uncertainty"] = Adversarial_selected_uncertainty

        push!(results_list, result_dict)
        # Check if the upper bound is close to the lower bound
        if isapprox(upper_bound, lower_bound; atol=1E-5)
            println("Optimal solution found")
            break
        end

        # Check if the gap improvement is below a threshold
        if abs(last_gap - gap) < gap_improvement_threshold
            println("Convergence detected: gap improvement below threshold.")
            break
        end

        ## end if current iter count > 100
        if current_iter_count > 100
            println("Convergence didnt occur for iter=100: maximum iteration count reached.")
            break
        end

        last_gap = gap
        master_uncertainty_set[num_master_initial_uncertainties + current_iter_count] = Adversarial_selected_uncertainty
    end
    println("Lower bound: ", lower_bound, " Upper bound: ", upper_bound)
    return results_list
end


# num_master_initial_uncertainties = 3

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

# full_exact_results = solve_RO_problem_exact(model_env_ms, model_env_adv, global_uncertainty_matrix, num_master_initial_uncertainties, num_items, f, p_bar, t, p_hat, C, w, gamma)


#Storage for results
results = OrderedDict{Int, Any}()
time_dict = OrderedDict{Int, Any}()

num_master_initial_uncertainties = 3

model_env_ms = Gurobi.Env()
model_env_adv = Gurobi.Env()

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

    full_exact_CCG_results = full_RO_CCG_exact(model_env_ms, model_env_adv, global_uncertainty_matrix, num_master_initial_uncertainties, num_items, f, p_bar, t, p_hat, C, w, gamma)

    end_time = Dates.now()
    time_dict[seed] = end_time - start_time
    results[seed] = full_exact_CCG_results
end

results_obj =  string("results_CCG_exact",num_items ,".json")

store_path = joinpath(parent_dir, "Three Stage RO", storage_folder, results_obj)

open(store_path, "w") do io
    JSON.print(io, results)
end

results_time =  string("times_CCG_exact",num_items ,".json")
store_path = joinpath(parent_dir, "Three Stage RO", storage_folder, results_time)
# Save the time list to a JSON file
open(store_path, "w") do io
    JSON.print(io, time_dict)
end












