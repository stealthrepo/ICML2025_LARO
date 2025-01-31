### Master solves for RMP and NN approximated Phase 2

using DataStructures
using JuMP
using Random
using Gurobi
using JSON
using CSV
using DataFrames

parent_dir = pwd()
folder ="source"
file_name = "instance_generation.jl"
include(joinpath(parent_dir, folder, file_name))

include(joinpath(parent_dir, folder, "Master Solve.jl"))

file_name = "nn_to_MILP.jl"
include(joinpath(parent_dir, folder, file_name))


###########################################################################################################################################################################################################################################
### Initiate the data with I items
I = 10

num_train_uncern = 50

k = 30

folder_1 = "post train NN data"
folder_2 = "$(I) instance $(num_train_uncern) scenarios improved"   # add improved to the folder name if the improved version of the neural network is used

storage_folder = "results improved NN"                                 # for the earlier higer MSE the folder was "results" otherwise "results improved NN"
############################################################################################################################################################################################################################################

function check_array_elements(arr::Array{Float64}, tol::Float64=1e-5)
    for element in arr
        if !(isapprox(element, 0.0; tol) || isapprox(element, 1.0; tol))
            return false
        end
    end
    return true
end

function norm_1(v1, v2)
    return sum(abs.(v1 - v2))
end
############################################################################################################################################################################################################################################
### get the scaling factors for the problem instance, target and uncertainities that were used before training the neural network
file_name = "min_max_scalers_inst_$(I).json"
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
file_name = "model_weight_$(I)_instance_$(num_train_uncern)_scen.json"
## Join the path and open the csv file
path = joinpath(parent_dir,folder_1, folder_2, file_name)
weight, bias = json_to_weights(path)
#############################################################################################################################################################################################################################################
### Get the Global uncertainity set for I items
### The post training data can be loaded as a Dataframe, the uncertainities trained on the neural network is stored in this dataframe and is used as a global set of unceratinities for the problem instance

## Join the path and open the csv file
file_name = "post_NN_results_$(I)_instance_$(num_train_uncern)_scen.csv"
path = joinpath(parent_dir, folder_1, folder_2, file_name)

df = CSV.read(path, DataFrame)
input_uncern_column_names = sort([col for col in names(df) if contains(lowercase(col), "uncern")], by=custom_sort)       ## Sort the uncertainity columns custom sort since the column name is a string which contains numeric values
scaled_global_uncertainty_set = unique(Matrix(select(df, input_uncern_column_names)), dims=1)                            ## Get the unique values of the uncertainity set, all values here are scaled between 0 and 1
unscaled_global_uncertainty_set = scale_inverse(scaled_global_uncertainty_set', uncertainty_min, uncertainty_max)'     ## Scale the uncertainity set back to 
#############################################################################################################################################################################################################################################
#############################################################################################################################################################################################################################################

## previously robust_optimization changed to solve_master_decomposition_exact
function solve_master_decomposition_exact(seed::Int, k::Int, I::Int, instance_min, instance_max, target_min, target_max, uncertainty_min, uncertainty_max, weight, bias, unscaled_global_uncertainty_set)
    # Seed the random number generator for reproducibility
    Random.seed!(seed)
    
    # Create the data instance
    instance = knapsack_instances(seed, I; train=false)
    
    f = instance["f"]
    p_bar = instance["p_bar"]
    t = instance["t"]
    p_hat = instance["p_hat"]
    C = instance["C"]
    w = instance["w"]
    gamma = instance["budget_uncertainity_parameter"]
    
    # unscaled_instance_vector = vcat(f, p_bar, t, p_hat, C, w, gamma...)  # problem instance to be solved by RO, this is an unscaled instance
    # scaled_instance_vector = scale_inverse(unscaled_instance_vector, instance_min, instance_max)
    
    E_bar = OrderedDict{Int, Vector{Float64}}()

    indices = [i for i in 1:size(unscaled_global_uncertainty_set, 1) if sum(unscaled_global_uncertainty_set[i, :]) < gamma]
    valid_uncertainty_set = unscaled_global_uncertainty_set[indices, :]

    Random.seed!(seed)
    selected_indices = randperm(size(valid_uncertainty_set, 1))[1:k]
    
    for (i, index) in enumerate(selected_indices)
        E_bar[i] = valid_uncertainty_set[index, :]
    end
    
    #u_star = E_bar[rand(1:k)]
    norm_tol = 1e-3    # tolerance for checking the norm of two vectors
    tol = 1e-5         # tolerance for checking binary values
    
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)
    
    # Create a dictionary with the necessary arguments
    problem_args = Dict(
        "model" => model,
        "I" => I,
        "Master_uncern_set" => E_bar,
        "f" => f,
        "p_bar" => p_bar,
        "t" => t,
        "p_hat" => p_hat,
        "C" => C,
        "w" => w,
        "gamma" => gamma,
        "z_integral" => true,
        "weight" => weight,
        "bias" => bias,
        "instance_min" => instance_min,
        "instance_max" => instance_max,
        "target_min" => target_min,
        "target_max" => target_max,
        "uncertainty_min" => uncertainty_min,
        "uncertainty_max" => uncertainty_max,
        "embedding_relu" => false
    )
    ## Solve the relaxed master problem
    E_bar_matrix = relaxed_master_stage_1_iterative(problem_args)  ## In Master Solve.jl
    
    max_iterations = k
    
    results = []
    push!(results, E_bar)

    for i in 1:max_iterations
        
        optimize!(model)
        #println(value.(model[:z]))

        index = findall(x -> isapprox(x, 1.0; atol=1E-5), value.(model[:z]))
        #isapprox(z_i, 0, atol=1E-6)
        #@assert(length(index) == 1 && sum(value.(model[:z])) == 1, "TU of z not satisfied")
        @assert(length(index) == 1, "TU of z not satisfied => Multiple or no indices selected")
        @assert(isapprox(sum(value.(model[:z])),1, atol=1E-5), "TU of z not satisfied => Sum of z not equal to 1")
        #@assert(check_array_elements(value.(model[:z]), tol), "TU of z not satisfied => z not binary")

        for i in length(model[:z])
            @assert(isapprox(value.(model[:z])[i],0,atol=tol) || isapprox(value.(model[:z])[i],1,atol=tol), "TU of z not satisfied => z not binary")
        end

        x = abs.(value.(model[:X]))
        ua_k = value.(model[:ua])
        z_k = value.(model[:z])
        
        result = OrderedDict(
            "iteration" => i,
            "objective_value" => objective_value(model),
            "ua_k" => round.(ua_k, digits=3),
            "z" => z_k,
            "uncertainity_index" => index[1],
            "x" => x
        )
        ## Solve the Phase 2 approximately
        Master_stage_2 = forward_pass_new(x, E_bar_matrix, weight, bias, f, p_bar, t, p_hat, C, w, gamma, instance_min, instance_max, target_min, target_max, uncertainty_min, uncertainty_max; embedding_relu = false)
        
        u_hat_k = Master_stage_2["selected_uncertainty"]
        
        result["forward_pass_scenario"] = Master_stage_2["index"]
        result["forward_pass"] = Master_stage_2["all_forward_pass_objective_values"]
        result["u_hat_k"] = round.(u_hat_k, digits=3)
        result["forward_pass_objective"] = Master_stage_2["max_objective_value"]
        
        push!(results, result)
        
        if norm_1(ua_k, u_hat_k) <= norm_tol
            result["converged"] = true
            break
        end
        
        @constraint(model, model[:z][index[1]] == 0)
        #println("Constraint added for index: ", index[1])
        
    end
    
    return results
end

true

results_dict = OrderedDict{Int, Any}()
time_list = OrderedDict{Int, Any}()

# Loop over seeds from 1 to 250
for seed in 1:250
    println("Running seed $seed...")
    start_time = Dates.now()
    results = solve_master_decomposition_exact(seed, k, I, instance_min, instance_max, target_min, target_max, uncertainty_min, uncertainty_max, weight, bias, unscaled_global_uncertainty_set)
    end_time = Dates.now()
    elapsed_time = end_time - start_time
    println("Time taken for seed $seed: ", elapsed_time)
    time_list[seed] = elapsed_time
    results_dict[seed] = results

end

results_obj =  string("results_decomposition_",I ,"_",k,".json")

store_path = joinpath(parent_dir, "Three Stage RO", storage_folder, results_obj)
open(store_path, "w") do io
    JSON.print(io, results_dict)
end

results_time =  string("times_decomposition_",I ,"_",k,".json")
store_path = joinpath(parent_dir, "Three Stage RO", storage_folder, results_time)
# Save the time list to a JSON file
open(store_path, "w") do io
    JSON.print(io, time_list)
end
# seed = 1
# k = 5
# I = 10
# robust_optimization(seed, k, I, instance_min, instance_max, target_min, target_max, uncertainty_min, uncertainty_max, weight, bias, unscaled_global_uncertainty_set)