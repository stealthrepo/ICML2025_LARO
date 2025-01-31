## not in use


include("instance_generation.jl")
include("nn_to_MILP.jl")
include("model_arch_dict.jl")
include("Master_Sub.jl")


using JuMP
using Gurobi
using DataFrames
using CSV
using JSON
using DataStructures

I = 30                                                        ## input 10 or 30
seed = 2                                                      ## Seed for random number generation, 1 to 200 used for training of the neural network. 201 to 250 used for testing
hidden_layer_bound = 10E3
folder = "$(I) instance 50 scenarios"

## Create an unscaled instance of the problem => instance = ['f', 'p_bar', 't', 'p_hat', 'C', 'w', 'gamma']
####### Instance Generation #######
RO_parameters = knapsack_instances(seed, 1, I)                                                                   ## the second arg is number of scenarios, its of no use here since scenario set is already generated during training
f = RO_parameters["f"]
p_bar = RO_parameters["p_bar"]
t = RO_parameters["t"]
p_hat = RO_parameters["p_hat"]
C = RO_parameters["C"]
w = RO_parameters["w"]
gamma = RO_parameters["budget_uncertainity_parameter"]

RO_instance = vcat(f, p_bar, t, p_hat, C, w, gamma)

############################################################################################################
############################################# Dataset ######################################################

function custom_sort(item)
    parse(Int, split(item, '_')[end])
end

file_name = "test_results_$(I)_instance_50_scen.csv" ## To genrate the uncertainity set from the trained data

## Join the path and open the csv file
path = joinpath(folder, file_name)
df = CSV.read(path, DataFrame)
input_uncern_columns = sort([col for col in names(df) if contains(lowercase(col), "uncern")], by=custom_sort)       ## Sort the uncertainity columns by the index
uncertainity_set_SP = unique(Matrix(select(df, input_uncern_columns)), dims=1)                                     ## Select the unique uncertainity rows from the dataframe and convert to matrix
uncertainity_set_SP = uncertainity_set_SP[sum.(eachrow(uncertainity_set_SP )) .<= gamma,:]                       ## Select the uncertainity vectors whose sum is less than gamma

############################################################################################################

############################################################################################################
################################ Weights, Biases and model architecture ####################################
file_name = "model_weight_$(I)_instance_50_scen.json"

## Join the path and open the csv file
path = joinpath(folder, file_name)
weight, bias = json_to_weights(path)

if I == 10
    model_architecture = instance_arch_10()
    #hidden_layer_bound = 10E3
else
    model_architecture = instance_arch_30()
    #hidden_layer_bound = 10E3
end
############################################################################################################

############################################################################################################
######################################### Scaling Factors ##################################################
file_name = "min_max_scalers_inst_$(I).json"

## Join the path and open the csv file
path = joinpath(folder, file_name)
min_max_scalers = JSON.parsefile(path)

uncertainity_max = min_max_scalers["scaler_uncern.data_max_"]
uncertainity_min = min_max_scalers["scaler_uncern.data_min_"]
target_max = min_max_scalers["scaler_target.data_max_"]
target_min = min_max_scalers["scaler_target.data_min_"]
instance_min = min_max_scalers["scaler_instance.data_min_"]
instance_max = min_max_scalers["scaler_instance.data_max_"]
############################################################################################################

############################################################################################################
############################################ Master Problem (MP) ###########################################

# ## Start with some uncertainity vector in the MP uncertainity set
uncertainity_set_MP = Dict()
uncertainity_set_MP[1] = uncertainity_set_SP[1,:]

## Create the MP model optimizer
local_iteration = Dict()
#gurobi_env = Gurobi.Env()

# i=1
# while i < 20
#     global model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(gurobi_env)))
#     set_optimizer_attribute(model, "OutputFlag", 0)
#     Master = Master_iteration(model, I, uncertainity_set_MP, weight, bias, f, p_bar, t, p_hat, C, w, gamma, model_architecture, hidden_layer_bound, uncertainity_min, uncertainity_max, target_min, target_max, instance_min, instance_max)
#     global local_store = Dict()
#     local_store["Master_Objective"] = Master["objective_value"]
#     local_store["Worst_Case_Uncertainity"] = Master["Worst_Case_Uncertainity (u)"]
#     local_store["Selected_Uncertainity"] = Master["uncertainity_vector"]
#     local_store["z"] = Master["z"]
#     local_store["x"] = Master["x"]
#     local_store["r"] = Master["r"]
#     local_store["y"] = Master["y"]
#     local_store["Check_Objective"] = Master["check_obj"]
#     local_store["Uncertainity_Index"] = Master["uncertainity_index"]

#     Subproblem = second_stage_alg(Master["x"], uncertainity_set_SP, weight, bias, f, p_bar, t, p_hat, C, w, gamma, inst_min, inst_max, target_min, target_max, uncern_min, uncern_max)
#     local_store["Subproblem_Objective"] = Subproblem["Second_stage_objective_value"]
#     local_store["Subproblem_Uncertainity"] = Subproblem["uncertainity_to_Master"]

#     global local_iteration[i] = local_store

#     if (Master["Worst_Case_Uncertainity (u)"] > Subproblem["Second_stage_objective_value"]) || isapprox(Master["Worst_Case_Uncertainity (u)"], Subproblem["Second_stage_objective_value"], atol=1e-2)
#         break
#     end
#     global i += 1
#     uncertainity_set_MP[i] = Subproblem["uncertainity_to_Master"]
# end
# println()
# println(i)

# for (keys, values) in local_iteration[1]
#     println(keys,"=>",values)
# end


# uncertainity_set_MP

function main_optimization()
    uncertainity_set_MP = Dict()
    uncertainity_set_MP[1] = uncertainity_set_SP[1, :]
    
    local_iteration = Dict()
    gurobi_env = Gurobi.Env()
    
    for i in 1:5
        model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(gurobi_env)))
        set_optimizer_attribute(model, "OutputFlag", 0)
        
        Master = Master_iteration(model, I, uncertainity_set_MP, weight, bias, f, p_bar, t, p_hat, C, w, gamma, model_architecture, hidden_layer_bound, uncertainity_min, uncertainity_max, target_min, target_max, instance_min, instance_max)
        
        local_store = OrderedDict{Any, Any}()
        local_store["Master_Objective"] = Master["objective_value"]
        local_store["Worst_Case_Uncertainty"] = Master["Worst_Case_Uncertainity (u)"]
        local_store["Selected_Uncertainty"] = Master["uncertainity_vector"]
        local_store["z"] = Master["z"]
        local_store["x"] = Master["x"]
        local_store["r"] = Master["r"]
        local_store["y"] = Master["y"]
        local_store["Check_Objective"] = Master["check_obj"]
        local_store["Uncertainity_Index"] = Master["uncertainity_index"]
        
        Subproblem = second_stage_alg(Master["x"], uncertainity_set_SP, weight, bias, f, p_bar, t, p_hat, C, w, gamma, instance_min, instance_max, target_min, target_max, uncertainity_min, uncertainity_max)
        
        local_store["Subproblem_Objective"] = Subproblem["Second_stage_objective_value"]
        local_store["Subproblem_Uncertainity"] = Subproblem["uncertainity_to_Master"]
        
        local_iteration[i] = local_store
        uncertainity_set_MP[i + 1] = Subproblem["uncertainity_to_Master"]
        println("Iteration: ", i)
        # Check convergence criteria
        # if Master["Worst_Case_Uncertainity (u)"] > Subproblem["Second_stage_objective_value"]
        #     break
        # end 
    end
    
    return local_iteration
end

local_iteration = main_optimization()



for (keys, values) in local_iteration[1]
    println(keys,"=>",values)
end

uncertainity_set_MP