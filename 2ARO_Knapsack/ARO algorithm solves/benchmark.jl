### code to create uncertainty benchmark for all the problem instances

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

############################################################################################################################################################################################
file_name = "nn_to_MILP.jl"
include(joinpath(parent_dir, folder, file_name))

############################################################################################################################################################################################
include(joinpath(parent_dir, folder, "Master Solve.jl"))

############################################################################################################################################################################################

###########################################################################################################################################################################################################################################
### Initiate the data with I items

num_items = 20
### Number of uncertainities to be selected in each iteration
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

## Get the Global uncertainity set for I items
## The post training data can be loaded as a Dataframe, the uncertainities trained on the neural network is stored in this dataframe and is used as a global set of unceratinities for the problem instance
# Join the path and open the csv file

file_name = "post_NN_results_$(num_items)_instance_$(num_train_uncern)_scen.csv"
path = joinpath(parent_dir, folder_1, folder_2, file_name)
df = CSV.read(path, DataFrame)
input_uncern_column_names = sort([col for col in names(df) if contains(lowercase(col), "uncern")], by=custom_sort)       ## Sort the uncertainity columns custom sort since the column name is a string which contains numeric values

scaled_global_uncertainty_matrix = unique(Matrix(select(df, input_uncern_column_names)), dims=1)                            ## Get the unique values of the uncertainity set, all values here are scaled between 0 and 1
unscaled_global_uncertainty_matrix = scale_inverse(scaled_global_uncertainty_matrix', uncertainty_min, uncertainty_max)'     ## Scale the uncertainity set back to the original scale
#############################################################################################################################################################################################################################################
#############################################################################################################################################################################################################################################
### budget_uncern_parameter = [0.1, 0.15, 0.2][rand(1:3)] * I
### For benchmarking the results, consider a limited number of uncertainties

# valid_global_uncertainty_matrix = unscaled_global_uncertainty_matrix[[i for i in 1:size(unscaled_global_uncertainty_matrix, 1) if sum(unscaled_global_uncertainty_matrix[i, :]) <= 6],:]

# becnhmarking_uncertainty_matrix = valid_global_uncertainty_matrix[randperm(size(valid_global_uncertainty_matrix, 1))[1:2500], :]

benchmarking_uncertainty_I_dict = OrderedDict()
gamma_list = [0.1, 0.15, 0.2]
Random.seed!(123)
for i in gamma_list
    global valid_global_uncertainty_matrix = unscaled_global_uncertainty_matrix[[i for i in 1:size(unscaled_global_uncertainty_matrix, 1) if sum(unscaled_global_uncertainty_matrix[i, :]) <= i*num_items],:]
    benchmarking_uncertainty_I_dict[i*num_items] = valid_global_uncertainty_matrix[randperm(size(valid_global_uncertainty_matrix, 1))[1:2500], :]
end

function save_dict_to_csv(dict, num_items, filepath)
    for (key, matrix) in dict
        # Convert the matrix to a DataFrame
        df = DataFrame(matrix, :auto)
        
        # Create a filename using the key

        filename = "num_items_$(num_items)_benchmarking_uncertainty_matrix_gamma_$(key).csv"
        store_path = joinpath(filepath, filename)
        # Save the DataFrame to a CSV file
        CSV.write(store_path, df)
    end
end

#save_dict_to_csv(benchmarking_uncertainty_I_dict, joinpath(parent_dir, folder_1, "benchmarking_uncertainties"))

Random.seed!(123)

# save 100 randomly selected seeds for the benchmarking uncertainties from 1 to 250 without replacement
seeds = sort(randperm(250)[1:50])
#benchmarking_uncertainty_I_dict["seed_list"] = seeds
#save_dict_to_csv(benchmarking_uncertainty_I_dict, num_items, joinpath(parent_dir, folder_1, "benchmarking_uncertainties"))

## save the seeds to a text file
file_name = "benchmarking_seed_list.txt"
path = joinpath(parent_dir, folder_1, "benchmarking_uncertainties", file_name)
open(path, "w") do io
    for seed in seeds
        println(io, seed)
    end
end

