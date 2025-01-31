using Plots
using Base64
using Images
using CSV
using DataFrames
using DataStructures
using LinearAlgebra
using Printf
using Statistics
using JSON
using JSON3


parent = "c:\\Users\\dube.rohit\\OneDrive - Texas A&M University\\EconDIspARO\\EconDispARO"

source_folder = "src"

storage_folder = "example_data\\24-bus system\\pre NN training"
## include source files 


include(joinpath(parent, source_folder, "model_constructors_new.jl"))
include(joinpath(parent, source_folder, "data_constructors.jl"))


## include data files
## open the 3 json file and load the data

generated_gen_data_24_bus = JSON.parsefile(joinpath(storage_folder, "generated_cost.json"))
# generated_load_24_bus = JSON.parsefile(joinpath(storage_folder, "generated_load.json"))
generated_edge_data_24_bus = JSON.parsefile(joinpath(storage_folder, "generated_edge.json"))

######################################################################################################
### the file for generated load will be too big, randomly select 50000 elements from the file

load_file_num = 1

file_load = JSON.parsefile(joinpath(storage_folder, "generated_load","uncertainty_dict_with_deviation_" * string(load_file_num) * ".json"))

######################################################################################################
### sort and convert the data to a dataframe

generated_gen_data = sort_values_by_keys(Dict(parse(Int, k) => DataFrame(v) for (k, v) in generated_gen_data_24_bus))

generated_load_data = OrderedDict()    

num_keys_load = length(file_load)

for key in 1:num_keys_load
    generated_load_data[key] = sort_values_by_keys(Dict(parse(Int, k) => v for (k, v) in file_load[string(key)]))
end

generated_edge_data = sort_values_by_keys(Dict(parse(Int, k) => v for (k, v) in generated_edge_data_24_bus))

true

model_env = Gurobi.Env()
num_buses = 24
num_gens = 12
time_period = 24
high_cost = 5000




    

