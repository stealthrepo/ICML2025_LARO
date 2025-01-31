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


parent = "c:\\Users\\dube.rohit\\OneDrive - Texas A&M University\\EconDIspARO\\EconDispARO"

source_folder = "src"

storage_folder = "example_data\\6-bus system\\pre NN training"
## include source files 


include(joinpath(parent, source_folder, "model_constructors_new.jl"))
include(joinpath(parent, source_folder, "data_constructors.jl"))


## include data files
## open the 3 json file and load the data

generated_gen_data_6_bus = JSON.parsefile(joinpath(storage_folder, "generated_cost.json"))
generated_load_6_bus = JSON.parsefile(joinpath(storage_folder, "generated_load.json"))
generated_edge_data_6_bus = JSON.parsefile(joinpath(storage_folder, "generated_edge.json"))

### sort and convert the data to a dataframe

generated_gen_data = Dict(parse(Int, k) => DataFrame(v) for (k, v) in generated_gen_data_6_bus)

generated_load_data = OrderedDict()    

num_keys_load = length(generated_load_6_bus)

for key in 1:num_keys_load
    generated_load_data[key] = sort_values_by_keys(Dict(parse(Int, k) => v for (k, v) in generated_load_6_bus[string(key)]))
end

generated_edge_data = sort_values_by_keys(Dict(parse(Int, k) => v for (k, v) in generated_edge_data_6_bus))

true

model_env = Gurobi.Env()
num_buses = 6
num_gens = 3
time_period = 24
pre_NN_training_data = Dict()
# model_results = Eco_Dis_model(model_env, num_buses, num_gens,
#                 power_generator_property_dict,
#                 bus_to_demand_dict, bus_to_generator_dict, edge_properties, 
#                 time_period)

### run the model for all combinations of the data

gen_data_df = generated_gen_data[1]

#gen_data = Power_Generator_Set(gen_data_df)

# bus_to_generator_dict = Group_Generators_by_Bus(gen_data_df.generator_no, gen_data_df.bus_no)

# edge_properties = generated_edge_data[1]

# bus_to_demand_dict = generated_load_data[1]

# Eco_Dis_model(model_env, num_buses, num_gens,
#             gen_data,
#             bus_to_demand_dict, bus_to_generator_dict, edge_properties, 
#             time_period)

    

