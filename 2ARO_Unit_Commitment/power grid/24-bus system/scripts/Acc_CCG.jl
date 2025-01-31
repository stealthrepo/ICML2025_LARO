import JSON
import DataFrames
import DataStructures
import LinearAlgebra
import Statistics
import Gurobi
import JuMP
import CSV
import BSON

parent_dir = "C:\\Users\\dube.rohit\\OneDrive - Texas A&M University\\EconDIspARO\\EconDispARO"
source_data = "src"

include(joinpath(parent_dir, source_data, "data_constructors.jl"))
include(joinpath(parent_dir, source_data, "model_constructors_new.jl"))
#include(joinpath(parent_dir, source_data, "NN_model_constructor_general.jl"))

generated_gen_data_24_bus = JSON.parsefile(joinpath(parent_dir, "example_data/24-bus system/pre NN training", "generated_cost.json"))
generated_load_24_bus = JSON.parsefile(joinpath(parent_dir, "example_data/24-bus system/pre NN training","generated_load_norm2_load.json"))
generated_edge_data_24_bus = JSON.parsefile(joinpath(parent_dir, "example_data/24-bus system/pre NN training", "generated_edge.json"))


ordered_col_names_gen_df = [:generator_no,	
                        :start_up_cost,	:shut_down_cost, 
                        :constant_cost_coeff, :linear_cost_coeff,
                        :min_power,	:max_power,
                        :min_up_time,:min_down_time,
                        :ramp_up_limit,:ramp_down_limit,
                        :start_ramp_limit,:shut_ramp_limit,
                        :bus_no]
generated_gen = sort_values_by_keys(OrderedDict(parse(Int, k) => select!(DataFrame(v),ordered_col_names_gen_df)  for (k, v) in generated_gen_data_24_bus))

generated_load = sort_values_by_keys(OrderedDict(parse(Int, k) => convert_keys_to_int(v) for (k, v) in generated_load_24_bus))

generated_edge = sort_values_by_keys(OrderedDict(parse(Int, k) => convert_keys_to_int(v) for (k, v) in generated_edge_data_24_bus))

generated_edge_properties = OrderedDict{Int, Any}()
keys_to_extract = ["edge_no", "susceptance", "min_capacity", "max_capacity"]

for (k, v) in generated_edge
    pivoted_data = OrderedDict{String, Any}()

    # Pivot data
    for key in keys_to_extract
        pivoted_data[key] = map(d -> d[key], values(v))
    end

    # Split node_tuple into node_1 and node_2
    pivoted_data["node_1"], pivoted_data["node_2"] = map(d -> d["node_tuple"][1], values(v)), map(d -> d["node_tuple"][2], values(v))

    generated_edge[k] = DataFrame(pivoted_data)

    generated_edge_properties[k] = Edge_Properties_Set(generated_edge[k])

end

###########################################################################################################
#*************** 6-bus system model with 3 generators and 24 time periods *******************************##
#****************** Acc-CCG method **********************************************************************##
###########################################################################################################

first_stage_env = Gurobi.Env()
second_stage_env = Gurobi.Env()
improved_stage_env = Gurobi.Env()

seed = 1
num_buses = 24
num_gens = size(generated_gen[1])[1] 
time_period = 24
num_edges = size(generated_edge[1])[1]
high_cost = 500

### load the load_no not allowed in a vector, these are stored in results//infeasible_keys_set.json
load_not_allowed = JSON.parsefile(joinpath(parent_dir, "example_data/24-bus system/results", "infeasible_keys_set.json"))

test_load = OrderedDict()
for k in 1:2
    if k in load_not_allowed
        continue
    end 
    test_load[k] = generated_load[k]
end


final_RMA_results = OrderedDict()
time_RMA_results = OrderedDict()

for (gen_key, gen_data) in generated_gen
    println("Started running the model for gen_key: ", gen_key)
    power_generator_property_dict = Power_Generator_Set(gen_data)
    bus_to_generator_dict = Group_Generators_by_Bus(gen_data.generator_no, gen_data.bus_no)

    for (edge_key, edge_data) in generated_edge_properties

        edge_properties = edge_data
        start_time = time()

        try
            result_RMA = Econ_Disp_Decomposed_CCG(seed, first_stage_env, second_stage_env, improved_stage_env,
                                                    num_buses, num_gens, num_edges,
                                                    power_generator_property_dict,
                                                    test_load, 
                                                    bus_to_generator_dict, 
                                                    edge_properties, 
                                                    high_cost,
                                                    time_period; warm_start_uncertainty_num = 5)
            
            final_RMA_results[(gen_key, edge_key)] = result_RMA
            time_RMA_results[(gen_key, edge_key)] = time() - start_time
            println("time taken to run per instance: ", time() - start_time)

        catch e
            #println("Error: ", e)
            println("Error in gen_key: ", gen_key, " edge_key: ", edge_key)
            continue
        end
    end
end

        
## save the final_RMA_results and time_RMA_results to a JSON file

JSON.open(joinpath(parent_dir, "example_data\\24-bus system\\results", "final_RMA_results.json"), "w") do io
    JSON.print(io, final_RMA_results)
end

JSON.open(joinpath(parent_dir, "example_data\\24-bus system\\results", "time_RMA_results.json"), "w") do io
    JSON.print(io, time_RMA_results)
end


first_stage_env = Gurobi.Env()
second_stage_env = Gurobi.Env()
improved_stage_env = Gurobi.Env()

seed = 1
num_buses = 24
num_gens = size(generated_gen[1])[1] 
time_period = 24
num_edges = size(generated_edge[1])[1]
high_cost = 500

test_load = OrderedDict()
load_not_allowed = JSON.parsefile(joinpath(parent_dir, "example_data/24-bus system/results", "infeasible_keys_set.json"))

for k in 1:2
    if k in load_not_allowed
        continue
    end 
    test_load[k] = generated_load[k]
end

gen_prop_no = 2
edge_prop_no = 1

power_generator_property_dict = Power_Generator_Set(generated_gen[gen_prop_no])
edge_properties = generated_edge_properties[edge_prop_no]
bus_to_generator_dict = Group_Generators_by_Bus(generated_gen[gen_prop_no].generator_no, generated_gen[gen_prop_no].bus_no)

result_ = Econ_Disp_Decomposed_CCG(seed, first_stage_env, second_stage_env, improved_stage_env,
                                        num_buses, num_gens, num_edges,
                                        power_generator_property_dict,
                                        test_load, ## dict with key as uncertainty number and value as bus to demand dict
                                        bus_to_generator_dict,
                                        edge_properties, 
                                        high_cost,
                                        time_period; warm_start_uncertainty_num = 5)