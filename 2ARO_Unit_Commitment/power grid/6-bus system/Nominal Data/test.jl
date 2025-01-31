import JSON
import DataFrames
import DataStructures
import LinearAlgebra
import Statistics
import Gurobi
import JuMP
import CSV

import BSON


parent_dir = pwd()
source_data = "src"

include(joinpath(parent_dir, source_data, "data_constructors.jl"))
include(joinpath(parent_dir, source_data, "model_constructors_new.jl"))

generated_gen_data_6_bus = JSON.parsefile(joinpath(parent_dir, "example_data/6-bus system/pre NN training", "generated_cost.json"))
generated_load_6_bus = JSON.parsefile(joinpath(parent_dir, "example_data/6-bus system/pre NN training", "generated_load.json"))
generated_edge_data_6_bus = JSON.parsefile(joinpath(parent_dir, "example_data/6-bus system/pre NN training", "generated_edge.json"))


ordered_col_names_gen_df = [:generator_no,	
                        :start_up_cost,	:shut_down_cost, 
                        :constant_cost_coeff, :linear_cost_coeff,
                        :min_power,	:max_power,
                        :min_up_time,:min_down_time,
                        :ramp_up_limit,:ramp_down_limit,
                        :start_ramp_limit,:shut_ramp_limit,
                        :bus_no]
generated_gen = sort_values_by_keys(OrderedDict(parse(Int, k) => select!(DataFrame(v),ordered_col_names_gen_df)  for (k, v) in generated_gen_data_6_bus))

generated_load = sort_values_by_keys(OrderedDict(parse(Int, k) => convert_keys_to_int(v) for (k, v) in generated_load_6_bus))

generated_edge = sort_values_by_keys(OrderedDict(parse(Int, k) => convert_keys_to_int(v) for (k, v) in generated_edge_data_6_bus))


#ordered_col_names_edge_df =[:susceptance, :min_capacity, :node_tuple, :max_capacity, :edge_no]

# Define keys to be extracted and node tuple keys
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

### remove the infeasible loads from the generated_load
infeasible_loads = [94, 192, 358, 440, 510, 673, 754, 827, 835, 966, 981]

# for infeasible_load in infeasible_loads
#     delete!(generated_load, infeasible_load)
#     
# end
generated_load[373] = generated_load[1] 

# for (key, value) in generated_load[373]
#     generated_load[373][key] = value .+ 0.1
# end

# generated_load[373] = sort_values_by_keys(generated_load[373])



#uncertainty_num_bus_to_demand_dict = OrderedDict(k=>OrderedDict(1=>v) for (k, v) in generated_load)

############################################################################################################
##*************** 6-bus system model with 3 generators and 24 time periods *******************************##
##*************** CCG method *****************************************************************************##
############################################################################################################

first_stage_env = Gurobi.Env()
second_stage_env = Gurobi.Env()

seed = 1
num_buses = 6
num_gens = 3
time_period = 24
num_edges = 7
high_cost = 500

# gen_prop_no = 2
# edge_prop_no = 1

# power_generator_property_dict = Power_Generator_Set(generated_gen[gen_prop_no])
# edge_properties = generated_edge_properties[edge_prop_no]
# bus_to_generator_dict = Group_Generators_by_Bus(generated_gen[gen_prop_no].generator_no, generated_gen[gen_prop_no].bus_no)

# result_CCG = Econ_Disp_CCG(seed, first_stage_env, second_stage_env, 
#                             num_buses, num_gens, num_edges,
#                             power_generator_property_dict,
#                             generated_load, ## dict with key as uncertainty number and value as bus to demand dict
#                             bus_to_generator_dict, 
#                             edge_properties, 
#                             high_cost,
#                             time_period)

### for loop for all the generated gen data and edge data
final_CCG_results = OrderedDict()
time_CCG_results = OrderedDict()

for (gen_key, gen_data) in generated_gen
    println("Started running the model for gen_key: ", gen_key)
    power_generator_property_dict = Power_Generator_Set(gen_data)
    bus_to_generator_dict = Group_Generators_by_Bus(gen_data.generator_no, gen_data.bus_no)

    for (edge_key, edge_data) in generated_edge_properties

        edge_properties = edge_data
        start_time = time()

        try
            result_CCG = Econ_Disp_CCG(seed, first_stage_env, second_stage_env, 
                        num_buses, num_gens, num_edges,
                        power_generator_property_dict,
                        generated_load, 
                        bus_to_generator_dict, 
                        edge_properties, 
                        high_cost,
                        time_period)
            
            final_CCG_results[(gen_key, edge_key)] = result_CCG
            time_CCG_results[(gen_key, edge_key)] = time() - start_time
            println("time taken to run per instance: ", time() - start_time)

        catch e
            println("Error: ", e)
            println("Error in gen_key: ", gen_key, " edge_key: ", edge_key)
            continue
        end
    end
end

## save the final_CCG_results and time_CCG_results to a JSON file

JSON.open(joinpath(parent_dir, "example_data\\6-bus system\\results", "final_CCG_results.json"), "w") do io
    JSON.print(io, final_CCG_results)
end

JSON.open(joinpath(parent_dir, "example_data\\6-bus system\\results", "time_CCG_results.json"), "w") do io
    JSON.print(io, time_CCG_results)
end

#BSON.@save "final_CCG_results.bson" final_CCG_results

############################################################################################################
##*************** 6-bus system model with 3 generators and 24 time periods *******************************##
##*************** exact method ***************************************************************************##
############################################################################################################

num_buses = 6
num_gens = 3
time_period = 24
num_edges = 7
high_cost = 500

model_env = Gurobi.Env()

# for loop for all the generated gen data and edge data

final_exact_results = OrderedDict()
time_exact_results = OrderedDict()

for (gen_key, gen_data) in generated_gen
    println("Started running the model for gen_key: ", gen_key)
    power_generator_property_dict = Power_Generator_Set(gen_data)
    bus_to_generator_dict = Group_Generators_by_Bus(gen_data.generator_no, gen_data.bus_no)

    for (edge_key, edge_data) in generated_edge_properties

        edge_properties = edge_data
        start_time = time()

        try
            result_exact = Econ_Disp_Model(model_env, num_buses, num_gens,
                            power_generator_property_dict,
                            generated_load, 
                            bus_to_generator_dict, 
                            edge_properties, 
                            high_cost, 
                            time_period)
            
            final_exact_results[(gen_key, edge_key)] = result_exact
            time_exact_results[(gen_key, edge_key)] = time() - start_time
            println("time taken to run per instance: ", time() - start_time)

        catch e
            println("Error: ", e)
            println("Error in gen_key: ", gen_key, " edge_key: ", edge_key)
            continue
        end
    end
end

## save the final_exact_results and time_exact_results to a JSON file

JSON.open(joinpath(parent_dir, "example_data\\6-bus system\\results", "final_exact_results.json"), "w") do io
    JSON.print(io, final_exact_results)
end

BSON.@save "final_exact_results.bson" final_exact_results

JSON.open(joinpath(parent_dir, "example_data\\6-bus system\\results", "time_exact_results.json"), "w") do io
    JSON.print(io, time_exact_results)
end

# result_exact = Econ_Disp_Model(model_env, num_buses, num_gens,
#                             power_generator_property_dict,
#                             generated_load, ## dict with key as uncertainty number and value as bus to demand dict
#                             bus_to_generator_dict, edge_properties, 
#                             high_cost, 
#                             time_period)




############################################################################################################
##*************** 6-bus system model with 3 generators and 24 time periods *******************************##
##*************** exact method with one uncertainty *****************************************************##
############################################################################################################


model_env_new = Gurobi.Env()
final_pre_NN_data = OrderedDict()
final_pre_NN_time = OrderedDict()

num_buses = 6
num_gens = 3
time_period = 24
num_edges = 7
high_cost = 500



for (gen_key, gen_data) in generated_gen
    println("Started running the model for gen_cost_key: ", gen_key)
    
    power_generator_property_dict = Power_Generator_Set(gen_data)
    bus_to_generator_dict = Group_Generators_by_Bus(gen_data.generator_no, gen_data.bus_no)

    for (edge_key, edge_data) in generated_edge_properties
        edge_properties = edge_data
        
        for (load_key, load_data) in generated_load
            gen_load_dict = OrderedDict()
            gen_load_dict[1] = load_data
            start_time = time()
            try
                
                model_results = Econ_Disp_Model(model_env_new, num_buses, num_gens,
                                                power_generator_property_dict,
                                                gen_load_dict, 
                                                bus_to_generator_dict, 
                                                edge_properties, 
                                                high_cost, 
                                                time_period)
                                        
                final_pre_NN_data[(gen_key, edge_key, load_key)] = model_results
                final_pre_NN_time[(gen_key, edge_key, load_key)] = time() - start_time
                
            catch e
                println("Error: ", e)
                println("Error in gen_cost_key: ", gen_key, " load_key: ", load_key, " edge_key: ", edge_key)
                continue  # Skip to the next iteration instead of breaking
            end
        end
    end
end

JSON.open(joinpath(parent_dir, "example_data\\6-bus system\\results", "final_pre_NN_data.json"), "w") do io
    JSON.print(io, final_pre_NN_data)
end

JSON.open(joinpath(parent_dir, "example_data\\6-bus system\\results", "final_pre_NN_time.json"), "w") do io
    JSON.print(io, final_pre_NN_time)
end

# num_buses = 6
# num_gens = 3
# time_period = 24
# num_edges = 7
# high_cost = 500

# gen_prop_no = 2
# edge_prop_no = 1
# model_env_new = Gurobi.Env()

# power_generator_property_dict = Power_Generator_Set(generated_gen[gen_prop_no])
# edge_properties = generated_edge_properties[edge_prop_no]
# bus_to_generator_dict = Group_Generators_by_Bus(generated_gen[gen_prop_no].generator_no, generated_gen[gen_prop_no].bus_no)

# gen_load_dict= OrderedDict(1=>generated_load[1])

# Econ_Disp_Model(model_env_new, num_buses, num_gens,
#                 power_generator_property_dict,
#                 gen_load_dict, 
#                 bus_to_generator_dict, 
#                 edge_properties, 
#                 high_cost, 
#                 time_period)

# ## save the final_pre_NN_data and final_pre_NN_time to a JSON file

# JSON.open(joinpath(parent_dir, "example_data\\6-bus system\\results", "final_pre_NN_data.json"), "w") do io
#     JSON.print(io, final_pre_NN_data)
# end

# JSON.open(joinpath(parent_dir, "example_data\\6-bus system\\results", "final_pre_NN_time.json"), "w") do io
#     JSON.print(io, final_pre_NN_time)
# end


############################################################################################################
##*************** 6-bus system model with 3 generators and 24 time periods *******************************##
##*************** relaxed master method ******************************************************************##
############################################################################################################

first_stage_env = Gurobi.Env()
second_stage_env = Gurobi.Env()
improved_stage_env = Gurobi.Env()

seed = 1
num_buses = 6
num_gens = 3
time_period = 24
num_edges = 7
high_cost = 500

# gen_prop_no = 2
# edge_prop_no = 3

# power_generator_property_dict = Power_Generator_Set(generated_gen[gen_prop_no])
# edge_properties = generated_edge_properties[edge_prop_no]
# bus_to_generator_dict = Group_Generators_by_Bus(generated_gen[gen_prop_no].generator_no, generated_gen[gen_prop_no].bus_no)

# result_ = Econ_Disp_Decomposed_CCG(first_stage_env, second_stage_env, improved_stage_env,
#                                         num_buses, num_gens, num_edges,
#                                         power_generator_property_dict,
#                                         generated_load, ## dict with key as uncertainty number and value as bus to demand dict
#                                         bus_to_generator_dict,
#                                         edge_properties, 
#                                         high_cost,
#                                         time_period; warm_start_uncertainty_num = 5)


## for loop for all the generated gen data and edge data

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
            result_RMA = Econ_Disp_Decomposed_CCG(first_stage_env, second_stage_env, improved_stage_env,
                                                    num_buses, num_gens, num_edges,
                                                    power_generator_property_dict,
                                                    generated_load, 
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

JSON.open(joinpath(parent_dir, "example_data\\6-bus system\\results", "final_RMA_results.json"), "w") do io
    JSON.print(io, final_RMA_results)
end

JSON.open(joinpath(parent_dir, "example_data\\6-bus system\\results", "time_RMA_results.json"), "w") do io
    JSON.print(io, time_RMA_results)
end

