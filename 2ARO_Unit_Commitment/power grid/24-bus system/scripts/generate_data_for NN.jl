import JSON
import DataFrames
import DataStructures
import LinearAlgebra
import Statistics
import Gurobi
import JuMP
import CSV
import BSON
using DataFrames
using Base.Iterators: product

parent_dir = "C:\\Users\\dube.rohit\\OneDrive - Texas A&M University\\EconDIspARO\\EconDispARO"
source_data = "src"

include(joinpath(parent_dir, source_data, "data_constructors.jl"))
include(joinpath(parent_dir, source_data, "model_constructors_new.jl"))
include(joinpath(parent_dir, source_data, "NN_model_constructor_general.jl"))

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

##################################

"""
    expand_group(a_start, a_end, b_vals, c_val)

Generate a DataFrame of all combinations of `a = a_start:a_end` and
`b` in `b_vals`, with a constant column `c = c_val`.
"""
function expand_group(a_start::Int, a_end::Int, b_vals::UnitRange{Int}, c_val::Int)
    # Generate (a, b) pairs via a comprehension
    combos = [(x, y) for x in a_start:a_end for y in b_vals]
    
    # Decompose the pairs into columns a and b, and fill c
    return DataFrame(
        a = first.(combos),
        b = last.(combos),
        c = fill(c_val, length(combos))
    )
end

# Correct grouping:
# - Group 1: a in 1:10000,   b in 1:10,    c=1
# - Group 2: a in 10001:20000, b in 11:20,   c=2
# - Group 3: a in 20001:30000, b in 21:30,   c=3
# - Group 4: a in 30001:40000, b in 31:40,   c=1
# - Group 5: a in 40001:50000, b in 41:50,   c=2

groups = [
    (1,      10000,  1:10,   1),
    (10001,  20000,  11:20,  2),
    (20001,  30000,  21:30,  3),
    (30001,  40000,  31:40,  1),
    (40001,  50000,  41:50,  2)
]

# Build each group DataFrame and concatenate them vertically
final_design = vcat([
    expand_group(a_s, a_e, b_vals, c_val)
    for (a_s, a_e, b_vals, c_val) in groups
]...)

@show size(final_design)  # Should be (500000, 3)

final_design = hcat(final_design.a, final_design.b, final_design.c)
##################################

############################################################################################################

############################################################################################################
##*************** 6-bus system model with 3 generators and 24 time periods *******************************##
##*************** exact method with one uncertainty *****************************************************##
##*************** for NN training data generation ********************************************************##
############################################################################################################
num_buses = 24
num_gens = size(generated_gen[1])[1]
time_period = 24
num_edges = length(generated_edge_properties[1])
high_cost = 500

final_pre_NN_data = OrderedDict()
final_pre_NN_time = OrderedDict()

model_env_new = Gurobi.Env()

for design in axes(final_design, 1)
    uncertainty_no = final_design[design, 1]
    gen_prop_no = final_design[design, 2]
    edge_prop_no = final_design[design, 3]

    power_generator_property_dict = Power_Generator_Set(generated_gen[gen_prop_no])
    edge_properties = generated_edge_properties[edge_prop_no]
    bus_to_generator_dict = Group_Generators_by_Bus(generated_gen[gen_prop_no].generator_no, generated_gen[gen_prop_no].bus_no)
    gen_load_dict= OrderedDict(1=>generated_load[uncertainty_no])

    start_time = time()
    try
        model_results = Econ_Disp_Model(model_env_new, num_buses, num_gens,
                                        power_generator_property_dict,
                                        gen_load_dict, 
                                        bus_to_generator_dict, 
                                        edge_properties, 
                                        high_cost, 
                                        time_period)
        total_time = time() - start_time
        # Store in the global dictionaries
        global final_pre_NN_data
        global final_pre_NN_time                     
        final_pre_NN_data[(gen_prop_no, edge_prop_no, uncertainty_no)] = model_results
        final_pre_NN_time[(gen_prop_no, edge_prop_no, uncertainty_no)] = time() - start_time
        
    catch e
        println("Error: ", e)
        println("Error in gen_cost_key: ", gen_prop_no, " uncertainty_no: ", uncertainty_no, " edge_prop_no: ", edge_prop_no)
        continue  # Skip to the next iteration instead of breaking
    end

    if design % 25000 == 0
        println("Completed design $design")
        json_filename = joinpath(parent_dir, "example_data\\24-bus system\\results", "final_pre_NN_data_norm2_" * string(design) * ".json")
        JSON.open(json_filename, "w") do io
            JSON.print(io, final_pre_NN_data)
        end
        json_filename = joinpath(parent_dir, "example_data\\24-bus system\\results", "final_pre_NN_time_norm2_" * string(design) * ".json")
        JSON.open(json_filename, "w") do io
            JSON.print(io, final_pre_NN_time)
        end
        ## empty the final_pre_NN_data and final_pre_NN_time for the next 10000 designs
        global final_pre_NN_data = OrderedDict()
        global final_pre_NN_time = OrderedDict()
        ## collect the garbage
        GC.gc()
    end
end





























# model_env_new = Gurobi.Env()
# BATCH_SIZE = 1000
# for load_batch in 1:50
#     ## select 1000 loads from the generated_load for each batch
#     start_index = (load_batch - 1) * BATCH_SIZE + 1
#     end_index = load_batch * BATCH_SIZE
#     selected_loads = collect(keys(generated_load))[start_index:end_index]
#     selected_loads_dict = OrderedDict()
#     for load_num in selected_loads
#         selected_loads_dict[load_num] = generated_load[load_num]
#     end

#     final_pre_NN_data = OrderedDict()
#     final_pre_NN_time = OrderedDict()

#     num_buses = 24
#     num_gens = size(generated_gen[1])[1]
#     time_period = 24
#     num_edges = length(generated_edge_properties[1])
#     high_cost = 500

#     for (load_key, load_data) in selected_loads_dict
#         gen_load_dict = OrderedDict()
#         gen_load_dict[1] = load_data

#         for (gen_key, gen_data) in generated_gen
#             #println("Started running the model for gen_cost_key: ", gen_key)
#             power_generator_property_dict = Power_Generator_Set(gen_data)
#             bus_to_generator_dict = Group_Generators_by_Bus(gen_data.generator_no, gen_data.bus_no)

#             for (edge_key, edge_data) in generated_edge_properties
#                 edge_properties = edge_data
#                 start_time = time()
#                 try
#                     model_results = Econ_Disp_Model(model_env_new, num_buses, num_gens,
#                                                     power_generator_property_dict,
#                                                     gen_load_dict, 
#                                                     bus_to_generator_dict, 
#                                                     edge_properties, 
#                                                     high_cost, 
#                                                     time_period)
                                            
#                     final_pre_NN_data[(gen_key, edge_key, load_key)] = model_results
#                     final_pre_NN_time[(gen_key, edge_key, load_key)] = time() - start_time
                    
#                 catch e
#                     println("Error: ", e)
#                     println("Error in gen_cost_key: ", gen_key, " load_key: ", load_key, " edge_key: ", edge_key)
#                     continue  # Skip to the next iteration instead of breaking
#                 end
#             end
#         end
#     end

#     JSON.open(joinpath(parent_dir, "example_data\\24-bus system\\results", "final_pre_NN_data_gen_key" * string(start_index) * string(end_index) * ".json"), "w") do io
#         JSON.print(io, final_pre_NN_data)
#     end
#     JSON.open(joinpath(parent_dir, "example_data\\24-bus system\\results", "final_pre_NN_time_gen_key" * string(start_index) * string(end_index) * ".json"), "w") do io
#         JSON.print(io, final_pre_NN_time)
#     end
# end

##############################################################################################################################################


#### run the randomly selected 50,000 loads on the nominal values of generated edge and gen data i.e. index 1.


# model_env_new = Gurobi.Env()
# final_pre_NN_data = OrderedDict()
# final_pre_NN_time = OrderedDict()

# num_buses = 24
# num_gens = size(generated_gen[1])[1]
# time_period = 24
# num_edges = length(generated_edge_properties[1])
# high_cost = 500

# ## now the for loop will only be on the generated load as the gen and edge data is fixed
# ### for loop for all the generated load data
# edge_data = generated_edge_properties[1]
# gen_data = generated_gen[1]
# power_generator_property_dict = Power_Generator_Set(gen_data)
# bus_to_generator_dict = Group_Generators_by_Bus(gen_data.generator_no, gen_data.bus_no)

# # for (load_key, load_data) in generated_load_nom
# #     gen_load_dict = OrderedDict()
# #     gen_load_dict[1] = load_data
# #     start_time = time()
# #     try
# #         model_results = Econ_Disp_Model(model_env_new, num_buses, num_gens,
# #                                             power_generator_property_dict,
# #                                             gen_load_dict, 
# #                                             bus_to_generator_dict, 
# #                                             edge_data, 
# #                                             high_cost, 
# #                                             time_period)
                                
# #         final_pre_NN_data[1,1,load_key] = model_results
# #         final_pre_NN_time[1,1,load_key] = time() - start_time
        
# #     catch e
# #         println("Error: ", e)
# #         println("Error in load_key: ", load_key)
# #         continue  # Skip to the next iteration instead of breaking
# #     end
# # end
# # Constants
# const BATCH_SIZE = 5000

# # Variables
# counter = 0
# file_index = 1

# # Batch containers
# batch_results_data = Dict()
# batch_results_time = Dict()

# for (load_key, load_data) in pairs(generated_load_nom)
#     # Create an OrderedDict for the Econ_Disp_Model call
#     gen_load_dict = OrderedDict()
#     gen_load_dict[1] = load_data
    
#     start_time = time()
#     try
#         # Call your model
#         model_results = Econ_Disp_Model(
#             model_env_new,
#             num_buses,
#             num_gens,
#             power_generator_property_dict,
#             gen_load_dict,
#             bus_to_generator_dict,
#             edge_data,
#             high_cost,
#             time_period
#         )
        
#         # Store results in the "batch" containers
#         batch_results_data[(1,1,load_key)] = model_results
#         batch_results_time[(1,1,load_key)] = time() - start_time

#     catch e
#         @warn "Caught error for load_key = $load_key" exception = e
#         continue  # Skip this load_key and move on
#     end
    
#     # Increment counter and check if we've hit the batch size
#     counter += 1
#     if counter % BATCH_SIZE == 0
        
#         # Write out this batch to separate JSON files
#         data_filename = joinpath(parent_dir, "example_data\\24-bus system\\results", "final_pre_NN_data_batch_nominal" * string(file_index) * ".json")
        
#         time_filename = joinpath(parent_dir, "example_data\\24-bus system\\results", "final_pre_NN_time_batch_nominal" * string(file_index) * ".json")
        
#         # Write data to a JSON file
#         # JSON.open(joinpath(parent_dir, "example_data\\24-bus system\\results", "final_pre_NN_data" * string(file_num) * ".json"), "w") do io
#         open(data_filename, "w") do f
#             # You might store just the batch_results_data as a dictionary
#             JSON.print(f, batch_results_data)
#         end

#         # Write time info to a separate JSON file
#         open(time_filename, "w") do f
#             # Similarly, store timing in its own dictionary
#             JSON.print(f, batch_results_time)
#         end
        
#         println("Wrote batch $(file_index) with $(BATCH_SIZE) items to $data_filename and $time_filename.")

#         # Prepare for the next batch
#         file_index += 1
#         batch_results_data = Dict()
#         batch_results_time = Dict()
#     end
# end

# # After the loop, if there are leftover entries,
# # we handle them by writing one final pair of JSON files.
# if counter % BATCH_SIZE != 0
#     data_filename = joinpath(parent_dir, "example_data\\24-bus system\\results", "final_pre_NN_data_batch_nominal" * string(file_index) * ".json")
        
#     time_filename = joinpath(parent_dir, "example_data\\24-bus system\\results", "final_pre_NN_time_batch_nominal" * string(file_index) * ".json")
    
#     open(data_filename, "w") do f
#         JSON.print(f, batch_results_data)
#     end
    
#     open(time_filename, "w") do f
#         JSON.print(f, batch_results_time)
#     end
    
#     println("Wrote remaining $(counter % BATCH_SIZE) items to $data_filename and $time_filename.")
# end



############################################################################################################
### load the uncertainty fie in a loop by using the file number 6 to 150 

# uncertainty_load = OrderedDict()
# index_key = 1   
# for file_num in 6:150
#     generated_load = JSON.parsefile(joinpath(parent_dir, "example_data/24-bus system/pre NN training/generated_load", "uncertainty_dict_with_deviation_" * string(file_num) *".json"))
#     ## randomly select 1/3 of loads without replacement from the 1000 loads in generated loads, length of generated_load is 1000 
#     selected_loads = collect(keys(generated_load))
#     selected_loads = sample(selected_loads, 333, replace = false)
#     for load_num in selected_loads
#         uncertainty_load[index_key] = generated_load[load_num]
#         index_key += 1
#     end
# end

# ## save the uncertainty_load to a JSON file

# JSON.open(joinpath(parent_dir, "example_data\\24-bus system\\pre NN training", "generated_load_for_nominal_training.json"), "w") do io
#     JSON.print(io, uncertainty_load)
# end
############################################################################################################






























# ############################################################################################################
# ##*************** 6-bus system model with 3 generators and 24 time periods *******************************##
# ##*************** CCG method *****************************************************************************##
# ############################################################################################################

# first_stage_env = Gurobi.Env()
# second_stage_env = Gurobi.Env()

# seed = 1
# num_buses = 6
# num_gens = 3
# time_period = 24
# num_edges = 7
# high_cost = 500

# # gen_prop_no = 2
# # edge_prop_no = 1

# # power_generator_property_dict = Power_Generator_Set(generated_gen[gen_prop_no])
# # edge_properties = generated_edge_properties[edge_prop_no]
# # bus_to_generator_dict = Group_Generators_by_Bus(generated_gen[gen_prop_no].generator_no, generated_gen[gen_prop_no].bus_no)

# # result_CCG = Econ_Disp_CCG(seed, first_stage_env, second_stage_env, 
# #                             num_buses, num_gens, num_edges,
# #                             power_generator_property_dict,
# #                             generated_load, ## dict with key as uncertainty number and value as bus to demand dict
# #                             bus_to_generator_dict, 
# #                             edge_properties, 
# #                             high_cost,
# #                             time_period)

# ### for loop for all the generated gen data and edge data
# final_CCG_results = OrderedDict()
# time_CCG_results = OrderedDict()

# for (gen_key, gen_data) in generated_gen
#     println("Started running the model for gen_key: ", gen_key)
#     power_generator_property_dict = Power_Generator_Set(gen_data)
#     bus_to_generator_dict = Group_Generators_by_Bus(gen_data.generator_no, gen_data.bus_no)

#     for (edge_key, edge_data) in generated_edge_properties

#         edge_properties = edge_data
#         start_time = time()

#         try
#             result_CCG = Econ_Disp_CCG(seed, first_stage_env, second_stage_env, 
#                         num_buses, num_gens, num_edges,
#                         power_generator_property_dict,
#                         generated_load, 
#                         bus_to_generator_dict, 
#                         edge_properties, 
#                         high_cost,
#                         time_period)
            
#             final_CCG_results[(gen_key, edge_key)] = result_CCG
#             time_CCG_results[(gen_key, edge_key)] = time() - start_time
#             println("time taken to run per instance: ", time() - start_time)

#         catch e
#             println("Error: ", e)
#             println("Error in gen_key: ", gen_key, " edge_key: ", edge_key)
#             continue
#         end
#     end
# end

# ## save the final_CCG_results and time_CCG_results to a JSON file

# JSON.open(joinpath(parent_dir, "example_data\\6-bus system\\results", "final_CCG_results.json"), "w") do io
#     JSON.print(io, final_CCG_results)
# end

# JSON.open(joinpath(parent_dir, "example_data\\6-bus system\\results", "time_CCG_results.json"), "w") do io
#     JSON.print(io, time_CCG_results)
# end

# #BSON.@save "final_CCG_results.bson" final_CCG_results

# ############################################################################################################
# ##*************** 6-bus system model with 3 generators and 24 time periods *******************************##
# ##*************** exact method ***************************************************************************##
# ############################################################################################################

# num_buses = 24
# num_gens = size(generated_gen[1])[1]
# time_period = 24
# num_edges = length(generated_edge_properties[1])
# high_cost = 500

# model_env = Gurobi.Env()

# # for loop for all the generated gen data and edge data

# final_exact_results = OrderedDict()
# time_exact_results = OrderedDict()

# for (gen_key, gen_data) in generated_gen
#     println("Started running the model for gen_key: ", gen_key)
#     power_generator_property_dict = Power_Generator_Set(gen_data)
#     bus_to_generator_dict = Group_Generators_by_Bus(gen_data.generator_no, gen_data.bus_no)

#     for (edge_key, edge_data) in generated_edge_properties

#         edge_properties = edge_data
#         start_time = time()

#         try
#             result_exact = Econ_Disp_Model(model_env, num_buses, num_gens,
#                             power_generator_property_dict,
#                             generated_load, 
#                             bus_to_generator_dict, 
#                             edge_properties, 
#                             high_cost, 
#                             time_period)
            
#             final_exact_results[(gen_key, edge_key)] = result_exact
#             time_exact_results[(gen_key, edge_key)] = time() - start_time
#             println("time taken to run per instance: ", time() - start_time)

#         catch e
#             println("Error: ", e)
#             println("Error in gen_key: ", gen_key, " edge_key: ", edge_key)
#             continue
#         end
#     end
# end

# ## save the final_exact_results and time_exact_results to a JSON file

# JSON.open(joinpath(parent_dir, "example_data\\24-bus system\\results", "final_exact_results.json"), "w") do io
#     JSON.print(io, final_exact_results)
# end

# BSON.@save "final_exact_results.bson" final_exact_results

# JSON.open(joinpath(parent_dir, "example_data\\24-bus system\\results", "time_exact_results.json"), "w") do io
#     JSON.print(io, time_exact_results)
# end

# # result_exact = Econ_Disp_Model(model_env, num_buses, num_gens,
# #                             power_generator_property_dict,
# #                             generated_load, ## dict with key as uncertainty number and value as bus to demand dict
# #                             bus_to_generator_dict, edge_properties, 
# #                             high_cost, 
# #                             time_period)






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


# ############################################################################################################
# ##*************** 6-bus system model with 3 generators and 24 time periods *******************************##
# ##*************** relaxed master method ******************************************************************##
# ############################################################################################################

# first_stage_env = Gurobi.Env()
# second_stage_env = Gurobi.Env()
# improved_stage_env = Gurobi.Env()

# seed = 1
# num_buses = 6
# num_gens = 3
# time_period = 24
# num_edges = 7
# high_cost = 500

# # gen_prop_no = 2
# # edge_prop_no = 3

# # power_generator_property_dict = Power_Generator_Set(generated_gen[gen_prop_no])
# # edge_properties = generated_edge_properties[edge_prop_no]
# # bus_to_generator_dict = Group_Generators_by_Bus(generated_gen[gen_prop_no].generator_no, generated_gen[gen_prop_no].bus_no)

# # result_ = Econ_Disp_Decomposed_CCG(first_stage_env, second_stage_env, improved_stage_env,
# #                                         num_buses, num_gens, num_edges,
# #                                         power_generator_property_dict,
# #                                         generated_load, ## dict with key as uncertainty number and value as bus to demand dict
# #                                         bus_to_generator_dict,
# #                                         edge_properties, 
# #                                         high_cost,
# #                                         time_period; warm_start_uncertainty_num = 5)


# ## for loop for all the generated gen data and edge data

# final_RMA_results = OrderedDict()
# time_RMA_results = OrderedDict()

# for (gen_key, gen_data) in generated_gen
#     println("Started running the model for gen_key: ", gen_key)
#     power_generator_property_dict = Power_Generator_Set(gen_data)
#     bus_to_generator_dict = Group_Generators_by_Bus(gen_data.generator_no, gen_data.bus_no)

#     for (edge_key, edge_data) in generated_edge_properties

#         edge_properties = edge_data
#         start_time = time()

#         try
#             result_RMA = Econ_Disp_Decomposed_CCG(first_stage_env, second_stage_env, improved_stage_env,
#                                                     num_buses, num_gens, num_edges,
#                                                     power_generator_property_dict,
#                                                     generated_load, 
#                                                     bus_to_generator_dict, 
#                                                     edge_properties, 
#                                                     high_cost,
#                                                     time_period; warm_start_uncertainty_num = 5)
            
#             final_RMA_results[(gen_key, edge_key)] = result_RMA
#             time_RMA_results[(gen_key, edge_key)] = time() - start_time
#             println("time taken to run per instance: ", time() - start_time)

#         catch e
#             #println("Error: ", e)
#             println("Error in gen_key: ", gen_key, " edge_key: ", edge_key)
#             continue
#         end
#     end
# end

        
# ## save the final_RMA_results and time_RMA_results to a JSON file

# JSON.open(joinpath(parent_dir, "example_data\\6-bus system\\results", "final_RMA_results.json"), "w") do io
#     JSON.print(io, final_RMA_results)
# end

# JSON.open(joinpath(parent_dir, "example_data\\6-bus system\\results", "time_RMA_results.json"), "w") do io
#     JSON.print(io, time_RMA_results)
# end

