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
include(joinpath(parent_dir, source_data, "NN_model_constructor_general.jl"))

generated_gen_data_24_bus = JSON.parsefile(joinpath(parent_dir, "example_data/24-bus system/pre NN training", "generated_cost.json"))
generated_load_24_bus = JSON.parsefile(joinpath(parent_dir, "example_data/24-bus system/pre NN training","generated_load_RO.json"))
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

############################################################################################################

generated_gen
generated_load
generated_edge

generated_edge_properties

generated_edge[1]
generated_edge_properties[1]

############################################################################################################

seed = 1
num_buses = 24
num_gens = size(generated_gen[1])[1]
time_period = 24
num_edges = length(generated_edge_properties[1])
high_cost = 500

generator_data = generated_gen[1]
bus_to_generator_dict = Group_Generators_by_Bus(generator_data.generator_no, generator_data.bus_no)
power_generator_property_dict = Power_Generator_Set(generator_data)

dict_bus_to_demand = OrderedDict()

for i in 1:20
    dict_bus_to_demand[i] = generated_load[i]
end

dict_bus_to_demand
dict_bus_to_demand[1]
keys(dict_bus_to_demand[1])

edge_properties = generated_edge_properties[1]

first_stage_env = Gurobi.Env()
improved_stage_env = Gurobi.Env()

num_heads = 16
len_of_features = 56
pos_enc_k = 24    #### this has to be retified, greater number can lead to a bug

lower_bound = -Inf
upper_bound = Inf

@assert(length(dict_bus_to_demand) > 0, "The uncertainty dict is empty")
@assert(length(power_generator_property_dict) == num_gens, "The number of generators in the power generator dict is not equal to the number of generators in the model")

generator_name_list = sort(collect(keys(power_generator_property_dict)))

start_up_cost = [power_generator_property_dict[key].start_up_cost for key in generator_name_list]
shut_down_cost = [power_generator_property_dict[key].shut_down_cost for key in generator_name_list]
constant_cost_coefficient = [power_generator_property_dict[key].constant_cost_coefficient for key in generator_name_list]
linear_cost_coefficient = [power_generator_property_dict[key].linear_cost_coefficient for key in generator_name_list]
min_power = [power_generator_property_dict[key].Min_electricty_output_limit for key in generator_name_list]
max_power = [power_generator_property_dict[key].Max_electricty_output_limit for key in generator_name_list]
min_up_time = [power_generator_property_dict[key].Min_up_time for key in generator_name_list]
min_down_time = [power_generator_property_dict[key].Min_down_time for key in generator_name_list]
ramp_up_limit = [power_generator_property_dict[key].Ramp_up_limit for key in generator_name_list]
ramp_down_limit = [power_generator_property_dict[key].Ramp_down_limit for key in generator_name_list]
start_ramp_limit = [power_generator_property_dict[key].Start_up_ramp_rate_limit for key in generator_name_list]
shut_ramp_limit = [power_generator_property_dict[key].Shut_down_ramp_rate_limit for key in generator_name_list]

@assert(length(start_up_cost) == num_gens, "The number of generators in the start up cost list is not equal to the number of generators in the model")

edge_properties = generated_edge_properties[1]

susceptance_matrix = Susceptance_Matrix_Generator(num_buses, edge_properties)     # num_buses x num_buses matrix
arc_incidence_matrix = Node_Arc_Incidence_Matrix_Generator(num_buses, edge_properties)

susceptance_vec = [edge_properties[edge].susceptance for edge in 1:num_edges]
max_edge_capacity = [edge_properties[edge].max_capacity for edge in sort(collect(keys(edge_properties)))]

master_dict_bus_to_demand_dict = OrderedDict()

Random.seed!(seed)
warm_start_uncertainty_num = 5
for i in 1:warm_start_uncertainty_num
    rand_key = rand(1:length(dict_bus_to_demand))
    master_dict_bus_to_demand_dict[rand_key] = dict_bus_to_demand[rand_key]
end

keys(master_dict_bus_to_demand_dict)

current_iteration = 1

optimality_gap = upper_bound - lower_bound

first_stage_result_ml_acc = First_Stage_ML_Acc_CCG(first_stage_env, 
                                                    num_buses,num_gens,
                                                    start_up_cost, shut_down_cost, 
                                                    constant_cost_coefficient, linear_cost_coefficient, 
                                                    min_power, max_power,
                                                    min_up_time, min_down_time, 
                                                    ramp_up_limit, ramp_down_limit, 
                                                    start_ramp_limit, shut_ramp_limit,
                                                    master_dict_bus_to_demand_dict, 
                                                    susceptance_matrix, arc_incidence_matrix, susceptance_vec, max_edge_capacity,
                                                    bus_to_generator_dict, 
                                                    high_cost,
                                                    time_period,
                                                    len_of_features, pos_enc_k)

first_stage_result = first_stage_result_ml_acc["first_phase_decision"]

second_stage_values = predict(first_stage_result, 
                                num_buses, len_of_features, pos_enc_k,
                                susceptance_matrix, 
                                bus_to_generator_dict, 
                                dict_bus_to_demand,   ### this dict is the dict of all the demand scenarios in the master set
                                power_generator_property_dict, 
                                high_cost)
