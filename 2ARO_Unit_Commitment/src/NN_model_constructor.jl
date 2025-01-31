using PyCall
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
import Gurobi
import JuMP
#################################################################################################
####################### load the data and preprocess it #########################################
parent_dir = pwd()
source_folder = "src"

include(joinpath(parent_dir, source_folder, "data_constructors.jl"))
include(joinpath(parent_dir, source_folder, "model_constructors_new.jl"))

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

generated_edge_properties = OrderedDict{Int, Any}()
keys_to_extract = ["edge_no", "susceptance", "min_capacity", "max_capacity"]

for (k, v) in generated_edge
    pivoted_data = OrderedDict{String, Any}()
    for key in keys_to_extract
        pivoted_data[key] = map(d -> d[key], values(v))
    end
    pivoted_data["node_1"], pivoted_data["node_2"] = map(d -> d["node_tuple"][1], values(v)), map(d -> d["node_tuple"][2], values(v))
    generated_edge[k] = DataFrame(pivoted_data)
    generated_edge_properties[k] = Edge_Properties_Set(generated_edge[k])
end

generated_load[373] = generated_load[1] 
#################################################################################################
####################### Python script for forward pass of the GAT model #########################
py"""

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

import numpy as np
import pandas as pd
import os

import sklearn.preprocessing as sk
import joblib

StandardScaler = sk.StandardScaler

folder = "example_data\\6-bus system"
file_folder = "post NN training"

scaler_node_c = joblib.load(os.path.join(folder, file_folder, "scaler_node.pkl"))
scaler_target_c = joblib.load(os.path.join(folder, file_folder, "scaler_target.pkl"))
weights = torch.load(os.path.join(folder, file_folder, "best_model.pth"))

def scale_node(unscaled_node_feats):
    unscaled_node_feats = np.array(unscaled_node_feats)
    new_node_feat = scaler_node_c.transform(unscaled_node_feats.reshape(-1, 
                        unscaled_node_feats.shape[-1])).reshape(unscaled_node_feats.shape)
    
    new_node_feat = Tensor(new_node_feat.astype(np.float32))
    return new_node_feat

def unscale_output(predictions):
    output = scaler_target_c.inverse_transform(predictions.detach().numpy())
    return output.flatten()

# def processed_susceptance(susceptance_matrix):
#     susceptance_matrix_copy = np.copy(np.array(susceptance_matrix))
#     ## convert a 6x6 matrix to a 1x6x6 matrix
#     #susceptance_matrix_copy = susceptance_matrix_copy.reshape(1, susceptance_matrix_copy.shape[0], susceptance_matrix_copy.shape[1])
#     for i in range(susceptance_matrix_copy.shape[0]):
#         np.fill_diagonal(susceptance_matrix_copy[i], -1 * np.diagonal(susceptance_matrix_copy[i]))
#     # Vectorized version without loop
#     epsilon = 1e-6
#     max_vals = np.max(susceptance_matrix_copy, axis=1, keepdims=True).reshape(susceptance_matrix_copy.shape[0],-1,1)  # Compute max along axis 1 (row-wise)
#     susceptance_matrix_copy = susceptance_matrix_copy / (max_vals + epsilon)
#     susceptance_matrix_copy = Tensor(susceptance_matrix_copy.astype(np.float64))
#     return susceptance_matrix_copy


def processed_susceptance(susceptance_matrix):
    susceptance_matrix_copy = np.copy(np.array(susceptance_matrix))
    ## convert the positive diagonal of 6x6 matrix to negative
    np.fill_diagonal(susceptance_matrix_copy, np.diagonal(susceptance_matrix_copy) * -1)
    susceptance_matrix_copy = - susceptance_matrix_copy
    epsilon = 1e-6
    max_vals = np.max(susceptance_matrix_copy, axis=1, keepdims=True)
    susceptance_matrix_copy = susceptance_matrix_copy / (max_vals + epsilon)
    susceptance_matrix_copy = Tensor(susceptance_matrix_copy.astype(np.float64))
    susceptance_matrix_copy = susceptance_matrix_copy.unsqueeze(0)

    return susceptance_matrix_copy




def gat_forward_pass(node_feats, edge_feats, weights, num_heads, gat_no):
    batch_size = node_feats.shape[0]
    num_nodes = node_feats.shape[1]
    c_in = node_feats.shape[2]
    c_out = 64
    
    node_feats = F.linear(node_feats, weights['gat_layer_'+str(gat_no)+'.projection.weight'], weights['gat_layer_'+str(gat_no)+'.projection.bias'])  # Linear projection
    #node_feats = F.leaky_relu(node_feats, negative_slope=0.2)  # LeakyReLU activation
    node_feats = F.layer_norm(node_feats, [weights["gat_layer_"+str(gat_no)+".layer_norm.weight"].shape[0]], weights['gat_layer_'+str(gat_no)+'.layer_norm.weight'], weights['gat_layer_'+str(gat_no)+'.layer_norm.bias'])  # LayerNorm

    node_feats = node_feats.view(batch_size, num_nodes, num_heads, -1)

    
    adj_matrix = (edge_feats != 0).int()
    #print(edge_feats.shape)
    # Weight transformation in GAT
    edge_feats = F.linear(edge_feats, weights['gat_layer_'+str(gat_no)+'.weight_linear_transform.weight'], weights['gat_layer_'+str(gat_no)+'.weight_linear_transform.bias'])
    edge_feats = F.relu(edge_feats)  # ReLU on weight matrix
    #print(edge_feats.shape)
    edge_feats_flat = edge_feats.view(batch_size * num_nodes, num_heads,-1)
    #print(edge_feats_flat.shape)
    
    edges = adj_matrix.nonzero(as_tuple=False)

    node_feats_flat = node_feats.view(batch_size * num_nodes, num_heads,-1)

    edge_indices_row = edges[:, 0] * num_nodes + edges[:, 1]
    edge_indices_col = edges[:, 0] * num_nodes + edges[:, 2]

    a_input = torch.cat(
                        [torch.index_select(node_feats_flat, index=edge_indices_row, dim=0),torch.index_select(node_feats_flat, index=edge_indices_col, dim=0) + torch.index_select(input=edge_feats_flat, index=edge_indices_col, dim=0),],
                        dim=-1,
                        )
    
    attn_logit = F.leaky_relu(torch.einsum("bhc,hc->bh", a_input, weights["gat_layer_"+str(gat_no)+".a"]), negative_slope=0.2) 
    attn_matrix = attn_logit.new_zeros(adj_matrix.shape + (num_heads,)).fill_(-9e15)
    attn_matrix[adj_matrix[..., None].repeat(1, 1, 1, num_heads) == 1] = attn_logit.reshape(-1)
    attn_probs = F.softmax(attn_matrix, dim=2)
    
    node_feats = torch.einsum("bijh,bjhc->bihc", attn_probs, node_feats)
    node_feats = node_feats.reshape(batch_size, num_nodes, -1)
    #print(node_feats.shape)
    #node_feats = node_feats.mean(dim=2)
    return node_feats

def forward_pass(node_feats_in, edge_feats_in, weights=weights, num_heads=16):
    

    node_feats = node_feats_in
    
    edge_feats = edge_feats_in
    #print(node_feats.shape)
    node_feats = gat_forward_pass(node_feats, edge_feats, weights, num_heads, 1)
    #print(node_feats.shape)
    edge_feats = edge_feats_in
    node_feats = gat_forward_pass(node_feats, edge_feats, weights, num_heads, 2)

    

    x = node_feats.view(node_feats.size(0), -1)

    x = F.linear(x, weights['mlp_model.fc1.weight'], weights['mlp_model.fc1.bias'])
    x = F.relu(x)
    
    x = F.linear(x, weights['mlp_model.fc2.weight'], weights['mlp_model.fc2.bias'])
    x = F.relu(x)
    
    x = F.linear(x, weights['mlp_model.fc3.weight'], weights['mlp_model.fc3.bias'])
    x = F.relu(x)
    
    # Final output layer
    x = F.linear(x, weights['mlp_model.fc4.weight'], weights['mlp_model.fc4.bias'])
    #print(x)
    output = unscale_output(x)
    ### return the output to a vector
    return output
    #return x

# def predict(node_feats, edge_feats, weights, num_heads=16):

#     node_feats_out = scale_node(node_feats)
#     edge_feats_out = processed_susceptance(edge_feats)

#     output = forward_pass(node_feats_out, edge_feats_out, weights, num_heads=num_heads)
#     return output

"""
#################################################################################################

function ML_Input_per_load(first_stage_decision, bus_to_generator_dict, bus_to_demand_dict, power_generator_property_dict, high_cost)
    ## create an empty num_nodes x 56 matrix
    input_matrix = zeros(6, 56)
    ## fill in the first 24 columns with the first stage decision of the generator output as per dict_bus_to_generator_dict
    for (bus, generator_list) in bus_to_generator_dict
        for generator in generator_list
            input_matrix[bus, 1:24] .= first_stage_decision[generator, :]
            input_matrix[bus, 49] = power_generator_property_dict[generator].start_up_cost
            input_matrix[bus, 50] = power_generator_property_dict[generator].shut_down_cost
            input_matrix[bus, 51] = power_generator_property_dict[generator].constant_cost_coefficient
            input_matrix[bus, 52] = power_generator_property_dict[generator].linear_cost_coefficient
            input_matrix[bus, 53] = power_generator_property_dict[generator].Min_electricty_output_limit
            input_matrix[bus, 54] = power_generator_property_dict[generator].Max_electricty_output_limit
            input_matrix[bus, 55] = power_generator_property_dict[generator].Ramp_up_limit
            input_matrix[bus, 56] = high_cost

        end
    end

    for (bus, demand) in bus_to_demand_dict
        input_matrix[bus, 25:48] .= demand
    end
    return input_matrix
end
    

function ML_Input_feats(model_results, bus_to_generator_dict, input_bus_to_demand_dict, power_generator_property_dict, high_cost)
    ## make an empty matrix of size 2 x 6 x 56
    output_matrix = zeros(length(input_bus_to_demand_dict), 6, 56)
    index = 1
    index_to_uncertainty_key = OrderedDict()
    for key in keys(input_bus_to_demand_dict)
        
        input = ML_Input_per_load(model_results, bus_to_generator_dict, input_bus_to_demand_dict[key], power_generator_property_dict, high_cost)

        # push!(output_matrix, input)
        # println(output_matrix)
        output_matrix[index, :, :] .= input
        index_to_uncertainty_key[index] = key
        index += 1
    end
    return output_matrix, index_to_uncertainty_key
end

function predict(first_stage_decision_from_jump,
                    scaled_processed_susceptance,
                    bus_to_generator_dict,
                    dict_bus_to_demand,
                    power_generator_property_dict,
                    high_cost,
                    num_heads)
    input_feats, index_to_key = ML_Input_feats(first_stage_decision_from_jump, bus_to_generator_dict, dict_bus_to_demand, power_generator_property_dict, high_cost)
    scaled_input = py"scale_node"(input_feats)
    processed_susceptance_matrix = scaled_processed_susceptance.repeat(length(dict_bus_to_demand),1,1)
    output = py"forward_pass"(scaled_input, processed_susceptance_matrix, py"weights", num_heads)
    return output, index_to_key
end


# function predict_gat(model_results, bus_to_generator_dict, input_bus_to_demand_dict, power_generator_property_dict, susceptance_matrix, high_cost, num_heads)
#     input_feats, index_to_key = ML_Input_feats(model_results, bus_to_generator_dict, input_bus_to_demand_dict, power_generator_property_dict, high_cost)
#     output = py"predict"(input_feats, susceptance_matrix, py"weights", num_heads)
#     return output, index_to_key
# end
#################################################################################################
#################################################################################################
## solve a single model with 2 loads

# model_env = Gurobi.Env()
# num_buses = 6
# num_gens = 3
# time_period = 24
# high_cost = 500

# generator_data = generated_gen[1]
# bus_to_generator_dict = Group_Generators_by_Bus(generator_data.generator_no, generator_data.bus_no)
# power_generator_property_dict = Power_Generator_Set(generator_data)

# dict_bus_to_demand = OrderedDict()
# dict_bus_to_demand[1] = generated_load[1]
# dict_bus_to_demand[2] = generated_load[2]
# dict_bus_to_demand[3] = generated_load[3]

# edge_properties = generated_edge_properties[1]

# model_env = Gurobi.Env()
# num_buses = 6
# num_gens = 3
# time_period = 24
# high_cost = 500



# model_results = Econ_Disp_Model(model_env, num_buses, num_gens,
#                                 power_generator_property_dict,
#                                 generated_load, 
#                                 bus_to_generator_dict, 
#                                 edge_properties, 
#                                 high_cost, 
#                                 time_period)

# first_stage_decision = model_results["gen_bin"]
# susceptance_matrix = model_results["susceptance_matrix"]
# num_heads = 16

# #predict(first_stage_decision, bus_to_generator_dict, dict_bus_to_demand, power_generator_property_dict, susceptance_matrix, high_cost, num_heads)

# ## check if the node feats are correctly scaled
# input_feats, index_to_key = ML_Input_feats(first_stage_decision, bus_to_generator_dict, generated_load, power_generator_property_dict, high_cost)
# scaled_input = py"scale_node"(input_feats)
# # print dimension of the scaled input

# ## process the susceptance matrix
# processed_susceptance_matrix = py"processed_susceptance"(susceptance_matrix).repeat(length(generated_load),1,1)

# ## check if the forward pass works

# output = py"forward_pass"(scaled_input, processed_susceptance_matrix, py"weights", num_heads)

# scaled_processed_susceptance = py"processed_susceptance"(susceptance_matrix)



# output, index_to_key = predict(first_stage_decision, scaled_processed_susceptance, bus_to_generator_dict, generated_load, power_generator_property_dict, high_cost, num_heads)



#################################################################################################
################### function for ML accelerated CCG model #######################################

################################################################################################################

function First_Stage_ML_Acc_CCG(first_phase_model_env, 
                                num_buses, num_gens, num_heads,
                                start_up_cost, shut_down_cost, 
                                constant_cost_coefficient, linear_cost_coefficient, 
                                min_power, max_power,
                                min_up_time, min_down_time, 
                                ramp_up_limit, ramp_down_limit, 
                                start_ramp_limit, shut_ramp_limit,
                                master_dict_bus_to_demand_dict, 
                                susceptance_matrix, scaled_susceptance_matrix, arc_incidence_matrix, susceptance_vec, max_edge_capacity,
                                bus_to_generator_dict, 
                                high_cost,
                                time_period)


    list_phase_1_and_2_keys = []

    result_dict = OrderedDict()
    

    first_phase_model = First_Stage_First_Phase(first_phase_model_env, num_buses, num_gens,
                                                    start_up_cost, shut_down_cost, 
                                                    constant_cost_coefficient, linear_cost_coefficient, 
                                                    min_power, max_power,
                                                    min_up_time, min_down_time, 
                                                    ramp_up_limit, ramp_down_limit, 
                                                    start_ramp_limit, shut_ramp_limit,
                                                    master_dict_bus_to_demand_dict,   ## dict with key as uncertainty number and value as bus to demand dict
                                                    susceptance_matrix, arc_incidence_matrix, susceptance_vec, max_edge_capacity,
                                                    bus_to_generator_dict, 
                                                    high_cost,
                                                    time_period)

    num_master_uncertainties = length(master_dict_bus_to_demand_dict)
    sorted_uncertainty_keys = sort(collect(keys(master_dict_bus_to_demand_dict)))

    for iter in 1:num_master_uncertainties
        
        optimize!(first_phase_model)

        first_phase_decision = value.(first_phase_model[:gen_bin])
        first_phase_objective_value = objective_value(first_phase_model)
        first_phase_uncertainty_key = [key for key in sorted_uncertainty_keys if isapprox(value(first_phase_model[Symbol("z_",key)]), 1.0, atol=1e-3)][1]
        
        second_phase_values, index_to_key  = predict(first_phase_decision, 
                                                        scaled_susceptance_matrix, 
                                                        bus_to_generator_dict, 
                                                        master_dict_bus_to_demand_dict, 
                                                        power_generator_property_dict, 
                                                        high_cost, num_heads)
        
        second_phase_uncertainty_index = argmax(second_phase_values)
        second_phase_objective_value = second_phase_values[second_phase_uncertainty_index]
        second_phase_uncertainty_key = index_to_key[second_phase_uncertainty_index]

        push!(list_phase_1_and_2_keys, (first_phase_uncertainty_key, second_phase_uncertainty_key))
        #println(second_phase_uncertainty_key)

        
        result_dict["objective_value"] = first_phase_objective_value     ## the objective value of the first phase
        result_dict["second_phase_objective_value"] = second_phase_objective_value
        result_dict["num_itertaions"] = iter
        result_dict["first_phase_decision"] = first_phase_decision
        result_dict["selected_demand_key"] = first_phase_uncertainty_key
        result_dict["uncertainty_key_combinations"] = list_phase_1_and_2_keys
        result_dict["susceptance_matrix"] = susceptance_matrix

        if (first_phase_uncertainty_key == second_phase_uncertainty_key) || (iter == length(master_dict_bus_to_demand_dict))
            break
        else
            @constraint(first_phase_model, first_phase_model[Symbol("z_",first_phase_uncertainty_key)] == 0)
        end

    end
    return result_dict
end

#################################################################################################

function ML_accelerated_CCG(seed, first_phase_model_env, improved_model_env,
                                num_buses, num_gens, num_edges, num_heads,
                                power_generator_property_dict,
                                uncertainty_num_to_bus_to_demand_dict, ## dict with key as uncertainty number and value as bus to demand dict
                                bus_to_generator_dict,
                                edge_properties, 
                                high_cost,
                                time_period; warm_start_uncertainty_num = 5)

    lower_bound = -Inf
    upper_bound = Inf

    @assert(length(uncertainty_num_to_bus_to_demand_dict) > 0, "The uncertainty dict is empty")
    @assert(length(power_generator_property_dict) == num_gens, "The number of generators in the power generator dict is not equal to the number of generators in the model")

    generator_name_list = sort(collect(keys(power_generator_property_dict)))

    ### Generator properties 
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

    ## Edge properties
    @assert(length(edge_properties) == num_edges, "The number of edges in the edge properties dict is not equal to the number of edges in the model")

    susceptance_matrix = Susceptance_Matrix_Generator(num_buses, edge_properties)     # num_buses x num_buses matrix
    scaled_processed_susceptance_matrix = py"processed_susceptance"(susceptance_matrix)
    arc_incidence_matrix = Node_Arc_Incidence_Matrix_Generator(num_buses, edge_properties)
    susceptance_vec = [edge_properties[edge].susceptance for edge in 1:num_edges]
    max_edge_capacity = [edge_properties[edge].max_capacity for edge in sort(collect(keys(edge_properties)))]

    master_dict_bus_to_demand_dict = OrderedDict()

    Random.seed!(seed)

    for i in 1:warm_start_uncertainty_num
        rand_key = rand(1:length(uncertainty_num_to_bus_to_demand_dict))
        master_dict_bus_to_demand_dict[rand_key] = uncertainty_num_to_bus_to_demand_dict[rand_key]
    end

    current_iteration = 0

    optimality_gap = upper_bound - lower_bound

    iter_end_tol = 1e-3

    result_dict = OrderedDict()

    while true
        current_iteration += 1

        first_stage_result_ml_acc = First_Stage_ML_Acc_CCG(first_phase_model_env, 
                                                            num_buses, num_gens, num_heads,
                                                            start_up_cost, shut_down_cost, 
                                                            constant_cost_coefficient, linear_cost_coefficient, 
                                                            min_power, max_power,
                                                            min_up_time, min_down_time, 
                                                            ramp_up_limit, ramp_down_limit, 
                                                            start_ramp_limit, shut_ramp_limit,
                                                            master_dict_bus_to_demand_dict, 
                                                            susceptance_matrix, scaled_processed_susceptance_matrix, arc_incidence_matrix, susceptance_vec, max_edge_capacity,
                                                            bus_to_generator_dict, 
                                                            high_cost,
                                                            time_period)

        first_stage_decision = first_stage_result_ml_acc["first_phase_decision"]
        lower_bound = first_stage_result_ml_acc["objective_value"]
        first_stage_selected_demand_key = first_stage_result_ml_acc["selected_demand_key"]

        second_stage_values, index_to_key = predict(first_stage_decision, 
                                                    scaled_processed_susceptance_matrix, 
                                                    bus_to_generator_dict, 
                                                    uncertainty_num_to_bus_to_demand_dict,
                                                    power_generator_property_dict, 
                                                    high_cost, num_heads)


        second_stage_uncertainty_index = argmax(second_stage_values)
        second_stage_objective = second_stage_values[second_stage_uncertainty_index]
        second_stage_uncertainty_key = index_to_key[second_stage_uncertainty_index]
        upper_bound = second_stage_objective
        
        result_dict[current_iteration] = OrderedDict()
        result_dict[current_iteration]["first_stage_decision"] = first_stage_decision
        result_dict[current_iteration]["first_stage_obj_key"] = first_stage_selected_demand_key
        result_dict[current_iteration]["lower_bound"] = lower_bound
        result_dict[current_iteration]["upper_bound"] = upper_bound
        result_dict[current_iteration]["optimality_gap"] = upper_bound - lower_bound
        result_dict[current_iteration]["first_stage_phase_1_2_keys"] = first_stage_result_ml_acc["uncertainty_key_combinations"]

        

        if isapprox(lower_bound, upper_bound, atol=iter_end_tol)
            ## add the second stage selected uncertainty to the master uncertainty dict
            master_dict_bus_to_demand_dict[second_stage_uncertainty_key] = uncertainty_num_to_bus_to_demand_dict[second_stage_uncertainty_key]

            ## solve the Robust Optimization problem with the master dict
            # improved_results = First_Stage_CCG(improved_model_env, num_buses, num_gens,
            #                                         start_up_cost, shut_down_cost, 
            #                                         constant_cost_coefficient, linear_cost_coefficient, 
            #                                         min_power, max_power,
            #                                         min_up_time, min_down_time, 
            #                                         ramp_up_limit, ramp_down_limit, 
            #                                         start_ramp_limit, shut_ramp_limit,
            #                                         master_dict_bus_to_demand_dict, 
            #                                         susceptance_matrix, arc_incidence_matrix, susceptance_vec, max_edge_capacity,
            #                                         bus_to_generator_dict, 
            #                                         high_cost,
            #                                         time_period)


            #result_dict["final_lower_bound"] = max(lower_bound, improved_results["objective_value"])
            result_dict["final_lower_bound"] = lower_bound
            result_dict["final_upper_bound"] = upper_bound
            break
        end

        ## if the new optimality gap is same as the old one approximately at a given tolerance then break
        if isapprox(optimality_gap, upper_bound - lower_bound, atol=iter_end_tol)
            master_dict_bus_to_demand_dict[second_stage_uncertainty_key] = uncertainty_num_to_bus_to_demand_dict[second_stage_uncertainty_key]

            # ## solve the Robust Optimization problem with the master dict
            # improved_results = First_Stage_CCG(improved_model_env, num_buses, num_gens,
            #                                         start_up_cost, shut_down_cost, 
            #                                         constant_cost_coefficient, linear_cost_coefficient, 
            #                                         min_power, max_power,
            #                                         min_up_time, min_down_time, 
            #                                         ramp_up_limit, ramp_down_limit, 
            #                                         start_ramp_limit, shut_ramp_limit,
            #                                         master_dict_bus_to_demand_dict, 
            #                                         susceptance_matrix, arc_incidence_matrix, susceptance_vec, max_edge_capacity,
            #                                         bus_to_generator_dict, 
            #                                         high_cost,
            #                                         time_period)


            # result_dict["final_lower_bound"] = max(lower_bound, improved_results["objective_value"])
            result_dict["final_lower_bound"] = lower_bound
            result_dict["final_upper_bound"] = upper_bound
            break
        end

        ## if the lower bound is greater than the upper bound then print error and break

        if current_iteration > 990
            println("Error: Bad convergence")
            master_dict_bus_to_demand_dict[second_stage_uncertainty_key] = uncertainty_num_to_bus_to_demand_dict[second_stage_uncertainty_key]

            ## solve the Robust Optimization problem with the master dict
            # improved_results = First_Stage_CCG(improved_model_env, num_buses, num_gens,
            #                                         start_up_cost, shut_down_cost, 
            #                                         constant_cost_coefficient, linear_cost_coefficient, 
            #                                         min_power, max_power,
            #                                         min_up_time, min_down_time, 
            #                                         ramp_up_limit, ramp_down_limit, 
            #                                         start_ramp_limit, shut_ramp_limit,
            #                                         master_dict_bus_to_demand_dict, 
            #                                         susceptance_matrix, arc_incidence_matrix, susceptance_vec, max_edge_capacity,
            #                                         bus_to_generator_dict, 
            #                                         high_cost,
            #                                         time_period)


            # result_dict["final_lower_bound"] = max(lower_bound, improved_results["objective_value"])
            result_dict["final_lower_bound"] = lower_bound
            result_dict["final_upper_bound"] = upper_bound
            break
        end

        optimality_gap = upper_bound - lower_bound

        master_dict_bus_to_demand_dict[second_stage_uncertainty_key] = sort_values_by_keys(uncertainty_num_to_bus_to_demand_dict[second_stage_uncertainty_key])
    end               
    return result_dict                                 
end


# #################################################################################################
## run the ML accelerated CCG model
first_stage_env = Gurobi.Env()
improved_stage_env = Gurobi.Env()

seed = 1
num_buses = 6
num_gens = 3
time_period = 24
num_edges = 7
high_cost = 500
num_heads = 16

final_ML_CCG_wo_imp_results = OrderedDict()
time_ML_CCG_wo_imp_results = OrderedDict()

for (gen_key, gen_data) in generated_gen
    println("Started running the model for gen_key: ", gen_key)
    power_generator_property_dict = Power_Generator_Set(gen_data)
    bus_to_generator_dict = Group_Generators_by_Bus(gen_data.generator_no, gen_data.bus_no)

    for (edge_key, edge_data) in generated_edge_properties

        edge_properties = edge_data
        start_time = time()

        try
            result_ML_CCG = ML_accelerated_CCG(seed, first_stage_env, improved_stage_env,
                                                num_buses, num_gens, num_edges, num_heads,
                                                power_generator_property_dict,
                                                generated_load, ## dict with key as uncertainty number and value as bus to demand dict
                                                bus_to_generator_dict,
                                                edge_properties, 
                                                high_cost,
                                                time_period; warm_start_uncertainty_num = 5)
            
            final_ML_CCG_wo_imp_results[(gen_key, edge_key)] = result_ML_CCG
            time_ML_CCG_wo_imp_results[(gen_key, edge_key)] = time() - start_time
            println("time taken to run per instance: ", time() - start_time)

        catch e
            println("Error: ", e)
            println("Error in gen_key: ", gen_key, " edge_key: ", edge_key)
            continue
        end
    end
end

#################################################################################################
## save the results
JSON.open(joinpath(parent_dir, "example_data/6-bus system/results", "final_ML_CCG_wo_imp_results.json"), "w") do io
    JSON.print(io, final_ML_CCG_wo_imp_results)
end

JSON.open(joinpath(parent_dir, "example_data/6-bus system/results", "time_ML_CCG_wo_imp_results.json"), "w") do io
    JSON.print(io, time_ML_CCG_wo_imp_results)
end



## check a single instance



# seed = 1
# generator_data = generated_gen[1]
# bus_to_generator_dict = Group_Generators_by_Bus(generator_data.generator_no, generator_data.bus_no)
# power_generator_property_dict = Power_Generator_Set(generator_data)

# dict_bus_to_demand = OrderedDict()
# dict_bus_to_demand[1] = generated_load[1]
# dict_bus_to_demand[2] = generated_load[2]
# dict_bus_to_demand[3] = generated_load[3]

# edge_properties = generated_edge_properties[1]

# first_stage_env = Gurobi.Env()
# improved_stage_env = Gurobi.Env()
# num_buses = 6
# num_gens = 3
# num_edges = 7
# time_period = 24
# high_cost = 500
# num_heads = 16

# model_results = ML_accelerated_CCG(seed, first_stage_env, improved_stage_env,
#                                     num_buses, num_gens, num_edges, num_heads,
#                                     power_generator_property_dict,
#                                     dict_bus_to_demand, ## dict with key as uncertainty number and value as bus to demand dict
#                                     bus_to_generator_dict,
#                                     edge_properties, 
#                                     high_cost,
#                                     time_period; warm_start_uncertainty_num = 5)
