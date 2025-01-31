using PyCall
println("PyCall Python version: ", PyCall.pyversion)
println("PyCall Python executable: ", PyCall.python)
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
parent_dir = "C:\\Users\\dube.rohit\\OneDrive - Texas A&M University\\EconDIspARO\\EconDispARO"
source_folder = "src"

include(joinpath(parent_dir, source_folder, "data_constructors.jl"))
include(joinpath(parent_dir, source_folder, "model_constructors_new.jl"))
###
py"""

import concurrent.futures
import os
import torch
nn = torch.nn
F = torch.nn.functional
Dataset = torch.utils.data.Dataset
 # ADDED IMPORT for Dataset
import numpy as np
import sklearn.preprocessing as sk
import joblib
import scipy.linalg as la
eigh = la.eigh
StandardScaler = sk.StandardScaler
DataLoader = torch.utils.data.DataLoader
TensorDataset = torch.utils.data.TensorDataset


# Load the model and scalers
parent = "C:\\Users\\dube.rohit\\OneDrive - Texas A&M University\\EconDIspARO\\EconDispARO"
folder = "example_data\\24-bus system"      ####### change this folder to the folder where the model and scalers are saved in
file_folder = "post NN training"


###################################################
## Load the scalers
###################################################

scaler_sus = joblib.load(os.path.join(parent, folder, file_folder, "sus_scaler.pkl"))
scaler_target = joblib.load(os.path.join(parent, folder, file_folder, "target_scaler.pkl"))
#print("scaler_target: ", scaler_target.mean_, scaler_target.var_)
scaler_feature = joblib.load(os.path.join(parent,folder, file_folder, "node_scaler.pkl"))
#print("scaler_feature: ", scaler_feature.mean_, scaler_feature.var_)

##################################################
## Scale the input data

##############################################
##### get and scale the input data ###########

def scale_input_features(input_data, scaler_feature):
    return scaler_feature.transform(input_data)

def unscale_target(target_data_from_nn, scaler_target):
    target_data = scaler_target.inverse_transform(target_data_from_nn)
    return target_data

def scale_repeat_susceptance(susceptance, scaler_sus, num_repeat):
    susceptance = np.array(susceptance)
    scale_repeat_susceptance = scaler_sus.transform(susceptance.reshape(-1,1)).reshape(-1, susceptance.shape[1])

    return np.tile(scale_repeat_susceptance, (num_repeat,1)).reshape(-1, scale_repeat_susceptance.shape[0], scale_repeat_susceptance.shape[1])

def repeat_eigenvec_positional_encoding(susceptance_matrix, num_repeat, k = 24):
    ## normalize the laplacian matrix
    susceptance_matrix = np.array(susceptance_matrix)
    D = np.diag(np.diag(susceptance_matrix))    #### this needs to be checked if diagonals are zeros or not
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0  # Handle division by zero
    A = D - susceptance_matrix
    norm_susceptance_matrix = np.eye(susceptance_matrix.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt

    ### find the eigenvalues and eigenvectors of the susceptance matrix
    eigenvalues, eigenvectors = eigh(norm_susceptance_matrix)
    eigenvalues = np.sort(eigenvalues)
    eigenvectors = eigenvectors[:, eigenvalues.argsort()]

    ### create the positional encoding matrix by selecting the first k eigenvectors and repeat foor num_repeat times
    eigen_pe = np.tile(eigenvectors, (num_repeat, 1)).reshape(-1, eigenvectors.shape[0], eigenvectors.shape[1])
    eigen_pe = eigen_pe[:, :,:k]
    return eigen_pe


#####  GAT model for general system ###########
###################################################
# Dataset Class
###################################################
##########***************##########################
###################################################
class GraphDataset(Dataset):
    def __init__(self, node_feats, eigen_pe, susceptance_matrix, target, indices):
        self.node_feats = node_feats
        self.eigen_pe = eigen_pe
        self.sus_matrix = susceptance_matrix
        self.target = target
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return (self.node_feats[i], self.eigen_pe[i], self.sus_matrix[i], self.target[i])

###################################################
class SusTransformer(nn.Module):
    def __init__(self, hidden_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, sus):
        return self.net(sus)

###################################################
###################################################
class GATLayer(nn.Module):
    def __init__(self, c_in, c_out, num_heads=4, concat_heads=True, alpha=0.2, dropout=0.1, residual = True, norm='layernorm', sus_hidden_dim=8):
        super().__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        self.residual = residual

        if self.concat_heads:
            assert c_out % num_heads == 0
            self.head_dim = c_out // num_heads
            self.final_out_dim = c_out
        else:
            self.head_dim = c_out
            self.final_out_dim = c_out

        self.projection = nn.Linear(c_in, self.num_heads*self.head_dim, bias=False)   ###  check
        self.a = nn.Parameter(torch.Tensor(num_heads, 2*self.head_dim + 1))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.sus_transform = SusTransformer(hidden_dim=sus_hidden_dim)

        if norm.lower() == 'layernorm':
            self.norm = nn.LayerNorm(self.final_out_dim)
        elif norm.lower() == 'batchnorm':
            self.norm = nn.BatchNorm1d(self.final_out_dim)
        else:
            raise ValueError("norm must be 'layernorm' or 'batchnorm'")

        if self.residual and c_in != self.final_out_dim:
            self.residual_transform = nn.Linear(c_in, self.final_out_dim, bias=False)  ## check
        else:
            self.residual_transform = None

        nn.init.xavier_uniform_(self.projection.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a, gain=1.414)

    def forward(self, node_feats, susceptance_matrix):
        B, N, _ = node_feats.size()
        node_feats_proj = self.projection(node_feats)
        node_feats_proj = node_feats_proj.view(B, N, self.num_heads, self.head_dim)

        hi = node_feats_proj.unsqueeze(2).expand(B, N, N, self.num_heads, self.head_dim)
        hj = node_feats_proj.unsqueeze(1).expand(B, N, N, self.num_heads, self.head_dim)

        sus_flat = susceptance_matrix.view(B*N*N, 1)
        sus_transformed = self.sus_transform(sus_flat)
        sus_transformed = sus_transformed.view(B, N, N, 1)
        sus_transformed = sus_transformed.unsqueeze(3).expand(B, N, N, self.num_heads, 1)

        a_input = torch.cat([hi, sus_transformed, hj], dim=-1)

        attn_logits = torch.einsum('bnmhx,hx->bnmh', a_input, self.a)
        attn_logits = self.leakyrelu(attn_logits)

        attn_probs = F.softmax(attn_logits, dim=2)
        attn_probs = self.dropout_layer(attn_probs)

        out = torch.einsum('bnmh,bnmhd->bnhd', attn_probs, hj)

        if self.concat_heads:
            out = out.reshape(B, N, self.num_heads*self.head_dim)
        else:
            out = out.mean(dim=2)

        if self.residual:
            if self.residual_transform is not None:
                res = self.residual_transform(node_feats)
            else:
                res = node_feats
            out = out + res

        if isinstance(self.norm, nn.BatchNorm1d):
            out_t = out.transpose(1,2)
            out_t = self.norm(out_t)
            out = out_t.transpose(1,2)
        else:
            out = self.norm(out)

        return out

###################################################
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.mlp(x)
###################################################
class GATwithMLP(nn.Module):
    def __init__(self, c_in=58, c_hidden=64, c_out=1, num_heads=4, sus_hidden_dim=16, alpha=0.2, dropout=0.1, norm='layernorm', k=24, mlp_hidden = 128, residual = True, concat_heads = True):
        super().__init__()
        class EigenPETransform(nn.Module):
            def __init__(self, k, out_dim):
                super().__init__()
                self.linear1 = nn.Linear(k, 128)  # First linear layer
                self.relu1 = nn.ReLU()  # Shared ReLU activation
                self.linear2 = nn.Linear(128, out_dim)  # Second linear layer
                self.relu2 = nn.ReLU()
            def forward(self, x):
                x = self.relu1(self.linear1(x))
                return self.relu2(self.linear2(x))
        
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.c_out = c_out
        self.num_heads = num_heads
        self.sus_hidden_dim = sus_hidden_dim
        self.alpha = alpha
        self.dropout = dropout
        self.norm = norm
        self.k = k
        self.mlp_hidden = mlp_hidden
        self.residual = residual
        self.concat_heads = concat_heads
    
    
        self.eigen_transform = EigenPETransform(k=self.k, out_dim=self.c_in)

        self.gat1 = GATLayer(
            self.c_in, self.c_hidden, num_heads=self.num_heads, concat_heads=self.concat_heads,
            alpha=self.alpha, dropout=self.dropout, residual = self.residual,
            norm=self.norm, sus_hidden_dim=self.sus_hidden_dim
        )
        self.gat2 = GATLayer(
            self.c_hidden, self.c_hidden, num_heads=self.num_heads, concat_heads=self.concat_heads,
            alpha=alpha, dropout=dropout, residual = self.residual,
            norm=norm, sus_hidden_dim=sus_hidden_dim
        )
        self.gat3 = GATLayer(
            self.c_hidden, self.c_hidden, num_heads=self.num_heads, concat_heads=self.concat_heads,
            alpha=self.alpha, dropout=self.dropout, residual = self.residual,
            norm=self.norm, sus_hidden_dim=self.sus_hidden_dim
        )
        self.mlp = MLPRegressor(input_dim=self.c_hidden, hidden_dim=self.mlp_hidden, output_dim=self.c_out)

    def forward(self, node_feats, eigen_pe, sus):

        # Transform eigen positional encodings and combine with node features
        eigen_pe_transformed = self.eigen_transform(eigen_pe)  # [B, N, c_in]
        x = node_feats + eigen_pe_transformed  # [B, N, c_in]

        # Pass through GAT layers
        x = self.gat1(x, sus)
        x = self.gat2(x, sus)
        x = self.gat3(x, sus)

        # Pool over nodes and pass through MLP
        x = x.mean(dim=1)  # Global pooling over nodes [B, c_hidden]
        out = self.mlp(x)  # Final prediction [B, c_out]
        return out
##########***************##########################
###################################################

# Instantiate checkpoint and model args and load weights
checkpoint = torch.load(os.path.join(parent, folder, file_folder, "GAT_model_best_2025-01-16.pth"))

model_args = checkpoint["model_args"]
model_state_dict = checkpoint["model_state_dict"]

GAT_model = GATwithMLP(**model_args)
GAT_model.load_state_dict(model_state_dict)

device = torch.device("cpu")
def predict_model_py_batched(node_feats, eigen_pe, sus, batch_size=500):
    GAT_model.eval()
    
    # B = node_feats.shape[0]  # total number of scenarios or data points
    # all_preds = []
    
    # with torch.no_grad():
    #     for start_idx in range(0, B, batch_size):
    #         end_idx = min(start_idx + batch_size, B)
            
    #         # Slice the batch
    #         batch_node_feats = node_feats[start_idx:end_idx]
    #         batch_eigen_pe   = eigen_pe[start_idx:end_idx]
    #         batch_sus        = sus[start_idx:end_idx]
            
    #         # Convert to torch tensors
    #         batch_node_feats_torch = torch.tensor(batch_node_feats, dtype=torch.float32).contiguous()
    #         batch_eigen_pe_torch   = torch.tensor(batch_eigen_pe,   dtype=torch.float32).contiguous()
    #         batch_sus_torch        = torch.tensor(batch_sus,        dtype=torch.float32).contiguous()
            
    #         # Forward pass
    #         batch_output = GAT_model(batch_node_feats_torch, batch_eigen_pe_torch, batch_sus_torch)
            
    #         # Convert back to numpy
    #         batch_output_np = batch_output.detach().cpu().numpy()
            
    #         # Collect predictions
    #         all_preds.append(batch_output_np)
    
    # # Concatenate all batch predictions along the first dimension
    # all_preds = np.concatenate(all_preds, axis=0)  # shape (B, 1)

    dataset = TensorDataset(torch.tensor(node_feats, dtype=torch.float32).contiguous(),
                            torch.tensor(eigen_pe, dtype=torch.float32).contiguous(),
                            torch.tensor(sus, dtype=torch.float32).contiguous())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            node_feats, eigen_pe, sus = batch
            output = GAT_model(node_feats, eigen_pe, sus)
            all_preds.append(output.detach().cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    return all_preds
# def predict_model_py(node_feats, eigen_pe, sus):
#     GAT_model.eval()
#     with torch.no_grad():
#         node_feats = torch.tensor(node_feats, dtype=torch.float32).contiguous()
#         eigen_pe   = torch.tensor(eigen_pe,   dtype=torch.float32).contiguous()
#         sus        = torch.tensor(sus,        dtype=torch.float32).contiguous()
#         output = GAT_model(node_feats, eigen_pe, sus)
#     return output.detach().cpu().numpy()
# def _predict_chunk(start_idx, end_idx, node_feats, eigen_pe, sus):
#     # Child process code
#     with torch.no_grad():
#         batch_node_feats = torch.tensor(node_feats[start_idx:end_idx], dtype=torch.float32)
#         batch_eigen_pe   = torch.tensor(eigen_pe[start_idx:end_idx],   dtype=torch.float32)
#         batch_sus        = torch.tensor(sus[start_idx:end_idx],        dtype=torch.float32)
#         batch_output = GAT_model(batch_node_feats, batch_eigen_pe, batch_sus)
#     return batch_output.detach().cpu().numpy()

# def predict_model_py_batched_mp(node_feats, eigen_pe, sus, batch_size=2000, nprocs=4):
#     GAT_model.eval()
#     B = node_feats.shape[0]
#     idx_ranges = [(i, min(i+batch_size, B)) for i in range(0, B, batch_size)]
#     results = []
#     with concurrent.futures.ProcessPoolExecutor(max_workers=nprocs) as executor:
#         future_to_chunk = {
#             executor.submit(_predict_chunk, start, end, node_feats, eigen_pe, sus): (start, end)
#             for (start, end) in idx_ranges
#         }
#         for future in concurrent.futures.as_completed(future_to_chunk):
#             results.append(future.result())
#     return np.concatenate(results, axis=0)


"""
###
#######################################################################################################
#######################################################################################################
######################################### finish python code ##########################################
py_predict = py"predict_model_py_batched"  # Grab the Python function by name
# py_predict = inference.predict_model_py_batched_mp


function nodes_feature_matrix(num_buses,length_features,
                                first_stage_decision, bus_to_generator_dict, 
                                bus_to_demand_dict, power_generator_property_dict, 
                                high_cost)

    ## create an empty num_buses x 56 matrix
    input_matrix = zeros(num_buses, length_features)     ## num nodes x 56
    ## the zeros take care of no generators or demand at a bus

    ## fill in the first 24 columns with the first stage decision of the generator output as per dict_bus_to_generator_dict
    for (bus, generator_list) in bus_to_generator_dict
        for generator in generator_list
            input_matrix[bus, 1:24] .= input_matrix[bus, 1:24] .+ first_stage_decision[generator, :]
            input_matrix[bus, 49] = input_matrix[bus, 49] + power_generator_property_dict[generator].start_up_cost
            input_matrix[bus, 50] = input_matrix[bus, 50] + power_generator_property_dict[generator].shut_down_cost
            input_matrix[bus, 51] = input_matrix[bus, 51] + power_generator_property_dict[generator].constant_cost_coefficient
            input_matrix[bus, 52] = input_matrix[bus, 52] + power_generator_property_dict[generator].linear_cost_coefficient
            input_matrix[bus, 53] = input_matrix[bus, 53] + power_generator_property_dict[generator].Min_electricty_output_limit
            input_matrix[bus, 54] = input_matrix[bus, 54] + power_generator_property_dict[generator].Max_electricty_output_limit
            input_matrix[bus, 55] = input_matrix[bus, 55] + power_generator_property_dict[generator].Min_up_time
            input_matrix[bus, 56] = input_matrix[bus, 56] + power_generator_property_dict[generator].Min_down_time
            input_matrix[bus, 57] = input_matrix[bus, 57] + power_generator_property_dict[generator].Ramp_up_limit
            input_matrix[bus, 58] = high_cost
        end
    end
    for (bus, demand) in bus_to_demand_dict
        input_matrix[bus, 25:48] .= demand
    end
    ### scale the input matrix using feature scaling
    input_matrix = py"scale_input_features"(input_matrix, py"scaler_feature")
    return input_matrix
end

################ input matrix for the ML model, for each load scenario, the input matrix is created ########################
##### this function takes in the first stage decisions, parameters of the problems and the dict of all the demand scenarios

function ML_Input_feats(num_buses, length_features, 
                        first_stage_decision_from_jump, 
                        bus_to_generator_dict, 
                        scenarios_bus_to_demand_dict, #### this is the dict of all the demand scenarios D, each of the scenarios have demand defined at each bus
                        power_generator_property_dict, 
                        high_cost)

    ## make an empty matrix of size num demand nodes x 24 x 56
    output_matrix = zeros(length(scenarios_bus_to_demand_dict), num_buses, length_features)
    index = 1
    index_to_uncertainty_key = OrderedDict()
    for key in keys(scenarios_bus_to_demand_dict)   ##
        input = nodes_feature_matrix(num_buses, length_features, 
                                        first_stage_decision_from_jump, bus_to_generator_dict, 
                                        scenarios_bus_to_demand_dict[key], 
                                        power_generator_property_dict, high_cost)
        output_matrix[index, :, :] .= input       ### scaled input matrix
        index_to_uncertainty_key[index] = key     ### the key is the index of the uncertainty scenario matching the index of the input matrix
        index += 1
    end
    return output_matrix, index_to_uncertainty_key   ### the output matrix is the input to the ML model
end

##### the susceptance matrix input is done in the RO CCG model itself, so just get the positional encodings for the susceptance matrixfunction predict()
function predict(first_stage_decision_from_jump,
                    num_buses, length_features,num_pos_enc, ## number of positional encodings of eigen vectors of the normalized laplacian (susceptance) matrix
                    susceptance_matrix,
                    bus_to_generator_dict,
                    scenarios_bus_to_demand_dict,
                    power_generator_property_dict,
                    high_cost)
    #println("predict function started")
    ### scale susceptance matrix
    num_uncertain_scenarios = length(scenarios_bus_to_demand_dict)
    scaled_sus = py"scale_repeat_susceptance"(susceptance_matrix, py"scaler_sus", num_uncertain_scenarios) ## scale the susceptance matrix
    #println("shape of scaled_sus: ", size(scaled_sus))
    eigen_pe = py"repeat_eigenvec_positional_encoding"(susceptance_matrix, num_uncertain_scenarios, num_pos_enc)  ## get the positional encoding for the susceptance matrix
    #println("shape of eigen: ", size(eigen_pe))
    ## get the scaled input matrix for the ML model
    input_matrix, index_to_uncertainty_key = ML_Input_feats(num_buses, length_features, 
                                                            first_stage_decision_from_jump, 
                                                            bus_to_generator_dict, 
                                                            scenarios_bus_to_demand_dict, 
                                                            power_generator_property_dict, 
                                                            high_cost)
    ##println("input_matrix to ML is done")
    ## get the predictions from the ML model
    predictions = py_predict(input_matrix, eigen_pe, scaled_sus)
    predictions = py"unscale_target"(predictions, py"scaler_target")
    ### convert the predictions to a julia list
    predictions = vec(predictions)
    return predictions, index_to_uncertainty_key
end


###################################################################################################################
###################################################################################################################

#################################################################################################
################### function for ML accelerated CCG model #######################################
################################################################################################################

function First_Stage_ML_Acc_CCG(first_phase_model_env, 
                                    num_buses, num_gens,
                                    start_up_cost, shut_down_cost, 
                                    power_generator_property_dict,
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
                                    length_features, num_pos_enc)

    list_phase_1_and_2_keys = []

    result_dict = OrderedDict()

    first_phase_model = First_Stage_First_Phase(first_phase_model_env, num_buses, num_gens,
                                                start_up_cost, shut_down_cost, 
                                                constant_cost_coefficient, linear_cost_coefficient, 
                                                min_power, max_power,
                                                min_up_time, min_down_time, 
                                                ramp_up_limit, ramp_down_limit, 
                                                start_ramp_limit, shut_ramp_limit,
                                                master_dict_bus_to_demand_dict,   ## dict with key as uncertainty number and value as bus to demand dict in master
                                                susceptance_matrix, arc_incidence_matrix, susceptance_vec, max_edge_capacity,
                                                bus_to_generator_dict, 
                                                high_cost,
                                                time_period)

    num_master_uncertainties = length(master_dict_bus_to_demand_dict)  ### number of demand scenarios in the master set
    sorted_uncertainty_keys = sort(collect(keys(master_dict_bus_to_demand_dict)))

    for iter in 1:num_master_uncertainties

        optimize!(first_phase_model)

        first_phase_decision = value.(first_phase_model[:gen_bin])
        first_phase_objective_value = objective_value(first_phase_model)
        first_phase_uncertainty_key = [key for key in sorted_uncertainty_keys if isapprox(value(first_phase_model[Symbol("z_",key)]), 1.0, atol=1e-3)][1]
        #println("first phase prediction started")
        second_phase_values, index_to_key  = predict(first_phase_decision, 
                                                        num_buses, length_features, num_pos_enc,
                                                        susceptance_matrix, 
                                                        bus_to_generator_dict, 
                                                        master_dict_bus_to_demand_dict,   ### this dict is the dict of all the demand scenarios in the master set
                                                        power_generator_property_dict, 
                                                        high_cost)

        second_phase_uncertainty_index = argmax(second_phase_values)

        second_phase_objective_value = second_phase_values[second_phase_uncertainty_index]
        ##println(second_phase_objective_value)

        second_phase_uncertainty_key = index_to_key[second_phase_uncertainty_index]   ### the key of the demand scenario with the highest objective value

        push!(list_phase_1_and_2_keys, (first_phase_uncertainty_key, second_phase_uncertainty_key))  ### at the last step the keys might match, if stopped before that then the keys will not match
        #println(second_phase_values)
        result_dict["objective_value"] = first_phase_objective_value     ## the objective value of the first phase
        result_dict["second_phase_objective_value"] = second_phase_objective_value
        result_dict["second phase values"] = second_phase_values
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


################################################################################################################################
################################################################################################################################

function ML_accelerated_CCG(seed, first_phase_model_env, improved_model_env,  ### use improved model later on
                                num_buses, num_gens, num_edges, length_features, num_pos_enc,
                                power_generator_property_dict,
                                uncertainty_num_to_bus_to_demand_dict, ## dict with key as uncertainty number and value as bus to demand dict
                                bus_to_generator_dict,
                                edge_properties, 
                                high_cost,
                                time_period; warm_start_uncertainty_num = 5)

    lower_bound = -Inf
    upper_bound =  Inf

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
    arc_incidence_matrix = Node_Arc_Incidence_Matrix_Generator(num_buses, edge_properties)
    # print("susceptance_matrix: ", susceptance_matrix)
    # print("arc_incidence_matrix: ", arc_incidence_matrix)
    susceptance_vec = [edge_properties[edge].susceptance for edge in 1:num_edges]
    max_edge_capacity = [edge_properties[edge].max_capacity for edge in sort(collect(keys(edge_properties)))]

    ### Dict to store the demand scenarios selected for the warm start and then append the selected uncertainty scenarios
    master_dict_bus_to_demand_dict = OrderedDict()

    Random.seed!(seed)

    for i in 1:warm_start_uncertainty_num
        rand_key = rand(1:length(uncertainty_num_to_bus_to_demand_dict))
        master_dict_bus_to_demand_dict[rand_key] = uncertainty_num_to_bus_to_demand_dict[rand_key]
    end
    ## print the selected keys
    #println("master_dict_bus_to_demand_dict: ", keys(master_dict_bus_to_demand_dict))

    current_iteration = 0
    optimality_gap = upper_bound - lower_bound
    iter_end_tol = 1e-3

    result_dict = OrderedDict()

    while true
        current_iteration += 1
        #println("current_iteration: ", current_iteration)
        first_stage_result_ml_acc = First_Stage_ML_Acc_CCG(first_phase_model_env, 
                                                            num_buses,num_gens,
                                                            start_up_cost, shut_down_cost, 
                                                            power_generator_property_dict,
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
                                                            length_features, num_pos_enc)

        first_stage_decision = first_stage_result_ml_acc["first_phase_decision"]
        lower_bound = first_stage_result_ml_acc["objective_value"]
        first_stage_selected_demand_key = first_stage_result_ml_acc["selected_demand_key"]
        #println("first stage selected demand key: ", first_stage_selected_demand_key)
        println("lower_bound: ", lower_bound)

        second_stage_values, index_to_key = predict(first_stage_decision, 
                                                        num_buses, length_features, num_pos_enc,
                                                        susceptance_matrix,
                                                        bus_to_generator_dict, 
                                                        uncertainty_num_to_bus_to_demand_dict,
                                                        power_generator_property_dict, 
                                                        high_cost)

        #println("second_stage_values: ", second_stage_values)
    
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

        #println("second_stage_uncertainty_key: ", second_stage_uncertainty_key)
        println("upper_bound: ", upper_bound)

        if isapprox(lower_bound, upper_bound, atol=iter_end_tol)
            master_dict_bus_to_demand_dict[second_stage_uncertainty_key] = uncertainty_num_to_bus_to_demand_dict[second_stage_uncertainty_key]

            # solve the Robust Optimization problem with the master dict
            improved_results = First_Stage_CCG(improved_model_env, num_buses, num_gens,
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
                                                    time_period)


            result_dict["final_lower_bound"] = max(lower_bound, improved_results["objective_value"])
            println("Optimal solution found due to convergence")
            result_dict["final_upper_bound"] = upper_bound
            break
        end

        if isapprox(optimality_gap, upper_bound - lower_bound, atol=iter_end_tol)
            println("Optimal solution found due to convergence of the optimality gap")
            #result_dict["final_lower_bound"] = lower_bound
                        ## solve the Robust Optimization problem with the master dict
            improved_results = First_Stage_CCG(improved_model_env, num_buses, num_gens,
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
                                                    time_period)


            result_dict["final_lower_bound"] = max(lower_bound, improved_results["objective_value"])
            result_dict["final_upper_bound"] = upper_bound
            break
        end

        if current_iteration > 990
            println("Error: Bad convergence")
            master_dict_bus_to_demand_dict[second_stage_uncertainty_key] = uncertainty_num_to_bus_to_demand_dict[second_stage_uncertainty_key]
                        ## solve the Robust Optimization problem with the master dict
            improved_results = First_Stage_CCG(improved_model_env, num_buses, num_gens,
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
                                                    time_period)


            result_dict["final_lower_bound"] = max(lower_bound, improved_results["objective_value"])
            result_dict["final_upper_bound"] = upper_bound
            break
        end

        optimality_gap = upper_bound - lower_bound
        #println("not converged: current optimality_gap: ", optimality_gap)
        #println("second_stage_uncertainty_key: ", second_stage_uncertainty_key)
        master_dict_bus_to_demand_dict[second_stage_uncertainty_key] = sort_values_by_keys(uncertainty_num_to_bus_to_demand_dict[second_stage_uncertainty_key])
    end               
    return result_dict                                 
end

true