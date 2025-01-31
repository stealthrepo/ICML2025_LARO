using JSON 
using JuMP
using Gurobi
using DataFrames
using CSV
using JSON
using DataStructures
using DecisionTree

###############################################################################################################################################################
function custom_sort(item)
    parse(Int, split(item, '_')[end])
end

###############################################################################################################################################################
function scale_vector(x, x_min, x_max)
    return (x .- x_min) ./ (x_max .- x_min)
end

function scale_inverse(x, x_min, x_max)
    return x .* (x_max .- x_min) .+ x_min
end
###############################################################################################################################################################



###############################################################################################################################################################
"""
    json_to_weights(path::String)

Converts a JSON file containing weights and biases into dictionaries.

# Arguments
- `path::String`: The path to the JSON file from  the Pytorch Moddule.

# Returns
- `weights_dict::Dict{String, Any}`: A dictionary containing the values as weights with key as the neural network layer name.
- `bias_dict::Dict{String, Any}`: A dictionary containing the values as biases with key as the neural network layer name.
"""
function json_to_weights(path::String)
    # Load JSON file
    weights = JSON.parsefile(path)

    
    for (key, value) in weights
        for i in 1:size(value)[1]
            value[i] = Float64.(value[i])
        end
        weights[key] = reduce(vcat, transpose(weights[key]))
    end

    weights_dict = Dict{String, Any}()
    bias_dict = Dict{String, Any}()

    # Regular expression pattern to match ".weight" and ".bias"
    pattern = r"\.(weight|bias)$"

    for (key, value) in weights
        # Remove ".weight" or ".bias" from the key
        key_base = replace(key, pattern => "")
        if contains(key, "weight")
            weights_dict[key_base] = value
        elseif contains(key, "bias")
            bias_dict[key_base] = value
        end
    end
    
    return weights_dict, bias_dict
end



###############################################################################################################################################################

function add_nn_layer_vars_constraints!(model, weight, bias, list_of_input_vectors, key, layer_relu, k; L = -10E3, U = 10E3)

    input_size = size(weight)[2]
    output_size = size(weight)[1]

    input_vector = vcat(list_of_input_vectors...)
    
    @assert(input_size == length(input_vector), "Error: Size of inputs are not the same")

    if layer_relu == true
        model[Symbol('x','_',key,'_','k',k)]  =  JuMP.@variable(model, [1:output_size], base_name = "x_$(key),_k,$(k)", lower_bound = 0)    
        model[Symbol('z','_',key,'_','k',k)]  =  JuMP.@variable(model, [1:output_size], base_name = "z_$(key),_k,$(k)",  Bin)   ## Binary variable for each layer to model the relu
    else
        model[Symbol('x','_',key,'_','k',k)]  =  JuMP.@variable(model, [1:output_size], base_name = "x_$(key),_k,$(k)")          ## No relu thus lower_bound cannot be set to 0
    end

    @assert(output_size == length(model[Symbol('x','_',key,'_','k',k)]), "Error: Size of outputs are not the same")


    for j in 1:output_size
        weight_sum = sum(weight[j,:] .* input_vector) + bias[j]
        if layer_relu
            JuMP.@constraint(model, L <= weight_sum <= U)
            JuMP.@constraint(model, model[Symbol('x','_',key,'_','k',k)][j]  >= weight_sum)
            JuMP.@constraint(model, model[Symbol('x','_',key,'_','k',k)][j]  <= weight_sum - L * (1 - model[Symbol('z','_',key,'_','k',k)][j])) 
            JuMP.@constraint(model, model[Symbol('x','_',key,'_','k',k)][j]  <= U * model[Symbol('z','_',key,'_','k',k)][j])
        else
            JuMP.@constraint(model, model[Symbol('x','_',key,'_','k',k)][j] == weight_sum)
        end
    end
end

###############################################################################################################################################################
"""
    NN_MILP!(model::JuMP.Model, model_architecture, weight, bias; k=1, bound_L = -10E5, bound_U = 10E5)
    The Jump model variables used in this function should all be declared outside of this function
TBW
"""

function NN_MILP!(model::JuMP.Model, model_architecture, weight, bias, k; bound_L=-10E5, bound_U=10E5)
    
    for (layer_key, layer_value) in model_architecture["hidden_layers"]
        in_vars = []
        if layer_value[1] in ["instance", "X"]
            push!(in_vars, model[Symbol(layer_value[1])])
            add_nn_layer_vars_constraints!(model, weight[layer_key], bias[layer_key], in_vars, layer_key, false, k; L=bound_L, U=bound_U)
        elseif layer_value[1] == "U"
            push!(in_vars, model[Symbol("U", k)])
            add_nn_layer_vars_constraints!(model, weight[layer_key], bias[layer_key], in_vars, layer_key, false, k; L=bound_L, U=bound_U)
        else
            for i in layer_value
                push!(in_vars, model[Symbol('x','_',i,'_','k',k)])
                #print(in_vars)
            end
            add_nn_layer_vars_constraints!(model, weight[layer_key], bias[layer_key], in_vars, layer_key, true, k; L=bound_L, U=bound_U)
        end
        #add_neural_network_constraint!(model, weight[layer_key], bias[layer_key], in_vars, out_x, out_bin, true, bound_L, bound_U)
    end

    # Define the constraints for the output layers
    for (layer_key, layer_value) in model_architecture["output_layers"]
        in_vars = [model[Symbol('x','_',layer_value[1],'_','k',k)]]
        add_nn_layer_vars_constraints!(model, weight[layer_key], bias[layer_key], in_vars, layer_key,false, k; L=bound_L, U=bound_U)
    end
    return model[Symbol('x','_',vcat(keys(model_architecture["output_layers"])...)[1],'_','k',k)]
end

###############################################################################################################################################################

function solved_and_feasible(model, ordered_model_architecture, I, instance_input, x_input, uncern_input, check_output, weight, bias,k,hidden_layer_bound)
    #### the input from the dataset is already scaled, use it in the model as it is

    model[Symbol("X")] = @variable(model, [1:I], base_name = "X")
    model[Symbol("instance")] = @variable(model, [1:length(instance_input)], base_name = "instance")
    model[Symbol("U",k)] = @variable(model, [1:I], base_name = "U$(k)")
    
    fix.(model[Symbol("instance")], instance_input; force = true)
    fix.(model[Symbol("X")], x_input; force = true)
    fix.(model[Symbol("U",k)], uncern_input; force = true)

    
    NN_output = NN_MILP!(model, ordered_model_architecture, weight, bias; k=k, bound_L=-hidden_layer_bound, bound_U=hidden_layer_bound)
    optimize!(model)
    ############################################################################################################################################

    println("The model is solved: ", is_solved_and_feasible(model))
    println("The NN and MILP solutions are approximately same: ",isapprox(value.(NN_output[1]), check_output; atol=1e-2))
    println("The NN output is: ", check_output, "   Unscaled: ", scale_inverse(check_output, target_min, target_max))
    println("The MILP output is: ", value.(NN_output[1]), "   Unscaled: ", scale_inverse(value.(NN_output[1]), target_min, target_max))
    if (is_solved_and_feasible(model)) && (isapprox(value.(NN_output)[1], check_output; atol=1e-2))
            println("The model is solved and the output is correct")
        else
            println("The model is not solved or the output is incorrect")
    end
end


###############################################################################################################################################################################################################################
function layer_alg(matrix, bias, input)
    if length(input) == size(matrix, 2)
        input = reshape(input, (1, length(input)))
    end
    result = matrix * transpose(input) .+ bias
    result_relu = max.(0, result)  # Apply ReLU operation
    return result_relu
end


"""
    Inputs:
    x_input: The input vector obtained from the Master problem
    uncern_mat: The uncertainity matrix obtained as a result of NN training, all uncertainities are scaled
    weight_dict: The dictionary containing the weights of the neural network
    bias_dict: The dictionary containing the biases of the neural network
    C: The capacity of the knapsack.....
TBW
"""
function forward_pass_old(x_input_MP, uncern_matrix, weight ,bias , f, p_bar, t, p_hat, C,  w, gamma, inst_min, inst_max, target_min, target_max, uncern_min, uncern_max)
    
    ## uncern inputs are taken from df where it is already scaled_instance_vector
    emd_un=layer_alg(weight["embedding_uncern"], bias["embedding_uncern"], uncern_matrix)
    ## The instance are created 
    scaled_instance_vector = scale_vector(vcat(f, p_bar, t, p_hat, C, w, gamma), inst_min, inst_max)
    emd_ins=layer_alg(weight["embedding_instance"], bias["embedding_instance"], scaled_instance_vector)
    ## The X is the same for all uncertainities
    emd_X=layer_alg(weight["embedding_X"], bias["embedding_X"], x_input_MP)
    ## Concatenate the outputs of the embedding layers
    out = transpose(vcat(repeat(vcat(emd_ins, emd_X),1, size(uncern_matrix)[1]), emd_un))

    # Loop through each layer defined in the weights dictionary
    for key in sort(collect(keys(weight)))  # Sort keys to ensure layers are applied in order
        if occursin("fc", key)  # Check if the key contains the substring "fc"
            weight_matrix = weight[key]
            bias_vector = bias[key]
            
            # Apply the layer operation: out = transpose(layer_alg(weight_matrix, bias_vector, out))
            out = transpose(layer_alg(weight_matrix, bias_vector, out))
        end
    end

    out = vcat(out...)

    #### out is scaled, unscale it to get the actual value
    out = scale_inverse(out, target_min, target_max)

    ##### Check if argmax or argmin is to be used
    index_out = argmax(out)
    ## Sending unscaled vectors and objective to MP
    ############################################################################################################################################
    uncern_to_MP = scale_inverse(uncern_matrix[index_out,:], uncern_min, uncern_max)                      ## This is the uncertainity vector that will be sent to MP
    
    objective_value = out[index_out]
    ############################################################################################################################################
    result_dict = Dict("index" => index_out, "selected_uncertainty" => uncern_to_MP, "max_objective_value" => objective_value)
    return result_dict
end


function layer_alg_new(matrix, bias, input; relu = true)
    if length(input) == size(matrix, 2)
        input = reshape(input, (1, length(input)))
    end
    result = matrix * transpose(input) .+ bias
    if relu
        result_relu = max.(0, result)  # Apply ReLU operation
        return result_relu
    else
        return result
    end
end


"""
    Inputs:
    x_input: The input vector obtained from the Master problem
    uncern_mat: The uncertainity matrix obtained as a result of NN training, all uncertainities are scaled
    weight_dict: The dictionary containing the weights of the neural network
    bias_dict: The dictionary containing the biases of the neural network
    C: The capacity of the knapsack.....
TBW
"""
function forward_pass_new(x_input_MP, unscaled_uncern_matrix, weight ,bias , f, p_bar, t, p_hat, C,  w, gamma, inst_min, inst_max, target_min, target_max, uncern_min, uncern_max; embedding_relu = true)
    
    scaled_uncern_matrix = scale_vector(unscaled_uncern_matrix', uncern_min, uncern_max)'
    emd_un=layer_alg_new(weight["embedding_uncern"], bias["embedding_uncern"], scaled_uncern_matrix; relu = embedding_relu)

    ## The instance are created 
    scaled_instance_vector = scale_vector(vcat(f, p_bar, t, p_hat, C, w, gamma), inst_min, inst_max)
    emd_ins=layer_alg_new(weight["embedding_instance"], bias["embedding_instance"], scaled_instance_vector; relu = embedding_relu)

    ## The X is the same for all uncertainities
    emd_X=layer_alg_new(weight["embedding_X"], bias["embedding_X"], x_input_MP; relu = embedding_relu)

    ## Concatenate the outputs of the embedding layers
    out = transpose(vcat(repeat(vcat(emd_ins, emd_X),1, size(scaled_uncern_matrix)[1]), emd_un))

    # Loop through each layer defined in the weights dictionary
    for key in sort(collect(keys(weight)))  # Sort keys to ensure layers are applied in order
        if occursin("fc", key)  # Check if the key contains the substring "fc"
            weight_matrix = weight[key]
            bias_vector = bias[key]
            
            # Apply the layer operation: out = transpose(layer_alg(weight_matrix, bias_vector, out))
            out = transpose(layer_alg_new(weight_matrix, bias_vector, out))       ### relu always true
        end
    end

    out = vcat(out...)

    #### out is scaled, unscale it to get the actual value
    out = scale_inverse(out, target_min, target_max)

    ##### Check if argmax or argmin is to be used
    index_out = argmax(out)
    ## Sending unscaled vectors and objective to MP
    ############################################################################################################################################
    #uncern_to_MP = scale_inverse(uncern_matrix[index_out,:], uncern_min, uncern_max)                      ## This is the uncertainity vector that will be sent to MP
    uncern_to_MP = unscaled_uncern_matrix[index_out,:]
    objective_value = out[index_out]
    ############################################################################################################################################
    result_dict = Dict("index" => index_out, "selected_uncertainty" => uncern_to_MP, "max_objective_value" => objective_value, "all_forward_pass_objective_values" => out)
    return result_dict
end
###############################################################################################################################################################################################################################



true