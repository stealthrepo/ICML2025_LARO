### Code to embed NN MILP in master problem
include("nn_to_MILP.jl")

## Create an unscaled instance of the problem => instance = ['f', 'p_bar', 't', 'p_hat', 'C', 'w', 'gamma']
## Initialize the 2RO problem with a random instance
## Create the uncertainity matrix using the trained neural network data, maintain unique set of uncertainity vectors in the matrix


## MP should always be provided an unscaled instance and uncertainity vector, scaling should be done inside the function



"""
    Master_iteration(model, I, Master_uncern_set, weight_dict, bias_dict, f, p_bar, t, p_hat, C, w, gamma, model_architecture, hidden_layer_bound, uncern_min, uncern_max, target_min, target_max, inst_min, inst_max)

Solve the master problem in a robust optimization framework using MILP embedding of neural network.

# Arguments
- `model`: JuMP model object.
- `I`: Number of items.
- `Master_uncern_set`: Dictionary of uncertainty sets.
- `weight_dict`: Dictionary of neural network weights.
- `bias_dict`: Dictionary of neural network biases.
- `f`: Vector of fixed costs.
- `p_bar`: Vector of average profits.
- `t`: Vector of transaction costs.
- `p_hat`: Vector of maximum profits.
- `C`: Capacity constraint.
- `w`: Vector of weights.
- `gamma`: Vector of risk parameters.
- `model_architecture`: Neural network architecture.
- `hidden_layer_bound`: Bound for hidden layer outputs.
- `uncern_min`: Minimum uncertainty value for scaling.
- `uncern_max`: Maximum uncertainty value for scaling.
- `target_min`: Minimum target value for scaling.
- `target_max`: Maximum target value for scaling.
- `inst_min`: Minimum instance value for scaling.
- `inst_max`: Maximum instance value for scaling.

# Returns
- `Master_result`: OrderedDict containing the results of the optimization, including objective value, selected uncertainties, and decision variables.
"""

function Master_iteration(model, I, Master_uncern_set, weight_dict, bias_dict, f, p_bar, t, p_hat, C, w, gamma, model_architecture, hidden_layer_bound, uncern_min, uncern_max, target_min, target_max, inst_min, inst_max)

    model[Symbol("X")] = @variable(model, [1:I], base_name = "X", lower_bound=0, upper_bound=1, Bin)

    @variable(model, r[1:I],lower_bound=0, upper_bound=1,Bin)
    @variable(model, y[1:I],lower_bound=0, upper_bound=1,Bin)

    ###################### This instance vector is to be fixed for all uncertainities and will be used as input to the neural network ################
    instance_input = vcat(f, p_bar, t, p_hat, C, w, gamma...)

    model[Symbol("instance")] = @variable(model, [1:length(instance_input)], base_name = "instance")                        
    fix.(model[Symbol("instance")], scale_vector(instance_input, inst_min, inst_max); force = true)              ## scale and fix the instance vector input of Neural Network constraint MIP

    @variable(model, l[1:length(Master_uncern_set)])                                  ## Output of the neural network after inverse scaling
    @variable(model, z[1:length(Master_uncern_set)], lower_bound=0, upper_bound=1, Bin)                             ## Binary variable to select the uncertainity vector
    @variable(model, ua[1:I])

    u_L = -1E10
    u_M = 0
    @variable(model, u_L <= u <= u_M)

    @constraint(model, sum(z) == 1)
    @constraint(model, sum(w[i]*y[i] + t[i]*r[i] for i in 1:I) <= C)
    @constraint(model, [i=1:I], y[i] <= model[Symbol("X")][i])
    @constraint(model, [i=1:I], r[i] <= y[i])

    @objective(model, Min, (sum((f[i]-p_bar[i]) * model[Symbol("X")][i] for i in 1:I) + sum((p_hat[i] * ua[i] - f[i]) * y[i] - p_hat[i] * ua[i] * r[i] for i in 1:I)))

    A=hcat(values(sort(Master_uncern_set))...) 

    @constraint(model, A * z .== ua)
    
    for (key,value) in Master_uncern_set

        model[Symbol("U", key)] = @variable(model, [1:I], base_name = "U$(key)")

        fix.(model[Symbol("U", key)], scale_vector(value, uncern_min, uncern_max); force = true)     ## scale and fix the uncertainity vector input of Neural Network constraint MIP
        
        @constraint(model, l[key] == scale_inverse(NN_MILP!(model, model_architecture, weight_dict, bias_dict, key; bound_L = -hidden_layer_bound, bound_U = hidden_layer_bound), target_min, target_max)[1])
        @constraint(model, u >= l[key])
        @constraint(model, u <= l[key] + (upper_bound(u) - lower_bound(u)) * (1 - z[key]))

    end

    optimize!(model)

    #println("Solution Exists: ", is_solved_and_feasible(model))

    Master_result = OrderedDict()
    Master_result["objective_value"] = objective_value(model)
    Master_result["Master_Uncertainties"] = Master_uncern_set
    Master_result["Worst_Case_Uncertainity (u)"] = value(u)
    Master_result["uncertainity_vector"] = value.(model[:ua])
    Master_result["z"] = value.(model[:z])
    Master_result["uncertainity_index"] = argmax(value.(model[:z]))
    Master_result["x"] = abs.(value.(model[Symbol("X")]))
    Master_result["r"] = abs.(value.(model[:r]))
    Master_result["y"] = abs.(value.(model[:y]))
    Master_result["check_obj"] = objective_value(model) - sum((f[i]-p_bar[i]) * Master_result["x"][i] for i in 1:I) 
    Master_result["value of contraint uncertainties"] = value.(model[:l])
    
    return Master_result
end




