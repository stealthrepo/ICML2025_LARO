
"""
    master_fixed_x_uncertainty(model_env, x_fixed, uncertainty, num_items, f, p_bar, t, p_hat, C, w)

Solve the optimization problem with fixed x values and given uncertainty.

# Arguments
- `model_env::Gurobi.Env`: The Gurobi environment.
- `x_fixed::Vector{Bool}`: Fixed vector of x values.
- `uncertainty::Vector{Float64}`: Vector of uncertainties.
- `num_items::Int`: Number of items.
- `f::Vector{Float64}`: Fixed cost of items.
- `p_bar::Vector{Float64}`: Vector of p_bar values.
- `t::Vector{Float64}`: Vector of t values.
- `p_hat::Vector{Float64}`: Vector of p_hat values.
- `C::Float64`: Capacity constraint.
- `w::Vector{Float64}`: Vector of weights.

# Returns
- `Float64`: The objective value of the optimization problem.
"""
function master_fixed_x_uncertainty(model_env, x_fixed, uncertainty, num_items, f, p_bar, t, p_hat, C, w)

    model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(model_env), "OutputFlag" => 0))

    @variable(model, r[1:num_items], Bin)
    @variable(model, y[1:num_items], Bin)

    @constraint(model, sum(w[j]*y[j] + t[j]*r[j] for j in 1:num_items) <= C)
    @constraint(model, [j=1:num_items], y[j] <= x_fixed[j])
    @constraint(model, [j=1:num_items], r[j] <= y[j])

    obj_component_1 = sum((f[j]-p_bar[j]) * x_fixed[j] for j in 1:num_items)
    obj_component_2 = sum((p_hat[j] * uncertainty[j] - f[j]) * y[j] - p_hat[j] * uncertainty[j] * r[j] for j in 1:num_items)

    @objective(model, Min, obj_component_1 + obj_component_2)

    optimize!(model)

    return objective_value(model)
end
#############################################################################################################################################################################################################################################


# function forward_pass_decision_tree(instance, scenario_matrix, top_k_rank, model)
#     instance_matrix = Matrix(hcat(instance))'
#     instance_broadcasted = repeat(instance_matrix, size(scenario_matrix, 1), 1)

#     input_matrix = hcat(instance_broadcasted, scenario_matrix)
    
#     objective = DecisionTree.predict(model, input_matrix)
#     max_objective = maximum(objective)
#     max_objective_index = argmax(objective)
    
#     return Dict("index" => max_objective_index, "selected_uncertainty" => scenario_matrix[max_objective_index,:], "max_objective_value" => max_objective)
# end


"""
    forward_pass_decision_tree(instance, scenario_matrix, top_k_rank, model)

Perform a forward pass through the decision tree model to select the top-k uncertainties.

# Arguments
- `instance::Vector{Float64}`: The instance vector.
- `scenario_matrix::Union{Matrix{Float64}, Dict}`: The scenario matrix or dictionary of scenarios.
- `top_k_rank::Int`: The number of top uncertainties to select.
- `model`: The decision tree model.

# Returns
- `Dict`: A dictionary containing the indices of the top-k uncertainties, the top-k uncertainties, and the top-k objective values.
"""
function forward_pass_decision_tree(instance, scenario_matrix, top_k_rank, model)

    if isa(scenario_matrix,Dict)
        scenario_matrix = hcat(collect(values(sort(scenario_matrix)))...)'   ## Convert the dictionary to a matrix
    end

    instance_matrix = Matrix(hcat(instance))'
    instance_broadcasted = repeat(instance_matrix, size(scenario_matrix, 1), 1)

    input_matrix = hcat(instance_broadcasted, scenario_matrix)

    objective = DecisionTree.predict(model, input_matrix)

    # Sort the objective values in descending order and get the top-k indices
    sorted_indices = sortperm(objective, rev=true)[1:top_k_rank]

    # Get the top-k uncertainties from scenario_matrix
    top_k_uncertainties = scenario_matrix[sorted_indices, :]

    # Get the top-k objective values
    top_k_objectives = objective[sorted_indices]

    return Dict("indices" => sorted_indices, "top_k_uncertainties" => top_k_uncertainties, "top_k_objectives" => top_k_objectives)
end


#############################################################################################################################################################################################################################################

"""
    master_stage_exact(model_exact_env, num_items, uncertainty_dict, f, p_bar, t, p_hat, C, w)

Solve the exact master problem with given uncertainties.

# Arguments
- `model_exact_env::Gurobi.Env`: The Gurobi environment.
- `num_items::Int`: Number of items.
- `uncertainty_dict::Dict`: Dictionary of uncertainties.
- `f::Vector{Float64}`: Fixed cost of items.
- `p_bar::Vector{Float64}`: Vector of p_bar values.
- `t::Vector{Float64}`: Vector of t values.
- `p_hat::Vector{Float64}`: Vector of p_hat values.
- `C::Float64`: Capacity constraint.
- `w::Vector{Float64}`: Vector of weights.

# Returns
- `OrderedDict`: A dictionary containing the results of the optimization.
"""
function master_stage_exact(model_exact_env, num_items, uncertainty_dict, f, p_bar, t, p_hat, C, w)

    model_exact = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(model_exact_env)))
    set_optimizer_attribute(model_exact, "OutputFlag", 0)

    #set_optimizer_attribute(model, "TimeLimit", 100)

    @variable(model_exact, X[1:num_items], Bin, lower_bound=0, upper_bound=1)
    @variable(model_exact, mu)

    num_master_uncertainty = length(uncertainty_dict)

    for i in 1:num_master_uncertainty

        uncertainty = uncertainty_dict[i]

        model_exact[Symbol("r", i)] = @variable(model_exact, [1:num_items], base_name = string("r",i), lower_bound=0, upper_bound=1, Bin)
        model_exact[Symbol("y", i)] = @variable(model_exact, [1:num_items], base_name = string("y",i), lower_bound=0, upper_bound=1, Bin)

        obj_component_1 = sum((f[j]-p_bar[j]) * X[j] for j in 1:num_items)
        obj_component_2 = sum((p_hat[j] * uncertainty[j] - f[j]) * model_exact[Symbol("y",i)][j] - p_hat[j] * uncertainty[j] * model_exact[Symbol("r",i)][j] for j in 1:num_items)

        model_exact[Symbol("c", i)] = @constraint(model_exact, obj_component_1 + obj_component_2 <= mu)

        @constraint(model_exact, sum(w[j]*model_exact[Symbol("y",i)][j] + t[j]*model_exact[Symbol("r",i)][j] for j in 1:num_items) <= C)

        @constraint(model_exact, [j=1:num_items], model_exact[Symbol("y",i)][j] <= X[j])
        @constraint(model_exact, [j=1:num_items], model_exact[Symbol("r",i)][j] <= model_exact[Symbol("y",i)][j])

        @objective(model_exact, Min, mu)
    end

    optimize!(model_exact)

    uncertainty_constraint_slack = Float64[]

    for i in 1:num_master_uncertainty
        uncertainty_constraint_slack = append!(uncertainty_constraint_slack, normalized_rhs(model_exact[Symbol("c", i)])-JuMP.value(model_exact[Symbol("c", i)]))
    end
        
    results = OrderedDict{Any, Any}()
    results["objective_value"] = objective_value(model_exact)
    results["X"] = value.(model_exact[:X])
    results["uncertainty_set"] = uncertainty_dict
    results["uncertainty_constraint_slack"] = uncertainty_constraint_slack

    selected_uncertainty_index = argmin(uncertainty_constraint_slack)
    results["selected_uncertainty"] = uncertainty_dict[selected_uncertainty_index]
    return results
end

#############################################################################################################################################################################################################################################

"""
Solves the second stage problem exactly for each uncertainty and returns the uncertainty that gives the maximum objective value
    max_enumeration_min_exact(model_env, x_fixed, uncertainty_matrix, num_items, f, p_bar, t, p_hat, C, w)
    model_env: Gurobi environment
    x_fixed: fixed vector of x values
    uncertainty_matrix: matrix of uncertainities, number of rows = number of uncertainities, number of columns = number of items
    num_items: number of items
    f: fixed cost of items

"""
function max_enumeration_min_exact(model_env, x_fixed, uncertainty_dict, num_items, f, p_bar, t, p_hat, C, w)

    num_uncertainties = length(uncertainty_dict)

    store_objective_values = []
    for uncertainty_index in 1:num_uncertainties

        uncertainty = uncertainty_dict[uncertainty_index]

        ss_model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(model_env)))
        set_optimizer_attribute(ss_model, "OutputFlag", 0)
        #set_optimizer_attribute(model, "TimeLimit", 600)

        @variable(ss_model, y[1:num_items], Bin, lower_bound=0, upper_bound=1)
        @variable(ss_model, r[1:num_items], Bin, lower_bound=0, upper_bound=1)

        @constraint(ss_model, sum(w[j]*y[j] + t[j]*r[j] for j in 1:num_items) <= C)
        @constraint(ss_model, [j=1:num_items], y[j] <= x_fixed[j])
        @constraint(ss_model, [j=1:num_items], r[j] <= y[j])

        obj_component_1 = sum((f[j]-p_bar[j]) * x_fixed[j] for j in 1:num_items)
        obj_component_2 = sum((p_hat[j] * uncertainty[j] - f[j]) * y[j] - p_hat[j] * uncertainty[j] * r[j] for j in 1:num_items)

        @objective(ss_model, Min, obj_component_1 + obj_component_2)

        optimize!(ss_model)
        #println("Objective value: ", objective_value(model))
        push!(store_objective_values, objective_value(ss_model))
    end

    max_objective_value_index = argmax(store_objective_values)
    second_stage_result = OrderedDict()
    second_stage_result["objective_value_master_second_stage"] = store_objective_values[max_objective_value_index]
    second_stage_result["selected_uncertainty"] = uncertainty_dict[max_objective_value_index]
    second_stage_result["all_objective_values"] = store_objective_values
    return second_stage_result
end

#############################################################################################################################################################################################################################################


"""
    master_relaxed(master_ms1_env, uncertainty_set, num_items, f, p_bar, t, p_hat, C, w)

Solve the relaxed master problem (Phase 1) with given uncertainties.

# Arguments
- `master_ms1_env::Gurobi.Env`: The Gurobi environment.
- `uncertainty_set::Dict`: Dictionary of uncertainties.
- `num_items::Int`: Number of items.
- `f::Vector{Float64}`: Fixed cost of items.
- `p_bar::Vector{Float64}`: Vector of p_bar values.
- `t::Vector{Float64}`: Vector of t values.
- `p_hat::Vector{Float64}`: Vector of p_hat values.
- `C::Float64`: Capacity constraint.
- `w::Vector{Float64}`: Vector of weights.

# Returns
- `Model`: The relaxed master problem model.
"""
function master_relaxed(master_ms1_env, uncertainty_set, num_items, f, p_bar, t, p_hat, C, w)

    num_uncertainties = length(uncertainty_set)

    model_ms1 = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(master_ms1_env)))
    set_optimizer_attribute(model_ms1, "OutputFlag", 0)

    model_ms1[Symbol("X")] = @variable(model_ms1, [1:num_items], base_name = "X", lower_bound=0, upper_bound=1, Bin)

    @variable(model_ms1, r[1:num_items], Bin)
    @variable(model_ms1, y[1:num_items], Bin)

    @variable(model_ms1, z[1:num_uncertainties], Bin, lower_bound=0, upper_bound=1)
    #### selected uncertainty vector 
    @variable(model_ms1, ua[1:num_items])

    @constraint(model_ms1, sum(z)==1)
    @constraint(model_ms1, sum(w[i]*y[i] + t[i]*r[i] for i in 1:num_items) <= C)
    @constraint(model_ms1, [i=1:num_items], y[i] <= model_ms1[Symbol("X")][i])
    @constraint(model_ms1, [i=1:num_items], r[i] <= y[i])

    uncertainty_matrix = hcat(collect(values(sort((uncertainty_set))))...)
    #### Calculate row-wise maximum and minimum
    max_values = mapslices(maximum, uncertainty_matrix, dims=2)
    min_values = mapslices(minimum, uncertainty_matrix, dims=2)
    max_u = vec(max_values)
    min_u = vec(min_values)
    
    @constraint(model_ms1, min_u .<= ua .<= max_u)
    
    @constraint(model_ms1, uncertainty_matrix * z .== ua)

    obj_comp_1 = sum((f[i]-p_bar[i]) * model_ms1[Symbol("X")][i] for i in 1:num_items)
    obj_comp_2 = sum((p_hat[i] * ua[i] - f[i]) * y[i] - p_hat[i] * ua[i] * r[i] for i in 1:num_items)
    @objective(model_ms1, Min, obj_comp_1 + obj_comp_2)

    return model_ms1
end

#############################################################################################################################################################################################################################################

"""
    master_decomposition_exact(model_env_ms1, model_env_ms2, num_items, f, p_bar, t, p_hat, C, w, master_uncertainty_set)

Solve the master decomposition problem (Relaxed Master Problem and Phase 2 exactly) with given uncertainties.

# Arguments
- `model_env_ms1::Gurobi.Env`: The Gurobi environment for the first stage.
- `model_env_ms2::Gurobi.Env`: The Gurobi environment for the second stage.
- `num_items::Int`: Number of items.
- `f::Vector{Float64}`: Fixed cost of items.
- `p_bar::Vector{Float64}`: Vector of p_bar values.
- `t::Vector{Float64}`: Vector of t values.
- `p_hat::Vector{Float64}`: Vector of p_hat values.
- `C::Float64`: Capacity constraint.
- `w::Vector{Float64}`: Vector of weights.
- `master_uncertainty_set::Dict`: Dictionary of uncertainties.

# Returns
- `Vector{OrderedDict}`: A list of dictionaries containing the results of each iteration.
"""
function master_decomposition_exact(model_env_ms1, model_env_ms2, num_items, f, p_bar, t, p_hat, C, w, master_uncertainty_set)

    num_master_uncertainty = length(master_uncertainty_set)

    master_model_first_stage = master_relaxed(model_env_ms1, master_uncertainty_set, num_items, f, p_bar, t, p_hat, C, w)

    master_result_list = []
    

    for iter_num in 1:num_master_uncertainty

        result_master_dict = OrderedDict()
        optimize!(master_model_first_stage)

        index = findall(val -> isapprox(val, 1.0; atol=1E-5), value.(master_model_first_stage[:z]))   ## Find the index of the selected uncertainty
        @assert(length(index) == 1, "TU of z not satisfied => Multiple or no indices selected")
        @assert(isapprox(sum(value.(master_model_first_stage[:z])),1, atol=1E-5), "TU of z not satisfied => Sum of z not equal to 1")
        for i in length(master_model_first_stage[:z])
            @assert(isapprox(value.(master_model_first_stage[:z])[i],0,atol=1E-5) || isapprox(value.(master_model_first_stage[:z])[i],1,atol=1E-5), "TU of z not satisfied => z not binary")
        end

        x_relaxed_master = abs.(value.(master_model_first_stage[:X]))
        ua_uncertainty_relaxed_master = value.(master_model_first_stage[:ua])
        z_uncertainty_list = value.(master_model_first_stage[:z])

        result_master_dict["iteration"] = iter_num
        result_master_dict["x_relaxed_master"] = x_relaxed_master
        result_master_dict["ua_relaxed_master"] = ua_uncertainty_relaxed_master
        result_master_dict["z_list_relaxed_master"] = z_uncertainty_list
        result_master_dict["objective_value_relaxed_master"] = objective_value(master_model_first_stage)

        master_second_stage_objectives = max_enumeration_min_exact(model_env_ms2, x_relaxed_master, master_uncertainty_set, num_items, f, p_bar, t, p_hat, C, w)

        result_master_dict["objective_value_master_second_stage"] = master_second_stage_objectives["objective_value_master_second_stage"]
        result_master_dict["uncertainty_second_stage"] = master_second_stage_objectives["selected_uncertainty"]
        result_master_dict["all_objective_values"] = master_second_stage_objectives["all_objective_values"]

        push!(master_result_list, result_master_dict)

        if isapprox(sum(abs.(result_master_dict["ua_relaxed_master"] - result_master_dict["uncertainty_second_stage"])), 0; atol=1e-4)
            # println("The selected uncertainty in the first stage is the same as the selected uncertainty in the second stage")
            break
        end

        @constraint(master_model_first_stage, master_model_first_stage[:z][index[1]] == 0)
    end
    return master_result_list
end

#############################################################################################################################################################################################################################################

"""
    master_decompose_NN(model_env_ms1, num_items, master_uncertainty_set, f, p_bar, t, p_hat, C, w, gamma, weight, bias, instance_min, instance_max, target_min, target_max, uncertainty_min, uncertainty_max)

Solve the master decomposition problem (RMP Phase 1 and NN approximation for Phase 2).

# Arguments
- `model_env_ms1::Gurobi.Env`: The Gurobi environment for the first stage.
- `num_items::Int`: Number of items.
- `master_uncertainty_set::Dict`: Dictionary of uncertainties.
- `f::Vector{Float64}`: Fixed cost of items.
- `p_bar::Vector{Float64}`: Vector of p_bar values.
- `t::Vector{Float64}`: Vector of t values.
- `p_hat::Vector{Float64}`: Vector of p_hat values.
- `C::Float64`: Capacity constraint.
- `w::Vector{Float64}`: Vector of weights.
- `gamma::Vector{Float64}`: Vector of gamma values.
- `weight::Matrix{Float64}`: Weight matrix for the neural network.
- `bias::Vector{Float64}`: Bias vector for the neural network.
- `instance_min::Vector{Float64}`: Minimum values for instance normalization.
- `instance_max::Vector{Float64}`: Maximum values for instance normalization.
- `target_min::Vector{Float64}`: Minimum values for target normalization.
- `target_max::Vector{Float64}`: Maximum values for target normalization.
- `uncertainty_min::Vector{Float64}`: Minimum values for uncertainty normalization.
- `uncertainty_max::Vector{Float64}`: Maximum values for uncertainty normalization.

# Returns
- `Vector{OrderedDict}`: A list of dictionaries containing the results of each iteration.
"""
function master_decompose_NN(model_env_ms1, num_items, master_uncertainty_set, f, p_bar, t, p_hat, C, w, gamma, weight, bias, instance_min, instance_max, target_min, target_max, uncertainty_min, uncertainty_max)

    num_master_uncertainty = length(master_uncertainty_set)
    master_uncertainty_matrix = hcat(collect(values(sort((master_uncertainty_set))))...)'   ## Convert the dictionary to a matrix
    master_model_first_stage = master_relaxed(model_env_ms1, master_uncertainty_set, num_items, f, p_bar, t, p_hat, C, w)

    master_result_list = []
    
    for iter_num in 1:num_master_uncertainty

        result_master_dict = OrderedDict()
        optimize!(master_model_first_stage)

        index = findall(val -> isapprox(val, 1.0; atol=1E-5), value.(master_model_first_stage[:z]))   ## Find the index of the selected uncertainty
        @assert(length(index) == 1, "TU of z not satisfied => Multiple or no indices selected")
        @assert(isapprox(sum(value.(master_model_first_stage[:z])),1, atol=1E-5), "TU of z not satisfied => Sum of z not equal to 1")
        for i in length(master_model_first_stage[:z])
            @assert(isapprox(value.(master_model_first_stage[:z])[i],0,atol=1E-5) || isapprox(value.(master_model_first_stage[:z])[i],1,atol=1E-5), "TU of z not satisfied => z not binary")
        end

        x_relaxed_master = abs.(value.(master_model_first_stage[:X]))
        ua_uncertainty_relaxed_master = value.(master_model_first_stage[:ua])
        z_uncertainty_list = value.(master_model_first_stage[:z])

        result_master_dict["iteration"] = iter_num
        result_master_dict["x_relaxed_master"] = x_relaxed_master
        result_master_dict["ua_relaxed_master"] = ua_uncertainty_relaxed_master
        result_master_dict["z_list_relaxed_master"] = z_uncertainty_list
        result_master_dict["objective_value_relaxed_master"] = objective_value(master_model_first_stage)

        master_second_stage_objectives = forward_pass_new(x_relaxed_master, master_uncertainty_matrix, weight ,bias , f, p_bar, t, p_hat, C,  w, gamma, instance_min, instance_max, target_min, target_max, uncertainty_min, uncertainty_max; embedding_relu = true)

        result_master_dict["objective_value_master_second_stage"] = master_second_stage_objectives["max_objective_value"]
        result_master_dict["uncertainty_second_stage"] = master_second_stage_objectives["selected_uncertainty"]
        result_master_dict["all_objective_values"] = master_second_stage_objectives["all_forward_pass_objective_values"]

        push!(master_result_list, result_master_dict)
        if isapprox(sum(abs.(result_master_dict["ua_relaxed_master"] - result_master_dict["uncertainty_second_stage"])), 0; atol=1e-4)
            #println("The selected uncertainty in the first stage is the same as the selected uncertainty in the second stage at iteration: ", iter_num)
            break
        end

        @constraint(master_model_first_stage, master_model_first_stage[:z][index[1]] == 0)
    end

    return master_result_list

end


#############################################################################################################################################################################################################################################
"""
Machine Learning Accelerated CCG with RMP Phase 1, NN approximated Phase 2 and NN approximated Adversarial Stage
"""

function fullRO_master_decomposition_NN_adv_NN(model_master_first_stage_env, model_adv_stage_env, model_master_exact_env,
                                               num_items, global_uncertainty_matrix, num_initial_master_uncertainty, seed, 
                                               f, p_bar, t, p_hat, C, w, gamma, weight, bias, 
                                               instance_min, instance_max, 
                                               target_min, target_max, 
                                               uncertainty_min, uncertainty_max; improved_master = true, ML_model_forward_master = nothing)
    solution_dict = OrderedDict()

    Random.seed!(seed)
    master_indices = rand(1:size(global_uncertainty_matrix, 1), num_master_initial_uncertainties)
    instance_vector = vcat(f, p_bar, t, p_hat, C, w, gamma...)
    num_ranked_uncertainty = 5
    master_uncertainty_set = OrderedDict()

    for (i, index) in enumerate(master_indices)
        master_uncertainty_set[i] = global_uncertainty_matrix[index, :]
    end

    lower_bound = -1E8
    upper_bound = 1E8

    #tol = 1E-3
    ## UB > LB + tol

    results_list = []

    current_iter_count = 0
    last_gap = Inf
    gap_improvement_threshold = 1E-4

    #uncertainty_last_iter = zeros(num_items)
    master_results_all_iters = []
    #master_exact_adv_uncern_results = OrderedDict()
    adv_selected_uncern_dict = OrderedDict()
    ML_selected_uncern_dict = OrderedDict()
    ML_ranked_obj_results = OrderedDict()

    while true

        results_dict = OrderedDict()

        #master_result_list = master_decompose_NN(model_master_first_stage_env, num_items, master_uncertainty_set ,f, p_bar, t, p_hat, C, w, gamma, weight, bias, instance_min, instance_max, target_min, target_max, uncertainty_min, uncertainty_max)
        
        master_result_list = master_decompose_NN(
            model_master_first_stage_env, 
            num_items, 
            master_uncertainty_set, 
            f, p_bar, t, p_hat, C, w, gamma, 
            weight, bias, 
            instance_min, instance_max, 
            target_min, target_max, 
            uncertainty_min, uncertainty_max
        )
        push!(master_results_all_iters, master_result_list)

    
        lower_bound = master_result_list[end]["objective_value_relaxed_master"]
        x_fixed = master_result_list[end]["x_relaxed_master"]
        uncertainty = master_result_list[end]["ua_relaxed_master"]

        #adv_result = forward_pass_new(x_fixed, global_uncertainty_matrix, weight, bias, f, p_bar, t, p_hat, C, w, gamma, instance_min, instance_max, target_min, target_max, uncertainty_min, uncertainty_max; embedding_relu = false)
        adv_result = forward_pass_new(
            x_fixed, global_uncertainty_matrix, 
            weight, bias, f, p_bar, t, p_hat, C, w, gamma, 
            instance_min, instance_max, 
            target_min, target_max, 
            uncertainty_min, uncertainty_max; 
            embedding_relu = false
        )

        uncertainty_adv = adv_result["selected_uncertainty"]

        #upper_bound = master_fixed_x_uncertainty(model_adv_stage_env, x_fixed, uncertainty_adv, num_items, f, p_bar, t, p_hat, C, w)
        upper_bound = master_fixed_x_uncertainty(
            model_adv_stage_env, x_fixed, uncertainty_adv, 
            num_items, f, p_bar, t, p_hat, C, w
        )

        gap = (upper_bound - lower_bound) / abs(lower_bound)
        println("Lower bound: ", lower_bound)
        println("Upper bound: ", upper_bound)
        #println("lowewr bound adv uncertainty: ", lower_bound_adv_uncertainty)
        println("Gap: ", gap)

        results_dict["master_objective"] = lower_bound
        results_dict["adv_objective"] = upper_bound
        results_dict["x_fixed"] = x_fixed
        results_dict["uncertainty_master"] = uncertainty
        results_dict["uncertainty_adv"] = uncertainty_adv
        current_iter_count += 1
        results_dict["total_iterations"] = current_iter_count
        #push!(results_list, results_dict)

        # Check if the upper bound is close to the lower bound
        if isapprox(upper_bound, lower_bound; atol=1E-5)
            println("Optimal solution found")
            results_dict["Improved_Master_Objective"] = upper_bound
            results_dict["Converged"] = 1
            push!(results_list, results_dict)
            break
        end


        adv_selected_uncern_dict[1] = uncertainty_adv

        # Check if the gap improvement is below a threshold
        if abs(last_gap - gap) < gap_improvement_threshold

            
            println("Convergence detected: gap improvement below threshold.")
            #println("check", master_exact_adv_uncern_results["objective_value"])
            
            master_uncertainty_matrix = hcat(collect(values(sort((master_uncertainty_set))))...)'   ## Convert the dictionary to a matrix
            
            if improved_master==true
                
                num_ranked_uncertainty = min(num_ranked_uncertainty, size(master_uncertainty_matrix, 1))

                ML_ranked_results = forward_pass_decision_tree(instance_vector, master_uncertainty_matrix, num_ranked_uncertainty, ML_model_forward_master)

                for (i, index) in enumerate(ML_ranked_results["indices"])
                    ML_selected_uncern_dict[1] = master_uncertainty_matrix[index, :]
                    master_exact_ranked_uncern_results = master_stage_exact(model_master_exact_env, num_items, ML_selected_uncern_dict, f, p_bar, t, p_hat, C, w)
                    ML_ranked_obj_results[index] = master_exact_ranked_uncern_results["objective_value"]
                    results_dict["ML_ranked_obj_results"] = ML_ranked_obj_results
                end

            end
            
            master_exact_adv_uncern_results = master_stage_exact(model_master_exact_env, num_items, adv_selected_uncern_dict, f, p_bar, t, p_hat, C, w)

            if improved_master==true
                results_dict["Improved_Master_Objective"] = max(maximum(values(ML_ranked_obj_results)), master_exact_adv_uncern_results["objective_value"], results_dict["master_objective"])
            else
                results_dict["Improved_Master_Objective"] = max(master_exact_adv_uncern_results["objective_value"], results_dict["master_objective"])
            end
            #results_dict["Improved_Master_Objective"] = max(maximum(values(results_dict["ML_ranked_obj_results"])), master_exact_adv_uncern_results["objective_value"], results_dict["master_objective"])
            
            results_dict["Converged"] = 0
            push!(results_list, results_dict)
            break
        end

        ## end if current iter count > 100
        if current_iter_count > 100
            println("Convergence didnt occur for iter=100: maximum iteration count reached.")
            results_dict["Improved_Master_Objective"] = upper_bound
            results_dict["Converged"] = 0
            push!(results_list, results_dict)
            break
        end

        last_gap = gap
        master_uncertainty_set[num_initial_master_uncertainty + current_iter_count] = uncertainty_adv
        
    end
    solution_dict["two_stage_iteration_list"] = results_list 
    #solution_dict["master_results_all_iters"] = master_results_all_iters
    solution_dict["final_master_uncertainty_set"] = master_uncertainty_set
    
    println("lower_bound: ", solution_dict["two_stage_iteration_list"][end]["master_objective"])
    println("upper_bound: ", solution_dict["two_stage_iteration_list"][end]["adv_objective"])
    
    return solution_dict
end

#############################################################################################################################################################################################################################################

