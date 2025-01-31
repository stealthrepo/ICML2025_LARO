using JuMP
using Gurobi
using DataStructures

# Define utility functions
function norm_1(v1, v2)
    return sum(abs.(v1 - v2))
end

function scale_vector(x, x_min, x_max)
    return (x .- x_min) ./ (x_max .- x_min)
end

function scale_inverse(x, x_min, x_max)
    return x .* (x_max .- x_min) .+ x_min
end

# Define the main function

function relaxed_master_stage_1_iterative(problem_args::Dict{String, Any})
    model = problem_args["model"]
    I = problem_args["I"]
    Master_uncern_set = problem_args["Master_uncern_set"]
    f = problem_args["f"]
    p_bar = problem_args["p_bar"]
    t = problem_args["t"]
    p_hat = problem_args["p_hat"]
    C = problem_args["C"]
    w = problem_args["w"]
    gamma = problem_args["gamma"]
    z_integral = problem_args["z_integral"]
    weight = problem_args["weight"]
    bias = problem_args["bias"]
    inst_min = problem_args["instance_min"]
    inst_max = problem_args["instance_max"]
    target_min = problem_args["target_min"]
    target_max = problem_args["target_max"]
    uncern_min = problem_args["uncertainty_min"]
    uncern_max = problem_args["uncertainty_max"]
    embedding_relu = problem_args["embedding_relu"]

    model[Symbol("X")] = @variable(model, [1:I], base_name = "X", lower_bound=0, upper_bound=1, Bin)

    @variable(model, r[1:I], Bin)
    @variable(model, y[1:I], Bin)

    @variable(model, z[1:length(Master_uncern_set)], Bin, lower_bound=0, upper_bound=1)
    #### selected uncertainty vector 
    @variable(model, ua[1:I])

    #### Problem constraints
    @constraint(model, sum(z) == 1)
    @constraint(model, sum(w[i]*y[i] + t[i]*r[i] for i in 1:I) <= C)
    @constraint(model, [i=1:I], y[i] <= model[Symbol("X")][i])
    @constraint(model, [i=1:I], r[i] <= y[i])

    #### Objective function
    @objective(model, Min, (sum((f[i]-p_bar[i]) * model[Symbol("X")][i] for i in 1:I) + sum((p_hat[i] * ua[i] - f[i]) * y[i] - p_hat[i] * ua[i] * r[i] for i in 1:I)))

    #### Uncertainty matrix
    A = hcat(collect(values(sort((Master_uncern_set))))...)
    
    #### Calculate row-wise maximum and minimum
    max_values = mapslices(maximum, A, dims=2)
    min_values = mapslices(minimum, A, dims=2)
    max_u = vec(max_values)
    min_u = vec(min_values)
    
    @constraint(model, min_u .<= ua .<= max_u)
    
    #### Constraint to select the uncertainty vector
    @constraint(model, A * z .== ua)
    return A'
end