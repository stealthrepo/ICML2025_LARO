using Distributions
using Random
using DataFrames
using CSV
using DataStructures
using JuMP
using OrderedCollections
using LinearAlgebra


############################################################################################### 
################################## Auumptions ################################################

### The power generators should be ordered and a int value should be assigned to each generator
### The bus should be ordered and a int value should be assigned to each bus

########################### Data Constructors #################################################
###############################################################################################
function Power_Generator(generator_no::Int, 
                        start_up_cost::Real, shut_down_cost::Real,
                        constant_cost_coefficient::Real, linear_cost_coefficient::Real, #quadratic_cost_coefficient::Real,
                        Min_electricty_output_limit::Real, Max_electricty_output_limit::Real,
                        Min_up_time::Real, Min_down_time::Real,
                        Ramp_up_limit::Real, Ramp_down_limit::Real,
                        Start_up_ramp_rate_limit::Real, Shut_down_ramp_rate_limit::Real, 
                        bus_no::Int)

    return (generator_no = generator_no, 
            start_up_cost = start_up_cost, shut_down_cost = shut_down_cost,
            constant_cost_coefficient = constant_cost_coefficient, linear_cost_coefficient = linear_cost_coefficient, #quadratic_cost_coefficient = quadratic_cost_coefficient,
            Min_electricty_output_limit = Min_electricty_output_limit, Max_electricty_output_limit = Max_electricty_output_limit,
            Min_up_time = Min_up_time, Min_down_time = Min_down_time,
            Ramp_up_limit = Ramp_up_limit, Ramp_down_limit = Ramp_down_limit,
            Start_up_ramp_rate_limit = Start_up_ramp_rate_limit, Shut_down_ramp_rate_limit = Shut_down_ramp_rate_limit, 
            bus_no = bus_no)
end

### Power_Generator_Set takes input and reads the data from the dataframe or matrix and each row is a power generator
### The function returns a dictionary of power generators
### The number of power generators is equal to the number of rows in the dataframe or matrix
### The number of columns in the dataframe or matrix is equal to the number of parameters of the power generator

#### The input data must be a DataFrame or a Matrix
#### The columns of the input data must be in the following order:
#### start_up_cost, shut_down_cost, Min_up_time, Min_down_time, Min_electricty_output_limit, Max_electricty_output_limit,
#### Ramp_up_limit, Ramp_down_limit, Start_up_ramp_up_rate_limit, Shut_down_ramp_down_rate_limit
#### The number of columns must be equal to the number of parameters of the power generator


function Power_Generator_Set(data)
    n_gens = size(data, 1)
    power_generators = OrderedDict()

    for i in 1:n_gens
        # If the input is a DataFrame, use indexing with column indices
        if isa(data, DataFrame)
            #@assert i==data[i, 1], "The generator number in the data is not equal to the index of the generator"
            power_generators[data[i, 1]] = Power_Generator(data[i, 1], data[i, 2], data[i, 3], data[i, 4], data[i, 5],
                                                    data[i, 6], data[i, 7], data[i, 8], data[i, 9], data[i, 10],
                                                    data[i, 11], data[i, 12], data[i, 13], data[i, 14]) #, data[i, 15])
        # If the input is a Matrix, we need to use direct indexing
        # Problem: In case some parameters are missing, the function will throw an error
        # Solution: Put zeros in place of missing parameters, but always have 10 columns
        else
            throw(ArgumentError("Input data must be either a Matrix or a DataFrame."))
        end
    end
    return power_generators
end


function Group_Generators_by_Bus(generator_col, bus_col)
    # Create a temporary dictionary
    bus_dict = Dict{Int, Vector{Int}}()

    # Iterate over the paired generator_no and bus_no columns
    for (generator_no, bus_no) in zip(generator_col, bus_col)
        push!(get!(bus_dict, bus_no, []), generator_no)
    end

    # Create an OrderedDict sorted by keys
    return OrderedDict(sort(bus_dict))
end
###############################################################################################

function Edge_Properties(edge_no, susceptance::Real, min_capacity::Real, max_capacity::Real, node_tuple::Tuple)
    return (edge_no = edge_no, susceptance = susceptance, min_capacity = min_capacity, max_capacity = max_capacity, node_tuple = node_tuple)
end

function Edge_Properties_Set(data)
    n_edges = size(data, 1)
    edge_properties = OrderedDict()

    for i in 1:n_edges
        if isa(data, DataFrame)
            edge_properties[data[i,1]] = Edge_Properties(data[i, 1], data[i,2],data[i, 3], data[i, 4], (data[i, 5], data[i, 6]))
        elseif isa(data, Matrix)
            edge_properties[data[i,1]] = Edge_Properties(data[i, 1], data[i,2],data[i, 3], data[i, 4], (data[i, 5], data[i, 6]))
        else
            throw(ArgumentError("Input data must be either a Matrix or a DataFrame."))
        end
    end
    return edge_properties
end

#############################################################################################################


##############################################################################################################
### function to produce arc incidence matrix
### The function takes the number of nodes and the edge dict as input
### The edge dict key is the edge number and the values are #susceptance, min_capacity, max_capacity, node_tuple

function Node_Arc_Incidence_Matrix_Generator(n_nodes::Int, edge_properties::OrderedDict)
    arc_incidence_matrix = zeros(Real, n_nodes, length(edge_properties))
    ## the order of keys of the edge_properties dict doesn't matter
    for (edge, properties) in edge_properties
        arc_incidence_matrix[properties.node_tuple[1], edge] = -1
        arc_incidence_matrix[properties.node_tuple[2], edge] = 1
    end
    return arc_incidence_matrix'
end

#arc_incidence_matrix = Node_Arc_Incidence_Matrix_Generator(n_nodes, edge_dict)

# # Print the arc incidence matrix to check if it is correct
# println(arc_incidence_matrix)

function Susceptance_Matrix_Generator(n_nodes::Int, edge_properties::OrderedDict)
    susceptance_matrix = zeros(Real, n_nodes, n_nodes)

    ## the order of keys of the edge_properties dict doesn't matter
    for (edge, properties) in edge_properties
        susceptance_matrix[properties.node_tuple[1], properties.node_tuple[2]] = - properties.susceptance   ### negative susteptance
        susceptance_matrix[properties.node_tuple[2], properties.node_tuple[1]] = - properties.susceptance   ### negative susteptance
    end

    ### The diagonal elements are the sum of the susceptance of the edges connected to the node

    for i in 1:n_nodes
        susceptance_matrix[i, i] = - sum(susceptance_matrix[i, :])    #### double negative => positive susceptance
    end

    return susceptance_matrix
end

##############################################################################################################

### function to segregate the power generator set according to the bus they are connected to
### input is the power generator set and the edge dict
### The bus to generator dict input has the bus number as the key and the value is a list of power generators connected to that bus
### return dict of dict, where the key is the bus number and the value is a dict of power generators connected to that bus

function Power_Generator_Set_per_Bus(power_generator_set, bus_to_generator_dict)
    power_generator_set_per_bus = Dict()
    for bus in keys(bus_to_generator_dict)
        power_generator_set_per_bus[bus] = Dict()
        for gen in bus_to_generator_dict[bus]
            power_generator_set_per_bus[bus][gen] = power_generator_set[gen]    
        end
    end
    return power_generator_set_per_bus  
end

##############################################################################################################
function Bus_Total_Demand(data)
    n_buses = size(data, 1)
    bus_to_demand_dict = OrderedDict()

    for i in 1:n_buses
        if isa(data, DataFrame)
            bus_to_demand_dict[data[i, 1]] = data[i, 2:end]
        elseif isa(data, Matrix)
            bus_to_demand_dict[data[i, 1]] = data[i, 2:end]
        else
            throw(ArgumentError("Input data must be either a Matrix or a DataFrame."))
        end
    end
    return bus_to_demand_dict
end


##############################################################################################################
############################ functions for data generation ###################################################

# Function to generate random vectors within a norm ball
# v is the center of the ball, r is the radius of the ball, n is the number of vectors to generate, p is the norm
# Function to generate random vectors within a norm ball and ensure non-negative values
function generate_vectors_in_norm_ball(v::Vector, r::Real, n::Int; p::Real = 2, seed = 0)
    Random.seed!(seed)
    dim = length(v)
    vectors = Matrix{Float64}(undef, dim, n)
    
    for i in 1:n
        # Generate random direction and random magnitude within the ball
        random_direction = normalize(randn(dim)) # Random unit vector
        random_magnitude = rand()^(1/dim) * r  # Scaled for uniform distribution in ball
        random_vector = random_magnitude * random_direction

        # Generate new vector by adding the random vector within the ball to the original vector
        new_vector = v .+ random_vector
        
        # Truncate any negative values to 0
        new_vector[new_vector .< 0] .= 0.0
        
        # Store the new vector
        vectors[:, i] = new_vector
    end
    
    return vectors
end

function generate_vectors_in_l1_ball(v::Vector{Float64}, fraction::Float64, n::Int; seed=0)
    rng = MersenneTwister(seed)
    dim = length(v)
    total = sum(v)
    radius = fraction * total  # e.g. fraction = 0.1 for 10%

    # Preallocate the output matrix
    vectors = Matrix{Float64}(undef, dim, n)

    for i in 1:n
        # Sample direction from the positive simplex (which is an L1 unit "sphere" in the positive orthant)
        expvals = rand.(rng, Exponential(1.0), dim)
        dir = expvals ./ sum(expvals)  # sum(dir) = 1

        # Choose a random radius for uniform distribution inside L1 ball
        # For an L1 ball, scaling by (rand())^(1/dim) ensures uniform distribution inside the ball
        R = radius * (rand(rng)^(1/dim))

        # Construct the increment vector
        Δ = dir .* R

        # Add the increments to the nominal vector
        new_vector = v .+ Δ

        # Store the result
        vectors[:, i] = new_vector
    end

    return vectors
end

function sort_values_by_keys(dict, key_type = Int)


    sorted_keys = sort(collect(keys(dict))) # Sort the keys
    
    new_dict = OrderedDict{Any, Any}()  
    # Rebuild the OrderedDict with sorted keys and their corresponding values
    for key in sorted_keys
        new_dict[key] = dict[key]
    end
    return new_dict
end

function convert_keys_to_int(dict)

    sorted_keys = sort(parse.(Int, collect(keys(dict)))) # Sort the keys

    new_dict = OrderedDict()
    # Rebuild the OrderedDict with sorted keys and their corresponding values
    for key in sorted_keys
        new_dict[key] = dict["$key"]
    end
    return new_dict
end
##############################################################################################################
"""
    generate_multi_node_scenarios_dict(
        demands::Dict{Int,Vector{<:Real}}, 
        alpha::Real; 
        num_samples::Int=100, 
        seed::Union{Nothing,Int}=nothing
    ) -> Dict{Int,Dict{Int,Vector{Float64}}}

Generate `num_samples` scenarios for multiple buses/nodes. Each bus ID in `demands`
has a nominal 24-hour demand vector (length 24). We perturb these vectors within an
L2-ball of radius `alpha * ||v_b||_2`, where `v_b` is the nominal demand for bus b.
    
# Arguments
- `demands`: A dictionary with bus IDs (Int) as keys, and 24-hour nominal demand 
  vectors (Vector{<:Real}) as values. For example, Vector{Int}, Vector{Float64}, etc.
- `alpha`: A fraction that scales the norm of each bus's 24-hour vector to define
  the radius of its uncertainty set (e.g. 0.1 for 10%).
- `num_samples`: Number of scenarios to generate.
- `seed`: (optional) sets the random seed for reproducibility.

# Returns
A `Dict{Int,Dict{Int,Vector{Float64}}}`, in which:
- The outer dictionary keys are scenario indices (1..num_samples).
- Each value is another dictionary mapping bus ID -> perturbed 24-hour demand vector.
"""
function generate_multi_node_scenarios_dict(
    demands::OrderedDict{}, 
    alpha::Real=0.3; 
    num_samples::Int=100, 
    seed::Union{Nothing,Int}=nothing
)
    # Optionally set the random seed for reproducibility
    if seed !== nothing
        Random.seed!(seed)
    end

    # Prepare the output: scenario_index => (bus_id => demand_vector)
    scenario_dict = Dict{Int,Dict{Int,Vector{Float64}}}()

    # Collect the bus IDs
    bus_ids = collect(keys(demands))

    # For each scenario
    for scenario_idx in 1:num_samples
        scenario_data = Dict{Int,Vector{Float64}}()
        # For each bus, perturb its nominal vector
        for b in bus_ids
            v = demands[b]
            # Convert nominal vector to Float64
            v_float = Vector{Float64}(v)

            # Compute the L2 radius for this bus
            r = alpha * norm(v_float, 2)

            # Random direction in R^24 (Gaussian => normalized)
            z = randn(length(v_float))
            z ./= norm(z, 2)

            # Random magnitude in [0, r]
            ρ = rand() * r

            # Perturbation
            perturbation = ρ .* z

            # Perturbed vector
            scenario_data[b] = max.(0.0, v_float .+ perturbation)
        end

        # Store in outer dictionary
        scenario_dict[scenario_idx] = scenario_data
    end

    return scenario_dict
end

##############################################################################################################

