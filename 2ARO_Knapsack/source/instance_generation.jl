using Random
using JuMP
using GLPK
using Distributions
using DataFrames
using CSV
using Dates
using Base.Filesystem
###########################################################################################################

function generate_numbers(lower, upper, num)
    return lower .+ (upper .- lower) .* rand(num)

end

############################################################################################################

"""
    generate_scenarios(sum_limit::Float64, num_numbers::Int, N::Int, seed::Int=123)

Generate random numbers that sum up to a given limit representing the budget uncertainty.

# Arguments
- `sum_limit::Float64`: The sum limit for each scenario vector.
- `num_numbers::Int`: The number of random numbers in each scenario vector.
- `N::Int`: The number of scenario vectors to generate.
- `seed::Int`: The seed for reproducibility.

# Returns
- `scenario_vectors::Array{Float64, 2}`: An array of scenario vectors, where each vector contains random numbers that sum up to the given limit.
"""
function generate_scenarios(seed::Int, sum_limit::Float64, num_numbers::Int, N::Int)
    # Set the seed for reproducibility
    Random.seed!(seed)
    # Array to store the scenario vectors
    scenario_vectors = Array{Float64, 2}(undef, N, num_numbers)
    
    for i in 1:N
        # Generate parameters for the Dirichlet distribution
        α = ones(Float64, num_numbers)
        
        # Sample from the Dirichlet distribution
        samples = rand(Dirichlet(α))
        
        # Scale the samples to ensure the sum constraint
        scaled_samples = samples .* sum_limit
        scaled_sum = sum(scaled_samples)
        
        # If the sum exceeds the limit, scale down the samples proportionally
        while scaled_sum > sum_limit
            scaled_samples .= scaled_samples .* (sum_limit / scaled_sum)
            scaled_sum = sum(scaled_samples)
        end
        
        scenario_vectors[i, :] = round.(scaled_samples, digits=4)
    end

    return scenario_vectors
end



#############################################################################################################


"""
Calculate the capacity of a knapsack and generate random data for the knapsack problem.

# Arguments
- `I::Int`: The number of items in the knapsack.
- `R::Int`: The range of weights and profits for the items.
- `H::Int`: The height of the knapsack.

# Returns
- `capacity`: The calculated capacity of the knapsack.
- `weights`: An array of randomly generated weights for the items.
- `profits_nominal`: An array of randomly generated nominal profits for the items.
- `profits_degradation`: An array of profits degradation value for the items, within a certain range of profit_downlim and profit_uplim.
- `add_units`: An array of additonal units of capacity taken from Knapsack Capacity if item is repaired.
- `outsource_cost`: An array of outsource costs for the items.
- `budget_uncern_parameter`: The budget uncertainty parameter.

"""
function knapsack_instances(seed::Int=1, I::Int=10, R::Int=1000,  H::Int=100, scenarios::Function=generate_scenarios; train::Bool=true, num_scenarios::Int=10)

    Random.seed!(seed)
    # Generate random weights and profits for the items
    weights = generate_numbers(1, R, I)                                                          #rand(1:R, I)
    profits_nominal = generate_numbers(1, R, I)                                                  #rand(1:R, I)
    
    # Generate random values for other variables
    h = [40, 80][rand(1:2)]
    delta = [0.1, 0.5, 1][rand(1:3)]

    profit_downlim = profits_nominal .* (1 - delta) / 2
    profit_uplim = profits_nominal .* (1 + delta) / 2

    #profits_degradation = round.([rand(profit_downlim[i]:profit_uplim[i]) for i in 1:I], digits=6)
    profits_degradation = generate_numbers(profit_downlim, profit_uplim, I)
    capacity = round(sum(weights) * h/(H+1), digits=6)

    #additional_units = round.([rand(1:weights[i]) for i in 1:I], digits=6)
    additional_units = generate_numbers(1, weights, I)
    #outsource_cost = round.([rand(1.1 *profits_nominal[i] : 1.5 * profits_nominal[i]) for i in 1:I], digits=6)
    outsource_cost = generate_numbers(1.1 * profits_nominal, 1.5 * profits_nominal, I)

    budget_uncern_parameter = [0.1, 0.15, 0.2][rand(1:3)] * I

    # Generate scenarios for the budget uncertainity, we will use the Dirichlet distribution and since sum is less than gamma, we will generate values between 0.8budget_uncern_parameter and budget_uncern_parameter uniformly
    info = Dict("C" => capacity, 
                "w" => weights, 
                "p_bar" => profits_nominal, 
                "p_hat" => profits_degradation, 
                "t" => additional_units, 
                "f" => outsource_cost, 
                "budget_uncertainity_parameter" => budget_uncern_parameter, 
                "h" => h, 
                "delta" => delta, 
                "profit_downlim" => profit_downlim, 
                "profit_uplim" => profit_uplim)

    if train
        gamma = generate_numbers(0.95*budget_uncern_parameter,budget_uncern_parameter,1)[1]                                          #rand(0.95*budget_uncern_parameter:budget_uncern_parameter)
        itemized_profit_uncertainity_matrix = scenarios(seed, gamma, I, num_scenarios)
        info["Matrix_of_Xi"] = itemized_profit_uncertainity_matrix
    end

    return info 
end



#############################################################################################################

#sample = knapsack_instances(12, 10, 10, 1000, 100)

true
