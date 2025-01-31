using Gurobi
using JuMP

include("data_constructors.jl")

################################################################################################################
##################  #########################

function Econ_Disp_Model(model_env, num_buses, num_gens,
                        power_generator_property_dict,
                        uncertainty_num_to_bus_to_demand_dict, ## dict with key as uncertainty number and value as bus to demand dict
                        bus_to_generator_dict, edge_properties, 
                        high_cost, 
                        time_period)

    model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(model_env)))
    set_optimizer_attribute(model, "OutputFlag", 0)

    ## define the number of edges using the edge properties dict
    num_edges = length(edge_properties)

    ## Define the vectors of generator properties,check the sequence
    ## start_up_cost, shut_down_cost, Min_up_time, Min_down_time, Min_electricty_output_limit, Max_electricty_output_limit,
    ## Ramp_up_limit, Ramp_down_limit, Start_up_ramp_up_rate_limit, Shut_down_ramp_down_rate_limit

    generator_name_list = sort(collect(keys(power_generator_property_dict)))
    @assert(length(generator_name_list) == num_gens, "The number of generators in the power generator dict is not equal to the number of generators in the model")

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

    ##############################################################################################################
    ## Define the first stage variables
    model[Symbol("gen_bin")] = @variable(model, [1:num_gens, 1:time_period], Bin, base_name="gen_bin", lower_bound=0, upper_bound=1)
    model[Symbol("gen_on")] = @variable(model, [1:num_gens, 1:time_period], Bin, base_name="gen_on", lower_bound=0, upper_bound=1)
    model[Symbol("gen_off")] = @variable(model, [1:num_gens, 1:time_period], Bin, base_name="gen_off", lower_bound=0, upper_bound=1)

    ##################################### first stage constraints ################################################

    ## Min up after switch ON and down time after switch OFF constraints
    for gen in 1:num_gens
        @constraint(model,[time_h = 1:min_up_time[gen]], model[:gen_bin][gen, 2:(time_period - min_up_time[gen])] - 
                                                        model[:gen_bin][gen, 1:(time_period - min_up_time[gen] - 1)] - 
                                                        model[:gen_bin][gen, (2+time_h):(time_period - min_up_time[gen] + time_h)] .<= 0)
    
        @constraint(model, [time_h = 1:min_down_time[gen]], -model[:gen_bin][gen, 2:(time_period - min_down_time[gen])] +
                                                        model[:gen_bin][gen, 1:(time_period - min_down_time[gen] - 1)] +
                                                        model[:gen_bin][gen, (2+time_h):(time_period - min_down_time[gen] + time_h)] .<= 1)
                                                        
    end

    ## logic constraints for turn on and off actions
    @constraint(model, on_logic, - model[:gen_bin][:, 1:(time_period-1)] + model[:gen_bin][:, 2:time_period] - model[:gen_on][:, 2:time_period] .<= 0)
    @constraint(model, off_logic, model[:gen_bin][:, 1:(time_period-1)] - model[:gen_bin][:, 2:time_period] - model[:gen_off][:, 2:time_period] .<= 0)
    ################################################################################################################
    ########## for each uncertainty scenario, define the second stage variables and constraints ####################
    ################################################################################################################

    susceptance_matrix = Susceptance_Matrix_Generator(num_buses, edge_properties)     # num_buses x num_buses matrix
    arc_incidence_matrix = Node_Arc_Incidence_Matrix_Generator(num_buses, edge_properties)
    susceptance_vec = [edge_properties[edge].susceptance for edge in 1:num_edges]

    ## first stage cost variables
    gen_start_up_cost = model[:gen_on] .* start_up_cost
    gen_shut_down_cost = model[:gen_off] .* shut_down_cost

    ## placeholder single variable for second stage cost

    model[Symbol("second_stage_cost")] = @variable(model, lower_bound = 0, base_name = "second_stage_cost")

    num_uncertainties = length(uncertainty_num_to_bus_to_demand_dict)

    for uncertainty_num in 1:num_uncertainties
        bus_to_demand_dict = uncertainty_num_to_bus_to_demand_dict[uncertainty_num]
        ## Define the second stage variables
        model[Symbol("gen_power_",uncertainty_num)] = @variable(model, [1:num_gens, 1:time_period], base_name = "gen_power_$(uncertainty_num)", lower_bound = 0)
        
        ## Power Deficit to be met by the generator of higher cost
        model[Symbol("gen_power_high_cost_",uncertainty_num)] = @variable(model, [1:num_gens, 1:time_period], base_name = "gen_power_high_cost_$(uncertainty_num)", lower_bound = 0)

        ## power generation upper and lower limits when the generator is ON
        @constraint(model, model[Symbol("gen_power_",uncertainty_num)] - model[:gen_bin] .* max_power.<= 0)
        @constraint(model, model[Symbol("gen_power_",uncertainty_num)] - model[:gen_bin] .* min_power.>= 0)

        ## Power output ramp up and ramp down limits

        @constraint(model, model[Symbol("gen_power_",uncertainty_num)][:, 2:time_period] - 
                        model[Symbol("gen_power_",uncertainty_num)][:, 1:(time_period-1)] - 
                        (2 .- model[:gen_bin][:, 1:(time_period-1)] - model[:gen_bin][:, 2:time_period]) .* start_ramp_limit - 
                        (1 .+ model[:gen_bin][:, 1:(time_period-1)] - model[:gen_bin][:, 2:time_period]) .* ramp_up_limit .<= 0)
        
        @constraint(model, - model[Symbol("gen_power_",uncertainty_num)][:, 2:time_period] + 
                        model[Symbol("gen_power_",uncertainty_num)][:, 1:(time_period-1)] - 
                        (2 .- model[:gen_bin][:, 1:(time_period-1)] - model[:gen_bin][:, 2:time_period]) .* shut_ramp_limit - 
                        (1 .- model[:gen_bin][:, 1:(time_period-1)] + model[:gen_bin][:, 2:time_period]) .* ramp_down_limit .<= 0)

        ## power and demand balance constraints

        ## bus to load dict: key=> bus number, value=> time series vector of load demand
        ## convert the bus to load dict to a matrix of size :: num_buses x time_period
        bus_to_demand_matrix = zeros(num_buses, time_period)

        ## make sure that the bus_to_demand_dict keys (bus numbers) are sorted in ascending order starting from 1 ending in num_buses
        for (bus, demand) in bus_to_demand_dict
            bus_to_demand_matrix[bus, 1:end] = demand
        end
        ## power balance constraint with demand = generation + generation from high cost generator
        #@constraint(model, [time in 1:time_period], sum(model[Symbol("gen_power_",uncertainty_num)][:, time]) .== sum(bus_to_demand_matrix[:, time]))  
        @constraint(model, [time in 1:time_period], sum(model[Symbol("gen_power_",uncertainty_num)][:, time]) + sum(model[Symbol("gen_power_high_cost_",uncertainty_num)][:, time]) .== sum(bus_to_demand_matrix[:, time]))

        ## transmission line capacity DC constraints

        ## Define the nodal angle variables for each node

        model[Symbol("theta_",uncertainty_num)] = @variable(model, [1:num_buses, 1:time_period], base_name = "theta_$(uncertainty_num)")

        ## bus to generator dict: key=> bus number, value=> list of generator number attached to the bus
        ## create a matrix of size :: num_buses x time period, where the value is the sum of the generator power output attached to the bus
        generator_power_per_bus_vec = []

        for bus in 1:num_buses
            if bus in keys(bus_to_generator_dict)
                generator_power_per_bus = @expression(model, sum(model[Symbol("gen_power_",uncertainty_num)][gen, :] for gen in bus_to_generator_dict[bus]))
                generator_power_high_cost_per_bus = @expression(model, sum(model[Symbol("gen_power_high_cost_",uncertainty_num)][gen, :] for gen in bus_to_generator_dict[bus]))
                push!(generator_power_per_bus_vec, generator_power_per_bus + generator_power_high_cost_per_bus)
                #push!(generator_power_per_bus_vec, generator_power_per_bus)
            else
                push!(generator_power_per_bus_vec, zeros(time_period))
            end
        end
        generator_power_per_bus_matrix = hcat(generator_power_per_bus_vec...)'

        ## The net power availability at each bus (production - demand) is equal to the susceptance matrix * theta
        ## The first value of theta is 0 (set as reference bus angle) for all time periods
        @constraint(model, model[Symbol("theta_",uncertainty_num)][1 , :] .== 0)

        @constraint(model, (generator_power_per_bus_matrix .- bus_to_demand_matrix) .- 
                            [dot(susceptance_matrix[row,:], model[Symbol("theta_",uncertainty_num)][:,col]) for row in 1:num_buses, col in 1:time_period] .== 0)

        ## Transmission line capacity constraints
        ## The power flow on each edge is equal to the susceptance of the edge * (theta_i - theta_j)
        ## line_flow = susceptance_diagonal_matrix * (arc_incidence_matrix * theta)

        model[Symbol("line_flow_",uncertainty_num)] = @variable(model, [1:num_edges, 1:time_period], base_name = "line_flow_$(uncertainty_num)")

        arc_incidence_mul_theta = [dot(arc_incidence_matrix[row, :], model[Symbol("theta_",uncertainty_num)][:, col]) for row in 1:num_edges, col in 1:time_period]

        @constraint(model, model[Symbol("line_flow_",uncertainty_num)] - susceptance_vec .* arc_incidence_mul_theta .== 0)

        ## The power flow on each edge should be less than the maximum capacity of the edge
        @constraint(model, model[Symbol("line_flow_",uncertainty_num)] .<= [edge_properties[edge].max_capacity for edge in sort(collect(keys(edge_properties)))])
        ## The power flow on each edge should be greater than the minimum capacity (just the direction changes so -maximum capacity) of the edge
        @constraint(model, model[Symbol("line_flow_",uncertainty_num)] .>= -[edge_properties[edge].max_capacity for edge in sort(collect(keys(edge_properties)))])

        ### constraint that the second stage cost variable is greater than the second stage cost caused by the current scenario
        gen_power_cost = model[Symbol("gen_power_",uncertainty_num)] .* linear_cost_coefficient .+ 
                        model[:gen_bin] .* constant_cost_coefficient .+
                        model[Symbol("gen_power_high_cost_",uncertainty_num)] .* high_cost
                       
        @constraint(model, model[:second_stage_cost] >= sum(gen_power_cost))
        
    end

    ## model objective function

    ## The objective function is the sum of the generator cost, start up cost, and shut down cost

    @objective(model, Min, sum(gen_start_up_cost .+ gen_shut_down_cost) + model[:second_stage_cost])
    optimize!(model)

    result_dict = OrderedDict()
    ### store the first stage results

    result_dict["gen_on"] = value.(model[:gen_on])
    result_dict["gen_off"] = value.(model[:gen_off])
    result_dict["gen_bin"] = abs.(value.(model[:gen_bin]))
    # result_dict["gen_deficit_high_cost"] = value.(model[Symbol("gen_power_high_cost_",uncertainty_num)])

    result_dict["first_stage_cost"] = value.(gen_start_up_cost .+ gen_shut_down_cost)

    result_dict["gen_power_list"] = []
    result_dict["theta_list"] = []
    result_dict["line_flow_list"] = []
    result_dict["second_stage_cost"] = []
    result_dict["full_first_and_second_stage_cost_per_uncertainty"] = []

    constant_cost = value.(model[:gen_bin]) .* constant_cost_coefficient

    for uncertainty_num in 1:num_uncertainties
        push!(result_dict["gen_power_list"], value.(model[Symbol("gen_power_",uncertainty_num)]))
        push!(result_dict["theta_list"], value.(model[Symbol("theta_",uncertainty_num)]))
        push!(result_dict["line_flow_list"], value.(model[Symbol("line_flow_",uncertainty_num)]))

        linear_cost = value.(model[Symbol("gen_power_",uncertainty_num)]) .* linear_cost_coefficient
        
        high_cost = value.(model[Symbol("gen_power_high_cost_",uncertainty_num)]) .* high_cost


        push!(result_dict["second_stage_cost"], linear_cost .+ constant_cost .+ high_cost)
        push!(result_dict["full_first_and_second_stage_cost_per_uncertainty"], sum(value.(gen_start_up_cost .+ gen_shut_down_cost) .+ linear_cost .+ constant_cost .+ high_cost))
    end

    result_dict["objective_value"] = objective_value(model)
    result_dict["susceptance_matrix"] = susceptance_matrix
    #result_dict["status"] = termination_status(model)
    #result_dict["model"] = model

    ## get the uncertainty number selected by the model
    return result_dict
end

################################################################################################################
####### ************************************  Inner Min Model  ******************************************** ####
################################################################################################################

function Inner_Min_Second_Stage(model_env, first_stage_decision, 
                                num_buses, num_gens,
                                susceptance_matrix, arc_incidence_matrix, susceptance_vec, max_edge_capacity,
                                bus_to_generator_dict, bus_to_demand_dict, 
                                linear_cost_coefficient, constant_cost_coefficient, min_power, max_power,
                                start_ramp_limit, shut_ramp_limit, ramp_up_limit, ramp_down_limit,
                                high_cost,
                                time_period)

    model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(model_env)))
    #println("#################################################")
    set_optimizer_attribute(model, "OutputFlag", 0)
    
    model[Symbol("gen_power")] = @variable(model, [1:num_gens, 1:time_period], base_name = "gen_power", lower_bound = 0)
    model[Symbol("gen_power_high_cost")] = @variable(model, [1:num_gens, 1:time_period], base_name = "gen_power_high_cost", lower_bound = 0)

    @constraint(model, model[:gen_power] -  first_stage_decision .* max_power.<= 0)
    @constraint(model, model[:gen_power] -  first_stage_decision .* min_power.>= 0)

    ## Power output ramp up and ramp down limits
    @constraint(model, model[:gen_power][:, 2:time_period] - 
                    model[:gen_power][:, 1:(time_period-1)] -
                    (2 .- first_stage_decision[:, 1:(time_period-1)] - first_stage_decision[:, 2:time_period]) .* start_ramp_limit -
                    (1 .+ first_stage_decision[:, 1:(time_period-1)] - first_stage_decision[:, 2:time_period]) .* ramp_up_limit .<= 0)

    @constraint(model, - model[:gen_power][:, 2:time_period] +
                    model[:gen_power][:, 1:(time_period-1)] -
                    (2 .- first_stage_decision[:, 1:(time_period-1)] - first_stage_decision[:, 2:time_period]) .* shut_ramp_limit -
                    (1 .- first_stage_decision[:, 1:(time_period-1)] + first_stage_decision[:, 2:time_period]) .* ramp_down_limit .<= 0)

    bus_to_demand_matrix = zeros(num_buses, time_period)

    for (bus, demand) in bus_to_demand_dict
        bus_to_demand_matrix[bus, 1:end] = demand
    end

    @constraint(model, [time in 1:time_period], sum(model[:gen_power][:, time] .+ model[:gen_power_high_cost][:, time]) .== sum(bus_to_demand_matrix[:, time]))  

    model[Symbol("theta")] = @variable(model, [1:num_buses, 1:time_period], base_name = "theta")

    generator_power_per_bus_vec = []
    for bus in 1:num_buses
        if bus in keys(bus_to_generator_dict)
            generator_power_per_bus = @expression(model, sum(model[:gen_power][gen, :] for gen in bus_to_generator_dict[bus]))
            generator_power_high_cost_per_bus = @expression(model, sum(model[:gen_power_high_cost][gen, :] for gen in bus_to_generator_dict[bus]))

            push!(generator_power_per_bus_vec, generator_power_per_bus .+ generator_power_high_cost_per_bus)
        else
            push!(generator_power_per_bus_vec, zeros(time_period))
        end
    end
    generator_power_per_bus_matrix = hcat(generator_power_per_bus_vec...)' 

    ## The first value of theta is 0 (set as reference bus angle) for all time periods
    @constraint(model, model[:theta][1 , :] .== 0)

    @constraint(model, (generator_power_per_bus_matrix .- bus_to_demand_matrix) .- 
                    [dot(susceptance_matrix[row,:], model[:theta][:,col]) for row in 1:num_buses, col in 1:time_period] .== 0)

    ## Transmission line capacity constraints
    model[Symbol("line_flow")] = @variable(model, [1:num_edges, 1:time_period], base_name = "line_flow")

    arc_incidence_mul_theta = [dot(arc_incidence_matrix[row, :], model[:theta][:, col]) for row in 1:num_edges, col in 1:time_period]

    @constraint(model, model[:line_flow] - susceptance_vec .* arc_incidence_mul_theta .== 0)

    ## The power flow on each edge should be less than the maximum capacity of the edge
    @constraint(model, model[:line_flow] .<= max_edge_capacity)    #[edge_properties[edge].max_capacity for edge in sort(collect(keys(edge_properties)))])
    @constraint(model, model[:line_flow] .>= -max_edge_capacity)   #[edge_properties[edge].max_capacity for edge in sort(collect(keys(edge_properties)))])

    ## model objective function
    gen_cost = @expression(model, (model[:gen_power] .* linear_cost_coefficient) .+ (first_stage_decision .* constant_cost_coefficient) .+ (model[:gen_power_high_cost] .* high_cost))

    @objective(model, Min, sum(gen_cost))

    optimize!(model)

    result_dict = OrderedDict()
    result_dict["gen_power"] = value.(model[:gen_power])
    result_dict["gen_power_high_cost"] = value.(model[:gen_power_high_cost])
    result_dict["theta"] = value.(model[:theta])
    result_dict["line_flow"] = value.(model[:line_flow])
    result_dict["objective_value"] = objective_value(model)
    result_dict["status"] = termination_status(model)
    result_dict["gen_cost"] = value.(gen_cost)
    result_dict["uncertainty_dict"] = bus_to_demand_dict

    #println("termination status:", termination_status(model))

    return result_dict
end
################################################################################################################
####### ************************************  First Stage Model  ****************************************** ####
################################################################################################################

function First_Stage_CCG(model_env, num_buses, num_gens,
                            start_up_cost, shut_down_cost, 
                            constant_cost_coefficient, linear_cost_coefficient, 
                            min_power, max_power,
                            min_up_time, min_down_time, 
                            ramp_up_limit, ramp_down_limit, 
                            start_ramp_limit, shut_ramp_limit,
                            uncertainty_num_to_bus_to_demand_dict, 
                            susceptance_matrix, arc_incidence_matrix, susceptance_vec, max_edge_capacity,
                            bus_to_generator_dict, 
                            high_cost,
                            time_period)

    model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(model_env)))
    set_optimizer_attribute(model, "OutputFlag", 0)

    ##############################################################################################################
    ## Define the first stage variables
    model[Symbol("gen_bin")] = @variable(model, [1:num_gens, 1:time_period], Bin, base_name="gen_bin", lower_bound=0, upper_bound=1)
    model[Symbol("gen_on")] = @variable(model, [1:num_gens, 1:time_period], Bin, base_name="gen_on", lower_bound=0, upper_bound=1)
    model[Symbol("gen_off")] = @variable(model, [1:num_gens, 1:time_period], Bin, base_name="gen_off", lower_bound=0, upper_bound=1)

    ##################################### first stage constraints ################################################

    ## Min up after switch ON and down time after switch OFF constraints
    for gen in 1:num_gens
        @constraint(model,[time_h = 1:min_up_time[gen]], model[:gen_bin][gen, 2:(time_period - min_up_time[gen])] - 
                                            model[:gen_bin][gen, 1:(time_period - min_up_time[gen] - 1)] - 
                                            model[:gen_bin][gen, (2+time_h):(time_period - min_up_time[gen] + time_h)] .<= 0)

        @constraint(model, [time_h = 1:min_down_time[gen]], -model[:gen_bin][gen, 2:(time_period - min_down_time[gen])] +
                                            model[:gen_bin][gen, 1:(time_period - min_down_time[gen] - 1)] +
                                            model[:gen_bin][gen, (2+time_h):(time_period - min_down_time[gen] + time_h)] .<= 1)
                                        
    end

    ## logic constraints for turn on and off actions
    @constraint(model, - model[:gen_bin][:, 1:(time_period-1)] + model[:gen_bin][:, 2:time_period] - model[:gen_on][:, 2:time_period] .<= 0)
    @constraint(model, model[:gen_bin][:, 1:(time_period-1)] - model[:gen_bin][:, 2:time_period] - model[:gen_off][:, 2:time_period] .<= 0)
    ######################################################################################################################
    ########## for each uncertainty scenario in master, define the recourse variables and constraints ####################
    ######################################################################################################################


    ## first stage cost variables
    gen_start_up_cost = model[:gen_on] .* start_up_cost
    gen_shut_down_cost = model[:gen_off] .* shut_down_cost

    ## placeholder single variable for second stage cost

    model[Symbol("second_stage_cost")] = @variable(model, lower_bound = 0, base_name = "second_stage_cost")

    num_uncertainties = length(uncertainty_num_to_bus_to_demand_dict)

    list_of_uncertainties = sort(collect(keys(uncertainty_num_to_bus_to_demand_dict)))

    for uncertainty_num in list_of_uncertainties

        bus_to_demand_dict = uncertainty_num_to_bus_to_demand_dict[uncertainty_num]
        ## Define the second stage variables
        model[Symbol("gen_power_",uncertainty_num)] = @variable(model, [1:num_gens, 1:time_period], base_name = "gen_power_$(uncertainty_num)", lower_bound = 0)
        model[Symbol("gen_power_high_cost_",uncertainty_num)] = @variable(model, [1:num_gens, 1:time_period], base_name = "gen_power_high_cost_$(uncertainty_num)", lower_bound = 0)

        ## power generation upper and lower limits when the generator is ON
        @constraint(model, model[Symbol("gen_power_",uncertainty_num)] - model[:gen_bin] .* max_power.<= 0)
        @constraint(model, model[Symbol("gen_power_",uncertainty_num)] - model[:gen_bin] .* min_power.>= 0)

        ## Power output ramp up and ramp down limits

        @constraint(model, model[Symbol("gen_power_",uncertainty_num)][:, 2:time_period] - 
            model[Symbol("gen_power_",uncertainty_num)][:, 1:(time_period-1)] - 
            (2 .- model[:gen_bin][:, 1:(time_period-1)] - model[:gen_bin][:, 2:time_period]) .* start_ramp_limit - 
            (1 .+ model[:gen_bin][:, 1:(time_period-1)] - model[:gen_bin][:, 2:time_period]) .* ramp_up_limit .<= 0)

        @constraint(model, - model[Symbol("gen_power_",uncertainty_num)][:, 2:time_period] + 
            model[Symbol("gen_power_",uncertainty_num)][:, 1:(time_period-1)] - 
            (2 .- model[:gen_bin][:, 1:(time_period-1)] - model[:gen_bin][:, 2:time_period]) .* shut_ramp_limit - 
            (1 .- model[:gen_bin][:, 1:(time_period-1)] + model[:gen_bin][:, 2:time_period]) .* ramp_down_limit .<= 0)

        bus_to_demand_matrix = zeros(num_buses, time_period)

        for (bus, demand) in bus_to_demand_dict
            bus_to_demand_matrix[bus, 1:end] = demand
        end

        @constraint(model, [time in 1:time_period], sum(model[Symbol("gen_power_",uncertainty_num)][:, time]) .+ 
                                                    sum(model[Symbol("gen_power_high_cost_",uncertainty_num)][:, time]) .== sum(bus_to_demand_matrix[:, time]))  ## power balance constraint

        model[Symbol("theta_",uncertainty_num)] = @variable(model, [1:num_buses, 1:time_period], base_name = "theta_$(uncertainty_num)")

        generator_power_per_bus_vec = []

        for bus in 1:num_buses
            if bus in keys(bus_to_generator_dict)
                generator_power_per_bus = @expression(model, sum(model[Symbol("gen_power_",uncertainty_num)][gen, :] for gen in bus_to_generator_dict[bus]))
                generator_power_high_cost_per_bus = @expression(model, sum(model[Symbol("gen_power_high_cost_",uncertainty_num)][gen, :] for gen in bus_to_generator_dict[bus]))
                push!(generator_power_per_bus_vec, generator_power_per_bus .+ generator_power_high_cost_per_bus)
            else
                push!(generator_power_per_bus_vec, zeros(time_period))
            end
        end

        generator_power_per_bus_matrix = hcat(generator_power_per_bus_vec...)'

        @constraint(model, model[Symbol("theta_",uncertainty_num)][1 , :] .== 0)

        @constraint(model, (generator_power_per_bus_matrix .- bus_to_demand_matrix) .- 
                [dot(susceptance_matrix[row,:], model[Symbol("theta_",uncertainty_num)][:,col]) for row in 1:num_buses, col in 1:time_period] .== 0)

        model[Symbol("line_flow_",uncertainty_num)] = @variable(model, [1:num_edges, 1:time_period], base_name = "line_flow_$(uncertainty_num)")

        arc_incidence_mul_theta = [dot(arc_incidence_matrix[row, :], model[Symbol("theta_",uncertainty_num)][:, col]) for row in 1:num_edges, col in 1:time_period]

        @constraint(model, model[Symbol("line_flow_",uncertainty_num)] - susceptance_vec .* arc_incidence_mul_theta .== 0)

        ## The power flow on each edge should be less than the maximum capacity of the edge
        @constraint(model, model[Symbol("line_flow_",uncertainty_num)] .<= max_edge_capacity)    
        ## The power flow on each edge should be greater than the minimum capacity (just the direction changes so -maximum capacity) of the edge
        @constraint(model, model[Symbol("line_flow_",uncertainty_num)] .>= -max_edge_capacity) 

        ### constraint that the second stage cost variable is greater than the second stage cost caused by the current scenario
        gen_power_cost = model[Symbol("gen_power_",uncertainty_num)] .* linear_cost_coefficient .+ 
                            model[:gen_bin] .* constant_cost_coefficient .+
                            model[Symbol("gen_power_high_cost_",uncertainty_num)] .* high_cost

        @constraint(model, model[:second_stage_cost] >= sum(gen_power_cost))

    end

    @objective(model, Min, sum(gen_start_up_cost .+ gen_shut_down_cost) + model[:second_stage_cost])
    optimize!(model)

    result_dict = OrderedDict()
    ### store the first stage results

    result_dict["gen_on"] = value.(model[:gen_on])
    result_dict["gen_off"] = value.(model[:gen_off])
    result_dict["gen_bin"] = value.(model[:gen_bin])

    result_dict["first_stage_cost"] = abs.(value.(gen_start_up_cost .+ gen_shut_down_cost))

    result_dict["gen_power_list"] = []
    result_dict["theta_list"] = []
    result_dict["line_flow_list"] = []
    result_dict["second_stage_cost"] = []

    constant_cost = value.(model[:gen_bin]) .* constant_cost_coefficient

    for uncertainty_num in list_of_uncertainties
        push!(result_dict["gen_power_list"], value.(model[Symbol("gen_power_",uncertainty_num)]))
        push!(result_dict["theta_list"], value.(model[Symbol("theta_",uncertainty_num)]))
        push!(result_dict["line_flow_list"], value.(model[Symbol("line_flow_",uncertainty_num)]))
        push!(result_dict["second_stage_cost"], value.(model[Symbol("gen_power_",uncertainty_num)]) .* linear_cost_coefficient .+ constant_cost .+ value.(model[Symbol("gen_power_high_cost_",uncertainty_num)]) .* high_cost)
    end

    result_dict["objective_value"] = objective_value(model)
    result_dict["status"] = termination_status(model)
    #result_dict["model"] = model
    #println(result_dict)
    #println("###############################first stage done################################################")
    return result_dict

end

################################################################################################################
################################################################################################################

function Second_Stage_CCG_Discrete(model_env, first_stage_decision, num_buses, num_gens,
                                    susceptance_matrix, arc_incidence_matrix, susceptance_vec, max_edge_capacity,
                                    bus_to_generator_dict, dict_of_demand_dicts,
                                    linear_cost_coefficient, constant_cost_coefficient, min_power, max_power,
                                    start_ramp_limit, shut_ramp_limit, ramp_up_limit, ramp_down_limit,
                                    high_cost,
                                    time_period)


    collect_infeasible_keys = []
    max_obj_list = []
    for (demand_key, bus_to_demand_dict) in dict_of_demand_dicts
        try
            
            second_stage_values = Inner_Min_Second_Stage(model_env, first_stage_decision, 
                                                            num_buses, num_gens,
                                                            susceptance_matrix, arc_incidence_matrix, susceptance_vec, max_edge_capacity,
                                                            bus_to_generator_dict, bus_to_demand_dict,
                                                            linear_cost_coefficient, constant_cost_coefficient, min_power, max_power,
                                                            start_ramp_limit, shut_ramp_limit, ramp_up_limit, ramp_down_limit,
                                                            high_cost,
                                                            time_period)
                                                    

            push!(max_obj_list, second_stage_values["objective_value"])
            
            #println("new uncertainty*************************************************************************")

        catch e
            
            #println(e)
            #println("demand key:", demand_key)
            push!(collect_infeasible_keys, demand_key)
            #println("Error in Inner Min Second Stage")
            continue
        end
    end
    max_obj_key = argmax(max_obj_list)
    #println(max_obj_key, " ", max_obj_list[max_obj_key])
    return max_obj_list, max_obj_key, collect_infeasible_keys
end    

################################################################################################################
################################################################################################################

function Econ_Disp_CCG(seed, first_stage_model_env, second_stage_model_env, 
                        num_buses, num_gens, num_edges,
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
    arc_incidence_matrix = Node_Arc_Incidence_Matrix_Generator(num_buses, edge_properties)
    susceptance_vec = [edge_properties[edge].susceptance for edge in 1:num_edges]
    max_edge_capacity = [edge_properties[edge].max_capacity for edge in sort(collect(keys(edge_properties)))]

    ## Randomly select "warm_start_uncertainty_num" number of the dict of bus to demands for the warm start, create a new orddred dict

    master_dict_bus_to_demand_dict = OrderedDict()

    Random.seed!(seed)

    for i in 1:warm_start_uncertainty_num
        rand_key = rand(1:length(uncertainty_num_to_bus_to_demand_dict))
        master_dict_bus_to_demand_dict[i] = uncertainty_num_to_bus_to_demand_dict[rand_key]
    end

    current_iteration = 0

    optimality_gap = upper_bound - lower_bound

    iter_end_tol = 1e-3

    result_dict = OrderedDict()

    ## create a set of infeasible keys for each instance
    per_instance_infeasible_keys = Set()

    while true
        current_iteration += 1
        first_stage_result = First_Stage_CCG(first_stage_model_env, num_buses, num_gens,
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
        #println("First Stage Result: ", first_stage_result["objective_value"])

        first_stage_decision = first_stage_result["gen_bin"]
        lower_bound = first_stage_result["objective_value"] 

        second_stage_values, max_obj_key, infeasible_keys = Second_Stage_CCG_Discrete(second_stage_model_env, first_stage_decision, num_buses, num_gens,
                                                                                        susceptance_matrix, arc_incidence_matrix, susceptance_vec, max_edge_capacity,
                                                                                        bus_to_generator_dict, uncertainty_num_to_bus_to_demand_dict,
                                                                                        linear_cost_coefficient, constant_cost_coefficient, min_power, max_power,
                                                                                        start_ramp_limit, shut_ramp_limit, ramp_up_limit, ramp_down_limit,
                                                                                        high_cost,
                                                                                        time_period)

        #println("the second stage is fine Second Stage Result: ", second_stage_values)


        #print("%%%%%%%%%%%%%")
        upper_bound = second_stage_values[max_obj_key]#["objective_value"]

        result_dict[current_iteration] = OrderedDict()
        result_dict[current_iteration]["first_stage_decision"] = first_stage_decision
        #result_dict[current_iteration]["second_stage_values"] = second_stage_values
        result_dict[current_iteration]["max_obj_key"] = max_obj_key
        result_dict[current_iteration]["lower_bound"] = lower_bound
        result_dict[current_iteration]["upper_bound"] = upper_bound
        result_dict[current_iteration]["optimality_gap"] = upper_bound - lower_bound
        
        #push!(per_instance_infeasible_keys, infeasible_keys)
        ## add the infeasible keys to the set
        union!(per_instance_infeasible_keys, infeasible_keys)

        ## if lower bound is equal to upper bound then break
        if isapprox(lower_bound, upper_bound, atol=iter_end_tol)
            println("Optimality gap is less than tolerance")
            result_dict["final_lower_bound"] = lower_bound
            result_dict["final_upper_bound"] = upper_bound
            break
        end
        
        ## if the new optimality gap is same as the old one approximately at a given tolerance then break
        if isapprox(optimality_gap, upper_bound - lower_bound, atol=iter_end_tol)
            println("Optimality gap is not changing")
            result_dict["final_lower_bound"] = lower_bound
            result_dict["final_upper_bound"] = upper_bound
            break
        end

        ## if the lower bound is greater than the upper bound then print error and break

        if current_iteration > 990
            println("Error: Bad convergence")
            result_dict["final_lower_bound"] = lower_bound
            result_dict["final_upper_bound"] = upper_bound
            break
        end

        optimality_gap = upper_bound - lower_bound

        ## update the uncertainty dict
        master_dict_bus_to_demand_dict[warm_start_uncertainty_num + current_iteration] = uncertainty_num_to_bus_to_demand_dict[max_obj_key]
        
    end
    println("Total Iterations: ", current_iteration)
    println("Final Lower Bound: ", result_dict["final_lower_bound"])
    println("Final Upper Bound: ", result_dict["final_upper_bound"])
    result_dict["infeasible_keys_set"] = per_instance_infeasible_keys
    return result_dict
end

################################################################################################################
################################################################################################################

function First_Stage_First_Phase(relaxed_model_env, num_buses, num_gens,
                                start_up_cost, shut_down_cost, 
                                constant_cost_coefficient, linear_cost_coefficient, 
                                min_power, max_power,
                                min_up_time, min_down_time, 
                                ramp_up_limit, ramp_down_limit, 
                                start_ramp_limit, shut_ramp_limit,
                                uncertainty_num_to_bus_to_demand_dict,   ## dict with key as uncertainty number and value as bus to demand dict
                                susceptance_matrix, arc_incidence_matrix, susceptance_vec, max_edge_capacity,
                                bus_to_generator_dict, 
                                high_cost,
                                time_period)

    model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(relaxed_model_env)))

    set_optimizer_attribute(model, "OutputFlag", 0)

    model[Symbol("gen_bin")] = @variable(model, [1:num_gens, 1:time_period], Bin, base_name="gen_bin", lower_bound=0, upper_bound=1)
    model[Symbol("gen_on")] = @variable(model, [1:num_gens, 1:time_period], Bin, base_name="gen_on", lower_bound=0, upper_bound=1)
    model[Symbol("gen_off")] = @variable(model, [1:num_gens, 1:time_period], Bin, base_name="gen_off", lower_bound=0, upper_bound=1)

    ##################################### first stage constraints ################################################

    ## Min up and down after switch ON and down time after switch OFF constraints
    for gen in 1:num_gens
        @constraint(model,[time_h = 1:min_up_time[gen]], model[:gen_bin][gen, 2:(time_period - min_up_time[gen])] - 
                                            model[:gen_bin][gen, 1:(time_period - min_up_time[gen] - 1)] - 
                                            model[:gen_bin][gen, (2+time_h):(time_period - min_up_time[gen] + time_h)] .<= 0)

        @constraint(model, [time_h = 1:min_down_time[gen]], -model[:gen_bin][gen, 2:(time_period - min_down_time[gen])] +
                                            model[:gen_bin][gen, 1:(time_period - min_down_time[gen] - 1)] +
                                            model[:gen_bin][gen, (2+time_h):(time_period - min_down_time[gen] + time_h)] .<= 1)
                                        
    end

    ## logic constraints for turn on and off actions
    @constraint(model, - model[:gen_bin][:, 1:(time_period-1)] + model[:gen_bin][:, 2:time_period] - model[:gen_on][:, 2:time_period] .<= 0)
    @constraint(model, model[:gen_bin][:, 1:(time_period-1)] - model[:gen_bin][:, 2:time_period] - model[:gen_off][:, 2:time_period] .<= 0)
    ######################################################################################################################

    ## first stage cost variables
    gen_start_up_cost = model[:gen_on] .* start_up_cost
    gen_shut_down_cost = model[:gen_off] .* shut_down_cost

    ######################################################################################################################

    model[Symbol("gen_power")] = @variable(model, [1:num_gens, 1:time_period], base_name = "gen_power", lower_bound = 0)
    model[Symbol("gen_power_high_cost")] = @variable(model, [1:num_gens, 1:time_period], base_name = "gen_power_high_cost", lower_bound = 0)

    @constraint(model, model[:gen_power] -  model[:gen_bin] .* max_power.<= 0)
    @constraint(model, model[:gen_power] -  model[:gen_bin] .* min_power.>= 0)

    ## Power output ramp up and ramp down limits
    @constraint(model, model[:gen_power][:, 2:time_period] - 
                    model[:gen_power][:, 1:(time_period-1)] -
                    (2 .- model[:gen_bin][:, 1:(time_period-1)] - model[:gen_bin][:, 2:time_period]) .* start_ramp_limit -
                    (1 .+ model[:gen_bin][:, 1:(time_period-1)] - model[:gen_bin][:, 2:time_period]) .* ramp_up_limit .<= 0)

    @constraint(model, - model[:gen_power][:, 2:time_period] +
                    model[:gen_power][:, 1:(time_period-1)] -
                    (2 .- model[:gen_bin][:, 1:(time_period-1)] - model[:gen_bin][:, 2:time_period]) .* shut_ramp_limit -
                    (1 .- model[:gen_bin][:, 1:(time_period-1)] + model[:gen_bin][:, 2:time_period]) .* ramp_down_limit .<= 0)

    ########################################################################################################################
    ###############**** Use binary variable z vector to select the uncertainty scenario ****################################
    ##############**** The z vector is a binary vector of size num_uncertainties ****########################################

    sorted_uncertainty_keys = sort(collect(keys(uncertainty_num_to_bus_to_demand_dict)))

    for key in sorted_uncertainty_keys
        model[Symbol("z_",key)] = @variable(model, base_name = "z_$(key)", binary = true)
    end

    ## constraint that only one z can be selected

    @constraint(model, sum([model[Symbol("z_",key)] for key in sorted_uncertainty_keys]) == 1)

    ### Convert each bus to demand dict to a bus to demand matrix

    bus_to_demand_matrix_dict = OrderedDict()

    for (key, bus_to_demand_dict) in uncertainty_num_to_bus_to_demand_dict
        bus_to_demand_matrix = zeros(num_buses, time_period)
        for (bus, demand) in bus_to_demand_dict
            bus_to_demand_matrix[bus, 1:end] = demand
        end
        bus_to_demand_matrix_dict[key] = bus_to_demand_matrix
    end

    ## Create model variable for the demand matrix and make a constraint that selects the demand matrix based on the z vector

    model[Symbol("selected_demand_matrix")] = @variable(model, [1:num_buses, 1:time_period], base_name = "selected_demand_matrix")

    ## broadcast the sum of each demand matrix multiplied by the z vector to the selected demand matrix

    ## z*Xi expression
    relaxed_sum = @expression(model, sum([model[Symbol("z_",key)] .* bus_to_demand_matrix_dict[key] for key in keys(uncertainty_num_to_bus_to_demand_dict)]))

    @constraint(model, model[:selected_demand_matrix] .== relaxed_sum)
   

    ########################################################################################################################

    ## Power balance constraint

    @constraint(model, [time in 1:time_period], sum(model[:gen_power][:, time]) .+ 
                                                sum(model[Symbol("gen_power_high_cost")][:, time]) .== sum(model[:selected_demand_matrix][:, time]))

    ########################################################################################################################

    ## Create the theta variable

    model[Symbol("theta")] = @variable(model, [1:num_buses, 1:time_period], base_name = "theta")

    generator_power_per_bus_vec = []

    for bus in 1:num_buses
        if bus in keys(bus_to_generator_dict)
            generator_power_per_bus = @expression(model, sum(model[:gen_power][gen, :] for gen in bus_to_generator_dict[bus]))
            generator_power_high_cost_per_bus = @expression(model, sum(model[:gen_power_high_cost][gen, :] for gen in bus_to_generator_dict[bus]))
            push!(generator_power_per_bus_vec, generator_power_per_bus .+ generator_power_high_cost_per_bus)
        else
            push!(generator_power_per_bus_vec, zeros(time_period))
        end
    end

    generator_power_per_bus_matrix = hcat(generator_power_per_bus_vec...)'

    ## The first value of theta is 0 (set as reference bus angle) for all time periods
    @constraint(model, model[:theta][1 , :] .== 0)

    @constraint(model, (generator_power_per_bus_matrix .- model[:selected_demand_matrix]) .- 
            [dot(susceptance_matrix[row,:], model[:theta][:,col]) for row in 1:num_buses, col in 1:time_period] .== 0)

    ## Transmission line capacity constraints
    model[Symbol("line_flow")] = @variable(model, [1:num_edges, 1:time_period], base_name = "line_flow")

    arc_incidence_mul_theta = [dot(arc_incidence_matrix[row, :], model[:theta][:, col]) for row in 1:num_edges, col in 1:time_period]

    @constraint(model, model[:line_flow] - susceptance_vec .* arc_incidence_mul_theta .== 0)

    ## The power flow on each edge should be less than the maximum capacity of the edge
    @constraint(model, model[:line_flow] .<= max_edge_capacity)  
    @constraint(model, model[:line_flow] .>= -max_edge_capacity)

    ## model objective function

    gen_cost = @expression(model, (model[:gen_power] .* linear_cost_coefficient) .+ (model[:gen_bin] .* constant_cost_coefficient) .+ (model[:gen_power_high_cost] .* high_cost))

    @objective(model, Min, sum(gen_cost) + sum(gen_start_up_cost) + sum(gen_shut_down_cost))

    return model
end

################################################################################################################

function First_Stage_ML_Acc_CCG(first_phase_model_env, second_phase_model_env, 
                                num_buses, num_gens,
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
        #println(first_phase_uncertainty_key)


        second_phase_results, second_phase_uncertainty_key = Second_Stage_CCG_Discrete(second_phase_model_env, first_phase_decision, num_buses, num_gens,
                                                                                        susceptance_matrix, arc_incidence_matrix, susceptance_vec, max_edge_capacity,
                                                                                        bus_to_generator_dict, master_dict_bus_to_demand_dict,
                                                                                        linear_cost_coefficient, constant_cost_coefficient, min_power, max_power,
                                                                                        start_ramp_limit, shut_ramp_limit, ramp_up_limit, ramp_down_limit,
                                                                                        high_cost,
                                                                                        time_period)
        #println(second_phase_uncertainty_key)
        second_phase_objective_value = second_phase_results[second_phase_uncertainty_key]

        push!(list_phase_1_and_2_keys, (first_phase_uncertainty_key, second_phase_uncertainty_key))
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

################################################################################################################

function Econ_Disp_Decomposed_CCG(seed, first_phase_model_env, second_stage_model_env, improved_model_env,
                                    num_buses, num_gens, num_edges,
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
    arc_incidence_matrix = Node_Arc_Incidence_Matrix_Generator(num_buses, edge_properties)
    susceptance_vec = [edge_properties[edge].susceptance for edge in 1:num_edges]
    max_edge_capacity = [edge_properties[edge].max_capacity for edge in sort(collect(keys(edge_properties)))]

    ## Randomly select "warm_start_uncertainty_num" number of the dict of bus to demands for the warm start, create a new orddred dict

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
        first_stage_result_relaxed = First_Stage_ML_Acc_CCG(first_phase_model_env, second_stage_model_env, 
                                                                num_buses, num_gens,
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

        first_stage_decision = first_stage_result_relaxed["first_phase_decision"]
        first_stage_selected_demand_key = first_stage_result_relaxed["selected_demand_key"]

        lower_bound = first_stage_result_relaxed["objective_value"] 

        #println("First Stage Objective Value: ", lower_bound)

        second_stage_values, max_obj_key = Second_Stage_CCG_Discrete(second_stage_model_env, first_stage_decision, num_buses, num_gens,
                                                                        susceptance_matrix, arc_incidence_matrix, susceptance_vec, max_edge_capacity,
                                                                        bus_to_generator_dict, uncertainty_num_to_bus_to_demand_dict,
                                                                        linear_cost_coefficient, constant_cost_coefficient, min_power, max_power,
                                                                        start_ramp_limit, shut_ramp_limit, ramp_up_limit, ramp_down_limit,
                                                                        high_cost,
                                                                        time_period)

        upper_bound = second_stage_values[max_obj_key]
        #println("Second Stage Objective Value: ", upper_bound)

        result_dict[current_iteration] = OrderedDict()
        result_dict[current_iteration]["first_stage_decision"] = first_stage_decision
        result_dict[current_iteration]["first_stage_obj_key"] = first_stage_selected_demand_key
        result_dict[current_iteration]["lower_bound"] = lower_bound
        result_dict[current_iteration]["upper_bound"] = upper_bound
        result_dict[current_iteration]["optimality_gap"] = upper_bound - lower_bound
        result_dict[current_iteration]["first_stage_phase_1_2_keys"] = first_stage_result_relaxed["uncertainty_key_combinations"]


        ## if lower bound is equal to upper bound then break
        if isapprox(lower_bound, upper_bound, atol=iter_end_tol)
            ## add the second stage selected uncertainty to the master uncertainty dict
            master_dict_bus_to_demand_dict[max_obj_key] = uncertainty_num_to_bus_to_demand_dict[max_obj_key]

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

        ## if the new optimality gap is same as the old one approximately at a given tolerance then break
        if isapprox(optimality_gap, upper_bound - lower_bound, atol=iter_end_tol)
            master_dict_bus_to_demand_dict[max_obj_key] = uncertainty_num_to_bus_to_demand_dict[max_obj_key]

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
            master_dict_bus_to_demand_dict[max_obj_key] = uncertainty_num_to_bus_to_demand_dict[max_obj_key]

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

        optimality_gap = upper_bound - lower_bound

        # if max_obj_key in keys(master_dict_bus_to_demand_dict)
        #     println("The max obj key is already in the master dict")
        #     master_dict_bus_to_demand_dict[max_obj_key] = uncertainty_num_to_bus_to_demand_dict[max_obj_key]

        #     ## solve the Robust Optimization problem with the master dict
        #     improved_results = First_Stage_CCG(improved_model_env, num_buses, num_gens,
        #                                             start_up_cost, shut_down_cost, 
        #                                             constant_cost_coefficient, linear_cost_coefficient, 
        #                                             min_power, max_power,
        #                                             min_up_time, min_down_time, 
        #                                             ramp_up_limit, ramp_down_limit, 
        #                                             start_ramp_limit, shut_ramp_limit,
        #                                             master_dict_bus_to_demand_dict, 
        #                                             susceptance_matrix, arc_incidence_matrix, susceptance_vec, max_edge_capacity,
        #                                             bus_to_generator_dict, 
        #                                             high_cost,
        #                                             time_period)


        #     result_dict["final_lower_bound"] = max(lower_bound, improved_results["objective_value"])
        #     result_dict["final_upper_bound"] = upper_bound
        #     break
        # else
            ## update the uncertainty dict
        master_dict_bus_to_demand_dict[max_obj_key] = sort_values_by_keys(uncertainty_num_to_bus_to_demand_dict[max_obj_key])
        # end
    end
    println("lower bound: ", result_dict["final_lower_bound"])
    println("upper bound: ", result_dict["final_upper_bound"])
    return result_dict

end

true




