using Gurobi
using JuMP

include("data_constructors.jl")

function DC_Transmission_Constraints(model, num_buses, # Number of buses: M
                                    num_edges, # Number of edges: N
                                    edge_properties, bus_to_load_dict, 
                                    power_generator_set_per_bus, 
                                    susceptance_matrix,  # Susceptance matrix of the network (N x N)
                                    arc_incidence_matrix, # Arc incidence matrix of the network (M x N)
                                    susceptance_diagonal_matrix, # Susceptance matrix of the edges (M x M)
                                    time_period)

    ##############################################################################################################
    ########################### Define the nodal angle variables #################################################

    ## theta: Nodal angle at bus at time : theta_bus_k
    for bus in keys(bus_to_load_dict)
        model[Symbol("theta_", bus)] = @variable(model, [1:time_period], base_name="theta_$(bus)")
    end

    ## Constraint 1: Fix the angle at the reference bus to 0 (node 1)
    first_bus = first(keys(bus_to_load_dict)) # Get the key of the first bus in the bus_to_load_dict
    println("The reference bus is: ", first_bus)
    @constraint(model, [time = 1 : time_period], model[Symbol("theta_", first_bus)][time] == 0)

    ##############################################################################################################
    ########################### Define the branch flow variables #################################################

    # Precompute branch flow limits for all edges
    branch_flow_max_vec = [edge_properties[edge].max_capacity for edge in 1:num_edges]
    branch_flow_min_vec = [edge_properties[edge].min_capacity for edge in 1:num_edges]

    for time in 1:time_period
        # Initialize vectors for net power injection and angles for a given time
        net_power_injection_vec = zeros(Float64, num_buses)
        angle_vec = [model[Symbol("theta_", bus)][time] for bus in 1:num_buses]

        # Compute net power injections at each bus using vectorized operations
        for (bus, load) in bus_to_load_dict
            net_power_injection_vec[bus] = sum(model[Symbol("gen_power_", bus, "_", gen)][time] 
                                               for gen in keys(power_generator_set_per_bus[bus])) - load[time]
        end

        # Nodal power balance constraint (excluding reference bus)
        @constraint(model, net_power_injection_vec[2:end] .== susceptance_matrix[2:end, 2:end] * angle_vec[2:end])

        # Branch flow constraints
        branch_flow = susceptance_diagonal_matrix * arc_incidence_matrix[:, 2:end] * angle_vec[2:end]
        @constraint(model, branch_flow .<= branch_flow_max_vec)
        @constraint(model, branch_flow .>= branch_flow_min_vec)
    end

end
    
###############################################################################################################################################################
###############################################################################################################################################################
###############################################################################################################################################################
"""
    Economic_Dispatch(
                        model_env, 
                        num_buses, num_generators, num_loads, 
                        power_generator_set, => Dict: Dict of all power generator properties
                        bus_to_generator_dict => Dict of List: Dict key as bus number and value as list of generators connected to that bus,
                        edge_dict, => Dict: Dict of edges and values as tuple of nodes that the edge connects
                        bus_to_load_dict
                        )

TBW
"""
function Economic_Dispatch(
                        model_env, 
                        num_buses, num_generators, num_loads, 
                        power_generator_set_per_bus, edge_properties, bus_to_load_dict,
                        time_period
                        )
    
    @assert num_buses == length(keys(power_generator_set_per_bus)) "Error: Number of buses (num_buses) does not match the number of keys in power_generator_set_per_bus."
    @assert num_generators == sum(length(keys(power_generator_set_per_bus[bus])) for bus in keys(power_generator_set_per_bus)) "Error: Number of generators (num_generators) does not match the sum of lengths of keys in power_generator_set_per_bus."

    ### All keys in power generator set per bus should be int, raise an error if not
    @assert all(isequal(typeof(bus), Int) for bus in keys(power_generator_set_per_bus)) "Error: Keys in power_generator_set_per_bus should be of type Int."

    ### The size of the number of buses should be equal to the maximum key in power generator set per bus
    @assert num_buses == maximum(keys(power_generator_set_per_bus)) "Error: Number of buses (num_buses) does not match the maximum key in power_generator_set_per_bus+> Missing Bus Integer"

    ### All keys in bus_to_load_dict should be int, raise an error if not
    @assert all(isequal(typeof(bus), Int) for bus in keys(bus_to_load_dict)) "Error: Keys in bus_to_load_dict should be of type Int."

    ### All keys in edge_properties should be int, raise an error if not
    @assert all(isequal(typeof(edge), Int) for edge in keys(edge_properties)) "Error: Keys in edge_properties should be of type Int."

    ### Make sure that the keys in bus_to_load_dict are the same as the keys in power_generator_set_per_bus
    ### In case the keys are not the same, raise an error
    @assert keys(power_generator_set_per_bus) == keys(bus_to_load_dict) "Error: Keys in power_generator_set_per_bus and bus_to_load_dict do not match."

    #############################################################################################################
    ############################## Define the model #############################################################
   
    ED_model = Gurobi.Model(env=model_env)
    set_attributes(ED_Model, "OutputFlag" => 0)
    
    ##############################################################################################################
    ########################### first stage variables ############################################################

    ## y_bus_generator: ON=1/OFF=0 status of generator at bus at time : gen_decision_bus_k_gen_j
    ## u_bus_generator: ON=1/OFF=0 generator switched ON at bus at time : gen_on_decision_bus_k_gen_j
    ## v_bus_generator: ON=1/OFF=0 generator switched OFF at bus at time : gen_off_decision_bus_k_gen_j
    
    

    for bus in keys(power_generator_set_per_bus)
        for gen in keys(power_generator_set_per_bus[bus])
            # y
            ED_model[Symbol("gen_bin_", bus, "_", gen)] = @variable(ED_model, [1:time_period], base_name="gen_bin_$(bus)_$(gen)", lower_bound=0, upper_bound=1, Bin)
            # u
            ED_model[Symbol("gen_on_", bus, "_", gen)] = @variable(ED_model, [1:time_period], base_name="gen_on_$(bus)_$(gen)", lower_bound=0, upper_bound=1, Bin)
            # v
            ED_model[Symbol("gen_off_", bus, "_", gen)] = @variable(ED_model, [1:time_period], base_name="gen_off_$(bus)_$(gen)", lower_bound=0, upper_bound=1, Bin)

        ## Constraint 1: Minimum up time and Minimum down time constraints

            for time in 2:time_period
                up_time = min(time_period, time + power_generator_set_per_bus[bus][gen].Min_up_time - 1)
                down_time = min(time_period, time + power_generator_set_per_bus[bus][gen].Min_down_time - 1)
                

                @constraint(ED_model, [t = (time+1) : up_time], 
                                        ED_model[Symbol("gen_bin_", bus, "_", gen)][time] - 
                                        ED_model[Symbol("gen_bin_", bus, "_", gen)][time-1] - 
                                        ED_model[Symbol("gen_bin_", bus, "_", gen)][t] <= 0)

                @constraint(ED_model, [t = (time+1) : down_time],
                                       -ED_model[Symbol("gen_bin_", bus, "_", gen)][time] + 
                                       ED_model[Symbol("gen_bin_", bus, "_", gen)][time-1] + 
                                       ED_model[Symbol("gen_bin_", bus, "_", gen)][t] <= 1)
            end

        ## Constraint 2: Generator ON/OFF status logic constraints
            @constraint(ED_model, [time = 2: time_period], 
                        ED_model[Symbol("gen_bin_", bus, "_", gen)][time - 1] - 
                        ED_model[Symbol("gen_bin_", bus, "_", gen)][time] + 
                        ED_model[Symbol("gen_on_", bus, "_", gen)][time] >= 0)

            @constraint(ED_model, [time = 2: time_period], 
                        - ED_model[Symbol("gen_bin_", bus, "_", gen)][time - 1] +
                        ED_model[Symbol("gen_bin_", bus, "_", gen)][time] + 
                        ED_model[Symbol("gen_off_", bus, "_", gen)][time] >= 0)
        end
    end

    ##############################################################################################################
    ########################### second stage variables and constraints ###########################################
    ## susceptance_matrix: Susceptance matrix of the network
    susceptance_matrix = Susceptance_Matrix_Generator(num_buses, edge_properties)
    
    ## arc_incidence_matrix: Arc incidence matrix of the network
    arc_incidence_matrix = Node_Arc_Incidence_Matrix_Generator(num_buses, edge_properties)

    ## susceptance_diagonal_matrix: Susceptance matrix of the edges
    susceptance_diagonal_matrix = zeros(Int, num_edges, num_edges)
    for edge in keys(edge_properties)
        susceptance_diagonal_matrix[edge, edge] = edge_properties[edge].susceptance
    end

    for bus in keys(power_generator_set_per_bus)
        for gen in keys(power_generator_set_per_bus[bus])

    ## variable: power generated by generator at bus at time : gen_power_bus_k_gen_j
            ED_model[Symbol("gen_power_", bus, "_", gen)] = @variable(ED_model, [1:time_period], base_name="gen_power_$(bus)_$(gen)", lower_bound=0)

    ## lower bound and upper bound constraints on power generated by generator depending on if the generator is ON or OFF

            @constraint(ED_Model, [time = 1:time_period], 
                        ED_model[Symbol("gen_power_", bus, "_", gen)][time] <= 
                        power_generator_set_per_bus[bus][gen].Max_electricty_output_limit * ED_model[Symbol("gen_bin_", bus, "_", gen)][time])

            @constraint(ED_Model, [time = 1:time_period],
                        ED_model[Symbol("gen_power_", bus, "_", gen)][time] >= 
                        power_generator_set_per_bus[bus][gen].Min_electricty_output_limit * ED_model[Symbol("gen_bin_", bus, "_", gen)][time])
    
    ## production ramp constraints during normal operation and when generators are switched ON or OFF

            @constraint(ED_Model, [time = 2:time_period], 
                        ED_model[Symbol("gen_power_", bus, "_", gen)][time] - 
                        ED_model[Symbol("gen_power_", bus, "_", gen)][time - 1] <= 
                        (2 - ED_model[Symbol("gen_bin_", bus, "_", gen)][time-1] - ED_model[Symbol("gen_bin_", bus, "_", gen)][time]) * power_generator_set_per_bus[bus][gen].Ramp_up_limit+
                        (1 + ED_model[Symbol("gen_bin_", bus, "_", gen)][time-1] - ED_model[Symbol("gen_bin_", bus, "_", gen)][time]) * power_generator_set_per_bus[bus][gen].Start_up_ramp_rate_limit)

            @constraint(ED_Model, [time = 2:time_period],
                        - ED_model[Symbol("gen_power_", bus, "_", gen)][time] + 
                        ED_model[Symbol("gen_power_", bus, "_", gen)][time - 1] <= 
                        (2 - ED_model[Symbol("gen_bin_", bus, "_", gen)][time-1] - ED_model[Symbol("gen_bin_", bus, "_", gen)][time]) * power_generator_set_per_bus[bus][gen].Ramp_down_limit +
                        (1 + ED_model[Symbol("gen_bin_", bus, "_", gen)][time-1] - ED_model[Symbol("gen_bin_", bus, "_", gen)][time]) * power_generator_set_per_bus[bus][gen].Shut_down_ramp_rate_limit)
        
           
            end
    end

    ## Total Power and Total Demand balance constraint for each time period
    for time in 1:time_period
        total_power = 0
        total_demand = 0
        for bus in keys(power_generator_set_per_bus)
            for gen in keys(power_generator_set_per_bus[bus])
                total_power += ED_model[Symbol("gen_power_", bus, "_", gen)][time]
            end
        end
        for load in keys(bus_to_load_dict)
            total_demand += bus_to_load_dict[load][time]
        end
        @constraint(ED_Model, total_power == total_demand)
    end


    ##############################################################################################################
    ########################### Transmission Constraint ##########################################################

    DC_Transmission_Constraints(ED_Model, num_buses, num_edges,
                                edge_properties, bus_to_load_dict, 
                                power_generator_set_per_bus, 
                                susceptance_matrix,  # Susceptance matrix of the network (N x N)
                                arc_incidence_matrix, # Arc incidence matrix of the network (M x N)
                                susceptance_diagonal_matrix, # Susceptance matrix of the edges (M x M)
                                time_period)

end

true



