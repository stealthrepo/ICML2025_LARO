###############################################################################################################
### The input of generator is the dict with key as generatoe number and value as the generator parameters #####
### Eco_Dis_model_Exact has single load demand for all buses ##################################################
function Eco_Dis_Model_Exact(model_env, num_buses, num_gens,
    power_generator_property_dict,
    bus_to_demand_dict, bus_to_generator_dict, edge_properties, 
    time_period)
model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(model_env)))
set_optimizer_attribute(model, "OutputFlag", 0)

num_edges = length(edge_properties)

## Define the vectors of generator properties

#### start_up_cost, shut_down_cost, Min_up_time, Min_down_time, Min_electricty_output_limit, Max_electricty_output_limit,
#### Ramp_up_limit, Ramp_down_limit, Start_up_ramp_up_rate_limit, Shut_down_ramp_down_rate_limit

generator_name_list = sort(collect(keys(power_generator_property_dict)))
@assert(length(generator_name_list) == num_gens, "The number of generators in the power generator dict is not equal to the number of generators in the model")

## assert the values of bus_to_demand_dict are of the same length as the time period
for (bus, demand) in bus_to_demand_dict
@assert(length(demand) == time_period, "The length of the demand vector for bus $bus is not equal to the time period")
end

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
##############################################################################################################
## Define the second stage variables

model[Symbol("gen_power")] = @variable(model, [1:num_gens, 1:time_period], base_name = "gen_power", lower_bound = 0)

susceptance_matrix = Susceptance_Matrix_Generator(num_buses, edge_properties)     # num_buses x num_buses matrix

arc_incidence_matrix = Node_Arc_Incidence_Matrix_Generator(num_buses, edge_properties)

susceptance_vec = [edge_properties[edge].susceptance for edge in 1:num_edges]

## Define the second stage constraints and matrices used for transmission line capacity constraints

## Power generation upper and lower limits when the generator is ON

@constraint(model, model[:gen_power] - model[:gen_bin] .* max_power.<= 0)
@constraint(model, model[:gen_power] - model[:gen_bin] .* min_power.>= 0)

## Power output ramp up and ramp down limits
@constraint(model, model[:gen_power][:, 2:time_period] - 
model[:gen_power][:, 1:(time_period-1)] -
(2 .- model[:gen_bin][:, 1:(time_period-1)] - model[:gen_bin][:, 2:time_period]) .* start_ramp_limit -
(1 .+ model[:gen_bin][:, 1:(time_period-1)] - model[:gen_bin][:, 2:time_period]) .* ramp_up_limit .<= 0)

@constraint(model, - model[:gen_power][:, 2:time_period] +
model[:gen_power][:, 1:(time_period-1)] -
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

@constraint(model, [time in 1:time_period], sum(model[:gen_power][:, time]) .== sum(bus_to_demand_matrix[:, time]))  ## power balance constraint
## transmission line capacity DC constraints

## Define the nodal angle variables for each node
model[Symbol("theta")] = @variable(model, [1:num_buses, 1:time_period], base_name = "theta")

## bus to generator dict: key=> bus number, value=> list of generator number attached to the bus
## create a matrix of size :: num_buses x time period, where the value is the sum of the generator power output attached to the bus

generator_power_per_bus_vec = []
for bus in 1:num_buses
if bus in keys(bus_to_generator_dict)
generator_power_per_bus = @expression(model, sum(model[:gen_power][gen, :] for gen in bus_to_generator_dict[bus]))
push!(generator_power_per_bus_vec, generator_power_per_bus)
else
push!(generator_power_per_bus_vec, zeros(time_period))
end
end
generator_power_per_bus_matrix = hcat(generator_power_per_bus_vec...)' 

## The net power availability at each bus (production - demand) is equal to the susceptance matrix * theta
## The first value of theta is 0 (set as reference bus angle) for all time periods
@constraint(model, model[:theta][1 , :] .== 0)

@constraint(model, (generator_power_per_bus_matrix .- bus_to_demand_matrix) .- 
[dot(susceptance_matrix[row,:], model[:theta][:,col]) for row in 1:num_buses, col in 1:time_period] .== 0)


## Transmission line capacity constraints
## The power flow on each edge is equal to the susceptance of the edge * (theta_i - theta_j)
## line_flow = susceptance_diagonal_matrix * (arc_incidence_matrix * theta)

model[Symbol("line_flow")] = @variable(model, [1:num_edges, 1:time_period], base_name = "line_flow")

arc_incidence_mul_theta = [dot(arc_incidence_matrix[row, :], model[:theta][:, col]) for row in 1:num_edges, col in 1:time_period]

@constraint(model, model[:line_flow] - susceptance_vec .* arc_incidence_mul_theta .== 0)

## The power flow on each edge should be less than the maximum capacity of the edge
@constraint(model, model[:line_flow] .<= [edge_properties[edge].max_capacity for edge in sort(collect(keys(edge_properties)))])
@constraint(model, model[:line_flow] .>= -[edge_properties[edge].max_capacity for edge in sort(collect(keys(edge_properties)))])
## model objective function

## The objective function is the sum of the generator cost, start up cost, and shut down cost
## The generator cost is the sum of the linear cost * power output + constant cost
gen_start_up_cost = model[:gen_on] .* start_up_cost
gen_shut_down_cost = model[:gen_off] .* shut_down_cost

gen_cost = model[:gen_power] .* linear_cost_coefficient .+ model[:gen_power] .* constant_cost_coefficient

@objective(model, Min, sum(gen_cost .+ gen_start_up_cost .+ gen_shut_down_cost))

optimize!(model)

result_dict = OrderedDict()
result_dict["gen_power"] = value.(model[:gen_power])
result_dict["gen_on"] = value.(model[:gen_on])
result_dict["gen_off"] = value.(model[:gen_off])
result_dict["gen_bin"] = value.(model[:gen_bin])
result_dict["theta"] = value.(model[:theta])
result_dict["line_flow"] = value.(model[:line_flow])
result_dict["objective_value"] = objective_value(model)
result_dict["start_up_cost"] = value.(gen_start_up_cost)
result_dict["shut_down_cost"] = value.(gen_shut_down_cost)
result_dict["gen_cost"] = value.(gen_cost)

result_dict["status"] = termination_status(model)
return result_dict

end