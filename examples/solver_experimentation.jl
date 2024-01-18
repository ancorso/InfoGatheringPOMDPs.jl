using InfoGatheringPOMDPs
using POMDPs
using POMDPTools
using NativeSARSOP
using JLD2
using CSV
using Random
using DataStructures
using Plots
default(framestyle = :box,  color_palette=:seaborn_deep6, fontfamily="Computer Modern", margin=5Plots.mm)

# Define the scenarios and corresponding paths to CSV files
scenario_csvs = OrderedDict(
        Symbol("Option 1") => "./examples/data/Geothermal Reservoir/DSSCEN1_50N_POMDP.csv", # Scenar
        Symbol("Option 2") => "./examples/data/Geothermal Reservoir/DSSCEN2_50N_POMDP.csv",
        Symbol("Option 3") => "./examples/data/Geothermal Reservoir/DSSCEN3_50N_POMDP.csv",
        Symbol("Option 4") => "./examples/data/Geothermal Reservoir/DSSCEN4_50N_POMDP.csv",
        Symbol("Option 5") => "./examples/data/Geothermal Reservoir/DSSCEN5_50N_POMDP.csv",
        Symbol("Option 6") => "./examples/data/Geothermal Reservoir/DSSCEN6_50N_POMDP.csv",
        Symbol("Option 7") => "./examples/data/Geothermal Reservoir/DSSCEN7_50N_POMPDP.csv",
        Symbol("Option 8") => "./examples/data/Geothermal Reservoir/DSSCEN8_50_POMDP.csv",
        Symbol("Option 9") => "./examples/data/Geothermal Reservoir/DSSCEN10_50N_POMPD.csv",
        Symbol("Option 10") => "./examples/data/Geothermal Reservoir/DSSCEN11_50N_POMPD .csv",
        Symbol("Option 11") => "./examples/data/Geothermal Reservoir/DSSCEN13_50N_POMDP.csv"
    )

# Define the set of geological and economic parameters so geo models can be separated
geo_params = ["par_P_Std", "par_P_Mean", "par_PMax", "par_AZ", "par_PMin", "par_PV", "par_SEED", "par_Zmax", "par_Zmin", "par_SEEDTrend", "par_C_WATER", "par_P_INIT", "par_FTM", "par_KVKH", "par_C_ROCK", "par_THCOND_RES", "par_HCAP_RES", "par_TempGrad"]
econ_params = ["par_CAPEXitem1", "par_CAPEXitem2", "par_CAPEXitem5", "par_CAPEXitem6", "par_UnitOPEXWater", "par_UnitOPEXWaterInj", "par_UnitOPEXActiveProducers", "par_UnitOPEXActiveWaterInjectors"]

# Parameter descriptions
var_description = OrderedDict( #TODO: check these
    :par_P_Std => "Porosity Std. Dev.",
    :par_P_Mean => "Porosity Mean",
    :par_PMax => "Variogram Anisotropy (major)",
    :par_PMin => "Variogram Anisotropy (minor)",
    :par_PV => "Variogram Anisotropy (vertical)",
    :par_AZ => "Variogram Azimuth",
    :par_FTM => "Fault Transmissibility Multiplier",
    :par_KVKH => "Permeability Ratio (vert/horiz)",
    :par_Zmax => "Surface Trend Z Max",
    :par_Zmin => "Surface Trend Z Min",
    :par_C_WATER => "Water Compressibility",
    :par_P_INIT => "Initial Reservoir Pressure",
    :par_C_ROCK => "Rock Compressibility",
    :par_THCOND_RES => "Rock Thermal Conductivity",
    :par_HCAP_RES => "Rock Heat Capacity",
    :par_TempGrad => "Temperature Gradient",
    :par_CAPEXitem1 => "CAPEX Injection Well",
    :par_CAPEXitem2 => "CAPEX Production Well",
    :par_CAPEXitem3 => "CAPEX Surface Facilities",
    :par_CAPEXitem4 => "CAPEX Flowlines",
    :par_CAPEXitem5 => "CAPEX Production Pump",
    :par_CAPEXitem6 => "CAPEX Injection Pump",
    :par_OPEXfixedtotal => "OPEX Fixed Total",
    :par_UnitOPEXWater => "OPEX Water",
    :par_UnitOPEXWaterInj => "OPEX Water Injectors",
    :par_UnitOPEXActiveProducers => "OPEX Active Producers",
    :par_UnitOPEXActiveWaterInjectors => "OPEX Active Water Injectors"
)

# Define which parameters are affected for the three-slim-well case
pairs_3Wells = [(:par_P_Std, 0.0025), (:par_P_Mean, 0.025), (:par_PMax, 1000), (:par_AZ, 45), (:par_PMin, 200), (:par_PV, 10), (:par_Zmax, 0.045), (:par_Zmin, 0.015)]

# Define the observation actions
obs_actions = [
    ObservationAction("Measure Water Compressibility", 14/365, -0.05, uniform(:par_C_WATER, 5e-5)),
    ObservationAction("Measure Initial Reservoir Pressure", 21/365, -0.1, uniform(:par_P_INIT, 5)),
    ObservationAction("Measure Fault Transmissibility Multiplier", 60/365, -2.0, uniform(:par_FTM, 0.015)),
    ObservationAction("Measure Permeability Ratio", 30/365, -0.05, uniform(:par_KVKH, 0.1)),
    ObservationAction("Measure Rock Compressibility", 30/365, -0.05, uniform(:par_C_ROCK, 5e-5)),
    ObservationAction("Measure Rock Thermal Conductivity", 30/365, -0.07, uniform(:par_THCOND_RES, 0.5)),
    ObservationAction("Measure Rock Heat Capacity", 30/365, -0.07, uniform(:par_HCAP_RES, 250)),
    ObservationAction("Measure Temperature Gradient", 21/365, -0.1, uniform(:par_TempGrad, 0.001)),
    ObservationAction("Drill 3 Wells", 270/365, -9, product_uniform(pairs_3Wells)),
    ObservationAction("Assess CAPEX Injection Well", 30/365, -1.2, uniform(:par_CAPEXitem1, 0.3)),
    ObservationAction("Assess CAPEX Production Well", 30/365, -1.2, uniform(:par_CAPEXitem2, 0.3)),
    ObservationAction("Assess CAPEX Surface Facilities", 60/365, -10, scenario_dependent_uniform(:par_CAPEXitem3, keys(scenario_csvs), 18.0)),
    ObservationAction("Assess CAPEX Flowlines", 60/365, -10, scenario_dependent_uniform(:par_CAPEXitem4, keys(scenario_csvs), 5.5)),
    ObservationAction("Assess CAPEX Production Pump", 30/365, -0.03, uniform(:par_CAPEXitem5, 0.01625)),
    ObservationAction("Assess CAPEX Injection Pump", 30/365, -0.02, uniform(:par_CAPEXitem6, 0.01)),
    ObservationAction("Assess OPEX Fixed Total", 30/365, -3.5, scenario_dependent_uniform(:par_OPEXfixedtotal, keys(scenario_csvs), 1.0)),
    ObservationAction("Assess OPEX Water", 30/365, -0.02, uniform(:par_UnitOPEXWater, 0.00975)),
    ObservationAction("Assess OPEX Water Injectors", 30/365, -0.02, uniform(:par_UnitOPEXWaterInj, 0.00975)),
    ObservationAction("Assess OPEX Active Producers", 30/365, -0.01, uniform(:par_UnitOPEXActiveProducers, 0.006)),
    ObservationAction("Assess OPEX Active Water Injectors", 30/365, -0.01, uniform(:par_UnitOPEXActiveWaterInjectors, 0.006)),
]

# Set the number of observation bins for each action
Nbins = fill(2, length(obs_actions[1:end]))
Nbins[findall([a.name == "Drill 3 Wells" for a in obs_actions])] .= 5

# Set the discount factor
discount_factor = 0.90 # Annual discount factor

## Create a POMDP with all the states:
Nsamples_per_bin=10
statevec = InfoGatheringPOMDPs.parseCSV(scenario_csvs, geo_params, econ_params)
Random.seed!(1)
discrete_obs = InfoGatheringPOMDPs.get_discrete_observations(statevec, obs_actions, Nbins;)
obs_dists = InfoGatheringPOMDPs.create_observation_distributions(statevec, obs_actions, discrete_obs, Nbins .* Nsamples_per_bin;)
pomdp = InfoGatheringPOMDPs.InfoGatheringPOMDP(statevec, obs_actions, keys(scenario_csvs), discrete_obs, obs_dists, (o) -> InfoGatheringPOMDPs.nearest_neighbor_mapping(o, discrete_obs), discount_factor)


## Try out and plot different solvers

# Lower bound:
αvec = onestep_alphavec_policy(pomdp)

# SARSOP
sarsop_solver = SARSOPSolver(max_time=10.,epsilon=0.5, init_lower = PreSolved(αvec), init_upper = NativeSARSOP.FastInformedBound(bel_res=1e-2, init_value = 0.0, max_time=100))
sarsop_planner = solve(sarsop_solver, pomdp);


# Plot some key metrics
avals = actionvalues(sarsop_planner, initialstate(pomdp))
plot(1:length(avals), avals, seriestype=:bar, xrotation=90, xticks=(1:length(avals), [a isa ObservationAction ? a.name : string(a) for a in actions(pomdp)]), ylabel="Value", label="", xlabel="Action", title="First Action Values for SARSOP", size=(600,600))
savefig("action_values.pdf")

using AbstractTrees
using D3Trees

struct ObsNode
    o
    condplan
end

struct CondPlan
    action
    value
    prob
    PES
    subplans
    CondPlan(action, value, prob, PES, subplans) = new(action, value, prob, PES, subplans)
end

AbstractTrees.children(node::ObsNode) = [node.condplan]
AbstractTrees.children(node::CondPlan) = node.subplans
D3Trees.text(node::ObsNode) ="$(collect(keys(node.o))[1]):$(collect(values(node.o))[1])"
D3Trees.text(node::CondPlan) = "Action:$(node.action isa Symbol ? node.action : node.action.name)\nValue:$(node.value)\nProb:$(node.prob)\nPES:$(node.PES)"

function CondPlan(pomdp, π, depth; prob=1.0, b=initialstate(pomdp), up=DiscreteUpdater(pomdp))
    a = action(π, b)
    v = value(π, b)
    PES = sum([bi*any([reward(pomdp, s, a) > 0 for a in pomdp.terminal_actions]) for (bi, s) in zip(b.b, states(pomdp))])
    children = []
    if depth > 0
        for o in observations(pomdp, a)
            po = sum([bi*obs_weight(pomdp, sp, a, sp, o) for (bi,sp) in zip(b.b, states(pomdp))])
            bp = update(up, b, a, o)
            plan = CondPlan(pomdp, π, depth-1; prob=prob*po, b=bp, up)
            push!(children, ObsNode(o, plan))
        end
    end
    return CondPlan(a, v, prob, PES, children)
end

plan = CondPlan(pomdp, sarsop_planner, 14)

inchrome(D3Tree(plan))
