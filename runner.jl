include("src/pomdp.jl")
include("src/policies.jl")
include("src/metrics.jl")

using POMDPs
using POMDPTools
using NativeSARSOP
using PointBasedValueIteration
using JLD2
using Random

scenario_csvs = Dict(
        :Scenario_1 => "data/Geothermal Reservoir/DSSCEN1_50N_POMDP.csv",
        :Scenario_2 => "data/Geothermal Reservoir/DSSCEN2_50N_POMDP.csv",
        :Scenario_3 => "data/Geothermal Reservoir/DSSCEN3_50N_POMDP.csv",
        :Scenario_4 => "data/Geothermal Reservoir/DSSCEN4_50N_POMDP.csv",
        :Scenario_5 => "data/Geothermal Reservoir/DSSCEN5_50N_POMDP.csv",
        :Scenario_6 => "data/Geothermal Reservoir/DSSCEN6_50N_POMDP.csv",
        :Scenario_7 => "data/Geothermal Reservoir/DSSCEN7_50N_POMPDP.csv",
        :Scenario_8 => "data/Geothermal Reservoir/DSSCEN8_50_POMDP.csv",
        :Scenario_10 => "data/Geothermal Reservoir/DSSCEN10_50N_POMPD.csv",
        :Scenario_11 => "data/Geothermal Reservoir/DSSCEN11_50N_POMPD .csv",
        :Scenario_13 => "data/Geothermal Reservoir/DSSCEN13_50N_POMDP.csv"
    )

geo_params = ["par_P_Std", "par_P_Mean", "par_PMax", "par_AZ", "par_PMin", "par_PV", "par_SEED", "par_Zmax", "par_Zmin", "par_SEEDTrend", "par_C_WATER", "par_P_INIT", "par_FTM", "par_KVKH", "par_C_ROCK", "par_THCOND_RES", "par_HCAP_RES", "par_TempGrad"]
econ_params = ["par_CAPEXitem1", "par_CAPEXitem2", "par_CAPEXitem5", "par_CAPEXitem6", "par_UnitOPEXWater", "par_UnitOPEXWaterInj", "par_UnitOPEXActiveProducers", "par_UnitOPEXActiveWaterInjectors"]

## For the three well case
pairs_3Wells = [(:par_P_Std, 0.0025), (:par_P_Mean, 0.025), (:par_PMax, 1000), (:par_AZ, 45), (:par_PMin, 200), (:par_PV, 10), (:par_Zmax, 0.045), (:par_Zmin, 0.015)]

obs_actions = [
    ObservationAction("Drill 3 Wells", 270/365, -9, product_uniform(pairs_3Wells)),
    ObservationAction("Water Compressibility", 14/365, -0.05, uniform(:par_C_WATER, 5e-5)),
    ObservationAction("Initial Reservoir Pressure", 21/365, -0.1, uniform(:par_P_INIT, 5)),
    ObservationAction("Fault Transmissibility Multiplier", 60/365, -2.0, uniform(:par_FTM, 0.015)),
    ObservationAction("Permeability Ratio", 30/365, -0.05, uniform(:par_KVKH, 0.1)),
    ObservationAction("Rock Compressibility", 30/365, -0.05, uniform(:par_C_ROCK, 5e-5)),
    ObservationAction("Rock Thermal Conductivity", 30/365, -0.07, uniform(:par_THCOND_RES, 0.5)),
    ObservationAction("Rock Heat Capacity", 30/365, -0.07, uniform(:par_HCAP_RES, 250)),
    ObservationAction("Temperature Gradient", 21/365, -0.1, uniform(:par_TempGrad, 0.001)),
    ObservationAction("Capex1", 30/365, -1.2, uniform(:par_CAPEXitem1, 1e-6)),
    ObservationAction("Capex2", 30/365, -1.2, uniform(:par_CAPEXitem2, 1e-6)),
    ObservationAction("Capex3", 60/365, -10, scenario_dependent_uniform(:par_CAPEXitem3, keys(scenario_csvs), 1e-6)),
    ObservationAction("Capex4", 60/365, -10, scenario_dependent_uniform(:par_CAPEXitem4, keys(scenario_csvs), 1e-6)),
    ObservationAction("Capex5", 30/365, -0.03, uniform(:par_CAPEXitem5, 1e-6)),
    ObservationAction("Capex6", 30/365, -0.02, uniform(:par_CAPEXitem6, 1e-6)),
    ObservationAction("OPEX Fixed Total", 30/365, -3.5, scenario_dependent_uniform(:par_OPEXfixedtotal, keys(scenario_csvs), 1e-6)),
    ObservationAction("OPEX Water", 30/365, -0.02, uniform(:par_UnitOPEXWater, 1e-6)),
    ObservationAction("OPEX Water Injection", 30/365, -0.02, uniform(:par_UnitOPEXWaterInj, 1e-6)),
    ObservationAction("OPEX Active Producers", 30/365, -0.01, uniform(:par_UnitOPEXActiveProducers, 1e-6)),
    ObservationAction("OPEX Active Water Injectors", 30/365, -0.01, uniform(:par_UnitOPEXActiveWaterInjectors, 1e-6)),
]

Nbins = [5, fill(2, length(obs_actions[2:end]))...]

pomdp, val, test = create_pomdp(scenario_csvs, geo_params, econ_params, obs_actions, Nbins, train_frac=0.8, val_frac=0.0, test_frac=0.2, rng=MersenneTwister(0))

# Solver for the expensive policies and save
sarsop_policy = solve(SARSOPSolver(), pomdp)
JLD2.@save "sarsop.jld2" sarsop_policy
sarsop_policy = JLD2.load("sarsop.jld2")["sarsop_policy"]

# Define the rest of the policies
min_particles = 50
best_current_option = BestCurrentOption(pomdp)
all = EnsureParticleCount(PlaybackPolicy(obs_actions, best_current_option), best_current_option, min_particles)
random = EnsureParticleCount(RandomPolicy(;pomdp), best_current_option, min_particles)
sarsop = EnsureParticleCount(sarsop_policy, best_current_option, min_particles)

# Evaluate and save
best_option_results = eval(pomdp, best_current_option, test)
all_results = eval(pomdp, all, test)
random_results = eval(pomdp, random, test)
sarsop_results = eval(pomdp, sarsop, test)

JLD2.@save "results.jld2" best_option_results all_results random_results sarsop_results


using Plots
stateindices = collect(1:length(states(pomdp)))
histogram(stateindices, weights=bs[1].b, bins=stateindices)
histogram(stateindices, weights=bs[2].b, bins=stateindices)
histogram(stateindices, weights=bs[end-1].b, bins=stateindices)
histogram(stateindices, weights=history[end].bp.b, bins=stateindices)
length(belief_hist(history))
length(history)
os = collect(observation_hist(history))
os[1]


as = collect(action_hist(history))
as[1]

sp, o, r = gen(pomdp, s0, as[1])

observation(pomdp, sp, as[1])

sp

pomdp.obs_dists[(sp, as[1])]

pomdp.obs_dists



# TODO: figure out how to display to
# -> Evaluate a single policy
# -> Compare policies for a fixed set of training states
# -> Compare policies for varying the training states
