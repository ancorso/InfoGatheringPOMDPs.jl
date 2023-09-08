include("src/pomdp.jl")
include("src/policies.jl")
include("src/metrics.jl")
include("src/plotting.jl")

using POMDPs
using POMDPTools
using NativeSARSOP
using JLD2
using Random
using DataStructures

# Define the save directory. This will through an error if the savedir already exists
savedir="results/test2/"
mkdir(savedir)

# Define the scenarios and corresponding paths to CSV files
scenario_csvs = OrderedDict(
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

# Define the set of geological and economic parameters so geo models can be separated
geo_params = ["par_P_Std", "par_P_Mean", "par_PMax", "par_AZ", "par_PMin", "par_PV", "par_SEED", "par_Zmax", "par_Zmin", "par_SEEDTrend", "par_C_WATER", "par_P_INIT", "par_FTM", "par_KVKH", "par_C_ROCK", "par_THCOND_RES", "par_HCAP_RES", "par_TempGrad"]
econ_params = ["par_CAPEXitem1", "par_CAPEXitem2", "par_CAPEXitem5", "par_CAPEXitem6", "par_UnitOPEXWater", "par_UnitOPEXWaterInj", "par_UnitOPEXActiveProducers", "par_UnitOPEXActiveWaterInjectors"]

# Define which parameters are affected for the three-slim-well case
pairs_3Wells = [(:par_P_Std, 0.0025), (:par_P_Mean, 0.025), (:par_PMax, 1000), (:par_AZ, 45), (:par_PMin, 200), (:par_PV, 10), (:par_Zmax, 0.045), (:par_Zmin, 0.015)]

# Define the observation actions
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

# Set the number of observation bins for each action
Nbins = [5, fill(2, length(obs_actions[2:end]))...]

# Set the discount factor
discount_factor = 0.99

# Create the pomdp, the validation and teest sets
pomdp, val, test = create_pomdp(scenario_csvs, geo_params, econ_params, obs_actions, Nbins, train_frac=0.8, val_frac=0.0, test_frac=0.2, rng=MersenneTwister(0), discount=discount_factor)

# Solve the POMDP using SARSOP
sarsop_policy = solve(SARSOPSolver(), pomdp) #<---- Uncomment this line to solve the policy
JLD2.@save joinpath(savedir, "sarsop.jld2") sarsop_policy

# Alternatively load the policy from file by uncommenting the following line
# sarsop_policy = JLD2.load(joinpath(savedir, "sarsop.jld2"))["sarsop_policy"] #<---- Uncomment this line to load the policy from file

# Define the rest of the policies
min_particles = 50
best_current_option = BestCurrentOption(pomdp)
all = EnsureParticleCount(PlaybackPolicy(obs_actions, best_current_option), best_current_option, min_particles)
random = EnsureParticleCount(RandomPolicy(;pomdp), best_current_option, min_particles)
onestepgreedy = EnsureParticleCount(OneStepGreedyPolicy(;pomdp), best_current_option, min_particles)
sarsop = EnsureParticleCount(sarsop_policy, best_current_option, min_particles)

# Evaluate the policies on the test set 
best_option_results = eval(pomdp, best_current_option, test) # <---- Uncomment this line to evaluate the policies
all_results = eval(pomdp, all, test) # <---- Uncomment this line to evaluate the policies
random_results = eval(pomdp, random, test) # <---- Uncomment this line to evaluate the policies
onestepgreedy_results = eval(pomdp, onestepgreedy, test) # <---- Uncomment this line to evaluate the policies
sarsop_results = eval(pomdp, sarsop, test) # <---- Uncomment this line to evaluate the policies

# Save the results
policy_results = [best_option_results, all_results, random_results, onestepgreedy_results, sarsop_results]
policy_names = ["Best Option Policy", "Observe-All Policy", "Random Policy", "One-Step Greedy Policy", "SARSOP Policy"]
JLD2.@save joinpath(savedir, "results.jld2") policy_results policy_names

# Alternatively, load from file by uncommenting the following lines
# results_file = JLD2.load(joinpath(savedir, "results.jld2")) # <---- Uncomment this line to load the results from file
# policy_results = results_file["policy_results"] # <---- Uncomment this line to load the results from file
# policy_names = results_file["policy_names"] # <---- Uncomment this line to load the results from file

# Save the results
for (policy_result, policy_name) in zip(policy_results, policy_names)
    p = policy_results_summary(pomdp, policy_result, policy_name)
    savefig(p, joinpath(savedir, policy_name * ".pdf"))
end
p = policy_comparison_summary(policy_results, policy_names)
savefig(p, joinpath(savedir, "policy_comparison.pdf"))


###########################################################################
# This section is to investigate the number of geological models needed   #
###########################################################################

# Create an array of pomdps, each with a different number of states
n_geologies = [5, 10, 20, 50, 100, 200]
fracs = n_geologies ./ 250
pomdps, val, test = create_pomdps_with_different_training_fractions(fracs, scenario_csvs, geo_params, econ_params, obs_actions, Nbins, val_frac=0.0, test_frac=0.2, rng=MersenneTwister(0), discount=discount_factor)

# Define which policies we want to use to compare the results
policies = [
    (pomdp) -> BestCurrentOption(pomdp),
    (pomdp) -> EnsureParticleCount(PlaybackPolicy(obs_actions, BestCurrentOption(pomdp)), BestCurrentOption(pomdp), min_particles),
    (pomdp) -> EnsureParticleCount(OneStepGreedyPolicy(;pomdp), BestCurrentOption(pomdp), min_particles),
    (pomdp) -> EnsureParticleCount(solve(SARSOPSolver(), pomdp), BestCurrentOption(pomdp), min_particles),
]
policy_names = ["Best Option Policy", "Observe-All Policy", "One-Step Greedy Policy", "SARSOP Policy"]

# Solve the policies and evaluate the results #<---- Uncomment the below lines to solve and eval the policies
results = Dict()
for (polfn, pol_name) in zip(policies, policy_names)
    println("Solving/Evaulating for policy: ", pol_name)
    results[pol_name] = Dict(:Ngeologies => [], :results =>[])
    for (Ngeology, pomdp) in zip(n_geologies, pomdps)
        println("Solving/Evaulating for POMDP with Ngeologies= ", Ngeology)
        push!(results[pol_name][:Ngeologies], Ngeology)
        push!(results[pol_name][:results], eval(pomdp, polfn(pomdp), test))
    end
end
JLD2.@save joinpath(savedir, "Nstates_results.jld2") results

# Alternatively, load from file by uncommenting the following lines
# results = JLD2.load(joinpath(savedir, "Nstates_results.jld2"))["results"] # <---- Uncomment this line to load the results from file

train_states_comparison_summary(results)
savefig(joinpath(savedir, "train_states_comparison.pdf"))
