using InfoGatheringPOMDPs
using POMDPs
using POMDPTools
using NativeSARSOP
using JLD2
using CSV
using Random
using DataStructures
using Plots
default(framestyle = :box,  color_palette=:seaborn_deep6, fontfamily="Computer Modern")

# Define the save directory. This will through an error if the savedir already exists
savedir="./results/v5/"
try mkdir(savedir) catch end

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

# Plot the scenario returns
p = scenario_returns(scenario_csvs, geo_params, econ_params)
savefig(p,  savefig(p, joinpath(savedir, "scenario_returns.pdf")))

# Parameter descriptions
var_description = OrderedDict( #TODO: check these
    :par_P_Std => "Porosity Std.",
    :par_P_Mean => "Porosity Mean",
    :par_PMax => "PMax",
    :par_PMin => "PMin",
    :par_AZ => "Variogram Azimuth",
    :par_PV => "PV",
    :par_Zmax => "Zmax",
    :par_Zmin => "Zmin",
    :par_C_WATER => "Water Compressibility",
    :par_P_INIT => "Initial Reservoir Pressure",
    :par_FTM => "Fault Transmissibility Multiplier",
    :par_KVKH => "Permeability Ratio",
    :par_C_ROCK => "Rock Compressibility",
    :par_THCOND_RES => "Rock Thermal Conductivity",
    :par_HCAP_RES => "Rock Heat Capacity",
    :par_TempGrad => "Temperature Gradient",
    :par_CAPEXitem1 => "Capex1",
    :par_CAPEXitem2 => "Capex2",
    :par_CAPEXitem3 => "Capex3",
    :par_CAPEXitem4 => "Capex4",
    :par_CAPEXitem5 => "Capex5",
    :par_CAPEXitem6 => "Capex6",
    :par_OPEXfixedtotal => "OPEX Fixed Total",
    :par_UnitOPEXWater => "OPEX Water",
    :par_UnitOPEXWaterInj => "OPEX Water Injection",
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
    ObservationAction("Assess Capex1", 30/365, -1.2, uniform(:par_CAPEXitem1, 1e-6)),
    ObservationAction("Assess Capex2", 30/365, -1.2, uniform(:par_CAPEXitem2, 1e-6)),
    ObservationAction("Assess Capex3", 60/365, -10, scenario_dependent_uniform(:par_CAPEXitem3, keys(scenario_csvs), 1e-6)),
    ObservationAction("Assess Capex4", 60/365, -10, scenario_dependent_uniform(:par_CAPEXitem4, keys(scenario_csvs), 1e-6)),
    ObservationAction("Assess Capex5", 30/365, -0.03, uniform(:par_CAPEXitem5, 1e-6)),
    ObservationAction("Assess Capex6", 30/365, -0.02, uniform(:par_CAPEXitem6, 1e-6)),
    ObservationAction("Assess OPEX Fixed Total", 30/365, -3.5, scenario_dependent_uniform(:par_OPEXfixedtotal, keys(scenario_csvs), 1e-6)),
    ObservationAction("Assess OPEX Water", 30/365, -0.02, uniform(:par_UnitOPEXWater, 1e-6)),
    ObservationAction("Assess OPEX Water Injection", 30/365, -0.02, uniform(:par_UnitOPEXWaterInj, 1e-6)),
    ObservationAction("Assess OPEX Active Producers", 30/365, -0.01, uniform(:par_UnitOPEXActiveProducers, 1e-6)),
    ObservationAction("Assess OPEX Active Water Injectors", 30/365, -0.01, uniform(:par_UnitOPEXActiveWaterInjectors, 1e-6)),
]

# Set the number of observation bins for each action
Nbins = [5, fill(2, length(obs_actions[2:end]))...]

# Set the discount factor
discount_factor = 0.94 # Annual discount factor

# Create the pomdp, the validation and teest sets
pomdps, test_sets = create_pomdps(
    scenario_csvs, 
    geo_params, 
    econ_params, 
    obs_actions, 
    Nbins, 
    rng=MersenneTwister(0), 
    discount=discount_factor
)

# Writeout the table of actions
generate_action_table(pomdps[1], var_description)

# Define the rest of the policies
min_particles = 50
scen7_pol(pomdp) = FixedPolicy([Symbol("Option 7")])
scen11_pol(pomdp) = FixedPolicy([Symbol("Option 10")])
scen13_pol(pomdp) = FixedPolicy([Symbol("Option 11")])
all_policy_geo(pomdp) = EnsureParticleCount(FixedPolicy(obs_actions, BestCurrentOption(pomdp)), BestCurrentOption(pomdp), min_particles)
all_policy_econ(pomdp) = EnsureParticleCount(FixedPolicy(reverse(obs_actions), BestCurrentOption(pomdp)), BestCurrentOption(pomdp), min_particles)
random_policy_10(pomdp) = EnsureParticleCount(RandPolicy(;pomdp, prob_terminal=0.1), BestCurrentOption(pomdp), min_particles)
random_policy_4(pomdp) = EnsureParticleCount(RandPolicy(;pomdp, prob_terminal=0.25), BestCurrentOption(pomdp), min_particles)
random_policy_2(pomdp) = EnsureParticleCount(RandPolicy(;pomdp, prob_terminal=0.5), BestCurrentOption(pomdp), min_particles)
# onestepgreedy_policy(pomdp) = EnsureParticleCount(OneStepGreedyPolicy(;pomdp), BestCurrentOption(pomdp), min_particles)
sarsop_policy(pomdp) = EnsureParticleCount(solve(SARSOPSolver(), pomdp), BestCurrentOption(pomdp), min_particles)

# combine policies into a list
policies = [scen7_pol, scen11_pol, scen13_pol, all_policy_geo, all_policy_econ, random_policy_10, random_policy_4, random_policy_2, sarsop_policy] # onestepgreedy_policy
policy_names = ["Scenario 7", "Scenario 11", "Scenario 13", "Observe-All Policy (Geo)", "Observe-All Policy (Econ)", "Random Policy (10)", "Random Policy (4)", "Random Policy (2)", "SARSOP Policy"] # "One-Step Greedy Policy"

# Evaluate the policies on the test set 
policy_results = [] # <---- Uncomment this block to evaluate the policies
for (policy, policy_name) in zip(policies, policy_names)
    println("Evaluating policy: ", policy_name, "...")
    push!(policy_results, eval_kfolds(pomdps, policy, test_sets))
end

# Save the results
JLD2.@save joinpath(savedir, "results.jld2") policy_results policy_names

# Alternatively, load from file by uncommenting the following lines
# results_file = JLD2.load(joinpath(savedir, "results.jld2")) # <---- Uncomment this line to load the results from file
# policy_results = results_file["policy_results"] # <---- Uncomment this line to load the results from file
# policy_names = results_file["policy_names"] # <---- Uncomment this line to load the results from file

# Plot the results
for (policy_result, policy_name) in zip(policy_results[5:end], policy_names[5:end])
    # Print out all of the policy metrics (both combined and some individual)
    pobs, pdev, pall = policy_results_summary(pomdps[1], policy_result, policy_name)
    try
        savefig(pall, joinpath(savedir, policy_name * "_summary.pdf"))
        savefig(pobs, joinpath(savedir, policy_name * "_observations.pdf"))
        # savefig(pobs, joinpath(savedir, policy_name * "_observations.tex"))
        savefig(pdev, joinpath(savedir, policy_name * "_development_selections.pdf"))
        # savefig(pdev, joinpath(savedir, policy_name * "_development_selections.tex"))

        # Plot the sankey diagram that shows the abandon, execute, observe flow
        p = policy_sankey_diagram(pomdps[1], policy_result, policy_name)
        savefig(p, joinpath(savedir, policy_name * "_sankey.pdf"))

        # Similar information to the sankey diagram but also includes expected regret
        df = trajectory_regret_metrics(pomdps[1], policy_result)
        CSV.write(joinpath(savedir,  policy_name * "_trajectory_regret_metrics.csv"), df)
    catch
        println("Error plotting for policy: ", policy_name)
    end
end

# Make direct comparisons across policies (figure and table)
p = policy_comparison_summary(policy_results, policy_names)
savefig(p, joinpath(savedir, "policy_comparison.pdf"))
policy_comparison_table(policy_results, policy_names)

# Compare just the PES CDFS across policies
p = pes_comparison(policy_results[[5,7,9]], policy_names[[5,7,9]])
vline!([policy_results[1][:PES][1]], label="Option 7", linestyle=:dash)
vline!([policy_results[2][:PES][1]], label="Option 10", linestyle=:dash)
vline!([policy_results[3][:PES][1]], label="Option 11", linestyle=:dash)
savefig(p, joinpath(savedir, "PES_comparison.pdf"))
# savefig(p, joinpath(savedir, "PES_comparison.tex"))

# Compare the expected loss across policies
p = expected_loss_comparison(policy_results[[5,7,9]], policy_names[[5,7,9]])
vline!([policy_results[1][:expected_loss][1]], label="Option 7", linestyle=:dash)
vline!([policy_results[2][:expected_loss][1]], label="Option 10", linestyle=:dash)
vline!([policy_results[3][:expected_loss][1]], label="Option 11", linestyle=:dash)
savefig(p, joinpath(savedir, "Expected_Loss_comparison.pdf"))
# savefig(p, joinpath(savedir, "Expected_Loss_comparison.tex"))

###########################################################################
# This section is to investigate the number of geological models needed   #
###########################################################################

# Create an array of pomdps, each with a different number of states
n_geologies = [5, 10, 20, 50, 100, 200]
fracs = n_geologies ./ 250
pomdps_per_geo, test_sets_per_geo = create_pomdps_with_different_training_fractions(fracs, scenario_csvs, geo_params, econ_params, obs_actions, Nbins, rng=MersenneTwister(0), discount=discount_factor)

# Solve the policies and evaluate the results #<---- Uncomment the below lines to solve and eval the policies
results = Dict()
for (policy, pol_name) in zip(policies, policy_names)
    results[pol_name] = Dict(:Ngeologies => [], :results =>[])
    for (Ngeology, pomdps, test_sets) in zip(n_geologies, pomdps_per_geo, test_sets_per_geo)
        println("Solving and evaluating for policy", pol_name, " with Ngeologies= ", Ngeology)
        push!(results[pol_name][:Ngeologies], Ngeology)
        push!(results[pol_name][:results], eval_kfolds(pomdps, policy, test_sets))
    end
end
JLD2.@save joinpath(savedir, "Nstates_results.jld2") results

# Alternatively, load from file by uncommenting the following lines
# results = JLD2.load(joinpath(savedir, "Nstates_results.jld2"))["results"] # <---- Uncomment this line to load the results from file

# Show how all metrics across all policies vary with number of subsurface realizations
train_states_comparison_summary(results)
savefig(joinpath(savedir, "subsurfaces_realizations_comparison.pdf"))

# Print the same for just the reward for just one policy
reward_vs_ngeolgies(results["SARSOP Policy"], "SARSOP Policy")
savefig(joinpath(savedir, "SARSOP_subsurface_realizations.pdf")) 
# savefig(joinpath(savedir, "SARSOP_subsurface_realizations.tex")) 