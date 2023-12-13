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

# Define random seeds
fix_solve_and_eval_seed = true # Whether the seed is set before each policy gen and evaluation. Seed is the eval index + test set. It is threadsafe. 
pomdp_gen_seed = 0 # seed used to control the generation of the pomdps
split_by = :geo # Split the test set for ensuring unique :geo, :econ, or :both

# Can parse 1 command line argument for the split_by 
if length(ARGS) > 0 
    if ARGS[1] == "geo"
        split_by = :geo
    elseif ARGS[1] == "econ"
        split_by = :econ
    elseif ARGS[1] == "both"
        split_by = :both
    else
        error("Invalid argument: ", ARGS[1])
    end
end

# Define the save directory. This will through an error if the savedir already exists
savedir="./results/final_splitby_$(split_by)/"
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
savefig(p, joinpath(savedir, "npvs.pdf"))

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
    :par_CAPEXitem1 => "Capex Injection Well",
    :par_CAPEXitem2 => "Capex Production Well",
    :par_CAPEXitem3 => "Capex Surface Facilities",
    :par_CAPEXitem4 => "Capex Flowlines",
    :par_CAPEXitem5 => "Capex Production Pump",
    :par_CAPEXitem6 => "Capex Injection Pump",
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
    ObservationAction("Assess Capex Injection Well", 30/365, -1.2, uniform(:par_CAPEXitem1, 0.3)),
    ObservationAction("Assess Capex Production Well", 30/365, -1.2, uniform(:par_CAPEXitem2, 0.3)),
    ObservationAction("Assess Capex Surface Facilities", 60/365, -10, scenario_dependent_uniform(:par_CAPEXitem3, keys(scenario_csvs), 18.0)),
    ObservationAction("Assess Capex Flowlines", 60/365, -10, scenario_dependent_uniform(:par_CAPEXitem4, keys(scenario_csvs), 5.5)),
    ObservationAction("Assess Capex Production Pump", 30/365, -0.03, uniform(:par_CAPEXitem5, 0.01625)),
    ObservationAction("Assess Capex Injection Pump", 30/365, -0.02, uniform(:par_CAPEXitem6, 0.01)),
    ObservationAction("Assess OPEX Fixed Total", 30/365, -3.5, scenario_dependent_uniform(:par_OPEXfixedtotal, keys(scenario_csvs), 1.0)),
    ObservationAction("Assess OPEX Water", 30/365, -0.02, uniform(:par_UnitOPEXWater, 0.00975)),
    ObservationAction("Assess OPEX Water Injectors", 30/365, -0.02, uniform(:par_UnitOPEXWaterInj, 0.00975)),
    ObservationAction("Assess OPEX Active Producers", 30/365, -0.01, uniform(:par_UnitOPEXActiveProducers, 0.006)),
    ObservationAction("Assess OPEX Active Water Injectors", 30/365, -0.01, uniform(:par_UnitOPEXActiveWaterInjectors, 0.006)),
]

# Set the number of observation bins for each action
Nbins = [5, fill(2, length(obs_actions[2:end]))...]

# Set the discount factor
discount_factor = 0.90 # Annual discount factor

# Create the pomdp, the validation and teest sets
pomdps, test_sets = create_pomdps(
    scenario_csvs, 
    geo_params, 
    econ_params, 
    obs_actions, 
    Nbins, 
    rng=MersenneTwister(pomdp_gen_seed), # Set the pomdp random seed 
    discount=discount_factor,
    split_by=split_by
)

# Writeout the table of actions
generate_action_table(pomdps[1], var_description)

# Define the rest of the policies
min_particles = 50
option7_pol(pomdp) = FixedPolicy([Symbol("Option 7")])
option11_pol(pomdp) = FixedPolicy([Symbol("Option 10")])
option13_pol(pomdp) = FixedPolicy([Symbol("Option 11")])
all_policy_geo(pomdp) = EnsureParticleCount(FixedPolicy(obs_actions, BestCurrentOption(pomdp)), BestCurrentOption(pomdp), min_particles)
all_policy_econ(pomdp) = EnsureParticleCount(FixedPolicy(reverse(obs_actions), BestCurrentOption(pomdp)), BestCurrentOption(pomdp), min_particles)
random_policy_10(pomdp) = EnsureParticleCount(RandPolicy(;pomdp, prob_terminal=0.1), BestCurrentOption(pomdp), min_particles)
random_policy_4(pomdp) = EnsureParticleCount(RandPolicy(;pomdp, prob_terminal=0.25), BestCurrentOption(pomdp), min_particles)
random_policy_2(pomdp) = EnsureParticleCount(RandPolicy(;pomdp, prob_terminal=0.5), BestCurrentOption(pomdp), min_particles)
voi_policy(pomdp) = EnsureParticleCount(OneStepGreedyPolicy(;pomdp), BestCurrentOption(pomdp), min_particles)
sarsop_policy(pomdp) = EnsureParticleCount(solve(SARSOPSolver(), pomdp), BestCurrentOption(pomdp), min_particles)

# combine policies into a list
policies = [option7_pol, option11_pol, option13_pol, all_policy_geo, all_policy_econ, random_policy_10, random_policy_4, random_policy_2, sarsop_policy] # voi_policy
policy_names = ["Option 7", "Option 11", "Option 13", "All Data Policy (Geo First)", "All Data Policy (Econ First)", "Random Policy (Pstop=0.1)", "Random Policy (Pstop=0.25)", "Random Policy (Pstop=0.5)", "SARSOP Policy"] # "VOI Policy"

# Evaluate the policies on the test set 
policy_results = [] # <---- Uncomment this block to evaluate the policies
for (policy, policy_name) in zip(policies, policy_names)
    println("Evaluating policy: ", policy_name, "...")
    push!(policy_results, eval_kfolds(pomdps, policy, test_sets, fix_seed = fix_solve_and_eval_seed))
end

# Save the results
JLD2.@save joinpath(savedir, "results.jld2") policy_results policy_names

# Alternatively, load from file by uncommenting the following lines
# results_file = JLD2.load(joinpath(savedir, "results.jld2")) # <---- Uncomment this line to load the results from file
# policy_results = results_file["policy_results"] # <---- Uncomment this line to load the results from file
# policy_names = results_file["policy_names"] # <---- Uncomment this line to load the results from file

# Plot the results
for (policy_result, policy_name) in zip(policy_results, policy_names)
    # Print out all of the policy metrics (both combined and some individual)
    pobs, pdev, pall = policy_results_summary(pomdps[1], policy_result, policy_name)
    try
        savefig(pall, joinpath(savedir, policy_name * "_summary.pdf"))
        savefig(pobs, joinpath(savedir, policy_name * "_data_acq_actions.pdf"))
        # savefig(pobs, joinpath(savedir, policy_name * "_data_acq_actions.tex"))
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
fracs = [0.02, 0.04, 0.08, 0.16, 0.32, 0.5, 0.8]
alls = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]

if split_by == :geo
    geo_fracs = fracs
    econ_fracs = alls
elseif split_by == :econ
    geo_fracs = alls
    econ_fracs = fracs
elseif split_by == :both
    geo_fracs = [fracs..., alls...]
    econ_fracs = [alls..., fracs...]
end

zip_fracs = [(g,e) for (g, e) in zip(geo_fracs, econ_fracs)]
pomdps_per_geo, test_sets_per_geo = create_pomdps_with_different_training_fractions(zip_fracs, scenario_csvs, geo_params, econ_params, obs_actions, Nbins; rng=MersenneTwister(0), discount=discount_factor, split_by)

# Solve the policies and evaluate the results #<---- Uncomment the below lines to solve and eval the policies
results = Dict()
for (policy, pol_name) in zip(policies, policy_names)
    results[pol_name] = Dict(:geo_frac => [], :econ_frac => [], :results =>[])
    for (frac, pomdps, test_sets) in zip(zip_fracs, pomdps_per_geo, test_sets_per_geo)
        geo_frac, econ_frac = frac[1], frac[2]
        println("Solving and evaluating for policy", pol_name, " with geo_frac= ", geo_frac, " econ_frac= ", econ_frac)
        push!(results[pol_name][:geo_frac], geo_frac)
        push!(results[pol_name][:econ_frac], econ_frac)
        push!(results[pol_name][:results], eval_kfolds(pomdps, policy, test_sets, fix_seed = fix_solve_and_eval_seed))
    end
end
JLD2.@save joinpath(savedir, "Nsamples_results.jld2") results

# Alternatively, load from file by uncommenting the following lines
# results = JLD2.load(joinpath(savedir, "Nstates_results.jld2"))["results"] # <---- Uncomment this line to load the results from file

# Show how all metrics across all policies vary with number of subsurface realizations
train_states_comparison_summary(results)
savefig(joinpath(savedir, "subsurfaces_realizations_comparison.pdf"))

# Print the same for just the reward for just one policy
reward_vs_ngeolgies(results["SARSOP Policy"], "SARSOP Policy")
savefig(joinpath(savedir, "SARSOP_subsurface_realizations.pdf")) 
# savefig(joinpath(savedir, "SARSOP_subsurface_realizations.tex")) 