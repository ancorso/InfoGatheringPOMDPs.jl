module InfoGatheringPOMDPs

using CSV
using DataFrames
using StatsBase
using Parameters
using Distributions
using Clustering
using POMDPs
using POMDPTools
using ProgressMeter
using Random
using LinearAlgebra
using Plots; default(fontfamily="Computer Modern", framestyle=:box)
using Plots.Measures


export DiscreteUp
include("discrete_updater.jl")

export uniform, product_uniform, scenario_dependent_uniform, 
       InfoGatheringPOMDP, ObservationAction, create_pomdps, 
       create_pomdps_with_different_training_fractions
include("pomdp.jl")

export BestCurrentOption, EnsureParticleCount, FixedPolicy, RandPolicy, OneStepGreedyPolicy
include("policies.jl")

export discounted_reward, observation_cost, number_observed, correct_scenario, correct_gonogo, PES, expected_loss, eval_single, eval_kfolds
include("metrics.jl")

export policy_results_summary, policy_comparison_summary, train_states_comparison_summary
include("plotting.jl")

end
