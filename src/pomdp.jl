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

@with_kw struct ObservationAction
    name = ""
    time = 1
    cost = 0
    obs_dist = (s) -> nothing
end

## The following functions are helpful for defining the observation distributions
# Use this one when there is a one-to-one mapping between observation and the state parameter it observes
function uniform(symbol, σ)
    return (s) -> Dict{Symbol, Distribution}(symbol => Distributions.Uniform(s[symbol]-σ, s[symbol]+σ))
end

# Use this one when one action observes multiple different state parameters
function product_uniform(symbol_σ_pairs)
    return (s) -> begin
        dists = Dict{Symbol, Distribution}()
        for (sym, σ) in symbol_σ_pairs
            dists[sym] = Distributions.Uniform(s[sym]-σ, s[sym]+σ)
        end
        return dists
    end
end

# Use this one when you have a parameter that is scenario-dependent
function scenario_dependent_uniform(symbol, scenarios, σ)
    return (s) -> begin
        dists = Dict{Symbol, Distribution}()
        for scen in scenarios
            sym = Symbol("$(symbol)_$(scen)")
            dists[sym] = Distributions.Uniform(s[sym]-σ, s[sym]+σ)
        end
        return dists
    end
end

# Be able to sample a dictionary of symbol => distribution
Base.rand(rng::AbstractRNG, d::Dict{Symbol, Distribution}) = Dict(k => rand(rng, v) for (k, v) in d)

# Function to parse a series of CSVs that represent different scenarios
function parseCSV(
    scenario_csvs,          # Dictionary mapping scenarios to filenames of the CSVs
    geo_params,             # Vector of symbols that represent the geological parameters
    econ_params,            # Vector of symbols that represent the economic parameters
	return_symbol = :NPV,  		 # The symbol that represents the value of the state/scenario
    )

    # Load the CSVs into dataframes
    Nscenarios = length(scenario_csvs)
    dfs = Dict(k => CSV.read(s, DataFrame) for (k, s) in scenario_csvs)

    # Get the column names that correspond to state variables
    columns = names(first(dfs)[2])
    state_symbols = Symbol.(columns[contains.(columns, "par_")])

    # Sanity check that all of the state data is exactly the same
    # This should notify about symbols that are scenario-dependent
    scenario_dependent_symbols = []
    for s in state_symbols
        if any([!(d1[!, s] == d2[!, s]) for (_, d1) in dfs for (_, d2) in dfs])
            println("Notice: $s is not the same across all scenarios, making scenario-dependent state")
            push!(scenario_dependent_symbols, s)
        end
    end

    shared_syms = setdiff(state_symbols, scenario_dependent_symbols)
	states = Dict{Symbol, Float64}[]
	sindices = Dict()

    # for keep track of the unique geological and economic realizations
    geos = Dict()
    econs = Dict()

	# Loop over the data from each scenario
	for (scenario, data) in dfs
        # Loop over each row (sample)
		for i=1:length(data[!, return_symbol])
            # Initialize the state with the values from the current data and get the index
			s = Dict(s => data[i, s] for s in shared_syms)
			if !haskey(sindices, s)
				push!(states, s)
				sindices[deepcopy(s)] = length(states)
			end
			sindex = sindices[s]

            # Keep track of the unique geological realizations
            geo = Dict(s => data[i, s] for s in geo_params)
            if !haskey(geos, geo)
                geos[deepcopy(geo)] = length(geos)
            end
            geoindex = geos[geo]
            states[sindex][:GeoID] = geoindex

            # Keep track of the unique economic realizations
            econ = Dict(s => data[i, s] for s in econ_params)
            if !haskey(econs, econ)
                econs[deepcopy(econ)] = length(econs)
            end
            econindex = econs[econ]
            states[sindex][:EconID] = econindex

			# store the NPV for the current scenario
			ret = data[i, return_symbol]
			states[sindex][scenario] = ret

			# Store the scenario dependent symbols
			for sym in scenario_dependent_symbols
				key = Symbol("$(sym)_$(scenario)")
				val = data[i, sym]
				states[sindex][key] = val
			end
		end
	end
	return states
end

# Function to get train, val, test splits over a set of indices
function get_train_val_test_indices(indices; train_frac=0.8, val_frac=0.1, test_frac=0.1, rng=Random.GLOBAL_RNG)
    @assert sum([train_frac, val_frac, test_frac]) <= 1.0

    N = length(indices)
    Ntrain = floor(Int, N*train_frac)
    Nval = floor(Int, N*val_frac)
    Ntest = floor(Int, N*test_frac)
    shuffled_indices = shuffle(rng, indices)

    train_indices = shuffled_indices[1:Ntrain]
    test_indices = shuffled_indices[end-Ntest+1:end]
    val_indices = shuffled_indices[end-Ntest-Nval+1:end-Ntest]
    return train_indices, val_indices, test_indices
end

# Function to split states into train, val, test
function split_states(states; train_frac=0.8, val_frac=0.1, test_frac=0.1, rng=Random.GLOBAL_RNG)
    geo_indices = unique([s[:GeoID] for s in states])
    # econ_indices = unique([s[:EconID] for s in states])

    train_geo, val_geo, test_geo = get_train_val_test_indices(geo_indices; train_frac, val_frac, test_frac, rng)
    # train_econ, val_econ, test_econ = get_train_val_test_indices(econ_indices; train_frac, val_frac, test_frac)

    train = [s for s in states if s[:GeoID] in train_geo] # && s[:EconID] in train_econ]
    val = [s for s in states if s[:GeoID] in val_geo]# && s[:EconID] in val_econ]
    test = [s for s in states if s[:GeoID] in test_geo]# && s[:EconID] in test_econ]

    return train, val, test
end

# Determine the set of discrete observations
function get_discrete_observations(states, obs_actions, Nbins=fill(2,length(obs_actions)), Nsamples_per_bin=500; rng=Random.GLOBAL_RNG)
    discrete_obs = []
    for (a, Nbin) in zip(obs_actions, Nbins)
        obss = [rand(rng, a.obs_dist(rand(rng, states))) for _=1:Nsamples_per_bin*Nbin]
        X=hcat([collect(values(o)) for o in obss]...)
        res = kmeans(X, Nbin; rng)
        append!(discrete_obs, [Dict(zip(keys(obss[1]), res.centers[:, i])) for i in 1:Nbin])
    end
    discrete_obs
end

# Function for mapping continuous observations to discrete observations
function nearest_neighbor_mapping(o, discrete_obs)
    min_dist = Inf
    min_obs = nothing
    ks = keys(o)
    for obs in discrete_obs
        ks != keys(obs) && continue # Don't consider observation that aren't of the same thing
        dist = sum([(o[k] - obs[k])^2 for k in ks])
        if dist < min_dist
            min_dist = dist
            min_obs = obs
        end
    end
    return min_obs
end

function create_observation_distributions(states, actions, discrete_obs, Nsamples=fill(10, lenth(actions)); rng=Random.GLOBAL_RNG)
    obs_dists = Dict()
    @showprogress "Creating discrete observation distributions..." for s in states
        for (a, Nsample) in zip(actions, Nsamples)
            os = [nearest_neighbor_mapping(rand(rng, a.obs_dist(s)), discrete_obs) for _=1:Nsample]
            probs = [sum([o == o′ for o′ in os])/Nsample for o in discrete_obs]
            nonzero_indices = findall(!iszero, probs)
            obs_dists[(s,a)] = SparseCat(discrete_obs[nonzero_indices], probs[nonzero_indices])
        end
    end
    obs_dists
end

struct InfoGatheringPOMDP <: POMDP{Any, Any, Any}
    states                  # Vector of states
    state_map               # Map from state to index
    actions                 # Vector of actions
    action_map              # Map from action to index
    terminal_actions        # Vector of terminal actions
    observations            # Vector of observations
    obs_map                 # Map from observation to index
    obs_dists               # Map from (state, action) to observation distribution
    obs_abstraction         # Function for mapping continuous observations to discrete observations
    discount_factor         # Discount factor

    # Constructor
    function InfoGatheringPOMDP(states, obs_actions, scenarios, observations, obs_dists, obs_abstraction, discount_factor)
        # Add extra state, action, obs symbols for handling terminal states
        states = [:terminal, states...]
        terminal_actions = [:abandon, scenarios...]
        actions = [terminal_actions..., obs_actions...]
        observations = [:terminal, observations...]

        # Create the index maps for quickly looking up indices
		state_map = Dict(s => i for (i,s) in enumerate(states))
        action_map = Dict(a => i for (i,a) in enumerate(actions))
        obs_map = Dict(o => i for (i,o) in enumerate(observations))
        
        new(states, state_map, actions, action_map, terminal_actions, observations, obs_map, obs_dists, obs_abstraction, discount_factor)
    end
end

POMDPs.states(m::InfoGatheringPOMDP) = m.states
POMDPs.stateindex(m::InfoGatheringPOMDP, s) = m.state_map[s]
POMDPs.actions(m::InfoGatheringPOMDP) = m.actions
POMDPs.actionindex(m::InfoGatheringPOMDP, a) = m.action_map[a]
POMDPs.observations(m::InfoGatheringPOMDP) = m.observations
function POMDPs.observations(m::InfoGatheringPOMDP, a) 
    filter(x->x==a, m.observations)
end
POMDPs.obsindex(m::InfoGatheringPOMDP, o) = m.obs_map[o]
POMDPs.discount(m::InfoGatheringPOMDP) = m.discount_factor
POMDPs.isterminal(m::InfoGatheringPOMDP, s) = s == :terminal
POMDPs.initialstate(m::InfoGatheringPOMDP) = DiscreteBelief(m, fill(1/length(m.states), length(m.states)))

function POMDPs.transition(m::InfoGatheringPOMDP, s, a)
    if a in m.terminal_actions
		return SparseCat([:terminal], [1.0])
	else
		return SparseCat([s], [1.0])
	end
end

function POMDPs.observation(m::InfoGatheringPOMDP, a, sp)
    return get(m.obs_dists, (sp,a), SparseCat([:terminal], [1.0]))
end

function POMDPs.reward(m::InfoGatheringPOMDP, s, a)
    if s == :terminal
        return 0.0
    elseif a == :abandon
        return 0.0
    elseif a in m.terminal_actions
        return s[a]
    else
        return a.cost
    end
end

function POMDPs.gen(m::InfoGatheringPOMDP, s, a, rng)
    sp = rand(rng, transition(m, s, a))
    if sp == :terminal
        o = :terminal
        r = reward(m, s, a)
    else
        # When using the gen function, we don't rely on our precomputed discrete obs distributions
        true_obs = rand(rng, a.obs_dist(sp))    
        o = m.obs_abstraction(true_obs)
    end

    r = reward(m, s, a)
    return (;sp, o, r)
end

function create_pomdp(scenario_csvs, geo_params, econ_params, obs_actions, Nbins; Nsamples_per_bin=10, train_frac=0.8, val_frac=0.1, test_frac=0.1, discount=0.9, rng=Random.GLOBAL_RNG)
    # Parse all of the states from the csv files
    statevec = parseCSV(scenario_csvs, geo_params, econ_params)

    # Split the states into train, val, test
    train, val, test = split_states(statevec; train_frac, val_frac, test_frac, rng)

    # Discretize the observations
    discrete_obs = get_discrete_observations(train, obs_actions, Nbins; rng)

    # Create categorical observation distributions
    obs_dists = create_observation_distributions(train, obs_actions, discrete_obs, Nbins .* Nsamples_per_bin; rng)

    # Make the POMDP and return the val and test sets
    pomdp = InfoGatheringPOMDP(train, obs_actions, keys(scenario_csvs), discrete_obs, obs_dists, (o) -> nearest_neighbor_mapping(o, discrete_obs), discount)
    return pomdp, val, test
end

function create_pomdps_with_different_training_fractions(train_fracs, scenario_csvs, geo_params, econ_params, obs_actions, Nbins; Nsamples_per_bin=10, val_frac=0.1, test_frac=0.1, discount=0.9, rng=Random.GLOBAL_RNG)
    test = nothing
    val = nothing
    pomdps = []
    for train_frac in train_fracs
        pomdp, v, t = create_pomdp(scenario_csvs, geo_params, econ_params, obs_actions, Nbins; Nsamples_per_bin, train_frac, val_frac, test_frac, discount, rng)
        isnothing(val) && (val = v)
        isnothing(test) && (test = t)
        push!(pomdps, pomdp)
    end
    return pomdps, val, test
end

