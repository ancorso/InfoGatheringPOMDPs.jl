# Pulling this in from POMDPs.jl because the error the default version generates reports the whole belief
# which was very slow when handling the error in my custom simulator

"""
    DiscreteUp

An updater type to update discrete belief using the discrete Bayesian filter.

# Constructor
    DiscreteUp(pomdp::POMDP)

# Fields
- `pomdp <: POMDP`
"""
mutable struct DiscreteUp{P<:POMDP} <: Updater
    pomdp::P
end

uniform_belief(up::DiscreteUp) = uniform_belief(up.pomdp)

function POMDPs.initialize_belief(bu::DiscreteUp, dist::Any)
    state_list = ordered_states(bu.pomdp)
    ns = length(state_list)
    b = zeros(ns)
    belief = DiscreteBelief(bu.pomdp, state_list, b)
    for s in support(dist)
        sidx = stateindex(bu.pomdp, s)
        belief.b[sidx] = pdf(dist, s)
    end
    return belief
end

function POMDPs.update(bu::DiscreteUp, b::DiscreteBelief, a, o)
    pomdp = bu.pomdp
    state_space = b.state_list
    bp = zeros(length(state_space))

    for (si, s) in enumerate(state_space)

        if pdf(b, s) > 0.0
            td = transition(pomdp, s, a)

            for (sp, tp) in weighted_iterator(td)
                spi = stateindex(pomdp, sp)
                op = obs_weight(pomdp, s, a, sp, o) # shortcut for observation probability from POMDPModelTools

                bp[spi] += op * tp * b.b[si]
            end
        end
    end

    bp_sum = sum(bp)

    if bp_sum == 0.0
        error("Failed discrete belief update: new probabilities sum to zero.")
    end

    # Normalize
    bp ./= bp_sum

    return DiscreteBelief(pomdp, b.state_list, bp)
end

POMDPs.update(bu::DiscreteUp, b::Any, a, o) = update(bu, initialize_belief(bu, b), a, o)