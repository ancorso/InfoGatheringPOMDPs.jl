using POMDPs
using LinearAlgebra

reset_policy!(policy) = nothing

struct BestCurrentOption <: Policy
    pomdp::InfoGatheringPOMDP
end

function POMDPs.action(p::BestCurrentOption, b::DiscreteBelief)
    action_values = [sum([b*reward(p.pomdp, s, a) for (b, s) in zip(b.b, pomdp.states)]) for a in pomdp.terminal_actions]
    return pomdp.terminal_actions[argmax(action_values)]
end

@with_kw struct EnsureParticleCount <: Policy
    policy::Policy
    final_action_policy::Policy
    min_particle_count::Int = 50
end

function reset_policy!(policy::EnsureParticleCount)
    reset_policy!(policy.policy)
    reset_policy!(policy.final_action_policy)
end

function POMDPs.action(p::EnsureParticleCount, b::DiscreteBelief)
    if sum(b.b .> 0) <= p.min_particle_count
        return action(p.final_action_policy, b)
    else
        return action(p.policy, b)
    end
end

@with_kw mutable struct PlaybackPolicy <: Policy
    actions::Vector
    backup_policy::Policy
    i::Int = 1
    PlaybackPolicy(actions, backup_policy) = new(actions, backup_policy, 1)
end

function reset_policy!(policy::PlaybackPolicy)
    policy.i = 1
    reset_policy!(policy.backup_policy)
end

function POMDPs.action(p::PlaybackPolicy, b)
    if p.i > length(p.actions)
        a = action(p.backup_policy, b)
    else
        a = p.actions[p.i]
        p.i += 1
    end
    return a
end


@with_kw struct RandomPolicy <: Policy
    pomdp::InfoGatheringPOMDP
    best_current_option::BestCurrentOption = BestCurrentOption(pomdp)
end

function POMDPs.action(p::RandomPolicy, b)
    a = rand(actions(p.pomdp))
    if a in pomdp.terminal_actions
        return action(p.best_current_option, b)
    else
        return a
    end
end

@with_kw struct OneStepGreedyPolicy <: Policy
    pomdp::InfoGatheringPOMDP
end

function lookahead(𝒫, U, b, a, up)
    r = sum(reward(𝒫, s, a)*b.b[i] for (i,s) in enumerate(states(𝒫))) 
    Posa(o,s,a) = sum(obs_weight(𝒫, s, a, s′, o)*ps′ for (s′, ps′) in transition(𝒫, s, a)) 
    Poba(o,b,a) = sum(b.b[i]*Posa(o,s,a) for (i,s) in enumerate(states(𝒫)))
    return r + discount(𝒫)*sum([Poba(o,b,a)*U(update(up, b, a, o).b) for o in observations(𝒫, a)], init=0) 
end 

function greedy(𝒫, U, b) 
    As = actions(𝒫)
    up = DiscreteUp(𝒫)
    u, a = findmax(a->lookahead(𝒫, U, b, a, up), As) 
    return (a=As[a], u=u) 
end 

function greedy(π, b) 
    U(b) = utility(π, b) 
    return greedy(π.pomdp, U, b) 
end

function utility(π::OneStepGreedyPolicy, b)
    return maximum([b ⋅ [reward(π.pomdp, s, a) for s in states(π.pomdp)] for a in π.pomdp.terminal_actions])
end

POMDPs.action(π::OneStepGreedyPolicy, b::DiscreteBelief) = greedy(π, b).a
