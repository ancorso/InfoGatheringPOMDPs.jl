POMDPs.action(p::T, b; i) where T<:Policy = action(p, b)

struct BestCurrentOption <: Policy
    pomdp::InfoGatheringPOMDP
end

function POMDPs.action(p::BestCurrentOption, b::DiscreteBelief)
    action_values = [sum([b*reward(p.pomdp, s, a) for (b, s) in zip(b.b, p.pomdp.states)]) for a in p.pomdp.terminal_actions]
    return p.pomdp.terminal_actions[argmax(action_values)]
end

@with_kw struct EnsureParticleCount <: Policy
    policy::Policy
    final_action_policy::Policy
    min_particle_count::Int = 50
end

function POMDPs.action(p::EnsureParticleCount, b::DiscreteBelief; i=nothing)
    if sum(b.b .> 0) <= p.min_particle_count
        return action(p.final_action_policy, b; i)
    else
        return action(p.policy, b; i)
    end
end

struct FixedPolicy <: Policy
    actions::Vector
    backup_policy::Policy 
    FixedPolicy(actions, backup_policy = FunctionPolicy((b)->error("No action defined for this policy."))) = new(actions, backup_policy)
end

function POMDPs.action(p::FixedPolicy, b; i=nothing)
    if i > length(p.actions)
        a = action(p.backup_policy, b; i)
    else
        a = p.actions[i]
    end
    return a
end

@with_kw struct RandPolicy <: Policy
    prob_terminal::Float64 = 0.1
    pomdp::InfoGatheringPOMDP
    best_current_option::BestCurrentOption = BestCurrentOption(pomdp)
end

function POMDPs.action(p::RandPolicy, b)
    if rand() < p.prob_terminal
        return action(p.best_current_option, b)
    else
        return rand(setdiff(actions(p.pomdp), p.pomdp.terminal_actions))
    end
end

@with_kw struct OneStepGreedyPolicy <: Policy
    pomdp::InfoGatheringPOMDP
end

function lookahead(𝒫, U, b, a, up)
    r = sum(reward(𝒫, s, a)*b.b[i] for (i,s) in enumerate(states(𝒫))) 
    Posa(o,s,a) = sum(obs_weight(𝒫, s, a, s′, o)*ps′ for (s′, ps′) in transition(𝒫, s, a)) 
    Poba(o,b,a) = sum(b.b[i]*Posa(o,s,a) for (i,s) in enumerate(states(𝒫)))
    return r + discount(𝒫, a)*sum([Poba(o,b,a)*U(update(up, b, a, o).b) for o in observations(𝒫, a)], init=0) 
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
