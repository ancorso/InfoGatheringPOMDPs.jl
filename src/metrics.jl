function observation_cost(history; kwargs...)
    if length(history) == 1
        return 0
    end
    return sum(h.r for h in history[1:end-1])
end

function number_observed(history; kwargs...)
    return length(history) - 1
end

function correct_scenario(history; pomdp, verbose=false)
    true_state = history[1].s
    scenario = history[end].a


    returns = [reward(pomdp, true_state, a) for a in pomdp.terminal_actions]
    correct_scenario = pomdp.terminal_actions[argmax(returns)]

    if verbose
        println("Chosen scenario: ", scenario)
        println("best scenario: ", correct_scenario)
    end
    return scenario == correct_scenario
end


function correct_gonogo(history; pomdp)
    true_state = history[1].s
    scenario = history[end].a

    returns = [reward(pomdp, true_state, a) for a in pomdp.terminal_actions]
    correct_scenario = pomdp.terminal_actions[argmax(returns)]

    if correct_scenario == :abandon
        return scenario == :abandon
    else
        if scenario == :abandon
            return false
        end
        return true_state[scenario] >= 0
    end
end


function PES(history; pomdp)
    last_belief = history[end].b
    final_action = history[end].a

    return sum([b*(reward(pomdp, s, final_action) > 0) for (b, s) in zip(last_belief.b, states(pomdp))])
end

function expected_loss(history; pomdp)
    last_belief = history[end].b
    final_action = history[end].a

    return sum([b*(reward(pomdp, s, final_action) < 0)*reward(pomdp, s, final_action) for (b, s) in zip(last_belief.b, states(pomdp))])
end

function eval_single(pomdp, policy, s, updater = DiscreteUpdater(pomdp), b0 = initialstate(pomdp))
    reset_policy!(policy)
    history = simulate(HistoryRecorder(), pomdp, policy, updater, b0, s)
    results = Dict()
    results[:reward] = discounted_reward(history)
    results[:obs_cost] = observation_cost(history)
    results[:num_obs] = number_observed(history)
    results[:correct_scenario] = correct_scenario(history; pomdp)
    results[:correct_gonogo] = correct_gonogo(history; pomdp)
    results[:PES] = PES(history; pomdp)
    results[:expected_loss] = expected_loss(history; pomdp)
    results[:actions] = collect(action_hist(history))
    results[:final_action] = history[end].a
    return results
end

function eval(pomdp, policy, states, updater = DiscreteUpdater(pomdp), b0 = initialstate(pomdp))
    results = []
    @showprogress for s in states
        push!(results, eval_single(pomdp, policy, s, updater, b0))
    end
    reset_policy!(policy)
    return Dict(k => [r[k] for r in results] for k in keys(results[1]))
end