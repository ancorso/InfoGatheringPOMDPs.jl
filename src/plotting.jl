function initial_expected_loss(pomdp) 
    mean([reward(pomdp, s, a) * (reward(pomdp, s, a) < 0.0) for a in pomdp.terminal_actions for s in states(pomdp)])
end

function symbol_histogram(unique_syms, symarray; normalize = true, kwargs...)
    counts = [sum(symarray .== s) for s in unique_syms]
    if normalize==true
        norm = sum(counts)
    elseif normalize isa Real
        norm = normalize
    else
        @warn "Unknown normalization type: $normalize"
        norm = 1
    end
    syms = string.(unique_syms)
    bottom_margin = maximum(length.(syms))*mm
    plot(1:length(syms), counts ./ norm, seriestype=:bar, xrotation=60, xticks=(1:length(syms), syms), ylabel="Frequency", label="", normalize=:pdf; bottom_margin, kwargs...)
end

# Plots a histogram of the action distribution for the aith action.
function action_distribution(pomdp, ai, results, policy_name)
    all_actions = actions(pomdp)
    obs_actions = [a.name for a in all_actions if a isa ObservationAction]
    acts = [as[ai].name for as in results[:actions]]
    symbol_histogram(obs_actions, acts, size=(600,600), title = "Action $(ai) for $(policy_name)", dpi=300)
end

function combine_actions(actions)
    combined_actions = []
    for as in actions
        actionset = []
        for a in as
            if a isa Symbol
                push!(actionset, string(a))
            elseif a isa ObservationAction
                push!(actionset, a.name)
            else
                error("Unknown action type: ", typeof(a))
            end
        end
        push!(combined_actions, unique(actionset)...)
    end
    return combined_actions
end

function histogram_with_cdf(data, bins=nothing; p=plot(), one_minus_cdf=true, ignore_hist=false, kwargs...)
    if !isnothing(bins)
        h = fit(Histogram, data, bins, closed=:left)
    else
        h = fit(Histogram, data, nbins=10, closed=:left)
    end
    h = StatsBase.normalize(h, mode=:probability)
    edges = h.edges[1]
    counts = h.weights
    cdf_values = cumsum(counts) ./ sum(counts)
    if ignore_hist
        plot!(p, ylabel="Probability", ylims=(0,1.1), legend=:topright, margin=5mm; kwargs...)
    else
        plot!(p, h, seriestype=:bar, ylabel="Probability", ylims=(0,1.1), label="Probability", legend=:topright, margin=5mm; kwargs...)
    end
    if one_minus_cdf
        plot!(edges[2:end], 1.0 .- cdf_values, label="1-CDF", lw=2; kwargs...)
    else
        plot!(edges[2:end], cdf_values, label="CDF", lw=2; kwargs...)
    end
    return p
end

function policy_results_summary(pomdp, results, policy_name)
    p_final_action = symbol_histogram(pomdp.terminal_actions, results[:final_action], xlabel = "Final Action", title="$policy_name  - Final Action")
    
    p_pes = histogram_with_cdf(results[:PES], 0:0.1:1, xlims=(0,1), xlabel="PES", title="$policy_name - PES")
    
    rexp = initial_expected_loss(pomdp)
    p_expected_loss = histogram_with_cdf(results[:expected_loss], range(rexp, 0, length=100), xlims=(rexp,0), xlabel="Expected Loss", title="$policy_name - Expected Loss", one_minus_cdf=false)

    all_actions = actions(pomdp)
    obs_actions = [a.name for a in all_actions if a isa ObservationAction]
    Nsamps = length(results[:final_action])
    pobs = symbol_histogram(obs_actions, combine_actions(results[:actions]), title="$policy_name  - Data Acquisition Actions", normalize=Nsamps)

    first_actions = unique([traj[1] isa Symbol ? string(traj[1]) : traj[1].name for traj in results[:actions]])
    first_actions_str = join(first_actions, "\n")
    regret = mean(results[:obs_cost][results[:final_action] .== :abandon])

    p_data = plot(legend=false, grid=false, axis=false, ticks=nothing, border=:none, size=(1000,400))
    annotate!(-0.1,0.9,("Mean Discounted Reward: $(round(mean(results[:reward]), digits=2))", :left, 10))
    annotate!(-0.1,0.8,("Mean Data Acquisition Cost: $(round(mean(results[:obs_cost]), digits=2))", :left, 10))
    annotate!(-0.1,0.7,("Mean Data Acquisition Duration: $(round(mean(results[:obs_duration]), digits=2))", :left, 10))
    annotate!(-0.1,0.6,("Mean Regret: $(round(regret, digits=2))", :left, 10))
    annotate!(-0.1,0.5,("Mean Number of Data Acquisition Actions: $(round(mean(results[:num_obs]), digits=2))", :left, 10))
    annotate!(-0.1,0.4,("Mean Correct Scenario: $(round(mean(results[:correct_scenario]), digits=2))", :left, 10))
    annotate!(-0.1,0.3,("Mean Correct Go/NoGo: $(round(mean(results[:correct_gonogo]), digits=2))", :left, 10))
    annotate!(-0.1,0.3 - (0.1*length(first_actions)),("First action(s): $(first_actions_str)", :left, 10))

    pall = plot(p_final_action, pobs, p_data, p_pes, p_expected_loss, layout=(2,3), size=(1400,800), left_margin=5mm, right_margin=5mm)
    plot!(pobs, size=(600,600)), p_final_action, pall
end

function pes_comparison(policy_results, policy_names)
    p = plot()
    for (results, name) in zip(policy_results, policy_names)
        histogram_with_cdf(
            results[:PES], 
            0:0.1:1, 
            xlims=(0,1), 
            ylims=(0,1),
            p=p, 
            ignore_hist=true, 
            xlabel="Probability of Economic Success", 
            label=name, 
            legend=:bottomleft,
            lw=1)
    end
    p
end

function expected_loss_comparison(policy_results, policy_names)
    p = plot()
    for (results, name) in zip(policy_results, policy_names)
        histogram_with_cdf(
            results[:expected_loss], 
            range(-200, 0, length=100),
            ylims=(0,1),
            p=p, 
            ignore_hist=true, 
            xlabel="Expected Loss (M)", 
            label=name, 
            legend=:topleft,
            one_minus_cdf=false,
            lw=1)
    end
    p
end

function policy_sankey_diagram(pomdp, results, policy_name; max_length=10)
    # Turn it into a set of source and destination nodes
    src = []
    dst = []
    weights = []
    # node_labels = ["Walk Away", "Develop"]
    node_labels = ["", ""]
    node_colors = [:red, :green]
    action_sets = [[:abandon], setdiff(pomdp.terminal_actions, [:abandon])]
    Nterm = length(action_sets)
    max_traj_length = maximum(length.(results[:actions]))
    for i=1:max_length
        for (ai, a) in enumerate(action_sets)
            append!(src, i+Nterm)
            append!(dst, ai)
            if i<max_length
                append!(weights, sum([traj[i] in a for traj in results[:actions] if length(traj) >= i]))
            else
                total_Na = 0
                for j=i:max_traj_length
                    total_Na += sum([traj[j] in a for traj in results[:actions] if length(traj) >= j])
                end
                append!(weights, total_Na)
            end
        end
        push!(node_labels, "$i")
        push!(node_colors, :gray)
        if i < max_length
            append!(src, i+Nterm)
            append!(dst, i+Nterm+1)
            append!(weights, sum([traj[i] isa ObservationAction for traj in results[:actions] if length(traj) >= i]))
        end
    end

    # plot the results
    sankey(src, dst, weights, size=(500,450);label_size=8, node_colors, node_labels, label_position=:top, compact=true, edge_color=:gradient)
end

function trajectory_regret_metrics(pomdp, results)
    ab_set = [:abandon]
    ex_set = setdiff(pomdp.terminal_actions, [:abandon])
    headers = ["N_abandon", " N_execute", " N_obs", " cumulative_spent", " abandon_chance", " execution_chance", " expected_regret"]
    df = DataFrame()
    for h in headers
        df[!, h] = Vector{Any}()
    end
    for i=1:maximum(length.(results[:actions]))
        Nab = 0
        Nob = 0
        Nex = 0
        # How much has been spent up to the current timestep
        obs_cost_to_i = 0

        # If you keep going what is the chance you walk away vs. execute
        abandon_chance = 0
        execution_chance = 0

        # If you keep going what is your expected regret in the cases you abandon
        expected_regret = 0
        Nex_regret = 0

        # Loop over trajectories
        for t in results[:actions]
            # Check that the current timestep is relevant for the curent trajectory
            if length(t) >= i 
                # Add the cost of this trajectory, but don't include the terminal cost/reward
                if length(t) > 1
                    obs_cost_to_i += sum(a.cost for a in t[1:i] if a isa ObservationAction)
                end

                # Add up the number of trajectories that abandon or execute
                Nab += t[i] in ab_set
                Nex += t[i] in ex_set

                # For all trajectories that continue on, how many end up abandoned? executed?
                if !(t[i] in pomdp.terminal_actions)
                    Nob += 1
                    if t[end] == :abandon
                        abandon_chance += 1
                        if length(t) > i+1
                            expected_regret += sum(a.cost for a in t[i+1:end] if a isa ObservationAction)
                            Nex_regret += 1
                        end
                    else
                        execution_chance += 1
                    end
                end

            end
        end
        abandon_chance /= Nob
        execution_chance /= Nob
        expected_regret /= Nex_regret
        obs_cost_to_i /= Nob
        
        push!(df, [Nab, Nex, Nob, obs_cost_to_i, abandon_chance, execution_chance, expected_regret])
    end
    df
end

function policy_comparison_summary(policy_results, policy_names)
    bottom_margin = maximum(length.(policy_names))*mm
    p_reward = bar(policy_names, [mean(r[:reward]) for r in policy_results], xrotation=60, bottom_margin=bottom_margin, ylabel="Mean Discounted Reward", title="Mean Discounted Reward", legend=false)
    p_obs_cost = bar(policy_names, [mean(r[:obs_cost]) for r in policy_results], xrotation=60, bottom_margin=bottom_margin, ylabel="Mean Data Acquisition Cost", title="Mean Data Acquisition Cost", legend=false, yflip=true)
    p_num_obs = bar(policy_names, [mean(r[:num_obs]) for r in policy_results], xrotation=60, bottom_margin=bottom_margin, ylabel="Mean Number of Data Acquisition Actions", title="Mean Number of Data Acquisition Actions", legend=false)
    p_correct_scenario = bar(policy_names, [mean(r[:correct_scenario]) for r in policy_results], xrotation=60, bottom_margin=bottom_margin, ylabel="Mean Correct Scenario", title="Mean Correct Scenario", legend=false)
    p_correct_gonogo = bar(policy_names, [mean(r[:correct_gonogo]) for r in policy_results], xrotation=60, bottom_margin=bottom_margin, ylabel="Mean Correct Go/NoGo", title="Mean Correct Go/NoGo", legend=false)
    plot(p_reward, p_obs_cost, p_num_obs, p_correct_scenario, p_correct_gonogo, layout=(2,3), legend=false, size=(1400,800), left_margin=5mm)
end

function report_mean(vals)
	mean_vals = mean(vals)
	stderr_vals = std(vals) / sqrt(length(vals))
	@sprintf("%.3f \\pm %.3f", mean_vals, stderr_vals) 
end

function policy_comparison_table(policy_results, policy_names)
    header = "Policy & NPV (M€) & Correct Go/No-Go & Correct Development Option & No. Data Acquisition Actions & Data Acquisition Cost (M€)\\\\"
    println(header)
    println("\\midrule")
    for (results, name) in zip(policy_results, policy_names)
        row = "$(name) & "
        row *= report_mean(results[:reward]) * " & "
        row *= report_mean(results[:correct_gonogo]) * " & " 
        row *= report_mean(results[:correct_scenario]) * " & "
        row *= report_mean(results[:num_obs]) * " & "
        row *= report_mean(results[:obs_cost])
        
        # row *= report_mean(results[:reward][results[:reward] .< 0]) * " & "
        
        row *= " \\\\"
        println(row)
    end
end

# function train_states_comparison_summary(results)
#     p_reward = plot()
#     p_obs_cost = plot()
#     p_num_obs = plot()
#     p_correct_scenario = plot()
#     p_correct_gonogo = plot()
#     p_legend = plot(legend=false, grid=false, axis=false, ticks=nothing, border=:none, size=(800,400))

#     xlab = "No. Subsurface Realizations"

#     for (policy_name, pol_results) in results
#         plot!(p_reward, pol_results[:geo_frac]*250, [mean(r[:reward]) for r in pol_results[:results]], xlabel=xlab, ylabel="Mean Discounted Reward", legend = false)
#         plot!(p_obs_cost, pol_results[:geo_frac]*250, [mean(r[:obs_cost]) for r in pol_results[:results]], xlabel=xlab, ylabel="Mean Data Acquisition Cost", legend = false)
#         plot!(p_num_obs, pol_results[:geo_frac]*250, [mean(r[:num_obs]) for r in pol_results[:results]], xlabel=xlab, ylabel="Mean Number of Data Acquisition Actions", legend = false)
#         plot!(p_correct_scenario, pol_results[:geo_frac]*250, [mean(r[:correct_scenario]) for r in pol_results[:results]], xlabel=xlab, ylabel="Mean Correct Scenario", legend = false)
#         plot!(p_correct_gonogo, pol_results[:geo_frac]*250, [mean(r[:correct_gonogo]) for r in pol_results[:results]], xlabel=xlab, ylabel="Mean Correct Go/NoGo", legend = false)
#         plot!(p_legend, [], [], label=policy_name, legend=:topleft)
#     end
#     plot(p_reward, p_obs_cost, p_num_obs, p_correct_scenario, p_correct_gonogo, p_legend, layout=(2,3),size=(1400,800), margin=5mm)
# end

function reward_vs_ngeolgies(pol_results, policy_name; p=plot())
    xlab = "No. Geological Realizations"
    max_econ = maximum(pol_results[:econ_frac])
    xs = 250*pol_results[:geo_frac][pol_results[:econ_frac] .== max_econ]
    perm = sortperm(xs)
    ys = [mean(r[:reward]) for (r, e) in zip(pol_results[:results], pol_results[:econ_frac]) if e == max_econ]
    plot!(p, xs[perm], ys[perm], xlabel=xlab, ylabel="Mean Discounted Reward", legend = false, title=policy_name)
end

function reward_vs_necon(pol_results, policy_name; p=plot())
    xlab = "No. Economic Realizations"
    max_geo = maximum(pol_results[:geo_frac])
    xs = 50*pol_results[:econ_frac][pol_results[:geo_frac] .== max_geo]
    perm = sortperm(xs)
    ys = [mean(r[:reward]) for (r, g) in zip(pol_results[:results], pol_results[:geo_frac]) if g == max_geo]
    plot!(p, xs[perm], ys[perm], xlabel=xlab, ylabel="Mean Discounted Reward", legend = false, title=policy_name)
end

function generate_action_table(pomdp, var_desc)
    header = "Name & Cost (M€) & Duration (Years) & Observed Variables & Observation Uncertainty \\\\"
    println(header)
    println("\\midrule")
    for a in actions(pomdp)
        if a in pomdp.terminal_actions
            continue
        end
        dist = a.obs_dist(pomdp.states[2])
        o = rand(dist)

        # Check for the scenario-dependent options
        rows = []
        k1 = string(collect(keys(o))[1])
        if occursin("Option", k1) #TODO this is just a fix for now and wont work when Scenarios aren't called Scenario_X
            k = Symbol(replace(k1, r"_Option.*" => ""))
            d = collect(values(dist))[1]
            sigma = (d.b - d.a)/2
            row = var_desc[k] * " & \\num{" *  @sprintf("%.3g", sigma) * "} \\\\" 
            push!(rows, row)
        else
            for ((_, d), (k, _)) in zip(dist, o)
                @assert d isa Distributions.Uniform # Asssert this for now
                sigma = (d.b - d.a)/2
                row = var_desc[k] * " & \\num{" * @sprintf("%.3g", sigma) * "} \\\\"
                push!(rows, row)
            end
        end
        nrows = length(rows)
        if nrows == 1
            prefix = "$(a.name) & \\num{" * @sprintf("%.3g", a.cost) *"} & \\num{" * @sprintf("%.2g", a.time) * "} & "
        else
            prefix = "\\multirow{$nrows}{*}{$(a.name)} & \\multirow{$nrows}{*}{\\num{" * @sprintf("%.3g", a.cost) *"}} & \\multirow{$nrows}{*}{\\num{" * @sprintf("%.2g", a.time) * "}} & "
        end

        ## Now print the table
        for (i, row) in enumerate(rows)
            if i==1
                println(prefix * row)
            else
                println("& & & "*row)
            end
        end
    end
end


function scenario_returns(scenario_csvs, geo_params, econ_params)
    statevec = InfoGatheringPOMDPs.parseCSV(scenario_csvs, geo_params, econ_params)
    hists = [[s[scenario] for s in statevec] for scenario in keys(scenario_csvs)]
    violin(
        hists, 
        label ="", 
        xrot=60,
        xticks=(1:length(hists), string.(keys(scenario_csvs))), 
        ylabel="NPV (M)", title="Development Option Returns", 
        color=:blue, 
        alpha=0.3
    )
    hline!([0], color=:black, label="")
end

function print_human_sequences(actions, humans)
    for (i, hlist) in enumerate(humans)
        action_names = [a.name for a in actions[hlist]]
        println("\\item Human Expert $i: ", join(action_names, ", "))
    end
end
