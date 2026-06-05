function load_ps!(df::DataFrame; tol=false)
    folder = joinpath("data", "ps_$(tol ? "TOL" : "RH")")
    N_batch = tol ? 9 : 24
    names = ["full", "lesion-$(tol ? "beta" : "gamma_ao")", "lesion-gamma_r", "lesion-gamma_p", "only-$(tol ? "beta" : "gamma_ao")", "only-gamma_r", "only-gamma_p", "chance"]
    full_ps = Dict(name => [Vector{Float64}[] for _ in 1:N_batch] for name in names)

    for filename in readdir(folder)
        startswith(filename, ".") && continue
        batch_index = parse(Int, split(split(filename, ".")[1], "_")[3])
        ps_batch = load(joinpath(folder, filename))["all_ps"]
        for name in names
            full_ps[name][batch_index] = ps_batch[name]
        end
    end

    for name in names
        df[!, Symbol("p_$name")] = reduce(vcat, full_ps[name])
    end

    return df
end

function load_simulated_games(; tol=false)
    folder = joinpath("data", "simulated_games_$(tol ? "TOL" : "RH")")
    dfs = DataFrame[]
    for file in ProgressBar(sort(filter(filename -> !startswith(filename, "."), readdir(folder))))
        push!(dfs, DataFrame(CSV.File(joinpath(folder, file))))
    end
    return vcat(dfs...)
end

function puzzle_counts(df::DataFrame, sas)
    df = df[:, [:subj, :puzzle, :name, :repeat]]
    mean_counts = Dict{String, Vector{Float64}}()
    sem_counts = Dict{String, Vector{Float64}}()
    counts = Dict{String, Vector{Float64}}()

    for name in sort(unique(df.name))
        mean_counts[name] = Float64[]
        sem_counts[name] = Float64[]
        counts[name] = Float64[]
    end

    for name in ProgressBar(sort(unique(df.name)))
        df_name = df[df.name .== name, :]
        for puzzle in sort(unique(df_name.puzzle))
            df_puzzle = df_name[df_name.puzzle .== puzzle, :]
            mean_lengths = Float64[]
            std_lengths = Float64[]

            for subject in sort(unique(df_puzzle.subj))
                df_subj_puzzle = df_puzzle[df_puzzle.subj .== subject, :]
                lengths = [sum(df_subj_puzzle.repeat .== repeat) - 1 for repeat in sort(unique(df_subj_puzzle.repeat))]
                push!(mean_lengths, mean(lengths))
                push!(std_lengths, std(lengths))
            end

            push!(mean_counts[name], mean(mean_lengths))
            push!(sem_counts[name], sqrt(sem(mean_lengths)^2 + mean(std_lengths .^ 2) / length(std_lengths)))
            push!(counts[name], length(mean_lengths))
        end
    end

    mean_counts["data"] = Float64[]
    sem_counts["data"] = Float64[]
    counts["data"] = Float64[]
    for puzzle in sort(unique(df.puzzle))
        lengths = Int[]
        for subject in sort(unique(df.subj))
            !isempty(sas[subject][puzzle]) && push!(lengths, length(sas[subject][puzzle][1]) - 1)
        end
        push!(mean_counts["data"], mean(lengths))
        push!(sem_counts["data"], sem(lengths))
        push!(counts["data"], length(lengths))
    end

    return mean_counts, sem_counts, counts, sort(unique(df.puzzle))
end

function get_puzzle_stats(analysis_info, d_goals, problems; rh_problems=[])
    external_problem_ids = isempty(rh_problems) ? problems : rh_problems
    state_space_sizes = [length(d_goals[problem]) for problem in external_problem_ids]
    optimal_lengths = Int[]

    for (problem_idx, external_id) in zip(problems, external_problem_ids)
        _, _, game, _ = analysis_info[problem_idx]
        push!(optimal_lengths, d_goals[external_id][start_state(game)])
    end

    return state_space_sizes, optimal_lengths
end

function get_binned_results(independent_var, mean_counts, sem_counts, puzzles, state_space_sizes, optimal_lengths; nbins=4)
    df = DataFrame(puzzle=Int[], model=String[], mean_L=Float64[], sem_L=Float64[], sss=Int[], L_opt=Int[])
    for model_name in keys(mean_counts)
        for (i, puzzle) in enumerate(puzzles)
            push!(df, (puzzle, model_name, mean_counts[model_name][i], sem_counts[model_name][i], state_space_sizes[i], optimal_lengths[i]))
        end
    end

    transform!(df, independent_var => (x -> add_bin_number(x; nbins=nbins)) => :bin_number)
    grouped = groupby(df, [:bin_number, :model])
    independent = combine(grouped, [:sss, :L_opt] => calculate_mean_sem_1 => [:mean_sss, :std_sss, :mean_L_opt, :std_L_opt])
    dependent = combine(grouped, [:mean_L, :sem_L] => calculate_mean_sem_2 => [:m_L, :sem_L])
    return innerjoin(independent, dependent, on=[:bin_number, :model])
end
