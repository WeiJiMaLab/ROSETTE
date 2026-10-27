X_d_goal(move, row, d_goals) = row.d_goal

function y_d_goal(move, row, d_goals)
    move_idx = findfirst(==(move), row.all_moves)
    sampled_s = row.neighs[move_idx]
    return d_goals[row.puzzle][sampled_s] + 1
end

function y_p_in_tree(move, row, d_goals)
    _, _, _, parents_moves = row.tree
    return move ∈ keys(parents_moves)
end

function y_p_undo_RH(move, row, d_goals)
    row.prev_move == (0, 0) && return 0
    return move == (row.prev_move[1], -row.prev_move[2])
end

function y_p_undo_TOL(move, row, d_goals)
    row.prev_move == (0, 0) && return 0
    return move == (row.prev_move[2], row.prev_move[1])
end

function y_p_same_car(move, row, d_goals)
    row.prev_move == (0, 0) && return 0
    return move[1] == row.prev_move[1]
end

function y_p_same_bead(move, row, d_goals)
    row.prev_move == (0, 0) && return 0
    return move[1] == row.prev_move[2]
end

function y_d_tree(move, row, d_goals)
    _, _, _, parents_moves = row.tree
    if move in keys(parents_moves)
        return minimum(or_node[1] for or_node in parents_moves[move])
    end
    return 1e9
end

function y_d_tree_diff(move, row, d_goals)
    _, _, _, parents_moves = row.tree
    unique_depths = sort(unique(reduce(vcat, [[or_node[1] for or_node in parents_moves[m]] for m in keys(parents_moves)])))
    diff_depths = unique_depths .- unique_depths[1]
    if move in keys(parents_moves)
        ranked_depths = [diff_depths[findfirst(==(or_node[1]), unique_depths)] for or_node in parents_moves[move]]
        return minimum(ranked_depths)
    end
    return 1e9
end

function y_p_worse(move, row, d_goals)
    move_idx = findfirst(==(move), row.all_moves)
    sampled_s = row.neighs[move_idx]
    return d_goals[row.puzzle][sampled_s] > row.d_goal
end

function y_p_same(move, row, d_goals)
    move_idx = findfirst(==(move), row.all_moves)
    sampled_s = row.neighs[move_idx]
    return d_goals[row.puzzle][sampled_s] == row.d_goal
end

function y_p_better(move, row, d_goals)
    move_idx = findfirst(==(move), row.all_moves)
    sampled_s = row.neighs[move_idx]
    return d_goals[row.puzzle][sampled_s] < row.d_goal
end

function y_p_full(move, row, d_goals)
    move_idx = findfirst(==(move), row.all_moves)
    return row.p_full[move_idx]
end

function y_p_in_tree_better(move, row, d_goals)
    y_p_in_tree(move, row, d_goals) == 0 && return 1e9
    return y_p_better(move, row, d_goals)
end

y_red_back(move, row, d_goals) = move[1] == 1 && move[2] < 0

function calculate_summary_statistics(df::DataFrame, d_goals_prbs; tol=false, iters=100)
    models = ["chance", "only-$(tol ? "beta" : "gamma_ao")", "only-gamma_r", "only-gamma_p", "lesion-$(tol ? "beta" : "gamma_ao")", "lesion-gamma_r", "lesion-gamma_p", "full"]
    summary_stats = tol ?
        [X_d_goal, y_d_goal, y_p_undo_TOL, y_p_same_bead, y_p_worse, y_p_same, y_p_better, y_p_full] :
        [X_d_goal, y_d_goal, y_p_undo_RH, y_p_same_car, y_p_worse, y_p_same, y_p_better, y_p_full]

    df_stats = DataFrame(
        subject=String[],
        puzzle=String[],
        model=String[],
        X_d_goal=Int[],
        y_d_goal=Float64[],
        y_p_undo=Float64[],
        y_p_same_car=Float64[],
        y_p_worse=Float64[],
        y_p_same=Float64[],
        y_p_better=Float64[],
        y_p_full=Float64[],
    )

    df_moves = df[df.event .== "move", :]
    for row in ProgressBar(eachrow(df_moves))
        data_stats = [sum_stat(row.move, row, d_goals_prbs) for sum_stat in summary_stats]
        push!(df_stats, vcat([string(row.subject), string(row.puzzle), "data"], data_stats))

        for model in models
            ps = row[Symbol("p_$model")]
            sampled = zeros(length(summary_stats), iters)
            for i in 1:iters
                move = wsample(row.all_moves, ps)
                for (n, sum_stat) in enumerate(summary_stats)
                    sampled[n, i] = sum_stat(move, row, d_goals_prbs)
                end
            end
            stats = [mean(values[values .< 1e9]) for values in eachrow(sampled)]
            push!(df_stats, vcat([string(row.subject), string(row.puzzle), model], stats))
        end
    end

    return df_stats
end

function bin_stats_subj(df_stats::DataFrame, independent_var::Symbol; nbins=10, subject_level=false)
    input_columns = propertynames(df_stats)[4:end]
    subject_columns = reduce(vcat, [[Symbol("mean_$name"), Symbol("std_$name")] for name in input_columns])
    output_columns = reduce(vcat, [[Symbol("m_$name"), Symbol("sem_$name")] for name in input_columns])

    binned_by_subject = []
    for subj in unique(df_stats.subject)
        df_subj = df_stats[df_stats.subject .== subj, :]
        transform!(df_subj, independent_var => (x -> add_bin_number(x; nbins=nbins)) => :bin_number)
        grouped = groupby(df_subj, [:bin_number, :model])
        combined = combine(grouped, input_columns => calculate_mean_sem_1 => subject_columns)
        combined[!, :subject] .= subj
        push!(binned_by_subject, combined)
    end

    subject_stats = vcat(binned_by_subject...)
    subject_level && return subject_stats
    return combine(groupby(subject_stats, [:model, :bin_number]), subject_columns => calculate_mean_sem_2 => output_columns)
end

function bin_stats(df_stats::DataFrame, independent_var::Symbol; nbins=10)
    input_columns = propertynames(df_stats)[4:end]
    output_columns = reduce(vcat, [[Symbol("m_$name"), Symbol("sem_$name")] for name in input_columns])
    transform!(df_stats, independent_var => (x -> add_bin_number(x; nbins=nbins)) => :bin_number)
    return combine(groupby(df_stats, [:bin_number, :model]), input_columns => calculate_mean_sem_1 => output_columns)
end

function add_bin_number(d_goal; nbins=10)
    bin_n = zeros(Int64, length(d_goal))
    bins = nbins == 2 ? [0, 0.5, 1.0] : quantile(d_goal, 0:(1 / nbins):1)

    for i in 1:nbins
        idxs = i < nbins ? (d_goal .>= bins[i]) .& (d_goal .< bins[i + 1]) : (d_goal .>= bins[i]) .& (d_goal .<= bins[i + 1])
        bin_n[idxs] .= i
    end

    return bin_n
end

function calculate_mean_sem_1(cols...)
    values = []
    for col in cols
        filtered = col[col .< 1e9]
        push!(values, mean(filtered), sem(filtered))
    end
    return [Tuple(values)]
end

function calculate_mean_sem_2(cols...)
    values = []
    for i in 1:Int(length(cols) / 2)
        means = cols[2i - 1]
        sems = cols[2i]
        push!(values, mean(means), 1.96 * sqrt(sem(means)^2 + mean(sems .^ 2) / length(sems)))
    end
    return [Tuple(values)]
end
