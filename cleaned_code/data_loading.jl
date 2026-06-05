load_raw_data() = CSV.read(joinpath("data", "all_subjects.csv"), DataFrame)

function filter_subjects(df::DataFrame)
    filtered_data = DataFrame()
    rejected = 0

    for subject in unique(df.subject)
        subj_data = df[df.subject .== subject, :]
        elapsed_minutes = subj_data.t ./ (1000 * 60)
        elapsed_minutes .-= first(elapsed_minutes)
        deleteat!(elapsed_minutes, findall(<(0), elapsed_minutes))

        intervals = elapsed_minutes[2:end] - elapsed_minutes[1:end - 1]
        completed_enough = (last(elapsed_minutes) - first(elapsed_minutes) >= 60) ||
                           length(unique(subj_data.instance)) == 70

        if !completed_enough || ("win" ∉ subj_data.event) || minimum(intervals) < 0
            rejected += 1
            continue
        end

        filtered_data = vcat(filtered_data, subj_data)
    end

    println("Rejection ratio: $(rejected)/$(length(unique(df.subject)))")
    return filtered_data
end

function pre_process(df::DataFrame, d_goals_prbs)
    new_df = DataFrame(
        subject=String[],
        puzzle=String[],
        event=String[],
        move=Action[],
        prev_move=Action[],
        s_free=RushHourFreeState[],
        s_fixed=RushHourFixedState[],
        d_goal=Int[],
        Lopt=Int[],
        attempt=Int[],
        RT=Int[],
        timestamp=Int[],
    )

    for subj in unique(df.subject)
        subj_df = df[(df.subject .== subj) .& (df.event .∈ Ref(["start", "restart", "drag_end", "win"])), :]
        for prb in unique(subj_df.instance)
            prb_df = subj_df[subj_df.instance .== prb, :]
            s_free, s_fixed = load_prb_(prb)
            attempt = 0
            prev_t = first(prb_df).t
            prev_move = (0, 0)
            Lopt = parse(Int, prb[end - 1] == '_' ? prb[end] : prb[end - 1:end]) - 2

            prb_df[end, :event] != "win" && continue

            for (n, row) in enumerate(eachrow(prb_df))
                d_goal = d_goals_prbs[prb][board_to_int32((s_free, s_fixed))[2]]

                if row.event == "start"
                    attempt += 1
                    if attempt == 1
                        s_free, _ = load_prb_(prb)
                        push!(new_df, [subj, prb, "start", (0, 0), prev_move, copy(s_free), s_fixed, d_goal, Lopt, attempt, row.t - prev_t, row.t])
                    elseif last(new_df).event != "restart" && last(new_df).event != "start"
                        push!(new_df, [subj, prb, "restart", (0, 0), prev_move, copy(s_free), s_fixed, d_goal, Lopt, attempt - 1, row.t - prev_t, row.t])
                        s_free, _ = load_prb_(prb)
                    else
                        attempt -= 1
                        s_free, _ = load_prb_(prb)
                    end
                    prev_t = row.t
                    prev_move = (0, 0)
                    continue
                elseif row.event in ("win", "restart")
                    continue
                end

                if check_solved((s_free, s_fixed))
                    push!(new_df, [subj, prb, "win", (0, 0), prev_move, copy(s_free), s_fixed, d_goal, Lopt, attempt, 0, prev_t])
                    break
                end

                n < nrow(prb_df) && prb_df[n + 1, :move] == row.move && continue

                car_id = row.piece == 8 ? 1 : row.piece + 2
                amount = if s_fixed[car_id].dir == :x
                    1 + (row.target % BOARD_SIZE) - s_free[car_id]
                else
                    1 + (row.target ÷ BOARD_SIZE) - s_free[car_id]
                end
                move = (car_id, amount)

                push!(new_df, [subj, prb, "move", move, prev_move, copy(s_free), s_fixed, d_goal, Lopt, attempt, row.t - prev_t, row.t])
                make_move!(s_free, move)
                prev_t = row.t
                prev_move = move
            end
        end
    end

    new_df.first_move = circshift(new_df.event .== "start", 1)
    return new_df
end

function get_filtered_df(subjects, problems, df::DataFrame)
    filtered_df = DataFrame()
    for subj in subjects
        for prb in problems
            df_prb = df[(df.subject .== subj) .& (df.puzzle .== prb), :]
            solved_moves = DataFrame()
            for row in eachrow(df_prb)
                if row.event == "start" || row.event == "restart"
                    solved_moves = DataFrame()
                elseif row.event == "move"
                    push!(solved_moves, row)
                elseif row.event == "win"
                    filtered_df = vcat(filtered_df, solved_moves)
                end
            end
        end
    end
    return filtered_df
end

function load_rh_data()
    data = CSV.read(joinpath("data", "rh_data.csv"), DataFrame)
    subjects = unique(data.subject)
    problems = String.(unique(data.puzzle))
    problem_order = sortperm([parse(Int, x[end - 1] == '_' ? x[end] : x[end - 1:end]) for x in problems])
    problems = problems[problem_order]

    state_actions = Vector{Vector{Vector{Tuple{Int, Action}}}}[]
    for subject in subjects
        state_actions_subj = Vector{Vector{Tuple{Int, Action}}}[]
        for problem in problems
            df_problem = data[(data.puzzle .== problem) .& (data.subject .== subject), :]
            restart_idxs = findall(==("restart"), df_problem.event)
            restart_idxs = isempty(restart_idxs) ? [1] : restart_idxs
            df_moves = df_problem[restart_idxs[end] + 1:end - 1, :]

            if isempty(df_moves)
                push!(state_actions_subj, [])
                continue
            end
            length(unique(df_moves.event)) != 1 && println("Not just moves")

            start_pos, s_fixed = load_prb(problem)
            game = RushHour(get_base(5, RUSH_HOUR_CARS), start_pos, s_fixed)
            s = start_state(game)
            state_actions_game = Tuple{Int, Action}[]

            for row in eachrow(df_moves)
                move = _parse_rh_move(row[:move])
                push!(state_actions_game, (s, move))
                s = make_move(s, move, game)
            end
            push!(state_actions_game, (s, (0, 0)))
            push!(state_actions_subj, [state_actions_game])
        end
        push!(state_actions, state_actions_subj)
    end

    return subjects, problems, state_actions
end

function _parse_rh_move(move_text)::Action
    parts = split(move_text, ",")
    car = parse(Int, parts[1][2:end])
    amount_text = replace(parts[2], ")" => "", " " => "")
    return car, parse(Int, amount_text)
end

function games_to_idx(state_actions, analysis_info)
    state_actions_idx = Vector{Vector{Vector{Tuple{Int, Action}}}}[[] for _ in eachindex(state_actions)]

    for i in eachindex(analysis_info)
        state_to_idx, _, _, N_t = analysis_info[i]
        for j in eachindex(state_actions)
            if isempty(state_actions[j][i])
                push!(state_actions_idx[j], [])
                continue
            end

            indexed_games = [[(state_to_idx[state], action) for (state, action) in game] for game in state_actions[j][i]]
            if indexed_games[1][end][1] < 0
                indexed_games[1][end] = (N_t - indexed_games[1][end][1], indexed_games[1][end][2])
            end
            push!(state_actions_idx[j], indexed_games)
        end
    end

    return state_actions_idx
end

function get_ss(state_actions)
    return vcat([
        vcat([[state_actions[k][j][1][i][1] for i in 1:(length(state_actions[k][j][1]) - 1)] for j in eachindex(state_actions[k]) if !isempty(state_actions[k][j])]...)
        for k in eachindex(state_actions)
    ]...)
end

function get_fitting_infos(problems)
    isempty(problems) && return FittingStruct[]
    return first(problems) isa AbstractString ? get_rh_fitting_infos(problems) : get_tol_fitting_infos(problems)
end

function get_analysis_info(problems)
    isempty(problems) && return []
    return first(problems) isa AbstractString ? get_rh_analysis_info(problems) : get_tol_analysis_info(problems)
end

function get_rh_fitting_infos(problems)
    fitting_infos = FittingStruct[]
    for problem in ProgressBar(problems)
        start_pos, s_fixed = load_prb(problem)
        game = RushHour(get_base(5, RUSH_HOUR_CARS), start_pos, s_fixed)
        state_to_idx, idx_to_state, Q, R = get_QR_matrices(game)
        _, N_t = size(R)
        all_dicts, all_moves, all_idxs, all_sizes = get_and_or_dicts(N_t, idx_to_state, state_to_idx, game)
        push!(fitting_infos, FittingStruct(all_dicts, all_moves, all_idxs, Q, R, zeros(length(all_sizes), 3), all_sizes))
    end
    return fitting_infos
end

function get_rh_analysis_info(problems)
    analysis_info = []
    for problem in problems
        start_pos, s_fixed = load_prb(problem)
        game = RushHour(get_base(5, RUSH_HOUR_CARS), start_pos, s_fixed)
        state_to_idx, idx_to_state, _, R = get_QR_matrices(game)
        push!(analysis_info, (state_to_idx, idx_to_state, game, size(R, 2)))
    end
    return analysis_info
end

function get_tol_fitting_infos(problems)
    fitting_infos = FittingStruct[]
    for problem in problems
        game = TowerOfLondon(get_base(4, 6), vcat(problem[1]...), vcat(problem[2]...))
        state_to_idx, idx_to_state, Q, R = get_QR_matrices(game)
        _, N_t = size(R)
        all_moves, all_idxs, all_sizes = get_tol_heuristics(N_t, idx_to_state, state_to_idx, game)
        push!(fitting_infos, FittingStruct(Dict{Tuple{Int64, Tuple{Int64, Int64}}, Float64}[], all_moves, all_idxs, Q, R, zeros(N_t, 3), all_sizes))
    end
    return fitting_infos
end

function get_tol_analysis_info(problems)
    analysis_info = []
    for problem in problems
        game = TowerOfLondon(get_base(4, 6), vcat(problem[1]...), vcat(problem[2]...))
        state_to_idx, idx_to_state, _, R = get_QR_matrices(game)
        push!(analysis_info, (state_to_idx, idx_to_state, game, size(R, 2)))
    end
    return analysis_info
end

function load_tol_data(; include_rts::Bool=false)
    data = Vector{Vector{Tuple{Int, Int}}}[]
    rts = Vector{Vector{Int}}[]
    problems = []
    subject_count = 0

    for file in readdir(joinpath("data", "tower_of_london", "subject_data"))
        if file in (".DS_Store", "5fc2c0cfdb80f9025c6b4810_d2.csv", "5eaf76ae6665cc671414ab73_d2.csv")
            continue
        end
        subj_data = CSV.read(joinpath("data", "tower_of_london", "subject_data", file), DataFrame)
        subj_data.tol_problem_id[1] isa AbstractString || continue
        subject_count += 1

        games = Vector{Tuple{Int, Int}}[]
        subj_rts = Vector{Int}[]
        for i in 0:24
            cols = include_rts ? [:rt, :mouse_click, :current_position] : [:mouse_click, :current_position]
            game_i = subj_data[subj_data.tol_problem_id .== string(i), cols]
            N = nrow(game_i)
            moves = Tuple{Int, Int}[]
            game_rts = Int[]

            for j in 1:Int(N / 2)
                m_from = parse(Int, split(game_i[2 * j - 1, :mouse_click], "_")[3])
                m_to = parse(Int, split(game_i[2 * j, :mouse_click], "_")[3])
                m_from == m_to && continue
                push!(moves, (m_from, m_to))
                include_rts && push!(game_rts, parse(Int, game_i[2 * j, :rt]) + parse(Int, game_i[2 * j, :rt]))
            end

            push!(games, moves)
            include_rts && push!(subj_rts, game_rts)

            if subject_count == 1
                push!(problems, _parse_tol_problem(game_i[1, :current_position], game_i[N, :current_position]))
            end
        end

        push!(data, games)
        include_rts && push!(rts, subj_rts)
    end

    return include_rts ? (problems, data, rts) : (problems, data)
end

load_tol_data_with_rts() = load_tol_data(; include_rts=true)

function _parse_tol_problem(start_text, target_text)
    start_pos = [[0, 0, 0], [0, 0], [0]]
    target_pos = [[0, 0, 0], [0, 0], [0]]

    for (target, text) in ((start_pos, start_text), (target_pos, target_text))
        target[1][1] = parse(Int, text[3])
        target[1][2] = parse(Int, text[5])
        target[1][3] = parse(Int, text[7])
        target[2][1] = parse(Int, text[11])
        target[2][2] = parse(Int, text[13])
        target[3][1] = parse(Int, text[17])
    end

    return start_pos, target_pos
end

function prepare_tol_dataframe(state_actions, problems, rts=nothing)
    include_rts = rts !== nothing
    df = DataFrame(
        subject=Int[],
        puzzle=Int[],
        RT=Int[],
        event=String[],
        s=Int[],
        move=Action[],
        prev_move=Action[],
        all_moves=Vector{Action}[],
        d_goal=Int[],
        Lopt=Int[],
        first_move=Bool[],
        neighs=Vector{Int}[],
    )
    d_goals = Dict{Int, Int}[]

    for prb in ProgressBar(eachindex(problems))
        game = TowerOfLondon(get_base(4, 6), vcat(problems[prb][1]...), vcat(problems[prb][2]...))
        distances = get_distances(game)
        Lopt = distances[start_state(game)]

        for subj in eachindex(state_actions)
            prev_move = (0, 0)
            for i in eachindex(state_actions[subj][prb][1])
                s, move = state_actions[subj][prb][1][i]
                move == (0, 0) && continue
                neighs, all_moves = neighbours_actions(s, game)
                move ∉ all_moves && println("Move not found in all_moves at: $((subj, prb, i))")
                rt = include_rts ? rts[subj][prb][i] : 0
                push!(df, (subj, prb, rt, "move", s, move, prev_move, all_moves, distances[s], Lopt, i == 1, neighs))
                prev_move = move
            end
        end

        push!(d_goals, distances)
    end

    return df, d_goals
end

function _load_indexed_params(folder::AbstractString, rows::Int, cols::Int; batch_size::Int=1)
    params = zeros(rows, cols)
    for filename in readdir(folder)
        startswith(filename, ".") && continue
        i = parse(Int, split(split(filename, ".")[1], "_")[2])
        loaded = load(joinpath(folder, filename))["params"]
        if batch_size == 1
            params[i, :] = loaded
        elseif i * batch_size > rows
            params[(i - 1) * batch_size + 1:end, :] = loaded[1:batch_size - ((i * batch_size) - rows), :]
        else
            params[(i - 1) * batch_size + 1:i * batch_size, :] = loaded
        end
    end
    return params
end

load_rh_parameters() = _load_indexed_params(joinpath("data", "params"), 42, 7)
load_rh_parameters_reduced() = _load_indexed_params(joinpath("data", "params_reduced"), 42, 4)
load_tol_parameters() = _load_indexed_params(joinpath("data", "params_TOL"), 469, 7; batch_size=10)
load_tol_parameters_reduced() = _load_indexed_params(joinpath("data", "params_TOL_reduced"), 469, 3; batch_size=10)
load_tol_parameters_super_reduced() = _load_indexed_params(joinpath("data", "params_TOL_super_reduced"), 469, 2; batch_size=10)

function load_rh_lesion_parameters()
    names = ["gamma_ao", "gamma_r", "alpha_+", "alpha_-", "gamma_p", "delta_+", "delta_-"]
    return Dict(name => _load_indexed_params(joinpath("data", "params_rh_lesioned_10M", name), 42, 7) for name in names)
end

function load_tol_lesion_parameters()
    names = ["beta", "gamma_r", "alpha_+", "alpha_-", "gamma_p", "delta_+", "delta_-"]
    return Dict(name => _load_indexed_params(joinpath("data", "params_TOL_lesioned", name), 469, 7; batch_size=10) for name in names)
end

function load_all_ps(folder::AbstractString=joinpath("data", "all_ps"))
    all_ps = []
    files = sort(filter(filename -> !startswith(filename, "."), readdir(folder)))
    for filename in files
        push!(all_ps, load(joinpath(folder, filename))["all_ps"][1:end - 1])
    end
    push!(all_ps, [load(joinpath(folder, files[end]))["all_ps"][end]])
    return reduce(vcat, all_ps)
end
