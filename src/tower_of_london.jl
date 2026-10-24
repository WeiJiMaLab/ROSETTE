struct TowerOfLondon <: AbstractGame
    bs::Vector{Int}
    initial_position::Vector{Int}
    goal_position::Vector{Int}
end

s_type = Int
a_type = Tuple{Int, Int}

thought_type = Tuple{Int, Int, Int, Int}# (bead, type, ii, jj)
or_type = Tuple{Int, thought_type}# depth, subgoal
and_type = Tuple{Int, a_type}# depth, move

function make_move(s::Int, m::Tuple{Int, Int}, game::TowerOfLondon)::Int
    return move_piece(s, m, game)
end

function neighbours(s::Int, game::TowerOfLondon)::Vector{Int}
    ns = Int[]
    for i in 1:6
        # check if there is a disk above current position
        # if so, no need to check since it is blocked
        if i == 1 || i == 2 || i == 4 #2,3,5#3,5,6
            if check_digit(s, i+1, game) > 0 #-
                continue
            end
        end
        if check_digit(s, i, game) > 0
            # move onto first pole
            if check_digit(s, 6, game) == 0 && i !== 6 #1
                push!(ns, make_move(s, (i, 6), game))
            end
            # move onto second pole
            if check_digit(s, 5, game) == 0 && i !== 5 && i !== 4 #2,3
                # if top and bottom are free, move onto bottom
                if check_digit(s, 4, game) == 0 #3,2
                    push!(ns, make_move(s, (i, 4), game))
                else
                    push!(ns, make_move(s, (i, 5), game))
                end
            end
            # move onto third pole
            if check_digit(s, 3, game) == 0 && i !== 3 && i !== 2 && i !== 1 #1,2,3
                # if top, middle and bottom are free, move onto bottom
                if check_digit(s, 2, game) == 0 #2,3,1
                    if check_digit(s, 1, game) == 0
                        push!(ns, make_move(s, (i, 1), game))
                    else
                        push!(ns, make_move(s, (i, 2), game))
                    end
                else
                    push!(ns, make_move(s, (i, 3), game))
                end
            end
        end
    end
    return ns
end

function has_won(s::Int, game::TowerOfLondon)::Bool
    return s == goal_state(game)
end

function has_lost(s::Int, game::TowerOfLondon)::Bool
    return false
end

function get_tol_array(s::Int, game::TowerOfLondon)
    return [check_digit(s, i, game) for i in 1:6]
end

function draw_state(s::Int, game::TowerOfLondon)
    # return [[check_digit(s, i, game) for i in reverse(4:6)], [check_digit(s, i, game) for i in reverse(2:3)], [check_digit(s, 1, game)]]
    return [[check_digit(s, i, game) for i in 1:3], [check_digit(s, i, game) for i in 4:5], [check_digit(s, 6, game)]]
end

function get_L_opt(s::Int, game::TowerOfLondon; max_iters=100)
    s_start = s
    visited = Set{Int}(s_start)
    distances = Dict{Int, Int}()
    distances[s_start] = 0
    frontier = Int[s_start]
    for i in 1:max_iters
        if i % 100000 == 0
            println("Iteration: $i")
        end
        if isempty(frontier)
            break
        end
        s = popfirst!(frontier)
        d = distances[s]
        for neigh in neighbours(s, game)
            if neigh in visited
                continue
            end
            if is_terminal(neigh, game)
                return d+1
            end
            push!(visited, neigh)
            distances[neigh] = d + 1
            push!(frontier, neigh)
        end
    end
    throw("Could not find solution when computing Lopt")
end

function get_distances(game::TowerOfLondon)
    states, leafs = search(game)
    distances = Dict{Int, Int}()
    for s in leafs
        distances[s] = 0
    end
    for s in states
        distances[s] = get_L_opt(s, game)
    end
    return distances
end

function neighbours_actions(s::Int, game::TowerOfLondon)::Tuple{Vector{Int}, Vector{Tuple{Int, Int}}}
    neighs = Int[]
    actions = Tuple{Int, Int}[]
    for i in 1:6
        # check if there is a disk above current position
        # if so, no need to check since it is blocked
        if i == 1 || i == 2 || i == 4 #2,3,5#3,5,6
            if check_digit(s, i+1, game) > 0 #-
                continue
            end
        end
        if check_digit(s, i, game) > 0
            # move onto first pole
            if check_digit(s, 6, game) == 0 && i !== 6 #1
                push!(neighs, make_move(s, (i, 6), game))
                push!(actions, (i, 6))
            end
            # move onto second pole
            if check_digit(s, 5, game) == 0 && i !== 5 && i !== 4 #2,3
                # if top and bottom are free, move onto bottom
                if check_digit(s, 4, game) == 0 #3,2
                    push!(neighs, make_move(s, (i, 4), game))
                    push!(actions, (i, 4))
                else
                    push!(neighs, make_move(s, (i, 5), game))
                    push!(actions, (i, 5))
                end
            end
            # move onto third pole
            if check_digit(s, 3, game) == 0 && i !== 3 && i !== 2 && i !== 1 #1,2,3
                # if top, middle and bottom are free, move onto bottom
                if check_digit(s, 2, game) == 0 #2,3,1
                    if check_digit(s, 1, game) == 0
                        push!(neighs, make_move(s, (i, 1), game))
                        push!(actions, (i, 1))
                    else
                        push!(neighs, make_move(s, (i, 2), game))
                        push!(actions, (i, 2))
                    end
                else
                    push!(neighs, make_move(s, (i, 3), game))
                    push!(actions, (i, 3))
                end
            end
        end
    end
    return neighs, actions
end


function load_tol_data()
    data = Vector{Vector{Tuple{Int, Int}}}[]
    problems = []
    nn = 0
    for file in readdir("data/tower_of_london/subject_data")
        if file == ".DS_Store" || file == "5fc2c0cfdb80f9025c6b4810_d2.csv" || file == "5eaf76ae6665cc671414ab73_d2.csv"
            continue
        end
        subj_data = CSV.read("data/tower_of_london/subject_data/$file", DataFrame)
        if typeof(subj_data.tol_problem_id[1]) != String15
            continue
        end
        nn += 1
        games = Vector{Tuple{Int, Int}}[]
        for i in 0:24
            game_i = subj_data[subj_data.tol_problem_id .== string(i), [:mouse_click, :current_position]]
            N, _ = size(game_i)
            moves = Tuple{Int, Int}[]
            for j in 1:Int(N/2)
                m_from = parse(Int, split(game_i[2*j - 1, :mouse_click], "_")[3])
                m_to = parse(Int, split(game_i[2*j, :mouse_click], "_")[3])
                if m_from == m_to
                    continue
                end
                push!(moves, (m_from, m_to))
            end
            push!(games, moves)
            if nn == 1
                start_pos = [[0,0,0],[0,0],[0]]
                target_pos = [[0,0,0],[0,0],[0]]
                start_pos_ = game_i[1, :current_position]
                target_pos_ = game_i[N, :current_position]
                start_pos[1][1] = parse(Int, start_pos_[3])
                start_pos[1][2] = parse(Int, start_pos_[5])
                start_pos[1][3] = parse(Int, start_pos_[7])
                start_pos[2][1] = parse(Int, start_pos_[11])
                start_pos[2][2] = parse(Int, start_pos_[13])
                start_pos[3][1] = parse(Int, start_pos_[17])

                target_pos[1][1] = parse(Int, target_pos_[3])
                target_pos[1][2] = parse(Int, target_pos_[5])
                target_pos[1][3] = parse(Int, target_pos_[7])
                target_pos[2][1] = parse(Int, target_pos_[11])
                target_pos[2][2] = parse(Int, target_pos_[13])
                target_pos[3][1] = parse(Int, target_pos_[17])
                push!(problems, (start_pos, target_pos))
            end
        end
        push!(data, games)
    end
    return problems, data
end

function prepare_dataframe(state_actions, problems)
    df = DataFrame(subject=Int[], puzzle=Int[], event=String[], s=s_type[], move=a_type[], prev_move=a_type[], all_moves=Vector{a_type}[], d_goal=Int[], Lopt=Int[], first_move=Bool[], neighs=Vector{s_type}[])
    d_goals = Dict{s_type, Int}[]
    for prb in ProgressBar(eachindex(problems))
        game = TowerOfLondon(get_base(4, 6), vcat(problems[prb][1]...), vcat(problems[prb][2]...))
        s = start_state(game)
        distances = get_distances(game)
        Lopt = distances[s]
        for subj in eachindex(state_actions)
            s = start_state(game)
            subj_data = state_actions[subj][prb][1]
            prev_move = (0, 0)
            for i in eachindex(subj_data)
                s, move = subj_data[i]
                first_move = i == 1
                if move == (0, 0)
                    # push!(df, (subj, prb, "win", s, move, prev_move, [], distances[s], Lopt, first_move, []))
                    continue
                end
                neighs, all_moves = neighbours_actions(s, game)
                if move ∉ all_moves
                    println("Move not found in all_moves at: $((subj, prb, i))")
                end

                push!(df, (subj, prb, "move", s, move, prev_move, all_moves, distances[s], Lopt, first_move, neighs))

                s = make_move(s, move, game)
                prev_move = move
            end
        end
        push!(d_goals, distances)
    end
    return df, d_goals
end

function state_actions_tol(problems, old_games)
    state_actions = Vector{Vector{Vector{Tuple{Int, a_type}}}}[]
    N = length(old_games)
    for j in 1:N
        state_actions_subj = Vector{Vector{Tuple{Int, a_type}}}[]
        for i in 1:25
            state_actions_game = Tuple{Int, a_type}[]
            tol = TowerOfLondon(get_base(4, 6), vcat(problems[i][1]...), vcat(problems[i][2]...))
            moves = old_games[j][i]
            s = start_state(tol)
            #moves_ = Tuple{Int, Int}[]
            moves_ = Int[s]
            for move in moves
                if move[1] == move[2]
                    continue
                end
                move_from = nothing
                if move[1] == 1
                    if check_digit(s, 3, tol) != 0
                        move_from = 3
                    elseif check_digit(s, 2, tol) != 0
                        move_from = 2
                    else
                        move_from = 1
                    end
                elseif move[1] == 2
                    if check_digit(s, 5, tol) != 0
                        move_from = 5
                    else
                        move_from = 4
                    end
                else
                    move_from = 6
                end
                move_to = nothing
                if move[2] == 1
                    if check_digit(s, 1, tol) == 0
                        move_to = 1
                    elseif check_digit(s, 2, tol) == 0
                        move_to = 2
                    else
                        move_to = 3
                    end
                elseif move[2] == 2
                    if check_digit(s, 4, tol) == 0
                        move_to = 4
                    else
                        move_to = 5
                    end
                else
                    move_to = 6
                end
                movee = (move_from, move_to)
                push!(state_actions_game, (s, movee))
                s = make_move(s, movee, tol)
            end
            if !is_terminal(s, tol)
                println("Error at: $((i, j))")
            end
            push!(state_actions_game, (s, (0, 0)))
            push!(state_actions_subj, [state_actions_game])
        end
        push!(state_actions, state_actions_subj)
    end
    return state_actions
end

function get_tol_heuristics(N_t, idx_to_state, state_to_idx, game)
    all_moves = MVector{36, Tuple{Int64, Int64}}[]
    all_idxs = Vector{Int64}[]
    all_sizes = zeros(Int, N_t)

    target_arr = get_tol_array(goal_state(game), game)
    for i in 1:N_t
        moves = @MVector([(0, 0) for _ in 1:(4*9)])
        s = idx_to_state[i]
        s_arr = get_tol_array(s, game)
        all_sizes[i] = 3 - sum(s_arr .== target_arr .&& s_arr .!= 0)

        neighs, actions = neighbours_actions(s, game)
        for (n, action) in enumerate(actions)
            moves[n] = action
        end
        for (n, neigh) in enumerate(neighs)
            sp = state_to_idx[neigh]
            neighs[n] = sp < 0 ? N_t - sp : sp
        end

        push!(all_moves, moves)
        push!(all_idxs, neighs)
    end
    return all_moves, all_idxs, all_sizes
end

function get_fitting_infos(problems)
    fitting_infos = []
    for prb in problems
        game = TowerOfLondon(get_base(4, 6), vcat(prb[1]...), vcat(prb[2]...))

        state_to_idx, idx_to_state, Q, R = get_QR_matrices(game)
        N_a, N_t = size(R)
        P = zeros(N_t, 3)

        all_moves, all_idxs, all_sizes = get_tol_heuristics(N_t, idx_to_state, state_to_idx, game)

        fitting_info = FittingStruct([], all_moves, all_idxs, Q, R, P, all_sizes);
        push!(fitting_infos, fitting_info)
    end
    return fitting_infos
end

function get_analysis_info(problems)
    analysis_info = []
    for prb in problems
        game = TowerOfLondon(get_base(4, 6), vcat(prb[1]...), vcat(prb[2]...))
        state_to_idx, idx_to_state, Q, R = get_QR_matrices(game)
        push!(analysis_info, (state_to_idx, idx_to_state, game, size(R)[2]))
    end
    return analysis_info
end
