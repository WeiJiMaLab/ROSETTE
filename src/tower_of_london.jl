make_move(s::Int, move::Action, game::TowerOfLondon)::Int = move_piece(s, move, game)

function neighbours(s::Int, game::TowerOfLondon)::Vector{Int}
    neighs, _ = neighbours_actions(s, game)
    return neighs
end

has_won(s::Int, game::TowerOfLondon)::Bool = s == goal_state(game)
has_lost(s::Int, game::TowerOfLondon)::Bool = false

get_tol_array(s::Int, game::TowerOfLondon)::Vector{Int} = [check_digit(s, i, game) for i in 1:6]

function draw_state(s::Int, game::TowerOfLondon)
    return [
        [check_digit(s, i, game) for i in 1:3],
        [check_digit(s, i, game) for i in 4:5],
        [check_digit(s, 6, game)],
    ]
end

function get_L_opt(s::Int, game::TowerOfLondon; max_iters=100)
    visited = Set{Int}([s])
    distances = Dict{Int, Int}(s => 0)
    frontier = Int[s]

    for _ in 1:max_iters
        isempty(frontier) && break
        s = popfirst!(frontier)
        d = distances[s]
        for neigh in neighbours(s, game)
            neigh in visited && continue
            is_terminal(neigh, game) && return d + 1
            push!(visited, neigh)
            distances[neigh] = d + 1
            push!(frontier, neigh)
        end
    end

    throw(ErrorException("Could not find solution when computing Lopt"))
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

function neighbours_actions(s::Int, game::TowerOfLondon)::Tuple{Vector{Int}, Vector{Action}}
    neighs = Int[]
    actions = Action[]

    for from in 1:6
        if from in (1, 2, 4) && check_digit(s, from + 1, game) > 0
            continue
        end
        check_digit(s, from, game) == 0 && continue

        _push_tol_move!(neighs, actions, s, game, from, 6; unless=from == 6)

        if check_digit(s, 5, game) == 0 && from != 5 && from != 4
            to = check_digit(s, 4, game) == 0 ? 4 : 5
            push!(neighs, make_move(s, (from, to), game))
            push!(actions, (from, to))
        end

        if check_digit(s, 3, game) == 0 && !(from in (1, 2, 3))
            to = if check_digit(s, 2, game) == 0
                check_digit(s, 1, game) == 0 ? 1 : 2
            else
                3
            end
            push!(neighs, make_move(s, (from, to), game))
            push!(actions, (from, to))
        end
    end

    return neighs, actions
end

function _push_tol_move!(neighs::Vector{Int}, actions::Vector{Action}, s::Int, game::TowerOfLondon, from::Int, to::Int; unless::Bool=false)
    if !unless && check_digit(s, to, game) == 0
        push!(neighs, make_move(s, (from, to), game))
        push!(actions, (from, to))
    end
end

function state_actions_tol(problems, old_games)
    state_actions = Vector{Vector{Vector{Tuple{Int, Action}}}}[]

    for subject_games in old_games
        state_actions_subj = Vector{Vector{Tuple{Int, Action}}}[]
        for i in 1:25
            state_actions_game = Tuple{Int, Action}[]
            game = TowerOfLondon(get_base(4, 6), vcat(problems[i][1]...), vcat(problems[i][2]...))
            s = start_state(game)

            for pole_move in subject_games[i]
                pole_move[1] == pole_move[2] && continue
                move = _tower_pole_move_to_slots(s, pole_move, game)
                push!(state_actions_game, (s, move))
                s = make_move(s, move, game)
            end

            !is_terminal(s, game) && println("Error at: $((i, length(state_actions) + 1))")
            push!(state_actions_game, (s, (0, 0)))
            push!(state_actions_subj, [state_actions_game])
        end
        push!(state_actions, state_actions_subj)
    end

    return state_actions
end

function _tower_pole_move_to_slots(s::Int, pole_move::Action, game::TowerOfLondon)::Action
    from_pole, to_pole = pole_move
    from_slot = if from_pole == 1
        check_digit(s, 3, game) != 0 ? 3 : (check_digit(s, 2, game) != 0 ? 2 : 1)
    elseif from_pole == 2
        check_digit(s, 5, game) != 0 ? 5 : 4
    else
        6
    end

    to_slot = if to_pole == 1
        check_digit(s, 1, game) == 0 ? 1 : (check_digit(s, 2, game) == 0 ? 2 : 3)
    elseif to_pole == 2
        check_digit(s, 4, game) == 0 ? 4 : 5
    else
        6
    end

    return from_slot, to_slot
end

function get_tol_heuristics(N_t::Int, idx_to_state, state_to_idx, game::TowerOfLondon)
    all_moves = MVector{36, Tuple{Int64, Int64}}[]
    all_idxs = Vector{Int64}[]
    all_sizes = zeros(Int, N_t)
    target_arr = get_tol_array(goal_state(game), game)

    for i in 1:N_t
        moves = @MVector([(0, 0) for _ in 1:36])
        s = idx_to_state[i]
        s_arr = get_tol_array(s, game)
        all_sizes[i] = 3 - sum((s_arr .== target_arr) .& (s_arr .!= 0))

        neighs, actions = neighbours_actions(s, game)
        for (n, action) in enumerate(actions)
            moves[n] = action
        end
        for n in eachindex(neighs)
            idx = state_to_idx[neighs[n]]
            neighs[n] = idx < 0 ? N_t - idx : idx
        end

        push!(all_moves, moves)
        push!(all_idxs, neighs)
    end

    return all_moves, all_idxs, all_sizes
end
