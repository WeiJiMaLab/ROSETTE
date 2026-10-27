function get_and_or_tree(s::RushHourState; max_iter=100)
    train_of_thought = Vector{Subgoal}()
    parents_moves = DefaultDict{Action, Vector{AndNode}}([])
    AND = DefaultDict{AndNode, Vector{OrNode}}([])
    OR = DefaultDict{OrNode, Vector{AndNode}}([])

    arr = zeros(Int, BOARD_SIZE, BOARD_SIZE)
    board_to_arr!(arr, s)
    s_free, _ = s

    root_move = (1, BOARD_SIZE - (s_free[1] + 1))
    first_thought = MVector{5, Int}(1, root_move[2], 0, 0, 0)
    push!(train_of_thought, first_thought)

    OR_root = (1, first_thought)
    AND_root = (1, root_move)
    push!(OR[OR_root], AND_root)

    blocked_cars = zeros(BlockingCars)
    move_amounts = zeros(MoveAmounts)
    blocking_nodes = zeros(Int, 4, 5)
    get_blocking_nodes!(blocking_nodes, blocked_cars, move_amounts, s, arr, root_move)

    moves = @MVector([(0, 0) for _ in 1:36])
    possible_moves!(moves, s, arr)

    tree = [AND_root, AND, OR, parents_moves]
    add_to_tree!(AND_root, blocking_nodes, blocked_cars, move_amounts, train_of_thought, moves, tree, s, arr, 0; max_iter=max_iter)

    if isempty(AND)
        leaf = (1, zeros(Subgoal))
        push!(AND[AND_root], leaf)
    end

    AND_root, AND, OR, parents_moves = tree
    return [AND_root, Dict(AND), Dict(OR), Dict(parents_moves)]
end

function add_to_tree!(
    prev_AND::AndNode,
    blocking_nodes::BlockingNodes,
    blocked_cars::BlockingCars,
    move_amounts::MoveAmounts,
    train_of_thought::Vector{Subgoal},
    moves::RushHourMoves,
    tree,
    s::RushHourState,
    arr::RushHourBoard,
    recursion_depth::Int;
    max_iter=100,
)::Nothing
    recursion_depth > max_iter && throw(DomainError("max_iter depth reached"))

    _, AND, OR, parents_moves = tree
    d, _ = prev_AND

    for node in eachrow(blocking_nodes)
        node[1] == 0 && break
        or_node = (d + 1, node)
        or_node in AND[prev_AND] || push!(AND[prev_AND], or_node)
        node in train_of_thought && continue

        for j in 2:lastindex(node)
            node[j] == 0 && break
            next_move = (node[1], node[j])

            new_blocking_nodes = copy(blocking_nodes)
            get_blocking_nodes!(new_blocking_nodes, blocked_cars, move_amounts, s, arr, next_move)
            new_blocking_nodes[1, 1] == -1 && continue

            and_node = (d + 1, next_move)
            and_node in OR[or_node] || push!(OR[or_node], and_node)

            if new_blocking_nodes[1, 1] == 0
                leaf = (d + 1, zeros(Subgoal))
                leaf in AND[and_node] || push!(AND[and_node], leaf)
                and_node in parents_moves[next_move] || push!(parents_moves[next_move], and_node)
                continue
            end

            new_train = copy(train_of_thought)
            push!(new_train, node)
            add_to_tree!(and_node, new_blocking_nodes, blocked_cars, move_amounts, new_train, moves, tree, s, arr, recursion_depth + 1; max_iter=max_iter)
        end
    end

    return nothing
end

function get_blocking_nodes!(
    blocking_nodes::BlockingNodes,
    blocked_cars::BlockingCars,
    move_amounts::MoveAmounts,
    s::RushHourState,
    arr::RushHourBoard,
    move::Action,
)::Nothing
    fill!(blocking_nodes, 0)
    move_blocked_by!(blocked_cars, move, s, arr)

    for i in eachindex(blocked_cars)
        blocked_id = blocked_cars[i]
        blocked_id == 0 && break

        unblocking_moves!(move_amounts, move, blocked_id, s)
        if sum(move_amounts) == 0
            fill!(blocking_nodes, -1)
            break
        end

        blocking_nodes[i, 1] = blocked_id
        for j in eachindex(move_amounts)
            blocking_nodes[i, j + 1] = move_amounts[j]
        end
    end

    return nothing
end

function propagate_ps(x::Float64, tree)::Dict{AndNode, Float64}
    AND_root, AND, OR, _ = tree
    dict = DefaultDict{AndNode, Float64}(0.0)
    dict[(1, (-1, -1))] = x

    if length(AND) == length(OR) == 1
        dict[(1, AND_root[2])] = 1 - x
    else
        train = Vector{Subgoal}()
        push!(train, MVector{5, Int}([AND_root[2][1], AND_root[2][2], 0, 0, 0]))
        _propagate_and_or!(x, 1 - x, train, dict, AND_root, AND, OR)
    end

    return Dict(dict)
end

function _propagate_and_or!(
    γ::Float64,
    parent_probability::Float64,
    train_of_thought,
    dict,
    current_and::AndNode,
    AND,
    OR,
)::Nothing
    N_or = length(AND[current_and])
    p_ors = parent_probability .* ones(N_or) ./ N_or

    for (n, OR_node) in enumerate(AND[current_and])
        p_or = p_ors[n]

        if OR_node[2] in train_of_thought || OR_node ∉ keys(OR)
            dict[(OR_node[1], (-2, -2))] += (1 - γ) * p_or
            dict[(OR_node[1], (-1, -1))] += γ * p_or
            continue
        end

        push!(train_of_thought, OR_node[2])
        p_continue = (1 - γ) * p_or
        dict[(OR_node[1], (-1, -1))] += γ * p_or

        N_and = length(OR[OR_node])
        p_ands = p_continue .* ones(N_and) ./ N_and
        for (m, AND_next) in enumerate(OR[OR_node])
            p_and = p_ands[m]
            if AND[AND_next][1][2][1] == 0
                dict[AND_next] += p_and
            else
                _propagate_and_or!(γ, p_and, copy(train_of_thought), dict, AND_next, AND, OR)
            end
        end
    end

    return nothing
end

function get_and_or_dicts(N_t::Int, idx_to_state, state_to_idx, game::RushHour)
    all_dicts = Dict{Tuple{Int64, Tuple{Int64, Int64}}, Float64}[]
    all_moves = MVector{36, Tuple{Int64, Int64}}[]
    all_idxs = Vector{Int64}[]
    all_sizes = Int64[]

    arr = zeros(Int, BOARD_SIZE, BOARD_SIZE)
    moves = @MVector([(0, 0) for _ in 1:36])

    for s_idx in 1:N_t
        state_int = idx_to_state[s_idx]
        state = int_to_s(state_int, game)
        tree = get_and_or_tree(state)
        dict = propagate_ps(0.0, tree)

        board_to_arr!(arr, state)
        possible_moves!(moves, state, arr)
        idxs = moves_to_idxs(moves, state, state_to_idx, N_t)

        push!(all_dicts, dict)
        push!(all_moves, copy(moves))
        push!(all_idxs, idxs)
        push!(all_sizes, length(tree[3]))
    end

    return all_dicts, all_moves, all_idxs, all_sizes
end
