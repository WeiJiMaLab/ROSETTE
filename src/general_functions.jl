
abstract type AbstractGame end
states_type = Set{Int64}
leafs_type = Set{Int64}

function get_base(base::Int, n::Int)::Vector{Int}
    return base .^ (0:n-1)
end

function start_state(game::AbstractGame)::Int
    s = 0
    for (p, v) in enumerate(game.initial_position)
        s += v*game.bs[p]
    end
    return s
end

function goal_state(game::AbstractGame)::Int
    s = 0
    for (p, v) in enumerate(game.goal_position)
        s += v*game.bs[p]
    end
    return s
end

function is_first_player(s::Int, game::AbstractGame)::Bool
    if is_single_player(game)
        return true
    end
    return check_digit(s, game.check_turn_bit, game) == 0
end

function check_digit(s::Int, n::Int, game::AbstractGame)::Int
    return (s ÷ game.bs[n]) % game.bs[2]
end

function remove_piece(s::Int, m::Tuple{Int, Int}, game::AbstractGame)
    piece, from = m
    return s - piece*game.bs[from]
end

function add_piece(s::Int, m::Tuple{Int, Int}, game::AbstractGame)
    piece, to = m
    return s + piece*game.bs[to]
end

function move_piece(s::Int, m::Tuple{Int, Int}, game::AbstractGame)
    from, to = m
    piece = check_digit(s, from, game)
    s = remove_piece(s, (piece, from), game)
    s = add_piece(s, (piece, to), game)
    return s
end

function move_piece_capture(s::Int, m::Tuple{Int, Int}, game::AbstractGame)
    from, to = m
    piece = check_digit(s, from, game)
    s = remove_piece(s, (piece, from), game)
    captured_piece = check_digit(s, to, game)
    s = remove_piece(s, (captured_piece, to), game)
    s = add_piece(s, (piece, to), game)
    return s
end

function move_piece_amount(s::Int, m::Tuple{Int, Int}, game::AbstractGame)::Int
    # if encoding is such that each unit refers to a piece and not a location
    piece, amount = m
    return s + amount*game.bs[piece]
end

function swap_turn(s::Int, game::AbstractGame)
    if is_first_player(s, game)
        return s + game.bs[game.check_turn_bit]
    else
        return s - game.bs[game.check_turn_bit]
    end
end

function is_terminal(s::Int, game::AbstractGame)::Bool
    # if a two-player game, check if state is not a win, tie, or loss
    return win_tie_loss(s, game) != -2
end

function is_single_player(game::AbstractGame)::Bool
    return :check_turn_bit ∉ fieldnames(typeof(game))
end

function win_tie_loss(s::Int, game::AbstractGame)::Int
    if is_single_player(game)
        if has_won(s, game)
            return 1
        elseif has_lost(s, game)
            return -1
        else
            return -2
        end
    end
    # 1 is win, 0 is tie, -1 is loss, -2 is not terminal
    if has_lost(s, game)
        if is_first_player(s, game)
            return -1
        else
            return 1
        end
    end
    if has_tied(s, game)
        return 0
    else
        return -2
    end
end

function search(game::AbstractGame; max_iters=100000)::Tuple{states_type, leafs_type}
    s_start = start_state(game)
    states = Set{Int}(s_start)
    leafs = Set{Int}()
    frontier = Int[s_start]
    for i in 1:max_iters
        if i % 100000 == 0
            println("Iteration: $i")
        end
        if isempty(frontier)
            return states, leafs
        end
        s = popfirst!(frontier)
        for neigh in neighbours(s, game)
            if neigh in states
                continue
            end
            if is_terminal(neigh, game)
                push!(leafs, neigh)
                continue
            end
            push!(states, neigh)
            push!(frontier, neigh)
        end
    end
    return Set{Int}(), Set{Int}()
    #error("Maximum search depth reached")
end