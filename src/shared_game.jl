get_base(base::Int, n::Int)::Vector{Int} = base .^ (0:n - 1)

function start_state(game::AbstractGame)::Int
    s = 0
    for (piece, value) in enumerate(game.initial_position)
        s += value * game.bs[piece]
    end
    return s
end

goal_state(game::TowerOfLondon)::Int = sum(v * game.bs[piece] for (piece, v) in enumerate(game.goal_position))

check_digit(s::Int, n::Int, game::AbstractGame)::Int = (s ÷ game.bs[n]) % game.bs[2]

function is_first_player(s::Int, game::AbstractGame)::Bool
    return mod(sum(check_digit(s, n, game) for n in eachindex(game.bs)), 2) == 0
end

function remove_piece(s::Int, move::Action, game::AbstractGame)
    return s - move[1] * game.bs[move[2]]
end

function add_piece(s::Int, move::Action, game::AbstractGame)
    return s + move[1] * game.bs[move[2]]
end

function move_piece(s::Int, move::Action, game::AbstractGame)
    piece = check_digit(s, move[1], game)
    s = remove_piece(s, (piece, move[1]), game)
    return add_piece(s, (piece, move[2]), game)
end

function move_piece_capture(s::Int, move::Action, game::AbstractGame)
    piece = check_digit(s, move[1], game)
    piece_captured = check_digit(s, move[2], game)
    s = remove_piece(s, (piece_captured, move[2]), game)
    s = remove_piece(s, (piece, move[1]), game)
    return add_piece(s, (piece, move[2]), game)
end

function move_piece_amount(s::Int, move::Action, game::AbstractGame)::Int
    return check_digit(s, move[1], game) * (game.bs[move[2]] - game.bs[move[1]])
end

function swap_turn(s::Int, game::AbstractGame)
    return s + game.bs[end] * (1 - 2 * check_digit(s, lastindex(game.bs), game))
end

is_terminal(s::Int, game::AbstractGame)::Bool = has_won(s, game) || has_lost(s, game)
is_single_player(game::AbstractGame)::Bool = true

function win_tie_loss(s::Int, game::AbstractGame)::Int
    if has_won(s, game)
        return 1
    elseif has_lost(s, game)
        return -1
    elseif is_terminal(s, game)
        return 0
    end
    error("State is not terminal")
end

function search(game::AbstractGame; max_iters=100_000)::Tuple{Set{Int}, Set{Int}}
    start = start_state(game)
    states = Set{Int}([start])
    leafs = Set{Int}()
    frontier = Int[start]

    for i in 1:max_iters
        if i % 100_000 == 0
            println("Iteration: $i")
        end
        isempty(frontier) && return states, leafs

        s = popfirst!(frontier)
        for neigh in neighbours(s, game)
            neigh in states && continue
            if is_terminal(neigh, game)
                push!(leafs, neigh)
            else
                push!(states, neigh)
                push!(frontier, neigh)
            end
        end
    end

    return Set{Int}(), Set{Int}()
end
