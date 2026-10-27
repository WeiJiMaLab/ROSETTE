function load_prb(prb::AbstractString)::Tuple{Vector{Int}, Vector{Car}}
    data = JSON.parsefile(joinpath("data", "problems", "$prb.json"))
    s_free = zeros(Int, length(data["cars"]))
    s_fixed = Vector{Car}(undef, length(data["cars"]))

    for car in data["cars"]
        id = car["id"] == "r" ? 1 : parse(Int, car["id"]) + 2
        x = 1 + car["position"] % BOARD_SIZE
        y = 1 + car["position"] ÷ BOARD_SIZE
        dir = car["orientation"] == "horizontal" ? :x : :y
        len = Int8(car["length"])

        if dir == :x
            s_free[id] = x - 1
            s_fixed[id] = Car(Int8(y), dir, len)
        else
            s_free[id] = y - 1
            s_fixed[id] = Car(Int8(x), dir, len)
        end
    end

    return s_free, s_fixed
end

function load_prb_(prb::AbstractString)::RushHourState
    s_free_zero_based, s_fixed = load_prb(prb)
    s_free = @MVector zeros(Int, RUSH_HOUR_CARS)
    for i in eachindex(s_free_zero_based)
        s_free[i] = s_free_zero_based[i] + 1
    end
    return s_free, Tuple(s_fixed)
end

make_move(s::Int, move::Action, game::RushHour)::Int = s + move[2] * game.bs[move[1]]

function neighbours(s::Int, game::RushHour)::Vector{Int}
    return [make_move(s, move, game) for move in get_all_moves(s, game)]
end

function get_all_moves(s::Int, game::RushHour)::Vector{Action}
    moves = Action[]
    arr = draw_state(s, game)

    for id in eachindex(game.s_fixed)
        car = game.s_fixed[id]
        dim1 = check_digit(s, id, game) + 1

        for (amount, board_pos) in enumerate(reverse(1:dim1 - 1))
            if _rush_hour_square_empty(arr, car, board_pos)
                push!(moves, (id, -amount))
            else
                break
            end
        end

        for (amount, board_pos) in enumerate(dim1 + car.len:BOARD_SIZE)
            if _rush_hour_square_empty(arr, car, board_pos)
                push!(moves, (id, amount))
            else
                break
            end
        end
    end

    return moves
end

function _rush_hour_square_empty(arr::RushHourBoard, car::Car, dim1::Integer)::Bool
    return car.dir == :x ? arr[car.dim2, dim1] == 0 : arr[dim1, car.dim2] == 0
end

function has_won(s::Int, game::RushHour)::Bool
    red_x = check_digit(s, 1, game)
    red_x == 4 && return true

    arr = draw_state(s, game)
    return unique(arr[3, red_x + 3:BOARD_SIZE]) == [0]
end

has_lost(s::Int, game::RushHour)::Bool = false

function draw_state(s::Int, game::RushHour)::RushHourBoard
    arr = zeros(Int, BOARD_SIZE, BOARD_SIZE)

    for id in eachindex(game.s_fixed)
        car = game.s_fixed[id]
        dim1 = check_digit(s, id, game) + 1
        for offset in 0:car.len - 1
            if car.dir == :x
                arr[car.dim2, dim1 + offset] = id
            else
                arr[dim1 + offset, car.dim2] = id
            end
        end
    end

    return arr
end

function int_to_s(s_int::Int, game::RushHour)::RushHourState
    s_fixed = Tuple(game.s_fixed)
    s_free = @MVector zeros(Int, RUSH_HOUR_CARS)
    for i in 1:RUSH_HOUR_CARS
        s_free[i] = check_digit(s_int, i, game) + 1
    end
    return s_free, s_fixed
end

function s_to_int(s::RushHourState)::Int
    s_free, s_fixed = s
    game = RushHour(get_base(5, RUSH_HOUR_CARS), collect(s_free .- 1), collect(s_fixed))
    return start_state(game)
end

function board_to_int32(s::RushHourState)::Tuple{Int64, Int32}
    s_free, s_fixed = s
    length(s_free) > 13 && throw(ErrorException("Cannot convert states with more than 13 cars to Int32"))

    problem = Int64(0)
    state = Int32(0)
    for (n, car) in enumerate(s_fixed)
        problem += (car.dim2 - 1) * 6^(n - 1)
        problem += (car.dir == :x) * 2^(34 + n - 1)
        problem += (car.len == 2) * 2^(48 + n - 1)
    end
    for (n, position) in enumerate(s_free)
        state += (position - 1) * 5^(n - 1)
    end
    return problem, state
end

function make_move!(s_free::RushHourFreeState, move::Action)::Nothing
    s_free[move[1]] += move[2]
    return nothing
end

function make_move(s_free::RushHourFreeState, move::Action)::RushHourFreeState
    s_free_new = copy(s_free)
    make_move!(s_free_new, move)
    return s_free_new
end

function overlap(s1::Tuple{Int64, Car}, s2::Tuple{Int64, Car})::Bool
    dim11, car1 = s1
    dim12, car2 = s2

    if car1.dir == car2.dir
        return car1.dim2 == car2.dim2 && dim12 - car1.len < dim11 < dim12 + car2.len
    end

    return (dim11 + car1.len > car2.dim2 >= dim11) &&
           (dim12 + car2.len > car1.dim2 >= dim12)
end

function check_solved(s::RushHourState)::Bool
    s_free, s_fixed = s
    red_dim1 = s_free[1]
    red_car = s_fixed[1]
    ghost_car = Car(red_car.dim2, red_car.dir, Int8(BOARD_SIZE - (red_dim1 + red_car.len - 1)))
    ghost_state = (Int64(red_dim1 + red_car.len), ghost_car)

    for id in eachindex(s_free)
        if s_fixed[id].dir != ghost_car.dir && overlap((Int64(s_free[id]), s_fixed[id]), ghost_state)
            return false
        end
    end

    return true
end

function board_to_arr!(arr::RushHourBoard, s::RushHourState)::Nothing
    s_free, s_fixed = s
    fill!(arr, 0)

    for id in eachindex(s_free)
        car = s_fixed[id]
        dim1 = s_free[id]
        for offset in 0:car.len - 1
            if car.dir == :x
                arr[car.dim2, dim1 + offset] = id
            else
                arr[dim1 + offset, car.dim2] = id
            end
        end
    end

    return nothing
end

function possible_moves!(moves::RushHourMoves, s::RushHourState, arr::RushHourBoard)::Nothing
    fill!(moves, (0, 0))
    s_free, s_fixed = s
    out = 0

    for id in eachindex(s_free)
        for direction in (-1, 1)
            for amount in 1:4
                car = s_fixed[id]
                dim2 = car.dim2
                dim1 = s_free[id] + (direction == 1 ? car.len - 1 : 0) + direction * amount
                (dim1 < 1 || dim1 > BOARD_SIZE) && break

                occupied = car.dir == :x ? arr[dim2, dim1] : arr[dim1, dim2]
                occupied != 0 && break

                out += 1
                moves[out] = (id, direction * amount)
            end
        end
    end

    return nothing
end

function moves_to_idxs(moves::RushHourMoves, s::RushHourState, state_to_idx, N_t::Int)::Vector{Int}
    s_free, s_fixed = s
    idxs = Int[]

    for move in moves
        move == (0, 0) && break
        next_free = make_move(s_free, move)
        next_int = s_to_int((next_free, s_fixed))
        idx = state_to_idx[next_int]
        push!(idxs, idx < 0 ? N_t - idx : idx)
    end

    return idxs
end

function move_blocked_by!(blocked_cars::BlockingCars, move::Action, s::RushHourState, arr::RushHourBoard)::Nothing
    fill!(blocked_cars, 0)
    out = 0
    prev_car = 0
    s_free, s_fixed = s
    id, amount = move

    for step in 1:abs(amount)
        if s_fixed[id].dir == :x
            y = s_fixed[id].dim2
            x = amount > 0 ? s_free[id] + s_fixed[id].len - 1 + step : s_free[id] - step
        else
            x = s_fixed[id].dim2
            y = amount > 0 ? s_free[id] + s_fixed[id].len - 1 + step : s_free[id] - step
        end

        car = arr[y, x]
        (car == 0 || car == prev_car) && continue
        out += 1
        blocked_cars[out] = car
        prev_car = car
    end

    return nothing
end

function unblocking_moves!(move_amounts::MoveAmounts, move::Action, blocked_id::Int, s::RushHourState)::Nothing
    fill!(move_amounts, 0)
    out = 0
    s_free, s_fixed = s
    moving_id, amount = move

    behind = s_free[blocked_id] - 1
    in_front = BOARD_SIZE - (s_free[blocked_id] + s_fixed[blocked_id].len - 1)

    for id in eachindex(s_free)
        same_lane = s_fixed[id].dim2 == s_fixed[blocked_id].dim2
        same_dir = s_fixed[id].dir == s_fixed[blocked_id].dir
        if id != blocked_id && same_lane && same_dir
            if s_free[id] > s_free[blocked_id]
                in_front -= s_fixed[id].len
            else
                behind -= s_fixed[id].len
            end
        end
    end

    if s_fixed[moving_id].dir == s_fixed[blocked_id].dir
        if amount < 0
            needed = (s_free[blocked_id] + s_fixed[blocked_id].len - 1) - (s_free[moving_id] + amount)
            for i in needed + 1:behind
                out += 1
                move_amounts[out] = -i
            end
        else
            needed = (s_free[moving_id] + s_fixed[moving_id].len - 1 + amount) - s_free[blocked_id]
            for i in needed + 1:in_front
                out += 1
                move_amounts[out] = i
            end
        end
    else
        behind_needed = s_free[blocked_id] + s_fixed[blocked_id].len - 1 - s_fixed[moving_id].dim2
        for i in behind_needed + 1:behind
            out += 1
            move_amounts[out] = -i
        end

        front_needed = s_fixed[moving_id].dim2 - s_free[blocked_id]
        for i in front_needed + 1:in_front
            out += 1
            move_amounts[out] = i
        end
    end

    return nothing
end
