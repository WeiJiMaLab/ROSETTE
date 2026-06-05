function _trim_moves(moves)
    trimmed = Action[]
    for move in moves
        move == (0, 0) && break
        push!(trimmed, move)
    end
    return trimmed
end

function _infer_domain(fi)::Symbol
    return isempty(fi.all_dicts) ? :tol : :rh
end

function prepare_fitting_info(fi; domain::Symbol=_infer_domain(fi))
    n_absorbing, n_transient = size(fi.R)
    all_idxs = [Vector{Int}(idxs) for idxs in fi.all_idxs]
    all_moves = [_trim_moves(moves) for moves in fi.all_moves]
    all_sizes = Vector{Int}(fi.all_sizes)
    all_dicts = [Dict{Tuple{Int64, Tuple{Int64, Int64}}, Float64}(dict) for dict in fi.all_dicts]
    return FastFittingInfo(domain, n_transient, n_absorbing, all_idxs, all_moves, all_sizes, all_dicts)
end

function prepare_fitting_infos(fitting_infos; domain::Union{Nothing, Symbol}=nothing)
    if domain === nothing
        return [prepare_fitting_info(fi) for fi in fitting_infos]
    end
    return [prepare_fitting_info(fi; domain=domain) for fi in fitting_infos]
end

function prepare_dataset(state_actions)::FastDataset
    transitions = FastTransition[]
    for puzzle in eachindex(state_actions)
        isempty(state_actions[puzzle]) && continue
        for game in state_actions[puzzle]
            length(game) < 2 && continue
            previous_move = (0, 0)
            state, action = game[1]
            for i in 1:(length(game) - 1)
                next_state, next_action = game[i + 1]
                push!(transitions, FastTransition(puzzle, state, next_state, previous_move))
                previous_move = action
                state = next_state
                action = next_action
            end
        end
    end
    return FastDataset(transitions)
end

function probability_params(params, info::FastFittingInfo; parameterization::Symbol=:fit)
    length(params) == 3 || throw(ArgumentError("Expected exactly 3 main-model parameters, got $(length(params))"))

    if parameterization == :probability
        return (Float64(params[1]), Float64(params[2]), Float64(params[3]))
    elseif parameterization == :fit
        if info.domain == :rh
            return (10.0^(-params[1]), 10.0^(-params[2]), 10.0^(-params[3]))
        elseif info.domain == :tol
            return (Float64(params[1]), 10.0^(-params[2]), 10.0^(-params[3]))
        end
    end

    throw(ArgumentError("parameterization must be :fit or :probability, got $parameterization"))
end

function _valid_probability_params(info::FastFittingInfo, heuristic::Float64, gamma_r::Float64, gamma_p::Float64)::Bool
    if info.domain == :rh && !(1e-5 <= heuristic <= 1.0)
        return false
    end
    return 1e-5 <= gamma_r <= 1.0 && 1e-5 <= gamma_p <= 1.0
end

function _policy_key(info::FastFittingInfo, heuristic::Float64, gamma_r::Float64, previous_move::Action)
    move_key = info.domain == :tol ? (0, 0) : previous_move
    domain_key = info.domain == :rh ? 1 : 2
    return (domain_key, heuristic, gamma_r, move_key)
end

function _rh_action_probs(info::FastFittingInfo, state::Int, gamma_ao::Float64, previous_move::Action)
    idxs = info.all_idxs[state]
    moves = info.all_moves[state]
    move_probs = Dict{Action, Float64}(move => 0.0 for move in moves)
    move_probs[(-1, -1)] = 0.0
    move_probs[(-2, -2)] = 0.0

    excluded = previous_move == (0, 0) ? Set{Action}([(0, 0)]) : Set{Action}((previous_move[1], amount) for amount in -4:4)
    temp = Dict{Tuple{Int64, Tuple{Int64, Int64}}, Float64}()
    for (key, value) in info.all_dicts[state]
        if value > 0
            temp[key] = value * (1 - gamma_ao)^key[1]
        end
    end
    temp[(0, (-1, -1))] = 1 - sum(values(temp))

    p_excluded = 0.0
    for (and_node, value) in temp
        move = and_node[2]
        if move in excluded
            p_excluded += value
        else
            move_probs[move] = get(move_probs, move, 0.0) + value
        end
    end
    move_probs[(-2, -2)] += p_excluded

    probs = zeros(Float64, length(idxs))
    normalizer = 0.0
    for i in eachindex(idxs)
        prob = get(move_probs, moves[i], 0.0)
        probs[i] = prob
        normalizer += prob
    end
    normalizer += move_probs[(-1, -1)]

    if normalizer > 0
        cycle_prob = move_probs[(-2, -2)] / normalizer
        for i in eachindex(probs)
            probs[i] += probs[i] * cycle_prob
        end
        move_probs[(-1, -1)] += move_probs[(-1, -1)] * cycle_prob
    else
        probs .+= move_probs[(-2, -2)] / length(probs)
    end

    probs .+= move_probs[(-1, -1)] / length(probs)
    return probs
end

function _tol_action_probs(info::FastFittingInfo, state::Int, beta::Float64)
    idxs = info.all_idxs[state]
    weights = Vector{Float64}(undef, length(idxs))
    for i in eachindex(idxs)
        idx = idxs[i]
        value = idx <= info.n_transient ? info.all_sizes[idx] : 0
        weights[i] = exp(-beta * value)
    end
    weights ./= sum(weights)
    return weights
end

function _build_policy_bundle(info::FastFittingInfo, heuristic::Float64, gamma_r::Float64, previous_move::Action)
    action_probs = Vector{Vector{Float64}}(undef, info.n_transient)
    q = zeros(Float64, info.n_transient, info.n_transient)

    for state in 1:info.n_transient
        probs = info.domain == :rh ? _rh_action_probs(info, state, heuristic, previous_move) : _tol_action_probs(info, state, heuristic)
        action_probs[state] = probs
        idxs = info.all_idxs[state]
        for i in eachindex(idxs)
            idx = idxs[i]
            idx <= info.n_transient && (q[idx, state] = probs[i])
        end
    end

    p_win = (I - ((1 - gamma_r) * q')) \ (gamma_r * ones(info.n_transient))
    p_stop = vcat(1 .- p_win, ones(info.n_absorbing))
    final_probs = Vector{Union{Nothing, Vector{Float64}}}(nothing, info.n_transient)
    return PolicyBundle(action_probs, p_stop, final_probs)
end

function _policy_bundle!(cache::PolicyCache, info::FastFittingInfo, heuristic::Float64, gamma_r::Float64, previous_move::Action)
    key = _policy_key(info, heuristic, gamma_r, previous_move)
    return get!(cache, key) do
        _build_policy_bundle(info, heuristic, gamma_r, key[4])
    end
end

function _inf_sum_reduced(policy::Vector{Float64}, p_stop::Vector{Float64}, gamma_p::Float64; K::Int=10_000_000)
    result = zeros(Float64, length(policy))
    updated = copy(policy)
    continuation = 1.0
    weight = gamma_p
    multiplier = exp.(p_stop)

    @inbounds for _ in 0:K
        normalizer = 0.0
        for i in eachindex(updated)
            updated[i] *= multiplier[i]
            normalizer += updated[i]
        end
        for i in eachindex(updated)
            updated[i] /= normalizer
            result[i] += weight * updated[i]
        end
        abs(1 - sum(result)) < 1e-6 && break
        continuation *= 1 - gamma_p
        weight = gamma_p * continuation
    end

    return result
end

function _state_probs!(bundle::PolicyBundle, info::FastFittingInfo, state::Int, gamma_p::Float64; K::Int)
    cached = bundle.final_probs[state]
    cached !== nothing && return cached

    idxs = info.all_idxs[state]
    stop_values = [bundle.p_stop[idx] for idx in idxs]
    probs = _inf_sum_reduced(bundle.action_probs[state], stop_values, gamma_p; K=K)
    bundle.final_probs[state] = probs
    return probs
end

function _transition_probability!(cache::PolicyCache, info::FastFittingInfo, state::Int, next_state::Int, previous_move::Action, heuristic::Float64, gamma_r::Float64, gamma_p::Float64; K::Int)
    bundle = _policy_bundle!(cache, info, heuristic, gamma_r, previous_move)
    probs = _state_probs!(bundle, info, state, gamma_p; K=K)
    idx = findfirst(==(next_state), info.all_idxs[state])
    idx === nothing && return 0.0
    return probs[idx]
end

function nll_main(params, dataset::FastDataset, fitting_infos::Vector{FastFittingInfo}; K::Int=10_000_000, parameterization::Symbol=:fit)
    isempty(fitting_infos) && return 0.0
    heuristic, gamma_r, gamma_p = probability_params(params, first(fitting_infos); parameterization=parameterization)
    _valid_probability_params(first(fitting_infos), heuristic, gamma_r, gamma_p) || return Inf

    cache = PolicyCache()
    nll = 0.0
    for transition in dataset.transitions
        info = fitting_infos[transition.puzzle]
        prob = _transition_probability!(
            cache,
            info,
            transition.state,
            transition.next_state,
            transition.previous_move,
            heuristic,
            gamma_r,
            gamma_p;
            K=K,
        )
        prob <= 0 && return Inf
        nll -= log(prob)
    end
    return nll
end

function nll_main(params, state_actions, fitting_infos::Vector{FastFittingInfo}; kwargs...)
    return nll_main(params, prepare_dataset(state_actions), fitting_infos; kwargs...)
end
