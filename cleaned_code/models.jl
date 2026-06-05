function get_stopping_prob(α, f)
    !(1 > f > 0) && throw(DomainError("f must be between 0 and 1, is $f"))
    α == 1.0 && return 0.0
    return f ^ tan(α * π / 2)
end

function get_stopping_probabilities!(
    P::Matrix{Float64},
    γp::Float64,
    α::Float64,
    root_state::Int,
    all_sizes::Vector{Int},
)::Nothing
    V_init = -all_sizes[root_state]
    g = 1 - γp

    for s in eachindex(all_sizes)
        V = -all_sizes[s]
        diff = V - V_init
        if diff < 0
            stopping_prob = get_stopping_prob(α, diff / V)
            P[s, 1] = 0
            P[s, 2] = stopping_prob * g
        else
            P[s, 1] = 0
            P[s, 2] = 0
        end
        P[s, 3] = γp
    end

    return nothing
end

function get_stopping_probabilities!(
    P::Matrix{Float64},
    γp::Float64,
    αp::Float64,
    αn::Float64,
    root_state::Int,
    all_sizes::Vector{Int},
)::Nothing
    V_init = -all_sizes[root_state]
    g = 1 - γp

    for s in eachindex(all_sizes)
        V = -all_sizes[s]
        diff = V - V_init
        if diff < 0
            P[s, 1] = 0
            P[s, 2] = get_stopping_prob(αn, diff / V) * g
        elseif diff > 0
            P[s, 1] = get_stopping_prob(αp, diff / -V_init) * g
            P[s, 2] = 0
        else
            P[s, 1] = 0
            P[s, 2] = 0
        end
        P[s, 3] = γp
    end

    return nothing
end

function apply_γp(s_init::Int, γp::Float64, α::Float64, fi::FittingStruct)::Matrix{Float64}
    get_stopping_probabilities!(fi.P, γp, α, s_init, fi.all_sizes)
    return _absorption_feature_matrix(fi)
end

function apply_γp(s_init::Int, γp::Float64, αp::Float64, αn::Float64, fi::FittingStruct)::Matrix{Float64}
    get_stopping_probabilities!(fi.P, γp, αp, αn, s_init, fi.all_sizes)
    return _absorption_feature_matrix(fi)
end

function apply_γp(γr::Float64, fi::FittingStruct)::Vector{Float64}
    N_t, _ = size(fi.Q)
    p_win = (I - ((1 - γr) * fi.Q')) \ (γr * ones(N_t))
    return 1 .- p_win
end

function _absorption_feature_matrix(fi::FittingStruct)::Matrix{Float64}
    sm = 1 .- sum(fi.P, dims=2)
    RR = sm .* fi.R'
    R_win = sum(RR, dims=2)
    return (I - (sm .* fi.Q')) \ hcat(R_win, fi.P)
end

function p_s(neighs::Vector{Int64}, p_policy::Vector{Float64}, B::Matrix{Float64})::Vector{Float64}
    ps = zeros(length(neighs))
    N_t, _ = size(B)

    p_stop_all = sum(p_policy[i] * B[neighs[i], 4] for i in eachindex(neighs) if neighs[i] <= N_t)
    p_restart = sum(p_policy[i] * B[neighs[i], 3] for i in eachindex(neighs) if neighs[i] <= N_t)

    if abs(p_restart - 1.0) < 1e-12
        ps .= p_policy
    else
        for i in eachindex(neighs)
            n = neighs[i]
            p_stop = n <= N_t ? B[n, 4] : 0
            p_lose_tie = n <= N_t ? B[n, 3] : 0
            ps[i] = p_policy[i] * (1 - p_stop - p_lose_tie + p_stop_all) / (1 - p_restart)
        end
    end

    return ps
end

function apply_γao(temp_dict::Dict{AndNode, Float64}, dict::Dict{AndNode, Float64}, γ::Float64)::Nothing
    empty!(temp_dict)
    for (k, v) in dict
        v > 0 && (temp_dict[k] = v * (1 - γ)^k[1])
    end
    temp_dict[(0, (-1, -1))] = 1 - sum(values(temp_dict))
    return nothing
end

function apply_γao_new(temp_dict::Dict{AndNode, Float64}, dict::Dict{AndNode, Float64}, γ::Float64, ϵ::Float64)::Nothing
    empty!(temp_dict)
    for (k, v) in dict
        v > 0 && (temp_dict[k] = v * (1 - γ)^k[1])
    end
    sm = sum(values(temp_dict))
    for (k, v) in temp_dict
        temp_dict[k] = ϵ * v + (1 - ϵ) * (v / sm)
    end
    temp_dict[(0, (-1, -1))] = ϵ * (1 - sm)
    return nothing
end

function process_dict!(ps, move_dict, all_moves, dict, excl_moves)::Nothing
    fill!(ps, 0)
    empty!(move_dict)

    N_neigh = 0
    for move in all_moves
        move == (0, 0) && break
        N_neigh += 1
        move_dict[move] = 0.0
    end
    move_dict[(-1, -1)] = 0.0
    move_dict[(-2, -2)] = 0.0

    p_excl = 0.0
    for and_node in keys(dict)
        move = and_node[2]
        if move ∉ excl_moves
            move_dict[move] += dict[and_node]
        else
            p_excl += dict[and_node]
        end
    end
    move_dict[(-2, -2)] += p_excl

    Z = 0.0
    for n in 1:N_neigh
        move = all_moves[n]
        ps[n] = move_dict[move]
        Z += ps[n]
    end
    Z += move_dict[(-1, -1)]

    if Z > 0
        p_cycle = move_dict[(-2, -2)] / Z
        for n in 1:N_neigh
            ps[n] += ps[n] * p_cycle
        end
        move_dict[(-1, -1)] += move_dict[(-1, -1)] * p_cycle
    else
        evenly = move_dict[(-2, -2)] / N_neigh
        for n in 1:N_neigh
            ps[n] += evenly
        end
    end

    evenly = move_dict[(-1, -1)] / N_neigh
    for n in 1:N_neigh
        ps[n] += evenly
    end

    return nothing
end

function update_QR_matrices_and_or!(fi::FittingStruct, γao::Float64, prev_move::Action)::Nothing
    _, N_t = size(fi.R)
    ps = zeros(Float64, 36)
    temp_dict = Dict{AndNode, Float64}()
    move_dict = Dict{Action, Float64}()
    same_car_moves = prev_move == (0, 0) ? [(0, 0)] : [(prev_move[1], j) for j in -4:4]

    for s in 1:N_t
        apply_γao(temp_dict, fi.all_dicts[s], γao)
        process_dict!(ps, move_dict, fi.all_moves[s], temp_dict, same_car_moves)

        for n in eachindex(fi.all_idxs[s])
            neigh = fi.all_idxs[s][n]
            if neigh > N_t
                fi.R[neigh - N_t, s] = ps[n]
            else
                fi.Q[neigh, s] = ps[n]
            end
        end
    end

    return nothing
end

function update_QR_matrices_tol!(fi::FittingStruct, β::Float64, prev_move::Action)::Nothing
    _, N_t = size(fi.R)

    for s in 1:N_t
        neighs = fi.Q[:, s].nzind
        vals = fi.all_sizes[neighs]
        if !isempty(fi.R[:, s].nzind)
            neighs = vcat(neighs, N_t .+ fi.R[:, s].nzind)
            vals = vcat(vals, 0)
        end
        ps = exp.(-β .* vals) ./ sum(exp.(-β .* vals))

        for n in eachindex(neighs)
            neigh = neighs[n]
            if neigh > N_t
                fi.R[neigh - N_t, s] = ps[n]
            else
                fi.Q[neigh, s] = ps[n]
            end
        end
    end

    return nothing
end

function get_ll_progress_rh(θ::Vector{Float64}, s::Int, s_next::Int, prev_move::Action, fi::FittingStruct)::Float64
    γao, γp, α = θ
    update_QR_matrices_and_or!(fi, γao, prev_move)
    B = apply_γp(s, γp, α, fi)
    _, N_t = size(fi.R)
    policy = vcat(fi.Q[:, s].nzval, fi.R[:, s].nzval)
    neighs = vcat(fi.Q[:, s].nzind, N_t .+ fi.R[:, s].nzind)
    p = p_s(neighs, policy, B)
    return log(p[findfirst(==(s_next), neighs)])
end

function get_p_progress_rh(θ::Vector{Float64}, s::Int, prev_move::Action, fi::FittingStruct)
    nlogγao, nlogγp, α = θ
    θp = [10.0^(-nlogγao), 10.0^(-nlogγp), α]

    update_QR_matrices_and_or!(fi, θp[1], prev_move)
    B = apply_γp(s, θp[2], θp[3], fi)
    _, N_t = size(fi.R)
    policy = vcat(fi.Q[:, s].nzval, fi.R[:, s].nzval)
    neighs = vcat(fi.Q[:, s].nzind, N_t .+ fi.R[:, s].nzind)
    return p_s(neighs, policy, B), neighs
end

function get_nll_progress_rh(θ::Vector{Float64}, state_actions, fitting_infos)
    nlogγao, nlogγp, α = θ
    γao = 10.0^(-nlogγao)
    γp = 10.0^(-nlogγp)
    if γao < 0 || γp < 0 || γao > 1 || γp > 1 || α < 0 || α > 1
        return Inf
    end
    return _sum_negative_log_likelihood([γao, γp, α], state_actions, fitting_infos, get_ll_progress_rh)
end

function inf_sum(a::Real, δp::Real, δn::Real, p::AbstractVector{<:Real}, B::AbstractMatrix{<:Real}, K::Int)
    N = length(p)
    @assert size(B, 1) == N
    @assert size(B, 2) == 4

    S = zeros(eltype(p), N)
    new_policy = copy(p)
    cont_prob = 1.0
    weight = a
    logM = δp .* (B[:, 1] .+ B[:, 2]) .+ δn .* B[:, 3]
    logw = similar(new_policy)

    @inbounds for _ in 0:K
        @simd for i in 1:N
            logw[i] = log(new_policy[i]) + logM[i]
        end
        m = maximum(logw)
        denom = zero(eltype(p))
        @simd for i in 1:N
            denom += exp(logw[i] - m)
        end
        logZ = m + log(denom)
        @simd for i in 1:N
            new_policy[i] = exp(logw[i] - logZ)
            S[i] += weight * new_policy[i]
        end
        abs(1 - sum(S)) < 1e-6 && break

        cont_prob *= 1 - a
        weight = a * cont_prob
    end

    return S
end

function inf_sum_reduced(γp::Float64, policy::Vector{Float64}, p_stop::Vector{Float64}; K::Int=10_000_000)
    N = length(policy)
    @assert length(p_stop) == N
    S = zeros(N)
    new_policy = copy(policy)
    cont_prob = 1.0
    weight = γp
    M = exp.(p_stop)

    @inbounds for _ in 0:K
        Z = 0.0
        for i in 1:N
            new_policy[i] *= M[i]
            Z += new_policy[i]
        end
        for i in 1:N
            new_policy[i] /= Z
            S[i] += weight * new_policy[i]
        end
        abs(1 - sum(S)) < 1e-6 && break

        cont_prob *= 1 - γp
        weight = γp * cont_prob
    end

    return S
end

function get_p_full_rh(params, s::Int, prev_move::Action, fi::FittingStruct; K::Int=1_000_000)
    if length(params) == 6
        nlogγao, nlogγp, α, nlogγre, δp, δn = params
        γao, γp, γre = 10.0^(-nlogγao), 10.0^(-nlogγp), 10.0^(-nlogγre)
        θp = [γao, γp, α, γre, δp, δn]
        return _get_p_full_rh_probability_params(θp, s, prev_move, fi; K=K)
    elseif length(params) == 7
        nlogγao, nlogγp, αp, αn, nlogγre, δp, δn = params
        γao, γp, γre = 10.0^(-nlogγao), 10.0^(-nlogγp), 10.0^(-nlogγre)
        θp = [γao, γp, αp, αn, γre, δp, δn]
        return _get_p_full_rh_probability_params(θp, s, prev_move, fi; K=K)
    end
    throw(ArgumentError("Rush Hour full model expects 6 or 7 parameters, got $(length(params))"))
end

function _get_p_full_rh_probability_params(θ, s::Int, prev_move::Action, fi::FittingStruct; K::Int)
    γao = θ[1]
    γp = θ[2]
    γre = θ[end - 2]
    δp = θ[end - 1]
    δn = θ[end]

    update_QR_matrices_and_or!(fi, γao, prev_move)
    B = length(θ) == 6 ? apply_γp(s, γp, θ[3], fi) : apply_γp(s, γp, θ[3], θ[4], fi)

    N_a, N_t = size(fi.R)
    policy = vcat(fi.Q[:, s].nzval, fi.R[:, s].nzval)
    neighs = vcat(fi.Q[:, s].nzind, N_t .+ fi.R[:, s].nzind)
    absorbing = zeros(N_a, 4)
    absorbing[:, 1] .= 1.0
    p = inf_sum(γre, δp, δn, policy, vcat(B, absorbing)[neighs, :], K)
    return p, neighs
end

function get_p_full_tol(params, s::Int, prev_move::Action, fi::FittingStruct; K::Int=1_000_000)
    β, nlogγp, αp, αn, nlogγre, δp, δn = params
    γp, γre = 10.0^(-nlogγp), 10.0^(-nlogγre)

    update_QR_matrices_tol!(fi, β, prev_move)
    B = apply_γp(s, γp, αp, αn, fi)
    N_a, N_t = size(fi.R)
    policy = vcat(fi.Q[:, s].nzval, fi.R[:, s].nzval)
    neighs = vcat(fi.Q[:, s].nzind, N_t .+ fi.R[:, s].nzind)
    absorbing = zeros(N_a, 4)
    absorbing[:, 1] .= 1.0
    p = inf_sum(γre, δp, δn, policy, vcat(B, absorbing)[neighs, :], K)
    return p, neighs
end

function _is_rush_hour_fitting_info(fi::FittingStruct)::Bool
    return !isempty(fi.all_dicts)
end

function main_probability_params(params, fi::FittingStruct; parameterization::Symbol=:fit)::Vector{Float64}
    length(params) == 3 || throw(ArgumentError("Main model expects exactly 3 parameters, got $(length(params))"))

    if parameterization == :probability
        return Float64[params[1], params[2], params[3]]
    elseif parameterization == :fit
        if _is_rush_hour_fitting_info(fi)
            return Float64[10.0^(-params[1]), 10.0^(-params[2]), 10.0^(-params[3])]
        end
        return Float64[params[1], 10.0^(-params[2]), 10.0^(-params[3])]
    end

    throw(ArgumentError("parameterization must be :fit or :probability, got $parameterization"))
end

function _valid_main_probability_params(θ::Vector{Float64}, fi::FittingStruct)::Bool
    heuristic, γr, γp = θ
    if _is_rush_hour_fitting_info(fi) && !(1e-5 <= heuristic <= 1.0)
        return false
    end
    return 1e-5 <= γr <= 1.0 && 1e-5 <= γp <= 1.0
end

function _get_p_main_probability_params(θ::Vector{Float64}, s::Int, prev_move::Action, fi::FittingStruct; K::Int=10_000_000)
    heuristic, γr, γp = θ
    if _is_rush_hour_fitting_info(fi)
        update_QR_matrices_and_or!(fi, heuristic, prev_move)
    else
        update_QR_matrices_tol!(fi, heuristic, prev_move)
    end

    p_stop = apply_γp(γr, fi)
    N_a, N_t = size(fi.R)
    policy = vcat(fi.Q[:, s].nzval, fi.R[:, s].nzval)
    neighs = vcat(fi.Q[:, s].nzind, N_t .+ fi.R[:, s].nzind)
    append!(p_stop, ones(N_a))
    return inf_sum_reduced(γp, policy, p_stop[neighs]; K=K), neighs
end

function get_p_main(params, s::Int, prev_move::Action, fi::FittingStruct; K::Int=10_000_000, parameterization::Symbol=:probability)
    θ = main_probability_params(params, fi; parameterization=parameterization)
    _valid_main_probability_params(θ, fi) || throw(DomainError(params, "main-model probabilities must be in valid ranges"))
    return _get_p_main_probability_params(θ, s, prev_move, fi; K=K)
end

function get_ll_main(params, s::Int, s_next::Int, prev_move::Action, fi::FittingStruct; K::Int=10_000_000, parameterization::Symbol=:probability)::Float64
    p, neighs = get_p_main(params, s, prev_move, fi; K=K, parameterization=parameterization)
    return log(p[findfirst(==(s_next), neighs)])
end

function get_nll_main(params::Vector{Float64}, state_actions, fitting_infos; K::Int=10_000_000, parameterization::Symbol=:fit)
    fi = first(fitting_infos)
    θ = main_probability_params(params, fi; parameterization=parameterization)
    _valid_main_probability_params(θ, fi) || return Inf
    ll_function = (θp, s, s_next, prev_move, fitting_info) -> get_ll_main(θp, s, s_next, prev_move, fitting_info; K=K, parameterization=:probability)
    return _sum_negative_log_likelihood(θ, state_actions, fitting_infos, ll_function)
end

get_p(θ::Vector{Float64}, s::Int, prev_move::Action, fi::FittingStruct; kwargs...) = get_p_main(θ, s, prev_move, fi; kwargs...)
get_ll(θ::Vector{Float64}, s::Int, s_next::Int, prev_move::Action, fi::FittingStruct; kwargs...) = get_ll_main(θ, s, s_next, prev_move, fi; kwargs...)
get_nll(θ::Vector{Float64}, state_actions, fitting_infos; kwargs...) = get_nll_main(θ, state_actions, fitting_infos; kwargs...)

function get_p_reduced_rh(params, s::Int, prev_move::Action, fi::FittingStruct; K::Int=10_000_000, parameterization::Symbol=:probability)
    _is_rush_hour_fitting_info(fi) || throw(ArgumentError("Rush Hour reduced model requires a Rush Hour fitting info"))
    return get_p_main(params, s, prev_move, fi; K=K, parameterization=parameterization)
end

function get_p_reduced_tol(params, s::Int, prev_move::Action, fi::FittingStruct; K::Int=10_000_000, parameterization::Symbol=:probability)
    _is_rush_hour_fitting_info(fi) && throw(ArgumentError("Tower of London reduced model requires a Tower of London fitting info"))
    return get_p_main(params, s, prev_move, fi; K=K, parameterization=parameterization)
end

function get_p_new(params, s::Int, prev_move::Action, fi::FittingStruct; K::Int=10_000_000, model::Symbol=:main, parameterization::Symbol=:probability)
    if model == :main || (model == :auto && length(params) == 3)
        return get_p_main(params, s, prev_move, fi; K=K, parameterization=parameterization)
    elseif model == :rh_reduced
        return get_p_reduced_rh(params, s, prev_move, fi; K=K, parameterization=parameterization)
    elseif model == :tol_reduced
        return get_p_reduced_tol(params, s, prev_move, fi; K=K, parameterization=parameterization)
    elseif model == :rh_full || (model == :auto && !isempty(fi.all_dicts) && length(params) in (6, 7))
        return get_p_full_rh(params, s, prev_move, fi; K=K)
    elseif model == :tol_full || (model == :auto && isempty(fi.all_dicts) && length(params) == 7)
        return get_p_full_tol(params, s, prev_move, fi; K=K)
    end
    throw(ArgumentError("Could not infer model for $(length(params)) parameters"))
end

function get_ll_new(params, s::Int, s_next::Int, prev_move::Action, fi::FittingStruct; K::Int=10_000_000, model::Symbol=:main, parameterization::Symbol=:probability)
    p, neighs = get_p_new(params, s, prev_move, fi; K=K, model=model, parameterization=parameterization)
    return log(p[findfirst(==(s_next), neighs)])
end

function get_nll_new(params::Vector{Float64}, state_actions, fitting_infos; K::Int=10_000_000, model::Symbol=:main, parameterization::Symbol=:fit)
    if model == :main || (model == :auto && length(params) == 3) || model in (:rh_reduced, :tol_reduced)
        return get_nll_main(params, state_actions, fitting_infos; K=K, parameterization=parameterization)
    end
    ll_function = (θ, s, s_next, prev_move, fi) -> get_ll_new(θ, s, s_next, prev_move, fi; K=K, model=model, parameterization=parameterization)
    return _sum_negative_log_likelihood(params, state_actions, fitting_infos, ll_function)
end

function _sum_negative_log_likelihood(params, state_actions, fitting_infos, ll_function)
    nll = 0.0
    for j in eachindex(state_actions)
        isempty(state_actions[j]) && continue
        fitting_info = fitting_infos[j]
        for state_actions_game in state_actions[j]
            prev_move = (0, 0)
            s, a = state_actions_game[1]
            for i in 1:(length(state_actions_game) - 1)
                s_next, a_next = state_actions_game[i + 1]
                nll -= ll_function(params, s, s_next, prev_move, fitting_info)
                prev_move = a
                s = s_next
                a = a_next
            end
        end
    end
    return nll
end
