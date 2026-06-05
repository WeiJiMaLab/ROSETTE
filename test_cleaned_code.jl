using CSV
using DataFrames
using DataStructures
using LinearAlgebra
using Random
using SparseArrays
using StaticArrays
using Statistics
using StatsBase
using Test

include(joinpath(@__DIR__, "cleaned_code", "ROSETTE.jl"))
const New = ROSETTE

module OldRH
using CSV
using DataFrames
using DataStructures
using JLD2
using JSON
using LinearAlgebra
using Optim
using Plots
using ProgressBars
using SparseArrays
using StaticArrays
using Statistics
using StatsBase
struct FittingStruct
    all_dicts::Vector{Dict{Tuple{Int64, Tuple{Int64, Int64}}, Float64}}
    all_moves::Vector{MVector{36, Tuple{Int64, Int64}}}
    all_idxs::Vector{Vector{Int64}}
    Q::SparseMatrixCSC{Float64, Int64}
    R::SparseMatrixCSC{Float64, Int64}
    P::Matrix{Float64}
    all_sizes::Vector{Int64}
end
include(joinpath(@__DIR__, "src", "rush_hour.jl"))
include(joinpath(@__DIR__, "src", "rush_hour_utils.jl"))
include(joinpath(@__DIR__, "src", "loading_data.jl"))
include(joinpath(@__DIR__, "src", "model_fitting.jl"))
include(joinpath(@__DIR__, "src", "and_or_trees.jl"))
include(joinpath(@__DIR__, "src", "summary_statistics.jl"))

function update_QR_matrices_and_or!(fi::FittingStruct, γao::Float64, prev_move::a_type)::Nothing
    N_a, N_t = size(fi.R)
    ps = zeros(Float64, 4 * 9)
    temp_dict = Dict{and_type, Float64}()
    move_dict = Dict{a_type, Float64}()
    same_car_moves = prev_move == (0, 0) ? [(0, 0)] : [(prev_move[1], j) for j in -4:4]
    for s in 1:N_t
        dict = fi.all_dicts[s]
        moves = fi.all_moves[s]
        apply_γao(temp_dict, dict, γao)
        process_dict!(ps, move_dict, moves, temp_dict, same_car_moves)
        idxs = fi.all_idxs[s]
        for n in eachindex(idxs)
            neigh = idxs[n]
            if neigh > N_t
                fi.R[neigh - N_t, s] = ps[n]
            else
                fi.Q[neigh, s] = ps[n]
            end
        end
    end
    return nothing
end

function apply_γp(γr::Float64, fi::FittingStruct)::Vector{Float64}
    N_t, _ = size(fi.Q)
    p_win = (I - ((1 - γr) * fi.Q')) \ (γr * ones(N_t))
    return 1 .- p_win
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

function get_p_main(θ::Vector{Float64}, s::Int, prev_move::a_type, fi::FittingStruct)
    γao, γr, γp = θ
    update_QR_matrices_and_or!(fi, γao, prev_move)
    p_stop = apply_γp(γr, fi)
    N_a, N_t = size(fi.R)
    policy = vcat(fi.Q[:, s].nzval, fi.R[:, s].nzval)
    neighs = vcat(fi.Q[:, s].nzind, N_t .+ fi.R[:, s].nzind)
    append!(p_stop, ones(N_a))
    return inf_sum_reduced(γp, policy, p_stop[neighs]), neighs
end

function get_ll_main(θ::Vector{Float64}, s::Int, s_next::Int, prev_move::a_type, fi::FittingStruct)::Float64
    p, neighs = get_p_main(θ, s, prev_move, fi)
    return log(p[findfirst(==(s_next), neighs)])
end

function get_nll_main(θ::Vector{Float64}, state_actions, fitting_infos)
    nlogγao, nlogγr, nlogγp = θ
    γao = 10.0^(-nlogγao)
    γr = 10.0^(-nlogγr)
    γp = 10.0^(-nlogγp)
    if γao < 1e-5 || γr < 1e-5 || γp < 1e-5 || γao > 1 || γr > 1 || γp > 1
        return Inf
    end
    θp = [γao, γr, γp]
    nll = 0.0
    for j in eachindex(state_actions)
        isempty(state_actions[j]) && continue
        fitting_info = fitting_infos[j]
        for state_actions_game in state_actions[j]
            prev_move = (0, 0)
            s, a = state_actions_game[1]
            for i in 1:(length(state_actions_game) - 1)
                s_next, a_next = state_actions_game[i + 1]
                nll -= get_ll_main(θp, s, s_next, prev_move, fitting_info)
                prev_move = a
                s = s_next
                a = a_next
            end
        end
    end
    return nll
end
end

module OldTOL
using CSV
using DataFrames
using DataStructures
using LinearAlgebra
using ProgressBars
using SparseArrays
using StaticArrays
using Statistics
using StatsBase
abstract type AbstractGame end
struct FittingStruct
    all_dicts::Vector{Dict{Tuple{Int64, Tuple{Int64, Int64}}, Float64}}
    all_moves::Vector{MVector{36, Tuple{Int64, Int64}}}
    all_idxs::Vector{Vector{Int64}}
    Q::SparseMatrixCSC{Float64, Int64}
    R::SparseMatrixCSC{Float64, Int64}
    P::Matrix{Float64}
    all_sizes::Vector{Int64}
end
include(joinpath(@__DIR__, "src", "general_functions.jl"))
include(joinpath(@__DIR__, "src", "tower_of_london.jl"))
include(joinpath(@__DIR__, "src", "model_fitting.jl"))

function update_QR_matrices_tol!(fi::FittingStruct, β::Float64, prev_move::a_type)::Nothing
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

function apply_γp(γr::Float64, fi::FittingStruct)::Vector{Float64}
    N_t, _ = size(fi.Q)
    p_win = (I - ((1 - γr) * fi.Q')) \ (γr * ones(N_t))
    return 1 .- p_win
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

function get_p_main(θ::Vector{Float64}, s::Int, prev_move::a_type, fi::FittingStruct)
    β, γr, γp = θ
    update_QR_matrices_tol!(fi, β, prev_move)
    p_stop = apply_γp(γr, fi)
    N_a, N_t = size(fi.R)
    policy = vcat(fi.Q[:, s].nzval, fi.R[:, s].nzval)
    neighs = vcat(fi.Q[:, s].nzind, N_t .+ fi.R[:, s].nzind)
    append!(p_stop, ones(N_a))
    return inf_sum_reduced(γp, policy, p_stop[neighs]), neighs
end

function get_ll_main(θ::Vector{Float64}, s::Int, s_next::Int, prev_move::a_type, fi::FittingStruct)::Float64
    p, neighs = get_p_main(θ, s, prev_move, fi)
    return log(p[findfirst(==(s_next), neighs)])
end

function get_nll_main(θ::Vector{Float64}, state_actions, fitting_infos)
    β, nlogγr, nlogγp = θ
    γr = 10.0^(-nlogγr)
    γp = 10.0^(-nlogγp)
    if γr < 1e-5 || γp < 1e-5 || γr > 1 || γp > 1
        return Inf
    end
    θp = [β, γr, γp]
    nll = 0.0
    for j in eachindex(state_actions)
        isempty(state_actions[j]) && continue
        fitting_info = fitting_infos[j]
        for state_actions_game in state_actions[j]
            prev_move = (0, 0)
            s, a = state_actions_game[1]
            for i in 1:(length(state_actions_game) - 1)
                s_next, a_next = state_actions_game[i + 1]
                nll -= get_ll_main(θp, s, s_next, prev_move, fitting_info)
                prev_move = a
                s = s_next
                a = a_next
            end
        end
    end
    return nll
end
end

car_fields(car) = (Int(car.dim2), car.dir, Int(car.len))
car_fields(cars::AbstractVector) = [car_fields(car) for car in cars]
car_fields(cars::Tuple) = [car_fields(car) for car in cars]

function transition_targets(Q, R, idx_to_state, col)
    _, N_t = size(R)
    targets = Pair{Int, Float64}[]
    for (idx, value) in zip(Q[:, col].nzind, Q[:, col].nzval)
        push!(targets, idx_to_state[idx] => value)
    end
    for (idx, value) in zip(R[:, col].nzind, R[:, col].nzval)
        push!(targets, idx_to_state[N_t + idx] => value)
    end
    return sort(targets; by=x -> x.first)
end

function probability_targets(p, neighs, idx_to_state)
    return sort([idx_to_state[neighs[i]] => p[i] for i in eachindex(neighs)]; by=x -> x.first)
end

function indexed_target(state_to_idx, N_t, state)
    idx = state_to_idx[state]
    return idx < 0 ? N_t - idx : idx
end

function rh_fitting_info_old(problem)
    start_pos, s_fixed = OldRH.load_prb(problem)
    game = OldRH.RushHour(OldRH.get_base(5, 9), start_pos, s_fixed)
    state_to_idx, idx_to_state, Q, R = OldRH.get_QR_matrices(game)
    _, N_t = size(R)
    all_dicts, all_moves, all_idxs, all_sizes = OldRH.get_and_or_dicts(N_t, idx_to_state, state_to_idx, game)
    return OldRH.FittingStruct(all_dicts, all_moves, all_idxs, Q, R, zeros(length(all_sizes), 3), all_sizes), state_to_idx, idx_to_state, game
end

function rh_fitting_info_new(problem)
    start_pos, s_fixed = New.load_prb(problem)
    game = New.RushHour(New.get_base(5, New.RUSH_HOUR_CARS), start_pos, s_fixed)
    state_to_idx, idx_to_state, Q, R = New.get_QR_matrices(game)
    _, N_t = size(R)
    all_dicts, all_moves, all_idxs, all_sizes = New.get_and_or_dicts(N_t, idx_to_state, state_to_idx, game)
    return New.FittingStruct(all_dicts, all_moves, all_idxs, Q, R, zeros(length(all_sizes), 3), all_sizes), state_to_idx, idx_to_state, game
end

function tol_fitting_info_old(initial, goal)
    game = OldTOL.TowerOfLondon(OldTOL.get_base(4, 6), initial, goal)
    state_to_idx, idx_to_state, Q, R = OldTOL.get_QR_matrices(game)
    _, N_t = size(R)
    all_moves, all_idxs, all_sizes = OldTOL.get_tol_heuristics(N_t, idx_to_state, state_to_idx, game)
    return OldTOL.FittingStruct([], all_moves, all_idxs, Q, R, zeros(N_t, 3), all_sizes), state_to_idx, idx_to_state, game
end

function tol_fitting_info_new(initial, goal)
    game = New.TowerOfLondon(New.get_base(4, 6), initial, goal)
    state_to_idx, idx_to_state, Q, R = New.get_QR_matrices(game)
    _, N_t = size(R)
    all_moves, all_idxs, all_sizes = New.get_tol_heuristics(N_t, idx_to_state, state_to_idx, game)
    return New.FittingStruct(Dict{Tuple{Int64, Tuple{Int64, Int64}}, Float64}[], all_moves, all_idxs, Q, R, zeros(N_t, 3), all_sizes), state_to_idx, idx_to_state, game
end

@testset "Rush Hour parsing and state mechanics" begin
    problem = "prb1707_7"
    old_free, old_fixed = OldRH.load_prb(problem)
    new_free, new_fixed = New.load_prb(problem)

    @test new_free == old_free
    @test car_fields(new_fixed) == car_fields(old_fixed)

    old_game = OldRH.RushHour(OldRH.get_base(5, 9), old_free, old_fixed)
    new_game = New.RushHour(New.get_base(5, New.RUSH_HOUR_CARS), new_free, new_fixed)
    old_start = OldRH.start_state(old_game)
    new_start = New.start_state(new_game)

    @test new_start == old_start
    @test New.draw_state(new_start, new_game) == OldRH.draw_state(old_start, old_game)
    @test sort(New.get_all_moves(new_start, new_game)) == sort(OldRH.get_all_moves(old_start, old_game))
    @test sort(New.neighbours(new_start, new_game)) == sort(OldRH.neighbours(old_start, old_game))

    old_state = OldRH.int_to_s(old_start, old_game)
    new_state = New.int_to_s(new_start, new_game)
    @test collect(new_state[1]) == collect(old_state[1])
    @test car_fields(new_state[2]) == car_fields(old_state[2])
    @test New.s_to_int(new_state) == OldRH.s_to_int(old_state)
    @test New.board_to_int32(new_state) == OldRH.board_to_int32(old_state)
end

@testset "Tower of London mechanics" begin
    initial = [1, 0, 0, 0, 0, 0]
    goal = [0, 0, 0, 0, 0, 1]
    old_game = OldTOL.TowerOfLondon(OldTOL.get_base(4, 6), initial, goal)
    new_game = New.TowerOfLondon(New.get_base(4, 6), initial, goal)
    old_start = OldTOL.start_state(old_game)
    new_start = New.start_state(new_game)

    @test new_start == old_start
    @test New.goal_state(new_game) == OldTOL.goal_state(old_game)
    @test New.draw_state(new_start, new_game) == OldTOL.draw_state(old_start, old_game)
    @test sort(New.neighbours(new_start, new_game)) == sort(OldTOL.neighbours(old_start, old_game))

    old_neighs, old_actions = OldTOL.neighbours_actions(old_start, old_game)
    new_neighs, new_actions = New.neighbours_actions(new_start, new_game)
    @test sort(new_actions) == sort(old_actions)
    @test sort(new_neighs) == sort(old_neighs)

    move = (1, 6)
    @test New.make_move(new_start, move, new_game) == OldTOL.make_move(old_start, move, old_game)
    @test New.has_won(New.make_move(new_start, move, new_game), new_game) == OldTOL.has_won(OldTOL.make_move(old_start, move, old_game), old_game)
end

@testset "Tower of London main model" begin
    initial = [1, 0, 0, 0, 0, 0]
    goal = [0, 0, 0, 0, 0, 1]
    old_fi, old_state_to_idx, old_idx_to_state, old_game = tol_fitting_info_old(initial, goal)
    new_fi, new_state_to_idx, new_idx_to_state, new_game = tol_fitting_info_new(initial, goal)

    @test size(old_fi.Q) == size(new_fi.Q)
    @test size(old_fi.R) == size(new_fi.R)
    @test old_fi.all_sizes == new_fi.all_sizes

    old_start_col = old_state_to_idx[OldTOL.start_state(old_game)]
    new_start_col = new_state_to_idx[New.start_state(new_game)]

    old_beta_fi = deepcopy(old_fi)
    new_beta_fi = deepcopy(new_fi)
    OldTOL.update_QR_matrices_tol!(old_beta_fi, 1.0, (0, 0))
    New.update_QR_matrices_tol!(new_beta_fi, 1.0, (0, 0))
    for state in old_idx_to_state[1:min(5, size(old_fi.R, 2))]
        old_col = old_state_to_idx[state]
        new_col = new_state_to_idx[state]
        @test transition_targets(old_beta_fi.Q, old_beta_fi.R, old_idx_to_state, old_col) ==
              transition_targets(new_beta_fi.Q, new_beta_fi.R, new_idx_to_state, new_col)
    end

    θ_main_probability = [1.0, 0.2, 0.5]
    old_p, old_neighs = OldTOL.get_p_main(θ_main_probability, old_start_col, (0, 0), deepcopy(old_fi))
    new_p, new_neighs = New.get_p_main(θ_main_probability, new_start_col, (0, 0), deepcopy(new_fi))
    old_prob_targets = probability_targets(old_p, old_neighs, old_idx_to_state)
    new_prob_targets = probability_targets(new_p, new_neighs, new_idx_to_state)
    @test first.(new_prob_targets) == first.(old_prob_targets)
    for i in eachindex(old_prob_targets)
        @test last(new_prob_targets[i]) ≈ last(old_prob_targets[i]) atol=1e-10
    end

    target_state = first(old_prob_targets).first
    old_main_target = indexed_target(old_state_to_idx, size(old_fi.R, 2), target_state)
    new_main_target = indexed_target(new_state_to_idx, size(new_fi.R, 2), target_state)
    @test New.get_ll_main(θ_main_probability, new_start_col, new_main_target, (0, 0), deepcopy(new_fi)) ≈
          OldTOL.get_ll_main(θ_main_probability, old_start_col, old_main_target, (0, 0), deepcopy(old_fi)) atol=1e-10
    @test New.get_ll(θ_main_probability, new_start_col, new_main_target, (0, 0), deepcopy(new_fi)) ≈
          New.get_ll_main(θ_main_probability, new_start_col, new_main_target, (0, 0), deepcopy(new_fi)) atol=1e-12
    @test New.get_ll_new(θ_main_probability, new_start_col, new_main_target, (0, 0), deepcopy(new_fi)) ≈
          New.get_ll_main(θ_main_probability, new_start_col, new_main_target, (0, 0), deepcopy(new_fi)) atol=1e-12

    θ_main_log = [1.0, -log10(0.2), -log10(0.5)]
    old_state_actions = [[[(old_start_col, (0, 0)), (old_main_target, (0, 0))]]]
    new_state_actions = [[[(new_start_col, (0, 0)), (new_main_target, (0, 0))]]]
    @test New.get_nll_main(θ_main_log, new_state_actions, [deepcopy(new_fi)]) ≈
          OldTOL.get_nll_main(θ_main_log, old_state_actions, [deepcopy(old_fi)]) atol=1e-10
    @test New.get_nll(θ_main_log, new_state_actions, [deepcopy(new_fi)]) ≈
          New.get_nll_main(θ_main_log, new_state_actions, [deepcopy(new_fi)]) atol=1e-12
    @test New.get_nll_new(θ_main_log, new_state_actions, [deepcopy(new_fi)]) ≈
          New.get_nll_main(θ_main_log, new_state_actions, [deepcopy(new_fi)]) atol=1e-12
    @test isinf(New.get_nll_main([1.0, -1.0, 1.0], new_state_actions, [deepcopy(new_fi)]))
    @test isinf(New.get_nll_new([1.0, 6.0, 1.0], new_state_actions, [deepcopy(new_fi)]))

    fi_reused = deepcopy(new_fi)
    New.get_p_main([1.0, 0.2, 0.5], new_start_col, (0, 0), fi_reused)
    reused_p, reused_neighs = New.get_p_main([0.5, 0.3, 0.5], new_start_col, (0, 0), fi_reused)
    fresh_p, fresh_neighs = New.get_p_main([0.5, 0.3, 0.5], new_start_col, (0, 0), deepcopy(new_fi))
    @test reused_neighs == fresh_neighs
    @test reused_p ≈ fresh_p atol=1e-12
end

@testset "QR matrices, AND/OR policies, and likelihoods" begin
    problem = "prb1707_7"
    old_fi, old_state_to_idx, old_idx_to_state, old_game = rh_fitting_info_old(problem)
    new_fi, new_state_to_idx, new_idx_to_state, new_game = rh_fitting_info_new(problem)

    @test length(old_state_to_idx) == length(new_state_to_idx)
    @test size(old_fi.Q) == size(new_fi.Q)
    @test size(old_fi.R) == size(new_fi.R)

    old_start_col = old_state_to_idx[OldRH.start_state(old_game)]
    new_start_col = new_state_to_idx[New.start_state(new_game)]
    @test transition_targets(old_fi.Q, old_fi.R, old_idx_to_state, old_start_col) ==
          transition_targets(new_fi.Q, new_fi.R, new_idx_to_state, new_start_col)
    for state in old_idx_to_state[1:min(5, size(old_fi.R, 2))]
        old_col = old_state_to_idx[state]
        new_col = new_state_to_idx[state]
        @test transition_targets(old_fi.Q, old_fi.R, old_idx_to_state, old_col) ==
              transition_targets(new_fi.Q, new_fi.R, new_idx_to_state, new_col)
    end

    old_state = OldRH.int_to_s(OldRH.start_state(old_game), old_game)
    new_state = New.int_to_s(New.start_state(new_game), new_game)
    old_policy = OldRH.propagate_ps(0.2, OldRH.get_and_or_tree(old_state))
    new_policy = New.propagate_ps(0.2, New.get_and_or_tree(new_state))
    @test keys(old_policy) == keys(new_policy)
    for key in keys(old_policy)
        @test new_policy[key] ≈ old_policy[key] atol=1e-12
    end

    old_targets = vcat(old_fi.Q[:, old_start_col].nzind, size(old_fi.R, 2) .+ old_fi.R[:, old_start_col].nzind)
    new_targets = vcat(new_fi.Q[:, new_start_col].nzind, size(new_fi.R, 2) .+ new_fi.R[:, new_start_col].nzind)
    @test length(old_targets) == length(new_targets)

    θ_progress_probability = [0.2, 0.1, 0.5]
    @test New.get_ll_progress_rh(θ_progress_probability, new_start_col, new_targets[1], (0, 0), deepcopy(new_fi)) ≈
          OldRH.get_ll(θ_progress_probability, old_start_col, old_targets[1], (0, 0), deepcopy(old_fi)) atol=1e-10

    θ_progress_log = [1.0, 1.0, 0.5]
    old_state_actions = [[[(old_start_col, (0, 0)), (old_targets[1], (0, 0))]]]
    new_state_actions = [[[(new_start_col, (0, 0)), (new_targets[1], (0, 0))]]]
    @test New.get_nll_progress_rh(θ_progress_log, new_state_actions, [deepcopy(new_fi)]) ≈
          OldRH.get_nll(θ_progress_log, old_state_actions, [deepcopy(old_fi)]) atol=1e-10

    θ_main_probability = [0.2, 0.1, 0.5]
    old_p, old_neighs = OldRH.get_p_main(θ_main_probability, old_start_col, (0, 0), deepcopy(old_fi))
    new_p, new_neighs = New.get_p_main(θ_main_probability, new_start_col, (0, 0), deepcopy(new_fi))
    old_prob_targets = probability_targets(old_p, old_neighs, old_idx_to_state)
    new_prob_targets = probability_targets(new_p, new_neighs, new_idx_to_state)
    @test first.(new_prob_targets) == first.(old_prob_targets)
    for i in eachindex(old_prob_targets)
        @test last(new_prob_targets[i]) ≈ last(old_prob_targets[i]) atol=1e-10
    end
    target_state = first(old_prob_targets).first
    old_main_target = indexed_target(old_state_to_idx, size(old_fi.R, 2), target_state)
    new_main_target = indexed_target(new_state_to_idx, size(new_fi.R, 2), target_state)

    @test New.get_ll_main(θ_main_probability, new_start_col, new_main_target, (0, 0), deepcopy(new_fi)) ≈
          OldRH.get_ll_main(θ_main_probability, old_start_col, old_main_target, (0, 0), deepcopy(old_fi)) atol=1e-10
    @test New.get_ll(θ_main_probability, new_start_col, new_main_target, (0, 0), deepcopy(new_fi)) ≈
          New.get_ll_main(θ_main_probability, new_start_col, new_main_target, (0, 0), deepcopy(new_fi)) atol=1e-12
    @test New.get_ll_new(θ_main_probability, new_start_col, new_main_target, (0, 0), deepcopy(new_fi)) ≈
          New.get_ll_main(θ_main_probability, new_start_col, new_main_target, (0, 0), deepcopy(new_fi)) atol=1e-12

    θ_main_log = [-log10(θ_main_probability[1]), -log10(θ_main_probability[2]), -log10(θ_main_probability[3])]
    old_main_state_actions = [[[(old_start_col, (0, 0)), (old_main_target, (0, 0))]]]
    new_main_state_actions = [[[(new_start_col, (0, 0)), (new_main_target, (0, 0))]]]
    @test New.get_nll_main(θ_main_log, new_main_state_actions, [deepcopy(new_fi)]) ≈
          OldRH.get_nll_main(θ_main_log, old_main_state_actions, [deepcopy(old_fi)]) atol=1e-10
    @test New.get_nll(θ_main_log, new_main_state_actions, [deepcopy(new_fi)]) ≈
          New.get_nll_main(θ_main_log, new_main_state_actions, [deepcopy(new_fi)]) atol=1e-12
    @test New.get_nll_new(θ_main_log, new_main_state_actions, [deepcopy(new_fi)]) ≈
          New.get_nll_main(θ_main_log, new_main_state_actions, [deepcopy(new_fi)]) atol=1e-12
    @test isinf(New.get_nll_main([-1.0, 1.0, 1.0], new_main_state_actions, [deepcopy(new_fi)]))
    @test isinf(New.get_nll_new([1.0, 6.0, 1.0], new_main_state_actions, [deepcopy(new_fi)]))

    fi_reused = deepcopy(new_fi)
    New.get_p_main([0.2, 0.1, 0.5], new_start_col, (0, 0), fi_reused)
    reused_p, reused_neighs = New.get_p_main([0.3, 0.2, 0.5], new_start_col, (0, 0), fi_reused)
    fresh_p, fresh_neighs = New.get_p_main([0.3, 0.2, 0.5], new_start_col, (0, 0), deepcopy(new_fi))
    @test reused_neighs == fresh_neighs
    @test reused_p ≈ fresh_p atol=1e-12
end

@testset "Data and preprocessing" begin
    old_subjects, old_problems, old_state_actions = OldRH.load_rh_data()
    new_subjects, new_problems, new_state_actions = New.load_rh_data()
    @test collect(new_subjects[1:3]) == collect(old_subjects[1:3])
    @test collect(new_problems[1:5]) == collect(old_problems[1:5])
    @test new_state_actions[1][1] == old_state_actions[1][1]

    synthetic = CSV.read(
        IOBuffer("""
        subject,instance,event,t,piece,target,move
        S1,prb1707_7,start,0,0,0,start
        S1,prb1707_7,drag_end,1000,1,4,m1
        S1,prb1707_7,win,2000,0,0,win
        """),
        DataFrame,
    )
    d_goals = Dict("prb1707_7" => DefaultDict{Int32, Int}(0))
    old_processed = OldRH.pre_process(synthetic, d_goals)
    new_processed = New.pre_process(synthetic, d_goals)
    @test nrow(new_processed) == nrow(old_processed)
    @test new_processed.event == old_processed.event
    @test new_processed.move == old_processed.move
    @test new_processed.prev_move == old_processed.prev_move

    synthetic_restart = CSV.read(
        IOBuffer("""
        subject,instance,event,t,piece,target,move
        S1,prb1707_7,start,0,0,0,start
        S1,prb1707_7,drag_end,500,1,4,m1
        S1,prb1707_7,drag_end,700,1,4,m1
        S1,prb1707_7,start,1000,0,0,restart
        S1,prb1707_7,drag_end,1500,1,4,m2
        S1,prb1707_7,win,2500,0,0,win
        """),
        DataFrame,
    )
    old_restart = OldRH.pre_process(synthetic_restart, d_goals)
    new_restart = New.pre_process(synthetic_restart, d_goals)
    @test nrow(new_restart) == nrow(old_restart)
    @test new_restart.event == old_restart.event
    @test new_restart.move == old_restart.move
    @test new_restart.prev_move == old_restart.prev_move
    @test new_restart.attempt == old_restart.attempt
end

@testset "Summary statistics" begin
    df = DataFrame(
        :subject => ["S1"],
        :puzzle => ["P1"],
        :event => ["move"],
        :move => [(1, 1)],
        :prev_move => [(0, 0)],
        :all_moves => [[(1, 1), (2, 1)]],
        :neighs => [[2, 3]],
        :d_goal => [1],
        :p_full => [[0.7, 0.3]],
        Symbol("p_chance") => [[0.5, 0.5]],
        Symbol("p_only-gamma_ao") => [[0.6, 0.4]],
        Symbol("p_only-gamma_r") => [[0.6, 0.4]],
        Symbol("p_only-gamma_p") => [[0.6, 0.4]],
        Symbol("p_lesion-gamma_ao") => [[0.6, 0.4]],
        Symbol("p_lesion-gamma_r") => [[0.6, 0.4]],
        Symbol("p_lesion-gamma_p") => [[0.6, 0.4]],
    )
    d_goals = Dict("P1" => Dict(2 => 0, 3 => 2))

    Random.seed!(11)
    old_stats = OldRH.calculate_summary_statistics(df, d_goals; iters=5)
    Random.seed!(11)
    new_stats = New.calculate_summary_statistics(df, d_goals; iters=5)

    @test nrow(new_stats) == nrow(old_stats)
    @test names(new_stats) == names(old_stats)
    @test new_stats == old_stats
    @test new_stats[new_stats.model .== "data", :X_d_goal] == old_stats[old_stats.model .== "data", :X_d_goal]
    @test new_stats[new_stats.model .== "data", :y_d_goal] == old_stats[old_stats.model .== "data", :y_d_goal]

    old_binned = OldRH.bin_stats(old_stats, :X_d_goal; nbins=2)
    new_binned = New.bin_stats(new_stats, :X_d_goal; nbins=2)
    @test isequal(new_binned, old_binned)
end
