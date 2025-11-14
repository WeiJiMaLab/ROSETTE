# require julia 1.8.5
# require DataFrames v1.3.6
using ProgressBars
using DataStructures
using Plots
using StaticArrays
using DataFrames
using CSV
using JLD2
using Optim
using SparseArrays
using LinearAlgebra
using StatsBase

include("rush_hour.jl")


# FOR FIXED SIZE RUSH HOUR PUZZLES
L = 9
######
s_free_type = MVector{L, Int}
s_fixed_type = NTuple{L, Car}
s_type = Tuple{s_free_type, s_fixed_type}
a_type = Tuple{Int, Int}
arr_type = Matrix{Int}

moves_type = MVector{4*L, a_type}
blocked_cars_type = MVector{4, Int}
move_amounts_type = MVector{4, Int}
blocking_nodes_type = Matrix{Int}

subgoal_type = MVector{5, Int}# (car, moves that unblock)
or_type = Tuple{Int, subgoal_type}# depth, subgoal
and_type = Tuple{Int, a_type}# depth, move

struct FittingStruct
    all_dicts::Vector{Dict{Tuple{Int64, Tuple{Int64, Int64}}, Float64}}
    all_moves::Vector{MVector{36, Tuple{Int64, Int64}}}
    all_idxs::Vector{Vector{Int64}}
    Q::SparseMatrixCSC{Float64, Int64}
    R::SparseMatrixCSC{Float64, Int64}
    P::Matrix{Float64}
    all_sizes::Vector{Int64}
end

include("rush_hour_utils.jl")
include("loading_data.jl")
include("model_fitting.jl")
include("and_or_trees.jl")
include("summary_statistics.jl")
include("plotting.jl")


# s_free, s_fixed = load_prb("prb3217_11")
s_free, s_fixed = load_prb("prb1707_7")
game = RushHour(get_base(5, 9), s_free, [s_fixed...])
s = start_state(game)

draw_state(s, game)

state_to_idx, idx_to_state, Q, R = get_QR_matrices(game)
N_a, N_t = size(R)

all_moves = MVector{36, Tuple{Int64, Int64}}[]
all_idxs = Vector{Int64}[]
all_sizes = Int64[]

arr = zeros(Int, 6, 6)
moves = @MVector([(0, 0) for _ in 1:(4*9)])
for s_idx in 1:N_t
    s = idx_to_state[s_idx] 
    ss = int_to_s(s, game)
    board_to_arr!(arr, ss)
    possible_moves!(moves, ss, arr)
    idxs = moves_to_idxs(moves, ss, state_to_idx, N_t)
    push!(all_moves, copy(moves))
    push!(all_idxs, idxs)
    push!(all_sizes, Lopt(s, game))
end

P = zeros(length(all_sizes), 3)

all_dicts, _, _, all_sizes_2 = get_and_or_dicts(N_t, idx_to_state, state_to_idx, game);

fi = FittingStruct(all_dicts, all_moves, all_idxs, Q, R, P, all_sizes);
s_idx = state_to_idx[s]


γp = 0.01
α = 0.001

update_QR_matrices_and_or!(fi, 1.0, (2, 2))

B = apply_γp(s_idx, γp, α, fi)

N_a, N_t = size(fi.R)
policy = vcat(fi.Q[:, s_idx].nzval, fi.R[:, s_idx].nzval)
neighs = vcat(fi.Q[:, s_idx].nzind, N_t .+ fi.R[:, s_idx].nzind)

p_s(neighs, policy, B)

a = 1e-5

c = inf_sum(a, 10, -10, policy, B[neighs, :], 10_000_000)



analysis_info = load("data/processed_data/analysis_info.jld2")["analysis_info"];
fitting_infos = load("data/processed_data/fitting_infos.jld2")["fitting_infos"];
state_actions_idx = load("data/processed_data/state_actions_idx.jld2")["state_actions_idx"];

prbs = sort([52, 29, 64, 12, 37, 8, 63, 58, 35, 46])#, 57, 20, 22, 16, 69, 68, 41, 56, 49, 32])
fis = fitting_infos[prbs];
sas = [state_actions_idx[i][prbs] for i in 1:42];

g = x -> get_nll(x, sas[11], fis)
res_g = optimize(g, [1.0, 1.0, 0.5]; g_tol=0.0001, show_trace=true, iterations=200);

f = x -> get_nll_new(x, sas[11], fis)#11
res_f = optimize(f, [1.0, 1.0, 0.5, 1.0, 0.0, 0.0]; g_tol=0.0001, show_trace=true, iterations=200);

params = zeros(42, 6);
nlls = zeros(42);
Threads.@threads for i in ProgressBar(1:42)
    fis = deepcopy(fitting_infos[prbs])
    f = x -> get_nll_new(x, sas[i], fis)
    res_f = optimize(f, [1.0, 1.0, 0.5, 1.0, 0.0, 0.0]; g_tol=0.0001, show_trace=true, show_every=20, iterations=200);
    params[i, :] .= res_f.minimizer
    nlls[i] = res_f.minimum
end

df_ = filtered_df[filtered_df.subject .∈ subjects[[11, 13]] .&& filtered_df.puzzle .∈ Ref(problems[prbs]), :]
ps = []
# new_moves = []
# as = []
# nlogγao, nlogγp, α = (0.0, 10.0, 0.00001)
# nlogγao, nlogγp, α = th_g2
# nlogγao, nlogγp, α, nlogγre, u, v = (1.0, 2.0, 0.95, 5.0, 2.0, -2.0)
nlogγao, nlogγp, α, nlogγre, u, v = th2
γao = 10 ^ (-nlogγao)
γp = 10 ^ (-nlogγp)
γre = 10 ^ (-nlogγre)

b = reduce(vcat, [[sas[11][i][1][j+1] for j in 1:length(sas[11][i][1])-1] for i in 1:length(sas[11])])
nll = 0

for (n, row) in ProgressBar(enumerate(eachrow(df_)))
    prb_idx = findfirst(==(row.puzzle), problems)
    fi = fitting_infos[prb_idx]
    puzzle_info = analysis_info[prb_idx]
    state_to_idx = puzzle_info[1]
    idx_to_state = puzzle_info[2]
    game = puzzle_info[3]
    s = state_to_idx[row.s]
    s_next = b[n][1]

    # game = puzzle_info[3]
    # all_sizes = []
    # N_a, N_t = size(fi.R)
    # for s_idx in 1:N_t
    #     ss = idx_to_state[s_idx]
    #     push!(all_sizes, Lopt(ss, game))
    # end
    # fi = FittingStruct(fi.all_dicts, fi.all_moves, fi.all_idxs, fi.Q, fi.R, fi.P, fi.all_sizes)

    update_QR_matrices_and_or!(fi, γao, row.prev_move)

    B = apply_γp(s, γp, α, fi)

    N_a, N_t = size(fi.R)
    policy = vcat(fi.Q[:, s].nzval, fi.R[:, s].nzval)
    neighs = vcat(fi.Q[:, s].nzind, N_t .+ fi.R[:, s].nzind)
    # neighs = [state_to_idx[x] > 0 ? state_to_idx[x] : N_t-state_to_idx[x] for x in row.neighs]
    # policy = ones(Float64, length(neighs)) ./ length(neighs)

    # for s_idx in 1:N_a+N_t
    #     push!(all_sizes, 0)
    # end

    # p = p_s(neighs, policy, B)
    B_ = zeros(N_a, 4)
    B_[:, 1] .= 1.0
    B__ = vcat(B, B_)
    p = inf_sum(γre, u, v, policy, B__[neighs, :], 10_000_000)
    nll -= log(p[findfirst(==(s_next), neighs)])
    # println(p)
    # println(B__[neighs, :])
    # println(all_sizes[neighs])
    # push!(ps, p)
    # push!(new_moves, wsample(row.all_moves, p))
    # push!(as, all_sizes[neighs])
    # break
    aligned_p = zeros(length(p))
    for (m, neigh) in enumerate(neighs)
        s_int = idx_to_state[neigh]
        state = int_to_s(s_int, game)
        _, s_int_32 = board_to_int32(state)
        idx = findfirst(x->x==s_int_32, row.neighs)
        aligned_p[idx] = p[m]
    end
    push!(ps, aligned_p)
end

df_[!, :ps] = ps;
df_[!, :move] = new_moves;


df_stats = calculate_summary_statistics(df_, d_goals_prbs);
binned_stats = bin_stats(df_stats, :X_d_goal);

figA(binned_stats)
figB(df_stats)
figC(binned_stats)


function get_ll_new(θ::Vector{Float64}, s::Int, s_next::Int, prev_move::a_type, fi::FittingStruct)::Float64
    γao, γp, α, γre, u, v = θ

    update_QR_matrices_and_or!(fi, γao, prev_move)

    B = apply_γp(s, γp, α, fi)

    N_a, N_t = size(fi.R)
    policy = vcat(fi.Q[:, s].nzval, fi.R[:, s].nzval)
    neighs = vcat(fi.Q[:, s].nzind, N_t .+ fi.R[:, s].nzind)
    # policy = ones(Float64, length(neighs)) ./ length(neighs)

    # p = p_s(neighs, policy, B)
    B_ = zeros(N_a, 4)
    B_[:, 1] .= 1.0
    B__ = vcat(B, B_)
    # p = policy
    p = inf_sum(γre, u, v, policy, B__[neighs, :], 10_000_000)
    return log(p[findfirst(==(s_next), neighs)])
end

function get_nll_new(θ::Vector{Float64}, state_actions, fitting_infos)
    nlogγao, nlogγp, α, nlogγre, u, v = θ
    γao = 10 ^ (-nlogγao)
    γp = 10 ^ (-nlogγp)
    γre = 10 ^ (-nlogγre)
    θp = [γao, γp, α, γre, u, v]
    if γao < 0 || γp < 0 || γre < 0 || γao > 1 || γp > 1 || γre > 1 || α < 0 || α > 1# || u < 0 || v < 0
        return Inf
    end
    nll = 0.0
    for j in eachindex(state_actions)
        if isempty(state_actions[j])
            continue
        end
        fitting_info = fitting_infos[j]
        for game_index in eachindex(state_actions[j])
            state_actions_game = state_actions[j][game_index]
            a_prev = (0, 0)
            s, a = state_actions_game[1]
            for i in 1:length(state_actions_game)-1
                s_next, a_next = state_actions_game[i+1]

                nll -= get_ll_new(θp, s, s_next, a_prev, fitting_info)

                a_prev = a
                s = s_next
                a = a_next
            end
        end
    end
    return nll
end








function inf_sum(a::Real,
                    u::Real,
                    v::Real,
                    p::AbstractVector{<:Real}, 
                    B::AbstractMatrix{<:Real}, 
                    K::Int)
    N = length(p)
    @assert size(B, 1) == N
    @assert size(B, 2) == 4
    S = zeros(eltype(p), N)
    new_policy = copy(p)

    cont_prob = 1.0
    weight = a
    M = exp.(u .* B[:, 1] .+ v .* B[:, 3])

    @inbounds for k in 0:K
        @simd for i in 1:N
            S[i] += weight * new_policy[i]
        end
        if abs(1 - sum(S)) < 1e-6
            break
        end

        cont_prob *= (1 - a)
        weight = a * cont_prob

        denom = 0.0
        @simd for i in 1:N
            # new_policy[i] *= B[i, 1]
            # new_policy[i] *= u*B[i, 1] + v*B[i, 3] + B[i, 4]
            new_policy[i] *= M[i]
            denom += new_policy[i]
        end
        inv_denom = 1.0 / denom
        @simd for i in 1:N
            new_policy[i] *= inv_denom
        end
    end

    return S
end









function Lopt(s::Int, game::RushHour; max_iters=100000)
    visited = Set{Int}(s)
    frontier = Int[s]
    depths = Dict{Int, Int}(s => 0)
    for i in 1:max_iters
        if isempty(frontier)
            error("Could not solve puzzle")
        end
        s = popfirst!(frontier)
        d = depths[s]
        for neigh in neighbours(s, game)
            if neigh in visited
                continue
            end
            if is_terminal(neigh, game)
                return d+1
            end
            depths[neigh] = d + 1
            push!(visited, neigh)
            push!(frontier, neigh)
        end
    end
    error("Maximum search depth reached")
end