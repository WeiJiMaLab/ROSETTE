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
    if γao < 1e-10 || γp < 1e-10 || γre < 1e-10 || γao > 1 || γp > 1 || γre > 1 || α < 0 || α > 1# || u < 0 || v < 0
        return Inf
    end
    nll = 0.0
    count = 1
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
                count += 1
            end
        end
    end
    return nll
end

function get_nll_single_state(θ::Vector{Float64}, ii, state_actions, fitting_infos)
    nlogγao, nlogγp, α, nlogγre, u, v = θ
    γao = 10 ^ (-nlogγao)
    γp = 10 ^ (-nlogγp)
    γre = 10 ^ (-nlogγre)
    θp = [γao, γp, α, γre, u, v]
    if γao < 1e-10 || γp < 1e-10 || γre < 1e-10 || γao > 1 || γp > 1 || γre > 1 || α < 0 || α > 1# || u < 0 || v < 0
        return Inf
    end
    count = 0
    s, s_next, a_prev, prb_idx = nothing, nothing, nothing, nothing
    for i in eachindex(state_actions)
        if isempty(state_actions[i])
            continue
        end
        for j in 1:length(state_actions[i][1])-1
            count += 1
            if count == ii
                s_prev, a_prev = j > 1 ? state_actions[i][1][j - 1] : (0, (0, 0))
                s, a = state_actions[i][1][j]
                s_next, a_next = state_actions[i][1][j + 1]
                prb_idx = i
                break
            end
        end
    end
    fitting_info = fitting_infos[prb_idx]
    return -get_ll_new(θp, s, s_next, a_prev, fitting_info)
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
        
        @simd for i in 1:N
            S[i] += weight * new_policy[i]
        end
        if abs(1 - sum(S)) < 1e-6
            break
        end

        cont_prob *= (1 - a)
        weight = a * cont_prob
    end

    return S
end

include("rush_hour_utils.jl")
include("loading_data.jl")
include("model_fitting.jl")
include("and_or_trees.jl")
include("summary_statistics.jl")
# include("plotting.jl")

function fit_subject_cluster(n, N_prbs; iters=200)
    analysis_info = load("data/processed_data/analysis_info.jld2")["analysis_info"];
    fitting_infos = load("data/processed_data/fitting_infos.jld2")["fitting_infos"];
    state_actions_idx = load("data/processed_data/state_actions_idx.jld2")["state_actions_idx"];

    prbs = sort([52, 29, 64, 12, 37, 8, 63, 58, 35, 46])#, 57, 20, 22, 16, 69, 68, 41, 56, 49, 32])
    prbs = prbs[1:N_prbs]
    fis = fitting_infos[prbs];
    sas = [state_actions_idx[i][prbs] for i in 1:42];

    f = x -> get_nll_new(x, sas[i], fis)
    res_f = optimize(f, [1.0, 1.0, 0.5, 1.0, 0.0, 0.0]; g_tol=0.0001, show_trace=true, show_every=20, iterations=iters);
    params = res_f.minimizer
    #@save "params/params_$(n).jld2" params
end

function get_all_ps(n, N, df, analysis_info, params, fitting_infos, subjects, problems)
    start_N = n*N
    end_N = (n+1)*N
    
    println(start_N)
    println(end_N)
    flush(stdout)
    
    all_ps = Vector{Float64}[]
    for (n, row) in ProgressBar(enumerate(eachrow(df)))
        if n < start_N
            continue
        end
        if n > end_N
            continue
        end
        if row.puzzle ∉ problems
            continue
        end
        
        subj_ind = findfirst(x->x==row.subject, subjects)
        problem_ind = findfirst(x->x==row.puzzle, problems)
    
        state_to_idx, idx_to_state, game, N_t = analysis_info[problem_ind]
        s = state_to_idx[row.s]
        prev_move = row.prev_move
        # p, neighs = get_p(params[subj_ind, :], s, prev_move, fitting_infos[problem_ind])
        p, neighs = get_p_new(params[subj_ind, :], s, prev_move, fitting_infos[problem_ind])
        if any(p .< 0)
            println("Negative probabilities for subject $(row.subject), problem $(row.puzzle)")
            println(p)
        end
        if abs(1 - sum(p)) > 1e-6
            println("Probabilities do not sum to 1 for subject $(row.subject), problem $(row.puzzle)")
            println(sum(p))
        end

        aligned_p = zeros(length(p))
        for (m, neigh) in enumerate(neighs)
            s_int = idx_to_state[neigh]
            state = int_to_s(s_int, game)
            _, s_int_32 = board_to_int32(state)
            idx = findfirst(x->x==s_int_32, row.neighs)
            aligned_p[idx] = p[m]
        end
        push!(all_ps, aligned_p)
    end
    return all_ps
   #@save "all_ps_$(N).jld2" all_ps
end

function get_p_new(params, s, prev_move, fi)
    nlogγao, nlogγp, α, nlogγre, u, v = params
    γao = 10 ^ (-nlogγao)
    γp = 10 ^ (-nlogγp)
    γre = 10 ^ (-nlogγre)

    update_QR_matrices_and_or!(fi, γao, prev_move)

    B = apply_γp(s, γp, α, fi)

    N_a, N_t = size(fi.R)
    policy = vcat(fi.Q[:, s].nzval, fi.R[:, s].nzval)
    neighs = vcat(fi.Q[:, s].nzind, N_t .+ fi.R[:, s].nzind)

    B_ = zeros(N_a, 4)
    B_[:, 1] .= 1.0
    B__ = vcat(B, B_)
    p = inf_sum(γre, u, v, policy, B__[neighs, :], 10_000_000)
    return p, neighs
end

all_old_ps = get_all_ps(0, 100000, filtered_df, analysis_info[prbs], old_params, fitting_infos[prbs], subjects, problems[prbs])
all_ps = get_all_ps(0, 100000, filtered_df, analysis_info[prbs], params, fitting_infos[prbs], subjects, problems[prbs])

df_ = filtered_df[filtered_df.puzzle .∈ Ref(problems[prbs]), :]
df_[!, :ps] = all_ps;

df__ = filtered_df[filtered_df.puzzle .∈ Ref(problems[prbs]), :]
df__[!, :ps] = all_old_ps;

df_stats = calculate_summary_statistics(df_, d_goals_prbs)
binned_stats = bin_stats(df_stats, :X_d_goal)



figA(binned_stats)
figB(df_stats)
figC(binned_stats)


figA(binned_stats_old)
figB(df_stats_old)
figC(binned_stats_old)


# nlls_old = zeros(size(df__)[1]);
nlls = zeros(size(df_)[1]);
nlls_lesion = zeros(6, size(df_)[1]);
# for (n, row) in enumerate(eachrow(df__))
#     if row.subject == "A1F4N58CAX8IMK:35DR22AR5ES6RR89U7EJ1DXW91AX3P"
#         continue
#     end
#     next_move_index = findfirst(x -> x == row.move, row.all_moves)
#     nlls_old[n] -= log(row.ps[next_move_index])
# end
for (n, row) in enumerate(eachrow(df_))
    if row.subject == "A1F4N58CAX8IMK:35DR22AR5ES6RR89U7EJ1DXW91AX3P"
        continue
    end
    next_move_index = findfirst(x -> x == row.move, row.all_moves)
    nlls[n] -= log(row.ps[next_move_index])
end
for (n, row) in ProgressBar(enumerate(eachrow(df_)))
    if row.subject == "A1F4N58CAX8IMK:35DR22AR5ES6RR89U7EJ1DXW91AX3P"
        continue
    end
    next_move_index = findfirst(x -> x == row.move, row.all_moves)
    prb_idx = findfirst(x->x==row.puzzle, problems[prbs])
    subj_idx = findfirst(x->x==row.subject, subjects)
    fi = fitting_infos[prbs[prb_idx]];
    state_to_idx, idx_to_state, game, N_t = analysis_info[prbs[prb_idx]]
    s = state_to_idx[row.s]
    for i in 6:6
        p, neighs = get_p_new([params[subj_idx, 1:i-1]..., 0.0, params[subj_idx, i+1:end]...], s, row.prev_move, fi)
        aligned_p = zeros(length(p))
        for (m, neigh) in enumerate(neighs)
            s_int = idx_to_state[neigh]
            state = int_to_s(s_int, game)
            _, s_int_32 = board_to_int32(state)
            idx = findfirst(x->x==s_int_32, row.neighs)
            aligned_p[idx] = p[m]
        end
        nlls_lesion[i, n] -= log(aligned_p[next_move_index])
    end
end

r = df_[6064, :];
prb_idx = findfirst(x->x==r.puzzle, problems[prbs]);
subj_idx = findfirst(x->x==r.subject, subjects);
s = analysis_info[prbs][prb_idx][1][r.s];
game = analysis_info[prbs][prb_idx][3]


fi = fitting_infos[prbs[prb_idx]];
p, neighs = get_p_new(params[20, :], s, r.prev_move, fi)
p, neighs = get_p_new([1.0, 2.0, 0.9, 3.0, 0.05, 0.05], s, r.prev_move, fi);


update_QR_matrices_and_or!(fi, γao, r.prev_move)

B = apply_γp(s, γp, α, fi)

N_a, N_t = size(fi.R)
policy = vcat(fi.Q[:, s].nzval, fi.R[:, s].nzval)
neighs = vcat(fi.Q[:, s].nzind, N_t .+ fi.R[:, s].nzind)

B_ = zeros(N_a, 4)
B_[:, 1] .= 1.0
B__ = vcat(B, B_)
p = inf_sum(γre, u, v, policy, B__[neighs, :], 10_000_000)




prbs = sort([52, 29, 64, 12, 37, 8, 63, 58, 35, 46, 57, 20, 22, 16, 69, 68, 41, 56, 49, 32])
prbs = prbs[1:2]
fis = fitting_infos[prbs];
sas = [state_actions_idx[i][prbs] for i in 1:42];
N_states = sum(length.([sas[16][i][1] for i in 1:length(sas[16])]) .- 1)

N_states = 50
N_repeats = 20
params_single = zeros(6, N_states)
nlls_single = zeros(6, N_states)
nlls_orig = zeros(6, N_states)
landscape = zeros(6, N_states, N_repeats)
lims = [(0, 10), (0, 10), (0, 1), (0, 10), (-5, 5), (-5, 5)]
for i_param in ProgressBar(1:6)
    for i in 1:N_states
        ff = x -> get_nll_single_state([params[16, 1:i_param-1]..., x, params[16, i_param+1:end]...], i, sas[16], fis)
        res = optimize(ff, lims[i_param][1], lims[i_param][2]; iterations=50)
        params_single[i_param, i] = res.minimizer
        nlls_single[i_param, i] = res.minimum
        nlls_orig[i_param, i] = ff(params[16, i_param])
        for (j, param) in enumerate(lims[i_param][1]:((lims[i_param][2]-lims[i_param][1])/(N_repeats-1)):lims[i_param][2])
            landscape[i_param, i, j] = ff(param)
        end
    end
end
saslist = []
for i in eachindex(sas[16])
    if isempty(sas[16][i])
        continue
    end
    for j in 1:length(sas[16][i][1])-1
        push!(saslist, (i, sas[16][i][1][j][1], sas[16][i][1][j][2]))
    end
end

plot(layout=grid(2, 3), size=(600, 400), legend=false, grid=false);
for i in 1:6
    xs = lims[i][1]:((lims[i][2]-lims[i][1])/(N_repeats-1)):lims[i][2]
    plot!(xs, landscape[i, :, :]' .- mean(landscape[i, :, :]', dims=1), sp=i, c=:gray, alpha=0.5)
    # plot!(xs, sum(landscape[i, :, :]', dims=2) .- mean(sum(landscape[i, :, :]', dims=2)), c=:black, sp=i)
    vline!([params[16, i]], c=:red, sp=i)
end
plot!()

f = x -> get_nll_new(x, sas[1], fis)
g = x -> get_nll_gamma_re_only(x, params[16, :], sas[16], fis)
h = x -> get_nll_gamma_re_only_single_state(x, params[16, :], sas[16][1][1][1], sas[16][1][1][2], sas[16][1][1][3], fis[1])
res_ff = optimize(f, [1.0, 0.5, 1.0, 0.5, 1.0, 0.0, 0.0]; g_tol=0.0001, show_trace=true, show_every=40, iterations=400);
res_f = optimize(f, [1.0, 1.0, 0.5, 1.0, 0.0, 0.0]; g_tol=0.0001, show_trace=true, show_every=40, iterations=200);
res_g = optimize(g, ones(Float64, N_states); g_tol=0.0001, show_trace=true, show_every=100, iterations=1000);
res_h = optimize(h, 0, 10; iterations=50);
a = res_g.minimizer
b = res_h.minimizer

ps = []
nll = 0.0
for i in ProgressBar(1:8)
    if i == 1
        h = x -> get_nll_gamma_re_only_single_state(x, params[16, :], (0, (0, 0)), sas[16][1][1][i], sas[16][1][1][i+1], fis[1])
    else
        h = x -> get_nll_gamma_re_only_single_state(x, params[16, :], sas[16][1][1][i-1], sas[16][1][1][i], sas[16][1][1][i+1], fis[1])
    end
    res_h = optimize(h, 0, 10; iterations=50);
    push!(ps, res_h.minimizer)
    nll += res_h.minimum
end

prev_move = sas[16][1][1][end-2][2]
s = sas[16][1][1][end-1][1]

update_QR_matrices_and_or!(fi, 0.00000999, prev_move)#γao

B = apply_γp(s, 0.5, α, fi)#γp

N_a, N_t = size(fi.R)
policy = vcat(fi.Q[:, s].nzval, fi.R[:, s].nzval)
neighs = vcat(fi.Q[:, s].nzind, N_t .+ fi.R[:, s].nzind)

B_ = zeros(N_a, 4)
B_[:, 1] .= 1.0
B__ = vcat(B, B_)
p = inf_sum(0.01, u, v, policy, B__[neighs, :], 10_000_000)
















# function get_nll_gamma_re_only_single_state(nlogγre::Float64, other_params, state_action_prev, state_action, state_action_next, fitting_info)
#     nlogγao, nlogγp, α, _, u, v = other_params
#     γao = 10 ^ (-nlogγao)
#     γp = 10 ^ (-nlogγp)
#     γre = 10 ^ (-nlogγre)
#     if γre < 1e-10 || γre > 1
#         return Inf
#     end
#     s, a = state_action
#     s_next, a_next = state_action_next
#     s_prev, a_prev = state_action_prev
#     θp = [γao, γp, α, γre, u, v]
#     return -get_ll_new(θp, s, s_next, a_prev, fitting_info)
# end

# function get_nll_gamma_re_only(θ::Vector{Float64}, other_params, state_actions, fitting_infos)
#     nlogγao, nlogγp, α, _, u, v = other_params
#     γao = 10 ^ (-nlogγao)
#     γp = 10 ^ (-nlogγp)

#     γres = 10 .^ (-θ)
#     if any(γres .< 1e-10) || any(γres .> 1)
#         return Inf
#     end
#     nll = 0.0
#     count = 1
#     for j in eachindex(state_actions)
#         if isempty(state_actions[j])
#             continue
#         end
#         fitting_info = fitting_infos[j]
#         for game_index in eachindex(state_actions[j])
#             state_actions_game = state_actions[j][game_index]
#             a_prev = (0, 0)
#             s, a = state_actions_game[1]
#             for i in 1:length(state_actions_game)-1
#                 s_next, a_next = state_actions_game[i+1]
#                 θp = [γao, γp, α, γres[count], u, v]
#                 nll -= get_ll_new(θp, s, s_next, a_prev, fitting_info)

#                 a_prev = a
#                 s = s_next
#                 a = a_next
#                 count += 1
#             end
#         end
#     end
#     return nll
# end