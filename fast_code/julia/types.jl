const Action = Tuple{Int, Int}

struct FastFittingInfo
    domain::Symbol
    n_transient::Int
    n_absorbing::Int
    all_idxs::Vector{Vector{Int}}
    all_moves::Vector{Vector{Action}}
    all_sizes::Vector{Int}
    all_dicts::Vector{Dict{Tuple{Int64, Tuple{Int64, Int64}}, Float64}}
end

struct FastTransition
    puzzle::Int
    state::Int
    next_state::Int
    previous_move::Action
end

struct FastDataset
    transitions::Vector{FastTransition}
end

mutable struct PolicyBundle
    action_probs::Vector{Vector{Float64}}
    p_stop::Vector{Float64}
    final_probs::Vector{Union{Nothing, Vector{Float64}}}
end

const PolicyCache = Dict{Tuple{Int, Float64, Float64, Action}, PolicyBundle}
