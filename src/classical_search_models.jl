function get_p_bfs(θ::Vector{Float64}, s::Int, prev_move::Action, fi::FittingStruct)::Tuple{Vector{Float64}, Vector{Int}}
    logd, β, λ, _ = θ
    vs, neighs = classic_treesearch(s, round(Int, 10^logd), fi; bfs=true)
    return get_p_treesearch(β, λ, vs, neighs), neighs
end

function get_p_dfs(θ::Vector{Float64}, s::Int, prev_move::Action, fi::FittingStruct)::Tuple{Vector{Float64}, Vector{Int}}
    logd, β, λ, _ = θ
    vs, neighs = classic_treesearch(s, round(Int, 10^logd), fi; dfs=true)
    return get_p_treesearch(β, λ, vs, neighs), neighs
end

function get_p_best_first_search(θ::Vector{Float64}, s::Int, prev_move::Action, fi::FittingStruct)::Tuple{Vector{Float64}, Vector{Int}}
    logd, β, λ, _ = θ
    vs, neighs = classic_treesearch(s, round(Int, 10^logd), fi; bestfirst=true)
    return get_p_treesearch(β, λ, vs, neighs), neighs
end

function get_p_astar(θ::Vector{Float64}, s::Int, prev_move::Action, fi::FittingStruct)::Tuple{Vector{Float64}, Vector{Int}}
    logd, β, λ, _ = θ
    vs, neighs = classic_treesearch(s, round(Int, 10^logd), fi; astar=true)
    return get_p_treesearch(β, λ, vs, neighs), neighs
end

function get_p_iddfs(θ::Vector{Float64}, s::Int, prev_move::Action, fi::FittingStruct)::Tuple{Vector{Float64}, Vector{Int}}
    logd, β, λ, max_bound = θ
    vs, neighs = iterative_deepening(s, round(Int, max_bound), round(Int, 10^logd), fi; astar=false)
    return get_p_treesearch(β, λ, vs, neighs), neighs
end

function get_p_idastar(θ::Vector{Float64}, s::Int, prev_move::Action, fi::FittingStruct)::Tuple{Vector{Float64}, Vector{Int}}
    logd, β, λ, max_bound = θ
    vs, neighs = iterative_deepening(s, round(Int, max_bound), round(Int, 10^logd), fi; astar=true)
    return get_p_treesearch(β, λ, vs, neighs), neighs
end

function get_p_treesearch(β::Float64, λ::Float64, vs::Vector{Float64}, neighs::Vector{Int})::Vector{Float64}
    if all(isnan.(vs))
        return ones(length(neighs)) ./ length(neighs)
    end

    valid = .!isnan.(vs)
    sm_values = β .* vs[valid]
    p_valid = exp.(sm_values .- maximum(sm_values))
    p_valid ./= sum(p_valid)

    p_treesearch = zeros(length(neighs))
    p_treesearch[valid] = p_valid
    p_random = ones(length(neighs)) ./ length(neighs)
    return λ * p_random .+ (1 - λ) * p_treesearch
end

function classic_treesearch(
    s0::Int,
    d::Int,
    fi::FittingStruct;
    bfs=false,
    dfs=false,
    bestfirst=false,
    astar=false,
    max_iters=100_000,
)
    @assert bfs || dfs || bestfirst || astar "One of bfs, dfs, bestfirst, or astar must be true."

    P = vcat(fi.Q, fi.R)
    values = fi.all_sizes
    _, N_t = size(P)
    neighs = P[:, s0].nzind
    visited = Set{Int}([s0])
    goals = Set{Int}()

    if bestfirst || astar
        frontier = PriorityQueue{Tuple{Int, Int}, Int}()
        enqueue!(frontier, (s0, 0) => 0)
    else
        frontier = Deque{Tuple{Int, Int}}()
        push!(frontier, (s0, 0))
    end

    inv_graph = Dict{Int, Vector{Int}}()
    counter = 0

    for _ in 1:max_iters
        isempty(frontier) && break
        s, depth = bestfirst || astar ? dequeue!(frontier) : (bfs ? popfirst!(frontier) : pop!(frontier))

        if s > N_t
            push!(goals, s)
            continue
        end

        for sp in reverse(P[:, s].nzind)
            counter >= d && break
            counter += 1
            push!(get!(inv_graph, sp, Int[]), s)

            sp in visited && continue
            push!(visited, sp)
            if bestfirst
                enqueue!(frontier, (sp, depth + 1) => (sp > N_t ? 0 : values[sp]))
            elseif astar
                g = depth + 1
                h = sp > N_t ? 0 : values[sp]
                enqueue!(frontier, (sp, depth + 1) => g + h)
            else
                push!(frontier, (sp, depth + 1))
            end
        end
    end

    isempty(goals) && return fill(NaN, length(neighs)), neighs
    return backward_propagation(s0, goals, inv_graph, d, neighs), neighs
end

function backward_propagation(
    s0::Int,
    goals::Set{Int},
    inv_graph::Dict{Int, Vector{Int}},
    d::Int,
    neighs::Vector{Int};
    bound=10_000_000,
    max_iters=100_000,
)
    ds = Dict{Int, Int}()
    visited = Set{Int}()
    frontier = Deque{Int}()
    for goal in goals
        ds[goal] = 0
        push!(visited, goal)
        push!(frontier, goal)
    end

    counter = 0
    for _ in 1:max_iters
        isempty(frontier) && break
        s = popfirst!(frontier)
        s == s0 && continue
        depth = ds[s]
        depth + 1 > bound && continue

        for sp in inv_graph[s]
            counter >= d && break
            counter += 1
            sp in visited && continue
            push!(visited, sp)
            push!(frontier, sp)
            ds[sp] = depth + 1
        end
    end

    vs = zeros(length(neighs))
    for (i, sp) in enumerate(neighs)
        vs[i] = haskey(ds, sp) ? -ds[sp] : NaN
    end
    return vs
end

function iterative_deepening(s0::Int, max_bound::Int, d::Int, fi::FittingStruct; astar=false, max_iters=1000)
    P = vcat(fi.Q, fi.R)
    values = fi.all_sizes
    neighs = P[:, s0].nzind
    bound = astar ? values[s0] : 0
    d_done = 0
    prev_goals, goals = Set{Int}(), Set{Int}()
    prev_inv_graph, inv_graph = Dict{Int, Vector{Int}}(), Dict{Int, Vector{Int}}()

    for _ in 1:max_iters
        goals, inv_graph, new_bound, d_done = depth_limited_search(s0, bound, d_done, d, fi; astar=astar)
        if new_bound > max_bound || d_done >= d || new_bound == 10_000_000
            goals = prev_goals
            inv_graph = prev_inv_graph
            break
        end
        bound = new_bound
        prev_goals = goals
        prev_inv_graph = inv_graph
    end

    isempty(goals) && return fill(NaN, length(neighs)), neighs
    return backward_propagation(s0, goals, inv_graph, d, neighs; bound=astar ? 10_000_000 : max_bound), neighs
end

function depth_limited_search(s0::Int, bound::Int, d_done::Int, d::Int, fi::FittingStruct; astar=false, max_iters=100_000)
    P = vcat(fi.Q, fi.R)
    values = fi.all_sizes
    _, N_t = size(P)
    visited = Set{Int}([s0])
    goals = Set{Int}()
    inv_graph = Dict{Int, Vector{Int}}()
    latest_bound = 10_000_000
    counter = d_done

    if astar
        frontier = PriorityQueue{Tuple{Int, Int}, Int}()
        enqueue!(frontier, (s0, 0) => 0)
    else
        frontier = Deque{Tuple{Int, Int}}()
        push!(frontier, (s0, 0))
    end

    for _ in 1:max_iters
        isempty(frontier) && break
        s, depth = astar ? dequeue!(frontier) : pop!(frontier)

        if s > N_t
            push!(goals, s)
            continue
        end

        if astar
            f = depth + (s > N_t ? 0 : values[s])
            if f > bound
                latest_bound = min(f, latest_bound)
                continue
            end
        elseif depth > bound
            latest_bound = min(depth, latest_bound)
            continue
        end

        for sp in reverse(P[:, s].nzind)
            counter >= d && break
            counter += 1
            push!(get!(inv_graph, sp, Int[]), s)
            sp in visited && continue
            push!(visited, sp)

            if astar
                g = depth + 1
                h = sp > N_t ? 0 : values[sp]
                enqueue!(frontier, (sp, depth + 1) => g + h)
            else
                push!(frontier, (sp, depth + 1))
            end
        end
    end

    return goals, inv_graph, latest_bound, counter
end

function get_ll_classical(θ::Vector{Float64}, s::Int, s_next::Int, prev_move::Action, fi::FittingStruct, model_name::String)::Float64
    p, neighs = if model_name == "bfs"
        get_p_bfs(θ, s, prev_move, fi)
    elseif model_name == "dfs"
        get_p_dfs(θ, s, prev_move, fi)
    elseif model_name == "bestfirst"
        get_p_best_first_search(θ, s, prev_move, fi)
    elseif model_name == "astar"
        get_p_astar(θ, s, prev_move, fi)
    elseif model_name == "iddfs"
        get_p_iddfs(θ, s, prev_move, fi)
    elseif model_name == "idastar"
        get_p_idastar(θ, s, prev_move, fi)
    else
        error("Unknown model name: $model_name")
    end

    return log(p[findfirst(==(s_next), neighs)])
end

function get_nll_classical(θ::Vector{Float64}, state_actions, fitting_infos, model_name)
    logd, _, λ, max_bound = θ
    d = round(Int, 10^logd)
    max_bound = round(Int, max_bound)
    if λ < 0 || λ > 1 || max_bound < 0 || max_bound > 1000 || d > 1_000_000
        return Inf
    end

    ll = (params, s, s_next, prev_move, fi) -> get_ll_classical(params, s, s_next, prev_move, fi, model_name)
    return _sum_negative_log_likelihood(θ, state_actions, fitting_infos, ll)
end
