function get_QR_matrices(game::AbstractGame)
    states, leafs = search(game)
    N_t = length(states)
    N_a = length(leafs)

    Q = sparse(zeros(N_t, N_t))
    R = sparse(zeros(N_a, N_t))
    state_to_idx = Dict{Int, Int}()
    idx_to_state = zeros(Int, N_t + N_a)

    for (n, state) in enumerate(states)
        state_to_idx[state] = n
        idx_to_state[n] = state
    end
    for (n, state) in enumerate(leafs)
        state_to_idx[state] = -n
        idx_to_state[N_t + n] = state
    end

    for state in states
        from = state_to_idx[state]
        neighs = neighbours(state, game)
        for neigh in neighs
            to = state_to_idx[neigh]
            if neigh in leafs
                R[-to, from] = 1.0 / length(neighs)
            elseif neigh in states
                Q[to, from] = 1.0 / length(neighs)
            else
                error("state not found")
            end
        end
    end

    return state_to_idx, idx_to_state, Q, R
end
