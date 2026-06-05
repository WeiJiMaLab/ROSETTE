const RH_DEFAULT_PROBLEM_ORDER = [52, 29, 64, 12, 37, 8, 63, 58, 46, 57, 35, 22, 20, 16, 69, 68, 41, 56, 32, 49, 33, 2, 21, 36, 50, 51, 61, 65, 62, 14, 70, 23, 1, 25, 31, 4, 38, 39, 48, 6, 26, 45, 60, 55, 67, 53, 5, 54, 19, 27, 44, 28, 10, 42, 18, 13, 11, 40, 3, 9, 24, 15, 59, 43, 47, 30, 7, 66, 34, 17]

function fit_rh_subject(subject_index::Int, N_prbs::Int; iters=200, problem_order=RH_DEFAULT_PROBLEM_ORDER)
    fitting_infos = load(joinpath("data", "processed_data", "fitting_infos.jld2"))["fitting_infos"]
    state_actions_idx = load(joinpath("data", "processed_data", "state_actions_idx.jld2"))["state_actions_idx"]
    prbs = problem_order[1:N_prbs]
    fis = fitting_infos[prbs]
    sas = [state_actions_idx[i][prbs] for i in 1:42]
    objective = x -> get_nll_main(x, sas[subject_index], fis)
    return optimize(objective, [1.0, 2.0, 2.0]; g_tol=0.0001, show_trace=true, show_every=20, iterations=iters)
end

function fit_tol_subject_batch(batch_index::Int; iters=200, batch_size=10)
    fitting_infos = load(joinpath("data", "tower_of_london", "fitting_infos.jld2"))["fitting_infos"]
    state_actions_idx = load(joinpath("data", "tower_of_london", "state_actions_idx.jld2"))["state_actions_idx"]
    prbs = collect(1:25)
    fis = fitting_infos[prbs]
    sas = [state_actions_idx[i][prbs] for i in 1:469]
    params = zeros(batch_size, 3)
    nlls = zeros(batch_size)

    for i in 1:batch_size
        subj = (batch_index - 1) * batch_size + i
        subj > 469 && break
        objective = x -> get_nll_main(x, sas[subj], fis)
        result = optimize(objective, [1.0, 2.0, 2.0]; g_tol=0.0001, show_trace=true, show_every=20, iterations=iters)
        params[i, :] = result.minimizer
        nlls[i] = result.minimum
    end

    return params, nlls
end

function simulate_reduced_rh_games(subject_index::Int, repeats::Int, fitting_infos, state_actions_idx, params; max_length=1_000)
    model_games = Vector{Vector{Vector{Tuple{Int, Action}}}}()
    for puzzle in eachindex(state_actions_idx[subject_index])
        push!(model_games, Vector{Vector{Tuple{Int, Action}}}())
        isempty(state_actions_idx[subject_index][puzzle]) && continue

        sas = state_actions_idx[subject_index][puzzle][end]
        fi = fitting_infos[puzzle]
        _, N_t = size(fi.R)

        for _ in 1:repeats
            s = sas[1][1]
            prev_move = (0, 0)
            game = Tuple{Int, Action}[]
            for step in 1:max_length
                p, neighs = get_p_reduced_rh(params, s, prev_move, fi)
                next_idx = wsample(p)
                s_next = neighs[next_idx]
                prev_move = fi.all_moves[s][findfirst(==(s_next), fi.all_idxs[s])]
                push!(game, (s, prev_move))
                s = s_next
                if s > N_t
                    push!(game, (s, (0, 0)))
                    break
                elseif step == max_length
                    throw(ErrorException("Simulation reached max_length=$max_length"))
                end
            end
            push!(model_games[puzzle], game)
        end
    end
    return model_games
end
