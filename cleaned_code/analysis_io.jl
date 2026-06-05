function load_results(; tol=false)
    folder = joinpath("data", "all_fitted_results_$(tol ? "TOL" : "RH")")
    N_subj = tol ? 469 : 42
    N_batch = tol ? 10 : 1
    param_names = ["full", "lesion-$(tol ? "beta" : "gamma_ao")", "lesion-gamma_r", "lesion-gamma_p", "only-$(tol ? "beta" : "gamma_ao")", "only-gamma_r", "only-gamma_p"]
    return _load_result_batches(folder, N_subj, N_batch, param_names, 3; include_chance=true)
end

function load_newest_results(; tol=false)
    folder = joinpath("data", "newest_results_$(tol ? "TOL" : "RH")")
    N_subj = tol ? 469 : 42
    N_batch = tol ? 10 : 1
    param_names = ["full", "lesion-$(tol ? "beta" : "gamma_ao")", "lesion-gamma_r", "lesion-gamma_w", "lesion-gamma_p"]
    return _load_result_batches(folder, N_subj, N_batch, param_names, 4; include_chance=true)
end

function load_perstate_results(; tol=false)
    folder = joinpath("data", "fit_per_state_$(tol ? "TOL" : "RH")")
    N_subj = tol ? 469 : 42
    N_prb = tol ? 25 : 35
    N_batch = tol ? 10 : 1
    all_params = fill(NaN, 3, N_subj, N_prb)
    all_nlls = zeros(3, N_subj, N_prb)

    for filename in readdir(folder)
        startswith(filename, ".") && continue
        i = parse(Int, split(split(filename, ".")[1], "_")[2])
        batch = load(joinpath(folder, filename))
        params_batch = batch["new_params"]
        nlls_batch = batch["new_nlls"]
        if i * N_batch > N_subj
            keep = 1:N_batch - ((i * N_batch) - N_subj)
            all_params[:, (i - 1) * N_batch + 1:end, :] = params_batch[:, keep, :]
            all_nlls[:, (i - 1) * N_batch + 1:end, :] = nlls_batch[:, keep, :]
        else
            all_params[:, (i - 1) * N_batch + 1:i * N_batch, :] = params_batch
            all_nlls[:, (i - 1) * N_batch + 1:i * N_batch, :] = nlls_batch
        end
    end

    return all_params, all_nlls
end

function load_classical_results(; tol=false)
    folder = joinpath("data", "fitting_classical_$(tol ? "TOL" : "RH")")
    N_subj = tol ? 469 : 42
    N_batch = tol ? 10 : 1
    model_names = ["bfs", "dfs", "bestfirst", "astar", "iddfs", "idastar"]
    param_names = ["logd", "beta", "lambda", "max_bound"]
    names = String[]
    for model_name in model_names
        push!(names, "$(model_name)-full")
        for param_name in param_names, extra in ("only", "lesion")
            push!(names, "$(model_name)-$(extra)-$(param_name)")
        end
    end
    return _load_result_batches(folder, N_subj, N_batch, names, 4; include_chance=false)
end

function _load_result_batches(folder, N_subj, N_batch, names, N_params; include_chance=false)
    all_params = Dict(name => zeros(N_subj, N_params) for name in names)
    all_nlls = Dict(name => zeros(N_subj) for name in names)
    if include_chance
        all_params["chance"] = zeros(N_subj, N_params)
        all_nlls["chance"] = zeros(N_subj)
    end

    for filename in readdir(folder)
        startswith(filename, ".") && continue
        i = parse(Int, split(split(filename, ".")[1], "_")[2])
        batch = load(joinpath(folder, filename))
        params_batch = batch["all_params"]
        nlls_batch = batch["all_nlls"]
        start_row = (i - 1) * N_batch + 1
        stop_row = min(i * N_batch, N_subj)
        rows = start_row:stop_row
        keep = 1:length(rows)

        for name in names
            all_params[name][rows, :] = params_batch[name][keep, :]
            all_nlls[name][rows] = nlls_batch[name][keep]
        end
        if include_chance
            all_nlls["chance"][rows] = nlls_batch["chance"][keep]
        end
    end

    return all_params, all_nlls
end

function get_lengths()
    sas_rh = load(joinpath("data", "processed_data", "state_actions_idx.jld2"))["state_actions_idx"]
    sas_tol = load(joinpath("data", "tower_of_london", "state_actions_idx.jld2"))["state_actions_idx"]
    prbs_rh = [52, 29, 64, 12, 37, 8, 63, 58, 46, 57, 35, 22, 20, 16, 69, 68, 41, 56, 32, 49, 33, 2, 21, 36, 50, 51, 61, 65, 62, 14, 70, 23, 1, 25, 31, 4, 38, 39, 48, 6, 26, 45, 60, 55, 67, 53, 5, 54, 19, 27, 44, 28, 10, 42, 18, 13, 11, 40, 3, 9, 24, 15, 59, 43, 47, 30, 7, 66, 34, 17][1:35]
    lengths_rh = [[isempty(sas_rh[i][j]) ? 0 : length(sas_rh[i][j][1]) - 1 for j in prbs_rh] for i in 1:42]
    lengths_tol = [[isempty(sas_tol[i][j]) ? 0 : length(sas_tol[i][j][1]) - 1 for j in 1:25] for i in 1:469]
    return lengths_rh, lengths_tol
end

function get_trial_nlls(nlls, lengths)
    trial_nlls = Dict{String, Vector{Float64}}()
    summed_lengths = sum.(lengths)
    for (name, values) in nlls
        trial_nlls[name] = reduce(vcat, [[values[i] / summed_lengths[i] for _ in 1:summed_lengths[i]] for i in eachindex(values)])
    end
    return trial_nlls
end

function calculate_nlls(df; tol=false)
    names = ["chance", "only-$(tol ? "beta" : "gamma_ao")", "only-gamma_r", "only-gamma_p", "lesion-$(tol ? "beta" : "gamma_ao")", "lesion-gamma_r", "lesion-gamma_p", "full"]
    nlls = Dict(name => zeros(nrow(df)) for name in names)
    for (n, row) in enumerate(eachrow(df))
        move_idx = findfirst(==(row.move), row.all_moves)
        for name in names
            nlls[name][n] -= log(row[Symbol("p_$name")][move_idx])
        end
    end
    return nlls
end
