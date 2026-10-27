#!/usr/bin/env julia

# One-time export of the anonymous, indexed inputs used for the published fits.
# Run with: julia --project=. scripts/export_paper_inputs.jl /path/to/original/workspace
# The original workspace is needed only for this export. The resulting CSV files
# are the public, identifier-free inputs for the release.

using CSV
using DataFrames
using JLD2

length(ARGS) == 1 || error("Pass the original ROSETTE workspace as one argument")
const SOURCE = abspath(ARGS[1])
const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUT = joinpath(ROOT, "data", "published")
isdir(SOURCE) || error("Original workspace not found: $SOURCE")
mkpath(OUT)

function export_choices()
    choices = DataFrame(
        task=String[], subject=Int[], puzzle=Int[], attempt=Int[], step=Int[],
        state=Int[], action_1=Int[], action_2=Int[], terminal=Bool[],
    )
    for (task, relative_path) in (
        ("RH", joinpath("data", "processed_data", "state_actions_idx.jld2")),
        ("TOL", joinpath("data", "tower_of_london", "state_actions_idx.jld2")),
    )
        nested = load(joinpath(SOURCE, relative_path))["state_actions_idx"]
        expected_subjects = task == "RH" ? 42 : 469
        expected_puzzles = task == "RH" ? 70 : 25
        length(nested) == expected_subjects || error("Unexpected $task subject count")
        for (subject, puzzles) in enumerate(nested)
            length(puzzles) == expected_puzzles || error("Unexpected $task puzzle count")
            for (puzzle, attempts) in enumerate(puzzles)
                for (attempt, trajectory) in enumerate(attempts)
                    for (step, (state, action)) in enumerate(trajectory)
                        terminal = step == length(trajectory)
                        terminal == (action == (0, 0)) || error(
                            "Terminal marker mismatch at $task/$subject/$puzzle/$attempt/$step",
                        )
                        push!(choices, (
                            task, subject, puzzle, attempt, step, state,
                            action[1], action[2], terminal,
                        ))
                    end
                end
            end
        end
    end
    CSV.write(joinpath(OUT, "choices.csv"), choices)
    println("Wrote $(nrow(choices)) indexed choice/terminal rows")
end

function export_puzzles()
    rh = CSV.read(joinpath(SOURCE, "data", "rh_data.csv"), DataFrame; select=[:puzzle])
    problems = String.(unique(rh.puzzle))
    order = sortperm([
        parse(Int, name[end-1] == '_' ? name[end] : name[end-1:end])
        for name in problems
    ])
    problems = problems[order]
    length(problems) == 70 || error("Unexpected Rush Hour puzzle count")
    CSV.write(joinpath(OUT, "rh_puzzles.csv"), DataFrame(puzzle=1:70, file=problems))

    tol = load(joinpath(SOURCE, "data", "tower_of_london", "problems.jld2"))["problems"]
    length(tol) == 25 || error("Unexpected Tower of London puzzle count")
    cols = Dict{Symbol, Vector{Int}}(:puzzle => collect(1:25))
    for prefix in ("start", "goal"), slot in 1:6
        cols[Symbol("$(prefix)_$(slot)")] = Int[]
    end
    for (start, goal) in tol
        for (prefix, state) in (("start", start), ("goal", goal))
            slots = vcat(state...)
            length(slots) == 6 || error("Tower of London puzzle has wrong slot count")
            for slot in 1:6
                push!(cols[Symbol("$(prefix)_$(slot)")], slots[slot])
            end
        end
    end
    names = vcat(:puzzle, [Symbol("$(prefix)_$(slot)") for prefix in ("start", "goal") for slot in 1:6])
    CSV.write(joinpath(OUT, "tol_puzzles.csv"), DataFrame([name => cols[name] for name in names]))
end

function export_fits()
    rows = DataFrame(task=String[], subject=Int[], model=String[], nll=Float64[], n_parameters=Int[])
    parameter_rows = DataFrame(
        task=String[], subject=Int[], model=String[], parameter_index=Int[], value=Float64[],
    )
    for task in ("RH", "TOL")
        n_subjects = task == "RH" ? 42 : 469
        batch_size = task == "RH" ? 1 : 10
        heuristic = task == "RH" ? "gamma_ao" : "beta"
        model_counts = Dict(
            "full" => 3, "chance" => 0,
            "lesion-$heuristic" => 2, "lesion-gamma_r" => 2, "lesion-gamma_p" => 2,
            "only-$heuristic" => 1, "only-gamma_r" => 1, "only-gamma_p" => 1,
            "bfs-full" => 3, "dfs-full" => 3, "bestfirst-full" => 3,
            "astar-full" => 3, "iddfs-full" => 4, "idastar-full" => 4,
        )
        for (folder, models) in (
            ("all_fitted_results_$task", [
                "full", "chance", "lesion-$heuristic", "lesion-gamma_r", "lesion-gamma_p",
                "only-$heuristic", "only-gamma_r", "only-gamma_p",
            ]),
            ("fitting_classical_$task", [
                "bfs-full", "dfs-full", "bestfirst-full", "astar-full", "iddfs-full", "idastar-full",
            ]),
        )
            for batch in 1:cld(n_subjects, batch_size)
                path = joinpath(SOURCE, "data", folder, "results_$(batch).jld2")
                result = load(path)
                nlls = result["all_nlls"]
                params = result["all_params"]
                first_subject = (batch - 1) * batch_size + 1
                last_subject = min(batch * batch_size, n_subjects)
                for model in models
                    haskey(nlls, model) || error("Missing $model in $path")
                    for subject in first_subject:last_subject
                        row = subject - first_subject + 1
                        nll = Float64(nlls[model][row])
                        isfinite(nll) || error("Non-finite NLL: $task/$subject/$model")
                        push!(rows, (task, subject, model, nll, model_counts[model]))
                        if folder == "all_fitted_results_$task" && model != "chance"
                            for parameter_index in axes(params[model], 2)
                                push!(parameter_rows, (
                                    task, subject, model, parameter_index,
                                    Float64(params[model][row, parameter_index]),
                                ))
                            end
                        end
                    end
                end
            end
        end
    end
    nrow(rows) == (42 + 469) * 14 || error("Missing published NLLs")
    CSV.write(joinpath(OUT, "fit_nll.csv"), rows)
    CSV.write(joinpath(OUT, "fit_parameters.csv"), parameter_rows)
    println("Wrote $(nrow(rows)) fitted NLLs and $(nrow(parameter_rows)) parameter values")
end

export_choices()
export_puzzles()
export_fits()
