#!/usr/bin/env julia

# Reconstruct the model's indexed trajectories and per-puzzle fitting objects
# from the compact public tables. Use `TOL 1` for a quick smoke test. Building
# all Rush Hour state spaces is substantially slower and uses more memory.

using CSV, DataFrames, JLD2

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT,"src","ROSETTE.jl"))
using .ROSETTE

function public_problems(task::String)
    if task == "RH"
        rows=sort(CSV.read(joinpath(ROOT,"data","published","rh_puzzles.csv"),DataFrame),:puzzle)
        return String.(rows.file)
    elseif task == "TOL"
        rows=sort(CSV.read(joinpath(ROOT,"data","published","tol_puzzles.csv"),DataFrame),:puzzle)
        return [(
            [Int(row[Symbol("start_$slot")]) for slot in 1:6],
            [Int(row[Symbol("goal_$slot")]) for slot in 1:6],
        ) for row in eachrow(rows)]
    end
    error("Task must be RH or TOL")
end

function public_choices(task::String; max_puzzles=task=="RH" ? 70 : 25)
    rows=CSV.read(joinpath(ROOT,"data","published","choices.csv"),DataFrame)
    filter!(r -> r.task==task && r.puzzle<=max_puzzles,rows)
    n_subjects=task=="RH" ? 42 : 469
    nested=[[Vector{Tuple{Int,ROSETTE.Action}}[] for _ in 1:max_puzzles] for _ in 1:n_subjects]
    for group in groupby(rows,[:subject,:puzzle,:attempt])
        ordered=sort(group,:step)
        moves=Tuple{Int,ROSETTE.Action}[(Int(r.state),(Int(r.action_1),Int(r.action_2))) for r in eachrow(ordered)]
        push!(nested[group.subject[1]][group.puzzle[1]],moves)
    end
    return nested
end

function build_cache(task::String; max_puzzles=task=="RH" ? 70 : 25)
    problems=public_problems(task)[1:max_puzzles]
    choices=public_choices(task;max_puzzles=max_puzzles)
    fitting_infos=cd(ROOT) do
        get_fitting_infos(problems)
    end
    cache_dir=joinpath(ROOT,"data","cache")
    mkpath(cache_dir)
    path=joinpath(cache_dir,"$(lowercase(task))_fitting_inputs_$(max_puzzles).jld2")
    jldsave(path;problems,choices,fitting_infos)
    println("Saved $(length(fitting_infos)) fitting objects and $(length(choices)) participant trajectories to $path")
    return path
end

if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    length(ARGS) in (1,2) || error("Usage: julia --project=. scripts/rebuild_model_inputs.jl RH|TOL [N_puzzles]")
    task=uppercase(ARGS[1])
    n=length(ARGS)==2 ? parse(Int,ARGS[2]) : task=="RH" ? 70 : 25
    build_cache(task;max_puzzles=n)
end
