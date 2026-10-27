#!/usr/bin/env julia

# One-time extraction from the original simulation archive. The public output
# retains the trajectories used for the state-space illustrations, but not the
# complete multi-gigabyte collection of all models and repeats.

length(ARGS) == 1 || error("Pass the original ROSETTE workspace as one argument")
const SOURCE = abspath(ARGS[1])
const ROOT = normpath(joinpath(@__DIR__, ".."))
const DEST = joinpath(ROOT, "data", "published", "illustrated_simulations.csv")
const EXPECTED_HEADER = "subj,puzzle,name,repeat,sa_index,s,a1,a2"

function keep(task, puzzle, model, repeat)
    if task == "RH"
        return puzzle == 52 && model in ("full", "lesion-gamma_ao", "only-gamma_ao", "chance") && 1 <= repeat <= 10
    end
    return model == "full" && repeat == 8 && 1 <= puzzle <= 25
end

function export_task(io, task)
    folder = joinpath(SOURCE, "data", "simulated_games_$task")
    files = filter(f -> occursin(r"^model_games_\d+\.csv$", f), readdir(folder))
    sort!(files; by=f -> parse(Int, match(r"\d+", f).match))
    isempty(files) && error("No simulation files for $task")
    count = 0
    for file in files
        open(joinpath(folder, file)) do input
            readline(input) == EXPECTED_HEADER || error("Unexpected columns in $file")
            for raw in eachline(input)
                fields = split(raw, ',')
                length(fields) == 8 || error("Malformed row in $file")
                subject = parse(Int, fields[1])
                puzzle = parse(Int, fields[2])
                model = fields[3]
                repeat = parse(Int, fields[4])
                keep(task, puzzle, model, repeat) || continue
                println(io, join((task, subject, puzzle, model, repeat, fields[5], fields[6], fields[7], fields[8]), ','))
                count += 1
            end
        end
    end
    println("$task: wrote $count illustrated simulation steps from $(length(files)) archive files")
end

open(DEST, "w") do io
    println(io, "task,subject,puzzle,model,repeat,step,state,action_1,action_2")
    export_task(io, "RH")
    export_task(io, "TOL")
end
println("Wrote $DEST")
