#!/usr/bin/env julia

# Edge frequencies underlying the Rush Hour and Tower of London trajectory
# illustrations (Figure 10 and Figures S3-S4). Rendering of the graph layouts
# and spline jitter was finalized manually, so only counts are regenerated.

using CSV, DataFrames

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DATA = joinpath(ROOT, "data", "published")
const OUT = joinpath(ROOT, "paper", "generated")
mkpath(OUT)

counts = Dict{Tuple{String,Int,String,Int,Int},Int}()
function tally!(task,puzzle,model,states)
    for i in 1:(length(states)-1)
        key=(task,puzzle,model,states[i],states[i+1])
        counts[key]=get(counts,key,0)+1
    end
end

human = CSV.read(joinpath(DATA,"choices.csv"),DataFrame)
for group in groupby(human,[:task,:subject,:puzzle,:attempt])
    task=group.task[1]
    puzzle=group.puzzle[1]
    (task=="RH" && puzzle==52 || task=="TOL") || continue
    sorted=sort(group,:step)
    tally!(task,puzzle,"human",Int.(sorted.state))
end

model = CSV.read(joinpath(DATA,"illustrated_simulations.csv"),DataFrame)
for group in groupby(model,[:task,:subject,:puzzle,:model,:repeat])
    sorted=sort(group,:step)
    tally!(group.task[1],group.puzzle[1],string(group.model[1]),Int.(sorted.state))
end

result=DataFrame(task=String[],puzzle=Int[],model=String[],from_state=Int[],to_state=Int[],count=Int[])
for ((task,puzzle,model,from_state,to_state),count) in counts
    push!(result,(task,puzzle,model,from_state,to_state,count))
end
sort!(result,[:task,:puzzle,:model,:from_state,:to_state])
CSV.write(joinpath(OUT,"trajectory_edge_counts.csv"),result)
println("Wrote $(nrow(result)) directed transition counts")
