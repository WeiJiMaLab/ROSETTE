#!/usr/bin/env julia

# Regenerate all code-generated quantitative panels from the public tables.
# Schematic/artwork panels and stochastic trajectory layouts are retained as
# the final reviewed assets in paper/figures/ (see README provenance table).

const ROOT = normpath(joinpath(@__DIR__, ".."))
cd(ROOT) do
    for script in (
        "figure06_bic.jl",
        "figure07_summary.jl",
        "figure08_counterintuitive.jl",
        "figure09_excess_length.jl",
        "figure09_shortest_path_control.jl",
        "figure10_transition_counts.jl",
    )
        println("Running $script")
        run(`$(Base.julia_cmd()) --project=$ROOT $(joinpath(ROOT,"scripts",script))`)
    end
end
println("Generated panels are in paper/generated/. Published figures remain in paper/figures/.")
