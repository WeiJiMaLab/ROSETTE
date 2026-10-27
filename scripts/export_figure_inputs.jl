#!/usr/bin/env julia

# One-time export of the participant-level values behind Figures 7-9 and S1-S2/S7.
# Run with: ROSETTE_HEADLESS_TESTS=1 julia --project=. scripts/export_figure_inputs.jl /path/to/original/workspace

using CSV
using DataFrames

length(ARGS) == 1 || error("Pass the original ROSETTE workspace as one argument")
const SOURCE = abspath(ARGS[1])
const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUT = joinpath(ROOT, "data", "published")
mkpath(OUT)
ENV["ROSETTE_HEADLESS_TESTS"] = "1"
include(joinpath(ROOT, "src", "ROSETTE.jl"))
using .ROSETTE

function export_move_summary_bins()
    # Match the subject order used by load_rh_data() while never exporting
    # the source identifiers present in the private raw event table.
    rh_subjects = unique(CSV.read(
        joinpath(SOURCE, "data", "rh_data.csv"), DataFrame; select=[:subject],
    ).subject)
    rh_subject_index = Dict(string(subject) => index for (index, subject) in enumerate(rh_subjects))
    length(rh_subject_index) == 42 || error("Unexpected Rush Hour participant count")
    output = DataFrame(
        task=String[], subject=Int[], model=String[], bin_number=Int[],
        mean_distance=Float64[], sem_distance=Float64[],
        mean_better=Float64[], sem_better=Float64[],
        mean_same=Float64[], sem_same=Float64[],
        mean_worse=Float64[], sem_worse=Float64[],
    )
    columns = [:subject, :puzzle, :model, :X_d_goal, :y_p_better, :y_p_same, :y_p_worse]
    for (task, filename, bins) in (
        ("RH", "summary_stats_rh.csv", 10),
        ("TOL", "summary_stats_tol.csv", 5),
    )
        input = CSV.read(
            joinpath(SOURCE, "data", "processed_data", filename), DataFrame;
            select=columns,
        )
        subject_bins = ROSETTE.bin_stats_subj(input, :X_d_goal; nbins=bins, subject_level=true)
        for row in eachrow(subject_bins)
            subject_index = task == "RH" ? rh_subject_index[string(row.subject)] : parse(Int, string(row.subject))
            push!(output, (
                task, subject_index, string(row.model), row.bin_number,
                row.mean_X_d_goal, row.std_X_d_goal,
                row.mean_y_p_better, row.std_y_p_better,
                row.mean_y_p_same, row.std_y_p_same,
                row.mean_y_p_worse, row.std_y_p_worse,
            ))
        end
    end
    CSV.write(joinpath(OUT, "move_summary_subject_bins.csv"), output)
    println("Wrote $(nrow(output)) subject/model/bin rows for Figure 7 and S1-S2")
end

function export_counterintuitive_bins()
    participant = DataFrame(task=String[], subject=Int[], model=String[], bin_number=Int[],
        mean_distance=Float64[], sem_distance=Float64[],
        mean_counterintuitive=Float64[], sem_counterintuitive=Float64[])
    combined = DataFrame(task=String[], model=String[], bin_number=Int[],
        mean_distance=Float64[], ci95_distance=Float64[],
        mean_counterintuitive=Float64[], ci95_counterintuitive=Float64[])
    rh_subjects = unique(CSV.read(
        joinpath(SOURCE, "data", "rh_data.csv"), DataFrame; select=[:subject],
    ).subject)
    rh_index = Dict(string(subject) => index for (index, subject) in enumerate(rh_subjects))
    for (task, source_name, bins) in (
        ("RH", "counterintuitive_summary_stats_rh_worsening_preparatory.csv", 10),
        ("TOL", "counterintuitive_summary_stats_tol.csv", 5),
    )
        path = joinpath(SOURCE, "fast_code", "validation", "results_model_evidence_20260808", source_name)
        raw = CSV.read(path, DataFrame; select=[:subject,:puzzle,:model,:X_d_goal,:y_p_counterintuitive])
        by_subject = ROSETTE.bin_stats_subj(raw, :X_d_goal; nbins=bins, subject_level=true)
        for row in eachrow(by_subject)
            subject_index = task == "RH" ? rh_index[string(row.subject)] : parse(Int, string(row.subject))
            push!(participant, (task, subject_index, string(row.model), row.bin_number,
                row.mean_X_d_goal, row.std_X_d_goal,
                row.mean_y_p_counterintuitive, row.std_y_p_counterintuitive))
        end
        aggregated = ROSETTE.bin_stats_subj(raw, :X_d_goal; nbins=bins)
        for row in eachrow(aggregated)
            push!(combined, (task, string(row.model), row.bin_number,
                row.m_X_d_goal, row.sem_X_d_goal,
                row.m_y_p_counterintuitive, row.sem_y_p_counterintuitive))
        end
    end
    CSV.write(joinpath(OUT, "counterintuitive_subject_bins.csv"), participant)
    CSV.write(joinpath(OUT, "counterintuitive_summary.csv"), combined)
    println("Wrote $(nrow(participant)) participant-bin rows for Figure 8 and S7")
end

function copy_checked(source_name, target_name)
    source = joinpath(
        SOURCE, "fast_code", "validation", "results_model_evidence_20260808", source_name,
    )
    isfile(source) || error("Missing validated figure input: $source")
    table = CSV.read(source, DataFrame)
    CSV.write(joinpath(OUT, target_name), table)
    println("Wrote $(nrow(table)) rows to $target_name")
end

export_move_summary_bins()
export_counterintuitive_bins()
copy_checked("paper_excess_length_scatter.csv", "excess_length_by_puzzle.csv")
