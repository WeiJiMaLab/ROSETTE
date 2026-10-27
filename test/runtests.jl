using CSV, DataFrames, Statistics, Test

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "src", "ROSETTE.jl"))
using .ROSETTE

@testset "Published data integrity" begin
    data = joinpath(ROOT, "data", "published")
    choices = CSV.read(joinpath(data, "choices.csv"), DataFrame)
    fits = CSV.read(joinpath(data, "fit_nll.csv"), DataFrame)
    summary = CSV.read(joinpath(data, "move_summary_subject_bins.csv"), DataFrame)
    @test nrow(choices) == 140_592
    @test nrow(fits) == 7_154
    @test all(choices.subject .> 0)
    @test all(choices.puzzle .> 0)
    @test all(isfinite, fits.nll)
    @test all(fits.nll .>= 0)
    @test length(unique(choices[choices.task .== "RH", :subject])) == 42
    @test length(unique(choices[choices.task .== "TOL", :subject])) == 469
    @test length(unique(summary[summary.task .== "RH", :subject])) == 42
    @test length(unique(summary[summary.task .== "TOL", :subject])) == 469
    @test eltype(summary.subject) <: Integer
    @test all(combine(groupby(fits, [:task, :subject]), nrow => :count).count .== 14)
    @test nrow(CSV.read(joinpath(data, "excess_length_by_puzzle.csv"), DataFrame)) == 60
    @test nrow(CSV.read(joinpath(data, "puzzle_metadata.csv"), DataFrame)) == 60
end

@testset "Core game and policy" begin
    start_pos, fixed = load_prb("prb1707_7")
    @test length(start_pos) == 9
    rh = RushHour(get_base(5, ROSETTE.RUSH_HOUR_CARS), start_pos, fixed)
    @test !isempty(neighbours(start_state(rh), rh))

    tol = TowerOfLondon(get_base(4, 6), [1,0,0,0,0,0], [0,0,0,0,0,1])
    @test !is_terminal(start_state(tol), tol)
    state_to_idx, idx_to_state, Q, R = get_QR_matrices(tol)
    n_terminal, n_transient = size(R)
    @test n_terminal > 0 && n_transient > 0
    moves, idxs, sizes = ROSETTE.get_tol_heuristics(
        n_transient, idx_to_state, state_to_idx, tol,
    )
    fi = FittingStruct(
        Dict{Tuple{Int64, Tuple{Int64, Int64}}, Float64}[],
        moves, idxs, Q, R, zeros(n_transient, 3), sizes,
    )
    start_index = state_to_idx[start_state(tol)]
    probabilities, targets = get_p_main([1.0, 0.2, 0.5], start_index, (0,0), fi)
    @test length(probabilities) == length(targets)
    @test all(probabilities .>= 0)
    @test sum(probabilities) ≈ 1 atol=2e-6
end
