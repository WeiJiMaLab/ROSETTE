using Optim
using Test

include(joinpath(@__DIR__, "..", "cleaned_code", "ROSETTE.jl"))
include(joinpath(@__DIR__, "julia", "FastROSETTE.jl"))

const Clean = ROSETTE
const Fast = FastROSETTE

function rh_fitting_info(problem::String)
    start_pos, fixed = Clean.load_prb(problem)
    game = Clean.RushHour(Clean.get_base(5, Clean.RUSH_HOUR_CARS), start_pos, fixed)
    state_to_idx, idx_to_state, Q, R = Clean.get_QR_matrices(game)
    _, n_transient = size(R)
    all_dicts, all_moves, all_idxs, all_sizes = Clean.get_and_or_dicts(n_transient, idx_to_state, state_to_idx, game)
    fi = Clean.FittingStruct(all_dicts, all_moves, all_idxs, Q, R, zeros(length(all_sizes), 3), all_sizes)
    return fi, state_to_idx[Clean.start_state(game)]
end

function tol_fitting_info()
    initial = [1, 0, 0, 0, 0, 0]
    goal = [0, 0, 0, 0, 0, 1]
    game = Clean.TowerOfLondon(Clean.get_base(4, 6), initial, goal)
    state_to_idx, idx_to_state, Q, R = Clean.get_QR_matrices(game)
    _, n_transient = size(R)
    all_moves, all_idxs, all_sizes = Clean.get_tol_heuristics(n_transient, idx_to_state, state_to_idx, game)
    fi = Clean.FittingStruct(Dict{Tuple{Int64, Tuple{Int64, Int64}}, Float64}[], all_moves, all_idxs, Q, R, zeros(n_transient, 3), all_sizes)
    return fi, state_to_idx[Clean.start_state(game)]
end

function fixture_game(fi, start_state; max_steps=6)
    n_transient = size(fi.R, 2)
    states = Int[start_state]
    actions = Tuple{Int, Int}[]
    state = start_state

    for _ in 1:max_steps
        idxs = fi.all_idxs[state]
        next_pos = findfirst(<=(n_transient), idxs)
        next_pos === nothing && (next_pos = 1)
        push!(actions, fi.all_moves[state][next_pos])
        next_state = idxs[next_pos]
        push!(states, next_state)
        next_state > n_transient && break
        state = next_state
    end

    return [(states[i], i <= length(actions) ? actions[i] : (0, 0)) for i in eachindex(states)]
end

function repeated_state_actions(game, repeats)
    return [[copy(game) for _ in 1:repeats]]
end

function minimum_elapsed(f; reps=3)
    f()
    times = [@elapsed f() for _ in 1:reps]
    return minimum(times)
end

@testset "Fast unified main-model NLL" begin
    rh_fi, rh_start = rh_fitting_info("prb1707_7")
    rh_game = fixture_game(rh_fi, rh_start; max_steps=6)
    rh_state_actions = repeated_state_actions(rh_game, 12)
    rh_fast_infos = Fast.prepare_fitting_infos([rh_fi])
    rh_fast_data = Fast.prepare_dataset(rh_state_actions)
    rh_params = [-log10(0.2), -log10(0.1), -log10(0.5)]

    clean_rh = Clean.get_nll_main(rh_params, rh_state_actions, [deepcopy(rh_fi)])
    fast_rh = Fast.nll_main(rh_params, rh_fast_data, rh_fast_infos)
    @test fast_rh ≈ clean_rh atol=1e-8

    tol_fi, tol_start = tol_fitting_info()
    tol_game = fixture_game(tol_fi, tol_start; max_steps=4)
    tol_state_actions = repeated_state_actions(tol_game, 8)
    tol_fast_infos = Fast.prepare_fitting_infos([tol_fi])
    tol_fast_data = Fast.prepare_dataset(tol_state_actions)
    tol_params = [1.0, -log10(0.2), -log10(0.5)]

    clean_tol = Clean.get_nll_main(tol_params, tol_state_actions, [deepcopy(tol_fi)])
    fast_tol = Fast.nll_main(tol_params, tol_fast_data, tol_fast_infos)
    @test fast_tol ≈ clean_tol atol=1e-8
end

@testset "Fast fitting speed smoke test" begin
    rh_fi, rh_start = rh_fitting_info("prb1707_7")
    rh_game = fixture_game(rh_fi, rh_start; max_steps=6)
    rh_state_actions = repeated_state_actions(rh_game, 20)
    rh_fast_infos = Fast.prepare_fitting_infos([rh_fi])
    rh_fast_data = Fast.prepare_dataset(rh_state_actions)
    params = [-log10(0.2), -log10(0.1), -log10(0.5)]

    legacy_infos = [deepcopy(rh_fi)]
    legacy_time = minimum_elapsed(() -> Clean.get_nll_main(params, rh_state_actions, legacy_infos); reps=3)
    fast_time = minimum_elapsed(() -> Fast.nll_main(params, rh_fast_data, rh_fast_infos); reps=5)
    println("Legacy cleaned NLL time: $(round(legacy_time, sigdigits=4)) seconds")
    println("Fast unified NLL time:   $(round(fast_time, sigdigits=4)) seconds")
    println("Observed NLL speedup:    $(round(legacy_time / fast_time, sigdigits=4))x")
    @test fast_time < legacy_time

    initial = [1.0, 1.0, 1.0]
    initial_nll = Fast.nll_main(initial, rh_fast_data, rh_fast_infos)
    fit_time = @elapsed result = Fast.fit_main(initial, rh_fast_data, rh_fast_infos; iterations=8)
    println("Fast short fit time:     $(round(fit_time, sigdigits=4)) seconds")
    println("Initial NLL:             $(round(initial_nll, sigdigits=6))")
    println("Optimized NLL:           $(round(Optim.minimum(result), sigdigits=6))")

    @test isfinite(Optim.minimum(result))
    @test Optim.minimum(result) <= initial_nll + 1e-8
    @test length(Optim.minimizer(result)) == 3
end
