using CSV
using DataFrames
using LinearAlgebra
using Printf
using Random
using Statistics

const SOURCE_DIR = @__DIR__
const EXPECTED_RAW = Dict("tol" => 0.534159110662197, "rh" => 0.527317522306568)
const EXPECTED_RAW_P = Dict("tol" => 0.005951430690852164, "rh" => 0.001134081533841614)
const PERMUTATIONS = 50_000

# Two-sided Student-t tail via the regularized incomplete beta function.
# This keeps the analysis within the manuscript's Julia project dependencies.
function beta_fraction(a::Float64, b::Float64, x::Float64)
    tiny = floatmin(Float64) / eps(Float64)
    qab, qap, qam = a + b, a + 1, a - 1
    c = 1.0
    d = 1 - qab * x / qap
    abs(d) < tiny && (d = tiny)
    d = inv(d)
    answer = d
    for m in 1:200
        m2 = 2m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        abs(d) < tiny && (d = tiny)
        c = 1 + aa / c
        abs(c) < tiny && (c = tiny)
        d = inv(d)
        answer *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        abs(d) < tiny && (d = tiny)
        c = 1 + aa / c
        abs(c) < tiny && (c = tiny)
        d = inv(d)
        delta = d * c
        answer *= delta
        abs(delta - 1) <= 3e-14 && return answer
    end
    error("Incomplete beta fraction did not converge")
end

function regularized_beta(x::Float64, a::Float64, b::Float64)
    x == 0 && return 0.0
    x == 1 && return 1.0
    0 < x < 1 || throw(ArgumentError("beta input outside (0, 1)"))
    front = exp(
        ccall(:lgamma, Cdouble, (Cdouble,), a + b) -
        ccall(:lgamma, Cdouble, (Cdouble,), a) -
        ccall(:lgamma, Cdouble, (Cdouble,), b) +
        a * log(x) + b * log1p(-x)
    )
    if x < (a + 1) / (a + b + 2)
        return front * beta_fraction(a, b, x) / a
    end
    return 1 - front * beta_fraction(b, a, 1 - x) / b
end

function residualize(y::AbstractVector, z::AbstractVector)
    X = hcat(ones(length(z)), z)
    return y - X * (X \ y)
end

function partial_r(human::AbstractVector, model::AbstractVector, shortest::AbstractVector)
    return cor(residualize(human, shortest), residualize(model, shortest))
end

function permutation_p(human_residual, model_residual, observed_r, seed)
    rng = Xoshiro(seed)
    more_extreme = 0
    for _ in 1:PERMUTATIONS
        r = cor(human_residual, model_residual[randperm(rng, length(model_residual))])
        more_extreme += abs(r) >= abs(observed_r)
    end
    return (more_extreme + 1) / (PERMUTATIONS + 1)
end

function analyze_task(rows::DataFrame, seed::Int)
    human = Float64.(rows.human_mean)
    model = Float64.(rows.model_mean)
    shortest = Float64.(rows.optimal_length)
    n = length(human)
    h_residual = residualize(human, shortest)
    m_residual = residualize(model, shortest)
    raw_r = cor(human, model)
    raw_p = regularized_beta(1 - raw_r^2, (n - 2) / 2, 0.5)
    adjusted_r = cor(h_residual, m_residual)
    df = n - 3 # one covariate (L*) plus an intercept
    adjusted_p = regularized_beta(1 - adjusted_r^2, df / 2, 0.5)
    # Approximate Fisher-z interval for a partial correlation with one covariate.
    ci = tanh.(atanh(adjusted_r) .+ [-1, 1] .* (1.959963984540054 / sqrt(n - 4)))
    perm_p = permutation_p(h_residual, m_residual, adjusted_r, seed)
    leave_one_out = [begin
        keep = collect(1:n) .!= i
        partial_r(human[keep], model[keep], shortest[keep])
    end for i in 1:n]
    reduced_sse = sum(abs2, h_residual)
    full_design = hcat(ones(n), shortest, model)
    full_coefficients = full_design \ human
    full_residual = human - full_design * full_coefficients
    full_sse = sum(abs2, full_residual)
    coefficient_se = sqrt((full_sse / df) * inv(full_design' * full_design)[3, 3])
    coefficient_t = full_coefficients[3] / coefficient_se
    @assert isapprox(coefficient_t^2, adjusted_r^2 * df / (1 - adjusted_r^2);
                     rtol=1e-9)
    total_sse = sum(abs2, human .- mean(human))
    delta_r2 = (reduced_sse - full_sse) / total_sse
    result = (
        task=rows.task[1], puzzles=n, raw_r=raw_r, raw_p=raw_p,
        human_shortest_r=cor(human, shortest),
        model_shortest_r=cor(model, shortest),
        human_residual_vs_model_r=cor(h_residual, model),
        partial_r=adjusted_r, partial_df=df, partial_p=adjusted_p,
        fisher_ci_low=ci[1], fisher_ci_high=ci[2], permutation_p=perm_p,
        delta_r2=delta_r2,
        leave_one_out_min=minimum(leave_one_out),
        leave_one_out_max=maximum(leave_one_out),
    )
    detail = DataFrame(
        task=rows.task, puzzle=rows.puzzle, optimal_length=rows.optimal_length,
        human_excess=human, model_excess=model,
        human_n=rows.human_n, model_n=rows.model_n,
        human_residual=h_residual, model_residual=m_residual,
    )
    return result, detail
end

function main()
    points = CSV.read(joinpath(SOURCE_DIR, "paper_excess_length_scatter.csv"), DataFrame)
    metadata = CSV.read(joinpath(SOURCE_DIR, "puzzle_metadata.csv"), DataFrame)
    @assert nrow(points) == nrow(metadata) == 60
    @assert nrow(unique(points, [:task, :puzzle])) == 60
    @assert nrow(unique(metadata, [:task, :puzzle])) == 60
    rows = innerjoin(points, select(metadata, :task, :puzzle, :optimal_length);
                     on=[:task, :puzzle])
    @assert nrow(rows) == 60
    @assert all(rows.human_mean .>= 0) && all(rows.model_mean .>= 0)
    sort!(rows, [:task, :puzzle])

    summary = DataFrame()
    details = DataFrame()
    for (task, expected_n, seed) in (("tol", 25, 20261023), ("rh", 35, 20261024))
        subset = rows[rows.task .== task, :]
        @assert nrow(subset) == expected_n
        result, detail = analyze_task(subset, seed)
        @assert isapprox(result.raw_r, EXPECTED_RAW[task]; atol=1e-10)
        @assert isapprox(result.raw_p, EXPECTED_RAW_P[task]; atol=1e-10)
        @assert isapprox(result.delta_r2,
                         result.partial_r^2 * (1 - result.human_shortest_r^2);
                         atol=1e-10)
        push!(summary, result; cols=:union)
        append!(details, detail; cols=:union)
        @printf("%s n=%d raw r=%.6f, human-only r=%.6f, partial r(%d)=%.6f, p=%.6f, 95%% CI [%.6f, %.6f], permutation p=%.6f, delta R²=%.6f, LOO [%.6f, %.6f]\n",
                uppercase(task), result.puzzles, result.raw_r,
                result.human_residual_vs_model_r, result.partial_df,
                result.partial_r, result.partial_p,
                result.fisher_ci_low, result.fisher_ci_high, result.permutation_p,
                result.delta_r2, result.leave_one_out_min,
                result.leave_one_out_max)
    end
    CSV.write(joinpath(@__DIR__, "figure9_shortest_path_summary.csv"), summary)
    CSV.write(joinpath(@__DIR__, "figure9_shortest_path_puzzles.csv"), details)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
