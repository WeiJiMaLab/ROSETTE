function normalize_hist_counts(df, v1, v2, idv, lims)
    counts = zeros(length(lims) - 1)
    df = df[(lims[1] .<= df[!, idv]) .& (df[!, idv] .<= lims[end]), :]
    for i in 1:(length(lims) - 1)
        counts[i] = mean(df[(lims[i] .<= df[!, idv]) .& (df[!, idv] .< lims[i + 1]), v2])
    end
    counts ./= sum(counts)
    return counts
end

function figA(binned_stats)
    y = binned_stats[binned_stats.model .== "data", :m_y_p_better]
    plot(1:length(y), y, label="Better", xlabel="Distance to goal bin", ylabel="Probability", grid=false)
    plot!(1:length(y), binned_stats[binned_stats.model .== "data", :m_y_p_same], label="Same")
    plot!(1:length(y), binned_stats[binned_stats.model .== "data", :m_y_p_worse], label="Worse")
end

function figB(df_stats)
    grouped = combine(groupby(df_stats, :model), :y_p_full => mean => :mean_probability)
    bar(grouped.model, grouped.mean_probability, label=nothing, xrotation=45, ylabel="Mean probability", grid=false)
end

function figC(binned_stats)
    models = unique(binned_stats.model)
    plot(grid=false, xlabel="Distance to goal bin", ylabel="Mean p(full)")
    for model in models
        rows = binned_stats[binned_stats.model .== model, :]
        plot!(rows.bin_number, rows.m_y_p_full, yerror=rows.sem_y_p_full, label=model)
    end
    plot!()
end

function create_move_icon(move, s)
    return "$(move[1]):$(move[2])"
end

function draw_board(s::Int, game::RushHour; sp=nothing)
    arr = draw_state(s, game)
    heatmap!(arr; aspect_ratio=:equal, colorbar=false, yflip=true, sp=sp, label=nothing)
end

function draw_policy(p::AbstractVector, moves::Vector{Tuple{Int, Int}}, game::RushHour)
    bar(string.(moves), p, label=nothing, ylabel="Probability", xrotation=45, grid=false)
end

function draw_policies(p::AbstractVector, moves::Vector{Tuple{Int, Int}}, game::RushHour, insets::Vector{String})
    plt = draw_policy(p, moves, game)
    annotate!(plt, [(i, p[i], text(get(insets, i, ""), 6)) for i in eachindex(p)])
    return plt
end

function nll_lesion_plot(nlls, model_details, N_subj; tol=false, iters=100, sp=1, title="")
    names = collect(keys(nlls))
    display_names = [get(model_details, name, (name, 0))[1] for name in names]
    sampled_nlls = Dict(name => Float64[] for name in names)
    L = length(nlls["full"])

    for _ in ProgressBar(1:iters)
        sampled_idxs = rand(1:L, L)
        bic_full = model_details["full"][2] * N_subj * log(L) + 2 * sum(nlls["full"][sampled_idxs])
        for name in names
            bic_model = model_details[name][2] * N_subj * log(L) + 2 * sum(nlls[name][sampled_idxs])
            push!(sampled_nlls[name], (bic_model - bic_full) / 1000)
        end
    end

    delta = [mean(sampled_nlls[name]) for name in names]
    err = [1.96 * std(sampled_nlls[name]) for name in names]
    order = sortperm(delta; rev=true)
    bar!(1:length(names), delta[order]; yerror=err[order], sp=sp, title=title, ylabel="D BIC (x1,000)", label=nothing, permute=(:y, :x), xticks=(1:length(names), display_names[order]))
end
