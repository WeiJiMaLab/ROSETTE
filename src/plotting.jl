function normalize_hist_counts(df, v1, v2, idv, lims)
    df[!, :norm_counts] = zeros(Float64, size(df, 1))
    for a in unique(df[!, v1])
        for b in unique(df[!, v2])
            dummy = df[df[!, v1] .== a .&& df[!, v2] .== b, :].hist_counts
            df[df[!, v1] .== a .&& df[!, v2] .== b, :norm_counts] .= dummy ./ sum(dummy)
            for d in lims
                df_ = df[df[!, v1] .== a .&& df[!, v2] .== b, :]
                if d ∉ df_[!, idv]
                    push!(df, [a, b, d, 0, 0])
                end
            end
        end
    end
    return df
end

function figA(binned_stats)
    models = ["model"] 
    DVs = ["y_p_in_tree", "y_d_tree", "y_p_undo", "y_p_same_car"]
    IDV = "X_d_goal"
    MM = length(models)
    d = length(DVs)
    l = @layout [grid(1, d); a{0.001h}];
    plot(size=(744, 220), grid=false, layout=l, dpi=300, xflip=true,
        legendfont=font(12, "helvetica"), 
        xtickfont=font(10, "helvetica"), 
        ytickfont=font(10, "helvetica"), 
        titlefont=font(12, "helvetica"), 
        guidefont=font(12, "helvetica"), 
        right_margin=0Plots.mm, top_margin=1Plots.mm, bottom_margin=7Plots.mm, left_margin=5Plots.mm, 
        fontfamily="helvetica", tick_direction=:out, xlim=(0, 16));

    ylabels = ["Prop. sensible" "Depth in tree" "Prop. undos" "Prop. same car"];
    ytickss = [[0.75, 0.8, 0.85, 0.9, 0.95], [2.0, 3.0, 4.0, 5.0], ([0.0, 0.05, 0.1], ["0", "0.05", "0.1"]), ([0.0, 0.05, 0.1, 0.15], ["0", "0.05", "0.1", "0.15"])]
    # ylimss = [(0.75, 0.95), (2, 5), (0, 0.11), (0, 0.15)]
    ylimss = [(0.75, 1.0), (2, 5.5), (0, 0.15), (0, 0.19)]

    for i in 1:d
        for j in 1:MM
            df_data = binned_stats[binned_stats.model .== "data", :]
            df_model = binned_stats[binned_stats.model .== models[j], :]
            sort!(df_data, :bin_number)
            sort!(df_model, :bin_number)
            ylabel = ylabels[i]
            title = ""
            yticks = ytickss[i]
            xticks = [0, 5, 10, 15]
            ylims = ylimss[i]
            sp = i

            plot!(df_data[:, "m_"*IDV], df_data[:, "m_"*DVs[i]], yerr=df_data[:, "sem_"*DVs[i]], sp=sp, markercolor=:black, c=:white, msw=1.4, label=nothing, xflip=true, linewidth=1, markershape=:none, ms=4, ylabel=ylabel, xticks=xticks, yticks=yticks)
            plot!(df_model[:, "m_"*IDV], df_model[:, "m_"*DVs[i]], ribbon=df_model[:, "sem_"*DVs[i]], sp=sp, label=nothing, c=:gray, l=nothing, ylabel=ylabel, title=title, xticks=xticks, yticks=yticks, ylims=ylims)
            plot!(df_model[:, "m_"*IDV], df_model[:, "m_"*DVs[i]], ribbon=df_model[:, "sem_"*DVs[i]], sp=sp, label=nothing, c=:gray, l=nothing, ylabel=ylabel, title=title, xticks=xticks, yticks=yticks, ylims=ylims)
        end
    end
    plot!(xlabel="Distance to goal", showaxis=false, grid=false, sp=d + 1, top_margin=-15Plots.mm, bottom_margin=7Plots.mm)
    display(plot!())
end
# figA(binned_stats)
function figB(df_stats)
    models = ["model"]
    Vs = [:h_d_tree, :h_d_tree_diff]
    lims = [2:11, 1:9]
    MM = length(models)
    d = length(Vs)
    l = @layout [grid(1, d)];
    plot(size=(372, 200), grid=false, layout=l, dpi=300, xflip=false,
        legendfont=font(12, "helvetica"), 
        xtickfont=font(10, "helvetica"), 
        ytickfont=font(10, "helvetica"), 
        titlefont=font(12, "helvetica"), 
        guidefont=font(12, "helvetica"), 
        right_margin=0Plots.mm, top_margin=0Plots.mm, bottom_margin=1Plots.mm, left_margin=0Plots.mm, 
        fontfamily="helvetica", tick_direction=:out);

    xlabels = ["Depth" "D Depth"]
    ytickss = [([0.0, 0.1, 0.2], ["0", "0.1", "0.2"]), ([0.0, 0.2, 0.4], ["0", "0.2", "0.4"])]
    xtickss = [[2, 4, 6, 8, 10], [0, 2, 4, 6, 8]]
    for i in 1:d
        r = Vs[i]
        df_ = df_stats[df_stats.h_d_tree .< 1000, :]
        gdf = groupby(df_, [:subject, :model, r])
        count_df = combine(gdf, r => length => :hist_counts)
    
        count_df_norm = normalize_hist_counts(count_df, "subject", "model", r, lims[i])
    
        diff_gdf = groupby(count_df_norm, [:model, r])
        diff_df = combine(diff_gdf, :norm_counts => (x -> [(mean(x), sem(x))]) => [:hist_mean, :hist_sem])

        df_data = diff_df[diff_df.model .== "data", :]
        xlabel = xlabels[i]
        ylabel = i == 1 ? "Proportion" : "TEST"
        title = ""
        yticks = ytickss[i]
        xticks = xtickss[i]
        bar!(df_data[:, r], df_data[:, :hist_mean], yerr=1.96*df_data[:, :hist_sem], sp=i, c=:white, msw=1.4, label=nothing, linewidth=1, markershape=:none, ms=0, title=title, ylabel=ylabel, xticks=xticks, yticks=yticks, linecolor=:black, markercolor=:black, xlabel=xlabel, left_margin=i==1 ? 0Plots.mm : -4Plots.mm)
        for j in 1:MM
            df_model = diff_df[diff_df.model .== models[j], :]
            sort!(df_model, r)
            plot!(df_model[:, r], df_model[:, :hist_mean], ribbon=1.96*df_model[:, :hist_sem], sp=i, label=nothing, c=:gray, l=nothing)
            plot!(df_model[:, r], df_model[:, :hist_mean], ribbon=1.96*df_model[:, :hist_sem], sp=i, label=nothing, c=:gray, l=nothing)
        end
    end
    display(plot!())
end

function figC(binned_stats)
    models = ["data", "model"]
    DVs = ["m_y_p_worse", "m_y_p_same", "m_y_p_better"]
    IDV = "m_X_d_goal"
    MM = length(models)
    d = length(DVs)
    l = @layout [grid(1, MM); a{0.001h}];#a{0.001h}; 
    plot(size=(372, 200), grid=false, layout=l, dpi=300, xflip=true, link=:both,
        legendfont=font(12, "helvetica"), 
        xtickfont=font(10, "helvetica"), 
        ytickfont=font(10, "helvetica"), 
        titlefont=font(12, "helvetica"), 
        guidefont=font(12, "helvetica"), 
        right_margin=0Plots.mm, top_margin=1Plots.mm, bottom_margin=4Plots.mm, left_margin=0Plots.mm, 
        fontfamily="helvetica", tick_direction=:out, xlim=(0, 15), ylim=(0, 1), yticks=nothing)

    labels = ["Worse   " "Same   " "Better   "]
    #bar!([0 0 0], c=[palette(:default)[1] palette(:default)[2] palette(:default)[3]], labels=labels, legend_columns=length(labels), linewidth=0, sp=1, showaxis=false, grid=false, background_color_legend=nothing, foreground_color_legend=nothing, legend=:top, top_margin=-2Plots.mm);
    titles = ["Participants" "Model" "AND-OR"];
    xlabels = ["" "" ""]
    ylabels = ["Proportion" "" ""]
    yticks = [([0, 0.2, 0.4, 0.6, 0.8, 1], ["0", "0.2", "0.4", "0.6", "0.8", "1"]) for _ in 1:MM]
    xticks = [[0, 5, 10, 15] for _ in 1:MM]
    for i in 1:MM
        df_ = binned_stats[binned_stats.model .== models[i], :]
        areaplot!(df_[:, IDV] .- 1, [df_[:, DVs[1]] + df_[:, DVs[2]] + df_[:, DVs[3]], df_[:, DVs[2]] + df_[:, DVs[3]], df_[:, DVs[3]]], sp=i+0, xflip=true, label=nothing, xlabel=xlabels[i], ylabel=ylabels[i], title=titles[i], yticks=yticks[i], xticks=xticks[i], left_margin=i==1 ? 2Plots.mm : -1Plots.mm)    
    end
    plot!(xlabel="Distance to goal", showaxis=false, grid=false, sp=MM+1, top_margin=-12Plots.mm)
    display(plot!())
end

function draw_board(s::Int, game::RushHour; sp=nothing)
    board_arr = draw_state(s, game)
    cmap = [
        RGB(([255, 255, 255]./255)...),
        RGB(([202, 76, 60]./255)...),
        RGB(([185, 156, 105]./255)...), 
        RGB(([102, 152, 80]./255)...), 
        RGB(([80, 173, 202]./255)...), 
        RGB(([219, 130, 57]./255)...), 
        RGB(([81, 51, 154]./255)...), 
        RGB(([147, 190, 103]./255)...), 
        RGB(([126, 74, 51]./255)...), 
        RGB(([124, 124, 124]./255)...),
        RGB(([242, 217, 80]./255)...),
        RGB(([51, 51, 51]./255)...),
        RGB(([158, 156, 79]./255)...),
        RGB(([48, 83, 141]./255)...),
        RGB(([211, 128, 149]./255)...),
        RGB(([84, 255, 198]./255)...)
        ]
    if sp === nothing
        heatmap(board_arr, c = cmap[1:maximum(board_arr)+1], legend = false, yflip = true, xmirror=true, framestyle = :box, size=(200, 200))
        vline!(1.5:5.5, c=:black, linewidth=0.2)
        hline!(1.5:5.5, c=:black, linewidth=0.2)
    else
        heatmap!(board_arr, sp=sp, c = cmap[1:maximum(board_arr)+1], legend = false, yflip = true, xmirror=true, framestyle = :box)
        vline!(1.5:5.5, sp=sp, c=:black, linewidth=0.2)
        hline!(1.5:5.5, sp=sp, c=:black, linewidth=0.2)
    end
    for c in sort(unique(board_arr))[2:end]
        idxs = findall(x->x==c, board_arr)
        l = length(idxs)
        s = sum(idxs)
        annotate!(s[2]/l, s[1]/l, text(c, :white, 16))
    end
    plot!(xticks=[], yticks=[])
end

function draw_policy(p::AbstractVector, moves::Vector{Tuple{Int, Int}}, game::RushHour)
    cmap = [
        RGB(([255, 255, 255]./255)...),
        RGB(([202, 76, 60]./255)...),
        RGB(([185, 156, 105]./255)...),
        RGB(([102, 152, 80]./255)...),
        RGB(([80, 173, 202]./255)...),
        RGB(([219, 130, 57]./255)...),
        RGB(([81, 51, 154]./255)...),
        RGB(([147, 190, 103]./255)...),
        RGB(([126, 74, 51]./255)...),
        RGB(([124, 124, 124]./255)...),
        RGB(([242, 217, 80]./255)...),
        RGB(([51, 51, 51]./255)...),
        RGB(([158, 156, 79]./255)...),
        RGB(([48, 83, 141]./255)...),
        RGB(([211, 128, 149]./255)...),
        RGB(([84, 255, 198]./255)...)
    ]

    n = length(p)
    xs = 1:n
    max_p = maximum(p)

    ylim_top = max_p * 1.1
    plt1 = bar(xs, p,
              legend = false,
              color = :lightgray, linecolor = :black,
              xticks = false, framestyle = :box,
              ylabel = "Probability",
              ylim = (0, ylim_top),
              grid=false)
    plt2 = plot(legend=false, ylim=(-1.5, 1.5), axis=([], false))

    base_width_unit  = 0.2   # width for a length-1 horizontal car
    base_height_unit = 0.5   # height for a length-1 vertical car (in probability units)

    # Y position baseline for car icons (center of icon will be around here)
    y_icon_center = max_p * 0.25  # push icons up a bit from absolute bottom
    for (i, (car_id, move_amt)) in enumerate(moves)
        @assert 1 ≤ car_id ≤ length(game.s_fixed)
        car = game.s_fixed[car_id]
        color = cmap[car_id + 1]   # +1 because index 1 reserved for background in your board

        # Determine icon size in data coords depending on orientation & length
        if car.dir == :x
            # horizontal car -> wider rectangle, small height
            w = base_width_unit * car.len
            h = base_height_unit
        elseif car.dir == :y
            # vertical car -> narrow width, taller rectangle
            w = base_width_unit
            h = base_height_unit * car.len
        end

        # Rectangle center is (xs[i], y_icon_center). build polygon via Shape
        x0 = xs[i] - w/2
        x1 = xs[i] + w/2
        y0 = y_icon_center - h/2
        y1 = y_icon_center + h/2

        # use Shape for a filled polygon (robust across backends)
        poly = Shape([x0, x1, x1, x0], [y0, y0, y1, y1])
        plot!(plt2, poly, fillcolor = color, linecolor = :black,)

        # Car id text centered
        annotate!(plt2, xs[i], y_icon_center, text("$(car_id)", 9, :white, :bold, halign = :center, valign = :center),)

        # Build arrow string (repeated symbol for magnitude)
        arrow_symbol = if car.dir == :x
            move_amt > 0 ? "→" : "←"
        else
            move_amt > 0 ? "↓\n" : "↑\n"
        end
        arrows = repeat(arrow_symbol, abs(move_amt))
        if car.dir == :y
            arrows = chop(arrows)
        end

        # Place arrows: horizontal cars to left/right; vertical above/below
        if car.dir == :x
            # shift horizontally by half-width + margin
            dx = (w/2) + 0.12 * abs(move_amt)
            x_arrow = xs[i] + sign(move_amt) * dx
            annotate!(plt2, x_arrow, y_icon_center, text(arrows, 9, :black, halign = :center, valign = :center),)
        else
            # vertical cars: arrows above (move_amt>0) or below (move_amt<0)
            dy = (h/2) + 0.3 * abs(move_amt)
            y_arrow = y_icon_center + -sign(move_amt) * dy
            annotate!(plt2, xs[i], y_arrow, text(arrows, 9, :black, halign = :center, valign = :center),)
        end
    end

    plt = plot(plt1, plt2, size = (500, 300), layout=@layout([a{0.75h}; b{0.25h}]), link=:x, bottom_margin=[0Plots.mm 0Plots.mm], top_margin=[0Plots.mm 0Plots.mm])
    return plt
end


function draw_policies(p::AbstractVector, moves::Vector{Tuple{Int, Int}}, game::RushHour, insets::Vector{String})
    cmap = [
        RGB(([255, 255, 255]./255)...),
        RGB(([202, 76, 60]./255)...),
        RGB(([185, 156, 105]./255)...),
        RGB(([102, 152, 80]./255)...),
        RGB(([80, 173, 202]./255)...),
        RGB(([219, 130, 57]./255)...),
        RGB(([81, 51, 154]./255)...),
        RGB(([147, 190, 103]./255)...),
        RGB(([126, 74, 51]./255)...),
        RGB(([124, 124, 124]./255)...),
        RGB(([242, 217, 80]./255)...),
        RGB(([51, 51, 51]./255)...),
        RGB(([158, 156, 79]./255)...),
        RGB(([48, 83, 141]./255)...),
        RGB(([211, 128, 149]./255)...),
        RGB(([84, 255, 198]./255)...)
    ]

    n = length(p[1])
    xs = 1:n
    max_p = maximum(maximum.(p))

    ylim_top = max_p * 1.05
    hist_plts = []
    for (n, pp) in enumerate(p)
        hist_plt = bar(xs, pp,
                # label = insets[n],
                # foreground_color_legend=nothing, background_color_legend=nothing,
                legend=false,
                color = :lightgray, linecolor = :black,
                xticks = false, framestyle = :box,
                ylabel = "Probability",
                ylim = (0, ylim_top),
                grid=false)
        annotate!(hist_plt, 2, ylim_top*0.9, insets[n])
        push!(hist_plts, hist_plt)
    end


    plt2 = plot(legend=false, ylim=(-1.5, 1.5), axis=([], false))

    base_width_unit  = 0.2   # width for a length-1 horizontal car
    base_height_unit = 0.5   # height for a length-1 vertical car (in probability units)

    # Y position baseline for car icons (center of icon will be around here)
    y_icon_center = max_p * 0.25  # push icons up a bit from absolute bottom
    for (i, (car_id, move_amt)) in enumerate(moves)
        @assert 1 ≤ car_id ≤ length(game.s_fixed)
        car = game.s_fixed[car_id]
        color = cmap[car_id + 1]   # +1 because index 1 reserved for background in your board

        # Determine icon size in data coords depending on orientation & length
        if car.dir == :x
            # horizontal car -> wider rectangle, small height
            w = base_width_unit * car.len
            h = base_height_unit
        elseif car.dir == :y
            # vertical car -> narrow width, taller rectangle
            w = base_width_unit
            h = base_height_unit * car.len
        end

        # Rectangle center is (xs[i], y_icon_center). build polygon via Shape
        x0 = xs[i] - w/2
        x1 = xs[i] + w/2
        y0 = y_icon_center - h/2
        y1 = y_icon_center + h/2

        # use Shape for a filled polygon (robust across backends)
        poly = Shape([x0, x1, x1, x0], [y0, y0, y1, y1])
        plot!(plt2, poly, fillcolor = color, linecolor = :black,)

        # Car id text centered
        annotate!(plt2, xs[i], y_icon_center, text("$(car_id)", 9, :white, :bold, halign = :center, valign = :center),)

        # Build arrow string (repeated symbol for magnitude)
        arrow_symbol = if car.dir == :x
            move_amt > 0 ? "→" : "←"
        else
            move_amt > 0 ? "↓\n" : "↑\n"
        end
        arrows = repeat(arrow_symbol, abs(move_amt))
        if car.dir == :y
            arrows = chop(arrows)
        end

        # Place arrows: horizontal cars to left/right; vertical above/below
        if car.dir == :x
            # shift horizontally by half-width + margin
            dx = (w/2) + 0.12 * abs(move_amt)
            x_arrow = xs[i] + sign(move_amt) * dx
            annotate!(plt2, x_arrow, y_icon_center, text(arrows, 9, :black, halign = :center, valign = :center),)
        else
            # vertical cars: arrows above (move_amt>0) or below (move_amt<0)
            dy = (h/2) + 0.3 * abs(move_amt)
            y_arrow = y_icon_center + -sign(move_amt) * dy
            annotate!(plt2, xs[i], y_arrow, text(arrows, 9, :black, halign = :center, valign = :center),)
        end
    end
    hb = 1 / (3*length(p) + 1)
    ht = 3 / (3*length(p) + 1)
    hs = [fill(ht, length(p))... , hb]
    println(hs)
    plt = plot(hist_plts..., plt2, size = (500, 100 + 200*length(p)), layout=grid(length(p)+1, 1, heights=hs), link=:x, bottom_margin=[0Plots.mm 0Plots.mm], top_margin=[0Plots.mm 0Plots.mm])
    return plt
end