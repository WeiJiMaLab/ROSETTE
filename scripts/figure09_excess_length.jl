# Figure 9: human versus ROSETTE excess solution length.
#
# The figure is written as SVG directly from Julia. This preserves the compact
# vector styling of the original Julia/Plots scatter while avoiding a platform-
# specific GR/Qt runtime dependency during manuscript builds.

using CSV
using DataFrames
using Printf

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DATA_PATH = joinpath(ROOT, "data", "published", "excess_length_by_puzzle.csv")
const OUTPUT_DIR = isempty(ARGS) ? joinpath(ROOT, "paper", "generated") : abspath(ARGS[1])

const WIDTH = 446.0
const HEIGHT = 204.0
const PLOT_TOP = 18.0
const PLOT_BOTTOM = 165.0
const PLOT_WIDTH = 147.0

fmt(x) = @sprintf("%.3f", x)

function log_position(value, limits, start, extent; reverse=false)
    lo, hi = log10.(limits)
    proportion = (log10(value) - lo) / (hi - lo)
    reverse ? start + extent * (1 - proportion) : start + extent * proportion
end

function line(io, x1, y1, x2, y2; stroke="black", width=0.8, dash=nothing)
    dash_attr = isnothing(dash) ? "" : " stroke-dasharray=\"$(dash)\""
    println(io, "<line x1=\"$(fmt(x1))\" y1=\"$(fmt(y1))\" x2=\"$(fmt(x2))\" y2=\"$(fmt(y2))\" stroke=\"$stroke\" stroke-width=\"$width\"$dash_attr/>")
end

function text_element(io, x, y, value; size=6, anchor="middle", transform=nothing)
    transform_attr = isnothing(transform) ? "" : " transform=\"$transform\""
    println(io, "<text x=\"$(fmt(x))\" y=\"$(fmt(y))\" font-family=\"Helvetica, Arial, sans-serif\" font-size=\"$size\" text-anchor=\"$anchor\"$transform_attr>$value</text>")
end

function draw_panel(io, rows::DataFrame; task::String, xleft::Float64)
    tol = task == "tol"
    limits = tol ? (0.08, 14.0) : (0.8, 600.0)
    ticks = tol ? [(0.1, "0.1"), (1.0, "1"), (10.0, "10")] :
                  [(1.0, "1"), (10.0, "10"), (100.0, "100")]
    identity_limits = tol ? (0.08, 14.0) : (1.0, 150.0)
    title = tol ? "Tower of London" : "Rush Hour"
    panel_center = xleft + PLOT_WIDTH / 2
    plot_height = PLOT_BOTTOM - PLOT_TOP

    xmap(value) = log_position(value, limits, xleft, PLOT_WIDTH)
    ymap(value) = log_position(value, limits, PLOT_TOP, plot_height; reverse=true)

    # GR rendered the original nominal 10/8-pt guide/tick settings at roughly
    # 14/12 pt in the PDF. Keep the title at that rendered size, but reduce the
    # guides and ticks to 11/9 pt for this denser excess-length plot.
    text_element(io, panel_center, 15.0, title; size=14)
    line(io, xleft, PLOT_TOP, xleft, PLOT_BOTTOM; width=0.8)
    line(io, xleft, PLOT_BOTTOM, xleft + PLOT_WIDTH, PLOT_BOTTOM; width=0.8)

    for (value, label) in ticks
        x = xmap(value)
        y = ymap(value)
        line(io, x, PLOT_BOTTOM, x, PLOT_BOTTOM + 3.2; width=0.8)
        line(io, xleft - 3.2, y, xleft, y; width=0.8)
        text_element(io, x, PLOT_BOTTOM + 13.0, label; size=9)
        text_element(io, xleft - 5.3, y + 3.0, label; size=9, anchor="end")
    end

    text_element(io, panel_center, 199.0, "Human excess length"; size=11)
    ylabel_x = xleft - 34.0
    ylabel_y = (PLOT_TOP + PLOT_BOTTOM) / 2
    text_element(
        io, ylabel_x, ylabel_y, "ROSETTE excess length";
        size=11, transform="rotate(-90 $(fmt(ylabel_x)) $(fmt(ylabel_y)))",
    )

    # Identity line.
    line(
        io,
        xmap(identity_limits[1]), ymap(identity_limits[1]),
        xmap(identity_limits[2]), ymap(identity_limits[2]);
        stroke="#888888", width=1.0, dash="7 5",
    )

    # One-standard-error bars and puzzle means, matching the current analysis.
    for row in eachrow(rows)
        human_lo = max(row.human_mean - row.human_sem, limits[1])
        human_hi = min(row.human_mean + row.human_sem, limits[2])
        model_lo = max(row.model_mean - row.model_sem, limits[1])
        model_hi = min(row.model_mean + row.model_sem, limits[2])
        x = xmap(row.human_mean)
        y = ymap(row.model_mean)
        xlo, xhi = xmap(human_lo), xmap(human_hi)
        ylo, yhi = ymap(model_lo), ymap(model_hi)
        cap = 1.35

        line(io, xlo, y, xhi, y; width=0.7)
        line(io, xlo, y - cap, xlo, y + cap; width=0.7)
        line(io, xhi, y - cap, xhi, y + cap; width=0.7)
        line(io, x, ylo, x, yhi; width=0.7)
        line(io, x - cap, ylo, x + cap, ylo; width=0.7)
        line(io, x - cap, yhi, x + cap, yhi; width=0.7)
        println(io, "<circle cx=\"$(fmt(x))\" cy=\"$(fmt(y))\" r=\"2.05\" fill=\"black\" stroke=\"white\" stroke-width=\"0.55\"/>")
    end
end

function make_svg(points::DataFrame)
    io = IOBuffer()
    println(io, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"446pt\" height=\"204pt\" viewBox=\"0 0 446 204\">")
    println(io, "<rect width=\"446\" height=\"204\" fill=\"white\"/>")
    println(io, "<text x=\"0\" y=\"14\" font-family=\"Helvetica, Arial, sans-serif\" font-size=\"17\" font-weight=\"bold\">A</text>")
    println(io, "<text x=\"225\" y=\"14\" font-family=\"Helvetica, Arial, sans-serif\" font-size=\"17\" font-weight=\"bold\">B</text>")
    draw_panel(io, points[points.task .== "tol", :]; task="tol", xleft=55.0)
    draw_panel(io, points[points.task .== "rh", :]; task="rh", xleft=280.0)
    println(io, "</svg>")
    String(take!(io))
end

function main()
    points = CSV.read(DATA_PATH, DataFrame)
    nrow(points[points.task .== "tol", :]) == 25 || error("Expected 25 Tower of London points")
    nrow(points[points.task .== "rh", :]) == 35 || error("Expected 35 Rush Hour points")
    svg = make_svg(points)
    mkpath(OUTPUT_DIR)
    svg_path = joinpath(OUTPUT_DIR, "fig09_excess_length.svg")
    pdf_path = joinpath(OUTPUT_DIR, "fig09_excess_length.pdf")
    open(svg_path, "w") do file
        write(file, svg)
    end
    converter = Sys.which("rsvg-convert")
    isnothing(converter) && error("Install rsvg-convert to render the PDF")
    run(`$converter --format pdf --output $pdf_path $svg_path`)
    println("Wrote Figure 9 to $OUTPUT_DIR")
end

main()
