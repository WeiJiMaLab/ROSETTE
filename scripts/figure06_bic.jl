using CSV
using DataFrames
using Random
using Statistics

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DATA_PATH = joinpath(ROOT, "data", "published", "fit_nll.csv")
const OUTPUT_DIR = isempty(ARGS) ? joinpath(ROOT, "paper", "generated") : abspath(ARGS[1])

function load_nlls(; tol::Bool)
    task = tol ? "TOL" : "RH"
    n_subjects = tol ? 469 : 42
    heuristic = tol ? "beta" : "gamma_ao"
    names = [
        "full", "lesion-$heuristic", "lesion-gamma_r", "lesion-gamma_p",
        "only-$heuristic", "only-gamma_r", "only-gamma_p", "chance",
        "bfs-full", "dfs-full", "bestfirst-full", "astar-full", "iddfs-full", "idastar-full",
    ]
    output = Dict(name => zeros(n_subjects) for name in names)
    counts = Dict(name => zeros(Int, n_subjects) for name in names)
    data = CSV.read(DATA_PATH, DataFrame)
    for row in eachrow(data[data.task .== task, :])
        haskey(output, row.model) || error("Unexpected model $(row.model)")
        1 <= row.subject <= n_subjects || error("Unexpected subject $(row.subject)")
        output[row.model][row.subject] = row.nll
        counts[row.model][row.subject] += 1
    end
    all(all(counts[name] .== 1) for name in names) || error("Missing or duplicated fitted NLL")
    return output
end

function task_statistics(; tol::Bool, seed::Int, iterations::Int=1_000)
    n_subjects = tol ? 469 : 42
    heuristic = tol ? "beta" : "gamma_ao"
    nlls = load_nlls(; tol)
    parameter_counts = Dict(
        "full" => 3, "chance" => 0,
        "lesion-$heuristic" => 2, "lesion-gamma_r" => 2, "lesion-gamma_p" => 2,
        "only-$heuristic" => 1, "only-gamma_r" => 1, "only-gamma_p" => 1,
        "bfs-full" => 3, "dfs-full" => 3, "bestfirst-full" => 3, "astar-full" => 3,
        "iddfs-full" => 4, "idastar-full" => 4,
    )
    display_names = Dict(
        "full" => "Full model", "chance" => "Chance",
        "lesion-beta" => "Lesion β", "lesion-gamma_ao" => "Lesion γₐₒ",
        "lesion-gamma_r" => "Lesion γᵣ", "lesion-gamma_p" => "Lesion γₚ",
        "only-beta" => "Only β", "only-gamma_ao" => "Only γₐₒ",
        "only-gamma_r" => "Only γᵣ", "only-gamma_p" => "Only γₚ",
        "bfs-full" => "BFS", "dfs-full" => "DFS", "bestfirst-full" => "Best-First",
        "astar-full" => "A*", "iddfs-full" => "IDDFS", "idastar-full" => "IDA*",
    )
    names = collect(keys(nlls))
    draws = Dict(name => Vector{Float64}(undef, iterations) for name in names)
    rng = Xoshiro(seed)
    for iteration in 1:iterations
        indices = rand(rng, 1:n_subjects, n_subjects)
        bic_full = parameter_counts["full"] * n_subjects * log(n_subjects) +
                   2 * sum(nlls["full"][indices])
        for name in names
            bic = parameter_counts[name] * n_subjects * log(n_subjects) +
                  2 * sum(nlls[name][indices])
            draws[name][iteration] = (bic - bic_full) / 1_000
        end
    end
    means = Dict(name => mean(draws[name]) for name in names)
    errors = Dict(name => 1.96 * std(draws[name]) for name in names)
    ordered = sort(names; by=name -> means[name])
    return [(name=name, label=display_names[name], mean=means[name], error=errors[name]) for name in ordered]
end

xml_escape(text) = replace(text, "&" => "&amp;", "<" => "&lt;", ">" => "&gt;")

function svg_label(text)
    escaped = xml_escape(text)
    escaped = replace(escaped, "γₚ" => "γ<tspan baseline-shift=\"sub\" font-size=\"11\">p</tspan>")
    escaped = replace(escaped, "γᵣ" => "γ<tspan baseline-shift=\"sub\" font-size=\"11\">r</tspan>")
    return replace(escaped, "γₐₒ" => "γ<tspan baseline-shift=\"sub\" font-size=\"11\">ao</tspan>")
end

function draw_panel(io, rows; x0, x1, title, panel, ticks, xmax)
    top, bottom = 53.0, 430.0
    step = (bottom - top) / length(rows)
    bar_height = step * 0.76
    scale(value) = x0 + clamp(value, 0, xmax) / xmax * (x1 - x0)

    println(io, "<text x=\"$((x0+x1)/2)\" y=\"28\" text-anchor=\"middle\" class=\"title\">$title</text>")
    println(io, "<text x=\"$(x0-104)\" y=\"28\" class=\"panel\">$panel</text>")
    println(io, "<line x1=\"$x0\" y1=\"$top\" x2=\"$x0\" y2=\"$bottom\" class=\"axis\"/>")

    for (index, row) in enumerate(rows)
        y = top + (index - 0.5) * step
        xmean = scale(row.mean)
        xlow = scale(row.mean - row.error)
        xhigh = scale(row.mean + row.error)
        println(io, "<text x=\"$(x0-8)\" y=\"$(y+4.5)\" text-anchor=\"end\" class=\"label\">$(svg_label(row.label))</text>")
        println(io, "<rect x=\"$x0\" y=\"$(y-bar_height/2)\" width=\"$(max(0,xmean-x0))\" height=\"$bar_height\" class=\"bar\"/>")
        println(io, "<line x1=\"$xlow\" y1=\"$y\" x2=\"$xhigh\" y2=\"$y\" class=\"error\"/>")
        println(io, "<line x1=\"$x0\" y1=\"$(y-3.8)\" x2=\"$x0\" y2=\"$(y+3.8)\" class=\"tick\"/>")
    end
    for tick in ticks
        x = scale(tick)
        println(io, "<line x1=\"$x\" y1=\"$bottom\" x2=\"$x\" y2=\"$(bottom+5)\" class=\"tick\"/>")
        println(io, "<text x=\"$x\" y=\"$(bottom+20)\" text-anchor=\"middle\" class=\"ticklabel\">$tick</text>")
    end
    println(io, "<line x1=\"$x0\" y1=\"$bottom\" x2=\"$x1\" y2=\"$bottom\" class=\"axis\"/>")
    println(io, "<text x=\"$((x0+x1)/2)\" y=\"462\" text-anchor=\"middle\" class=\"guide\">Δ BIC</text>")
    println(io, "<text x=\"$x1\" y=\"462\" text-anchor=\"end\" class=\"multiplier\">×1000</text>")
end

function main()
    tol = task_statistics(; tol=true, seed=20260812)
    rh = task_statistics(; tol=false, seed=20260813)
    mkpath(OUTPUT_DIR)
    output = joinpath(OUTPUT_DIR, "fig06_bic.svg")
    open(output, "w") do io
        println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="775" height="473" viewBox="0 0 775 473">
<rect width="775" height="473" fill="white"/>
<style>
.title{font:20px Helvetica,Arial,sans-serif}.panel{font:bold 23px Helvetica,Arial,sans-serif}
.label{font:15.5px Helvetica,Arial,sans-serif}.ticklabel{font:13px Helvetica,Arial,sans-serif}
.guide{font:18px Helvetica,Arial,sans-serif}.multiplier{font:11px Helvetica,Arial,sans-serif}
.axis{stroke:#000;stroke-width:1.35}.tick{stroke:#000;stroke-width:1.1}.bar{fill:#fff;stroke:#000;stroke-width:1.1}
.error{stroke:#000;stroke-width:1.1;stroke-linecap:round}
</style>""")
        draw_panel(io, tol; x0=108.0, x1=365.0, title="Tower of London", panel="A", ticks=0:20:80, xmax=80.0)
        draw_panel(io, rh; x0=508.0, x1=758.0, title="Rush Hour", panel="B", ticks=0:5:25, xmax=25.0)
        println(io, "</svg>")
    end
    converter = Sys.which("rsvg-convert")
    isnothing(converter) && error("Install rsvg-convert to render the PDF")
    run(`$converter -f pdf -o $(joinpath(OUTPUT_DIR, "fig06_bic.pdf")) $output`)
    println("Wrote Figure 6 to $OUTPUT_DIR")
end

main()
