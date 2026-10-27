#!/usr/bin/env julia

# Reconstruct the participant-weighted distance-to-goal summaries in Figure 7.
# Inputs contain anonymous participant-bin means and within-participant SEMs.

using CSV, DataFrames, Statistics, Printf

const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUT = joinpath(ROOT, "paper", "generated")
mkpath(OUT)

source = CSV.read(joinpath(ROOT, "data", "published", "move_summary_subject_bins.csv"), DataFrame)
results = DataFrame(task=String[], model=String[], bin_number=Int[], metric=String[],
    distance=Float64[], mean=Float64[], ci95=Float64[], n_subjects=Int[])
for group in groupby(source, [:task, :model, :bin_number])
    for metric in ("better", "same", "worse")
        means = Float64.(group[!, Symbol("mean_", metric)])
        within = Float64.(group[!, Symbol("sem_", metric)])
        n = length(means)
        n > 0 || continue
        ci = 1.96 * sqrt((n == 1 ? 0.0 : std(means)^2 / n) + mean(within .^ 2) / n)
        push!(results, (group.task[1], group.model[1], group.bin_number[1], metric,
            mean(group.mean_distance), mean(means), ci, n))
    end
end
CSV.write(joinpath(OUT, "fig07_aggregates.csv"), results)

fmt(x) = @sprintf("%.2f", x)
esc(s) = replace(string(s), "&" => "&amp;", "<" => "&lt;")
function line(io, x1,y1,x2,y2; color="black", width=1.3)
    println(io, "<line x1='$x1' y1='$y1' x2='$x2' y2='$y2' stroke='$color' stroke-width='$width'/>")
end
function label(io,x,y,s; size=21, bold=false, anchor="middle", rotate=nothing)
    transform = rotate === nothing ? "" : " transform='rotate($rotate $x $y)'"
    weight = bold ? "bold" : "normal"
    println(io,"<text x='$x' y='$y' text-anchor='$anchor' font-size='$size' font-weight='$weight'$transform>$(esc(s))</text>")
end
function panel(io, task, metric, x0, y0, letter)
    pw, ph = 250, 190
    xmin, xmax = task == "RH" ? (0.0,17.0) : (0.0,8.0)
    ymin, ymax = task == "RH" ?
        (metric == "better" ? (0.35,1.0) : (0.0,0.45)) :
        (metric == "better" ? (0.55,1.05) : (0.0,0.25))
    px(v) = x0 + (xmax-v)/(xmax-xmin)*pw
    py(v) = y0 + (ymax-v)/(ymax-ymin)*ph
    line(io,x0,y0,x0,y0+ph)
    line(io,x0,y0+ph,x0+pw,y0+ph)
    xticks = task == "RH" ? [15,10,5,0] : [8,6,4,2,0]
    for v in xticks
        xx=px(v)
        line(io,xx,y0+ph,xx,y0+ph+7)
        label(io,xx,y0+ph+28,v;size=19)
    end
    yticks = task == "RH" ?
        (metric == "better" ? collect(0.4:0.1:1.0) : collect(0:0.1:0.4)) :
        (metric == "better" ? collect(0.6:0.1:1.0) : collect(0:0.05:0.25))
    for v in yticks
        yy=py(v)
        line(io,x0-7,yy,x0,yy)
        tick = v == 0 ? "0" : @sprintf("%.2g",v)
        label(io,x0-14,yy+7,tick;size=19,anchor="end")
    end
    label(io,x0+pw/2,y0+ph+66,"Distance to goal";size=24)
    label(io,x0-77,y0+ph/2,"Prop. $metric";size=24,rotate=-90)
    label(io,x0-103,y0-7,letter;size=30,bold=true,anchor="start")
    data = sort(results[(results.task.==task).&(results.model.=="data").&(results.metric.==metric),:],:bin_number)
    model = sort(results[(results.task.==task).&(results.model.=="full").&(results.metric.==metric),:],:bin_number)
    nrow(data)==nrow(model) || error("Data/model bin mismatch in $task/$metric")
    # Grey band and curve are model means with 95% intervals; black bars are data.
    upper = join(["$(fmt(px(r.distance))),$(fmt(py(clamp(r.mean+r.ci95,ymin,ymax))))" for r in eachrow(model)], " ")
    lower = join(["$(fmt(px(r.distance))),$(fmt(py(clamp(r.mean-r.ci95,ymin,ymax))))" for r in reverse(eachrow(model))], " ")
    println(io,"<polygon points='$upper $lower' fill='#bdbdbd' opacity='0.65'/>")
    points = join(["$(fmt(px(r.distance))),$(fmt(py(r.mean)))" for r in eachrow(model)]," ")
    println(io,"<polyline points='$points' fill='none' stroke='#929292' stroke-width='5' stroke-linecap='round'/>")
    for r in eachrow(data)
        xx=px(r.distance)
        hi=py(clamp(r.mean+r.ci95,ymin,ymax))
        lo=py(clamp(r.mean-r.ci95,ymin,ymax))
        line(io,xx,hi,xx,lo;width=1.2)
        line(io,xx-6,hi,xx+6,hi;width=1.2)
        line(io,xx-6,lo,xx+6,lo;width=1.2)
    end
end

svg = joinpath(OUT,"fig07_summary.svg")
open(svg,"w") do io
    println(io,"<svg xmlns='http://www.w3.org/2000/svg' width='1200' height='630' viewBox='0 0 1200 630'>")
    println(io,"<rect width='100%' height='100%' fill='white'/><g font-family='Helvetica,Arial,sans-serif'>")
    label(io,29,145,"Rush Hour";size=30,bold=true,rotate=-90)
    label(io,29,470,"Tower of London";size=30,bold=true,rotate=-90)
    for (row,task) in enumerate(("RH","TOL")), (col,metric) in enumerate(("better","same","worse"))
        panel(io,task,metric,155+390*(col-1),40+310*(row-1),"ABCDEF"[(row-1)*3+col])
    end
    line(io,168,77,190,77;width=6)
    label(io,198,84,"Data";size=22,anchor="start")
    line(io,168,105,190,105;color="#929292",width=6)
    label(io,198,112,"Model";size=22,anchor="start")
    println(io,"</g></svg>")
end
run(`rsvg-convert -f pdf -o $(joinpath(OUT,"fig07_summary.pdf")) $svg`)

# The supplementary panels use the same subject-level aggregation and compare
# the full model, three lesions, three single-component models, and chance.
const SUPP_COLORS = ["#929292","#0798e5","#de6b49","#36a24f",
    "#bb6bd7","#b08f10","#00a4a6","#ee5795"]
function supplemental_panel(io,task,metric,models,colors,x0,y0,letter; ylabel=false)
    pw,ph=180,150
    xmax=task=="RH" ? 17.0 : 8.0
    ymax=metric=="better" ? 1.05 : metric=="same" ? (task=="RH" ? 0.55 : 0.32) : (task=="RH" ? 0.4 : 0.55)
    ymin=metric=="better" ? (task=="RH" ? 0.15 : 0.3) : 0.0
    px(v)=x0+(xmax-v)/xmax*pw
    py(v)=y0+(ymax-v)/(ymax-ymin)*ph
    line(io,x0,y0,x0,y0+ph)
    line(io,x0,y0+ph,x0+pw,y0+ph)
    for v in (task=="RH" ? (15,10,5,0) : (8,6,4,2,0))
        xx=px(v); line(io,xx,y0+ph,xx,y0+ph+5); label(io,xx,y0+ph+21,v;size=15)
    end
    for v in (metric=="better" ? (0.4,0.6,0.8,1.0) : (0.0,0.1,0.2,0.3))
        yy=py(v)
        ymin<=v<=ymax || continue
        line(io,x0-5,yy,x0,yy)
        label(io,x0-9,yy+5,v==0 ? "0" : @sprintf("%.1f",v);size=15,anchor="end")
    end
    label(io,x0+pw/2,y0+ph+48,"Distance to goal";size=18)
    ylabel && label(io,x0-57,y0+ph/2,"Prop. $metric";size=18,rotate=-90)
    label(io,x0-72,y0-4,letter;size=25,bold=true,anchor="start")
    human=sort(results[(results.task.==task).&(results.model.=="data").&(results.metric.==metric),:],:bin_number)
    for r in eachrow(human)
        xx=px(r.distance)
        hi=py(clamp(r.mean+r.ci95,ymin,ymax))
        lo=py(clamp(r.mean-r.ci95,ymin,ymax))
        line(io,xx,hi,xx,lo)
        line(io,xx-5,hi,xx+5,hi)
        line(io,xx-5,lo,xx+5,lo)
    end
    for (model,color) in zip(models,colors)
        rows=sort(results[(results.task.==task).&(results.model.==model).&(results.metric.==metric),:],:bin_number)
        nrow(rows)>0 || error("Missing supplementary $task/$model/$metric")
        upper=join(["$(fmt(px(r.distance))),$(fmt(py(clamp(r.mean+r.ci95,ymin,ymax))))" for r in eachrow(rows)]," ")
        lower=join(["$(fmt(px(r.distance))),$(fmt(py(clamp(r.mean-r.ci95,ymin,ymax))))" for r in reverse(eachrow(rows))]," ")
        println(io,"<polygon points='$upper $lower' fill='$color' opacity='0.20'/>")
        points=join(["$(fmt(px(r.distance))),$(fmt(py(r.mean)))" for r in eachrow(rows)]," ")
        println(io,"<polyline points='$points' fill='none' stroke='$color' stroke-width='3.3' stroke-linecap='round'/>")
    end
end

for task in ("RH","TOL")
    heuristic=task=="RH" ? "gamma_ao" : "beta"
    groups=[
        (["full"],[SUPP_COLORS[1]]),
        (["lesion-$heuristic","lesion-gamma_r","lesion-gamma_p"],SUPP_COLORS[2:4]),
        (["only-$heuristic","only-gamma_r","only-gamma_p"],SUPP_COLORS[5:7]),
        (["chance"],[SUPP_COLORS[8]]),
    ]
    path=joinpath(OUT,"s0$(task=="RH" ? 1 : 2)_move_summary.svg")
    open(path,"w") do io
        println(io,"<svg xmlns='http://www.w3.org/2000/svg' width='1200' height='890' viewBox='0 0 1200 890'><rect width='100%' height='100%' fill='white'/><g font-family='Helvetica,Arial,sans-serif'>")
        for (row,metric) in enumerate(("better","same","worse")), (col,(models,colors)) in enumerate(groups)
            supplemental_panel(io,task,metric,models,colors,130+275*(col-1),30+270*(row-1),
                string("ABCDEFGHIJKL"[(row-1)*4+col]);ylabel=col==1)
        end
        heuristic_label=task=="RH" ? "γao" : "β"
        legend_models=["Full","Lesion $heuristic_label","Lesion γr","Lesion γp",
            "Only $heuristic_label","Only γr","Only γp","Chance"]
        for (index,(name,color)) in enumerate(zip(legend_models,SUPP_COLORS))
            col=(index-1)%4; row=(index-1)÷4
            xx=195+250*col; yy=832+31*row
            line(io,xx,yy,xx+35,yy;color=color,width=6)
            label(io,xx+43,yy+6,name;size=17,anchor="start")
        end
        println(io,"</g></svg>")
    end
    run(`rsvg-convert -f pdf -o $(replace(path,".svg"=>".pdf")) $path`)
end
println("Wrote Figure 7 and S1-S2 aggregates, SVGs, and PDFs")
