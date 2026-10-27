#!/usr/bin/env julia

# Rebuild the quantitative panels of Figure 8 and the alternative-model check
# in Figure S7. The board examples in Figure 8A-B are hand-composed artwork;
# the published composite PDF is retained under paper/figures/.

using CSV, DataFrames, Printf

const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUT = joinpath(ROOT, "paper", "generated")
mkpath(OUT)
const TABLE = CSV.read(joinpath(ROOT, "data", "published", "counterintuitive_summary.csv"), DataFrame)

fmt(x) = @sprintf("%.2f", x)
function text!(io,x,y,value; size=18,anchor="middle",bold=false,rotate=nothing)
    transform = isnothing(rotate) ? "" : " transform='rotate($rotate $x $y)'"
    println(io,"<text x='$x' y='$y' text-anchor='$anchor' font-size='$size' font-weight='$(bold ? "bold" : "normal")'$transform>$value</text>")
end
function line!(io,x1,y1,x2,y2; color="#111",width=1.4)
    println(io,"<line x1='$x1' y1='$y1' x2='$x2' y2='$y2' stroke='$color' stroke-width='$width'/>")
end
function draw_panel!(io, task, models, colors, x0, y0, width, height; label="", show_ylabel=true, legend=false)
    rh = task == "RH"
    xmax = rh ? 17.0 : 8.0
    ymax = rh ? 0.16 : 0.18
    px(v) = x0 + (xmax-v)/xmax*width
    py(v) = y0 + (ymax-v)/ymax*height
    line!(io,x0,y0,x0,y0+height)
    line!(io,x0,y0+height,x0+width,y0+height)
    for v in (rh ? (15,10,5,0) : (8,6,4,2,0))
        xx=px(v)
        line!(io,xx,y0+height,xx,y0+height+6)
        text!(io,xx,y0+height+25,v;size=17)
    end
    for v in (0.0,0.05,0.10,0.15)
        yy=py(v)
        line!(io,x0-6,yy,x0,yy)
        tick = v == 0 ? "0" : @sprintf("%.2f",v)
        text!(io,x0-10,yy+6,tick;size=17,anchor="end")
    end
    text!(io,x0+width/2,y0+height+60,"Distance to goal";size=22)
    show_ylabel && text!(io,x0-76,y0+height/2,"Counterintuitive move";size=22,rotate=-90)
    isempty(label) || text!(io,x0-96,y0-12,label;size=28,anchor="start",bold=true)

    for (model,color) in zip(models,colors)
        rows = sort(TABLE[(TABLE.task.==task).&(TABLE.model.==model),:],:bin_number)
        nrow(rows)>0 || error("Missing $task/$model")
        xs = [px(r.mean_distance) for r in eachrow(rows)]
        ys = [py(r.mean_counterintuitive) for r in eachrow(rows)]
        upper = [py(clamp(r.mean_counterintuitive+r.ci95_counterintuitive,0,ymax)) for r in eachrow(rows)]
        lower = [py(clamp(r.mean_counterintuitive-r.ci95_counterintuitive,0,ymax)) for r in eachrow(rows)]
        if model == "data"
            for (x,hi,lo) in zip(xs,upper,lower)
                line!(io,x,hi,x,lo;width=1.25)
                line!(io,x-6,hi,x+6,hi;width=1.25)
                line!(io,x-6,lo,x+6,lo;width=1.25)
            end
        else
            polygon = join(["$(fmt(x)),$(fmt(y))" for (x,y) in zip(xs,upper)]," ") * " " *
                join(["$(fmt(x)),$(fmt(y))" for (x,y) in zip(reverse(xs),reverse(lower))]," ")
            println(io,"<polygon points='$polygon' fill='$color' opacity='0.25'/>")
            points = join(["$(fmt(x)),$(fmt(y))" for (x,y) in zip(xs,ys)]," ")
            println(io,"<polyline points='$points' fill='none' stroke='$color' stroke-width='4' stroke-linecap='round'/>")
        end
    end
    if legend
        line!(io,x0+width-145,y0+26,x0+width-120,y0+26;width=5)
        text!(io,x0+width-112,y0+32,"Data";size=19,anchor="start")
        line!(io,x0+width-145,y0+54,x0+width-120,y0+54;color="#929292",width=5)
        text!(io,x0+width-112,y0+60,"Model";size=19,anchor="start")
    end
end
function save_svg(body,name,width,height)
    path=joinpath(OUT,name*".svg")
    open(path,"w") do io
        println(io,"<svg xmlns='http://www.w3.org/2000/svg' width='$width' height='$height' viewBox='0 0 $width $height'><rect width='100%' height='100%' fill='white'/><g font-family='Helvetica,Arial,sans-serif'>")
        body(io)
        println(io,"</g></svg>")
    end
    run(`rsvg-convert -f pdf -o $(joinpath(OUT,name*".pdf")) $path`)
end

save_svg("fig08_analysis_panels",1000,330) do io
    draw_panel!(io,"RH",["data","full"],["#111","#929292"],125,25,320,225;label="C",legend=true)
    draw_panel!(io,"TOL",["data","full"],["#111","#929292"],625,25,320,225;label="D")
    text!(io,19,140,"Rush Hour";size=23,bold=true,rotate=-90)
    text!(io,519,150,"Tower of London";size=23,bold=true,rotate=-90)
end

const LESION_COLORS = ["#0b98e6","#dc6b47","#36a54b"]
const ONLY_COLORS = ["#bb6bd9","#b09214","#00a6a7"]
save_svg("s07_counterintuitive_alternatives",1200,670) do io
    for (row,task) in enumerate(("RH","TOL"))
        heuristic = task == "RH" ? "gamma_ao" : "beta"
        groups = [
            (["full"],["#929292"]),
            (["lesion-$heuristic","lesion-gamma_r","lesion-gamma_p"],LESION_COLORS),
            (["only-$heuristic","only-gamma_r","only-gamma_p"],ONLY_COLORS),
            (["chance"],["#ed5794"]),
        ]
        for (col,(models,colors)) in enumerate(groups)
            letter = "ABCDEFGH"[(row-1)*4+col]
            draw_panel!(io,task,vcat("data",models),vcat("#111",colors),
                112+(col-1)*285,35+(row-1)*315,175,185;
                label=string(letter),show_ylabel=col==1)
        end
        text!(io,20,145+(row-1)*315,task == "RH" ? "Rush Hour" : "Tower of London";size=22,bold=true,rotate=-90)
    end
    text!(io,600,655,"Full    Lesions: heuristic / γr / γp    Only: heuristic / γr / γp    Chance";size=18)
end
println("Wrote Figure 8 quantitative panels and Figure S7 model comparisons")
