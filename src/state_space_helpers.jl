function get_state_mappings(analysis_info)
    all_maps = [analysis_info[i][2] for i in eachindex(analysis_info)]
    target_map = all_maps[1]
    mappings = Vector{Int}[]

    for map in all_maps
        mapping = Int[]
        for state in map
            push!(mapping, findfirst(==(state), target_map))
        end
        push!(mappings, mapping)
    end

    return mappings
end

function get_tol_points()
    hourglass_points = hcat([
        [0.0, 0.4],
        [0.2 * sqrt(3) / 2, 0.5],
        [0.2 * sqrt(3) / 2, 0.3],
        [-0.2 * sqrt(3) / 2, 0.5],
        [-0.2 * sqrt(3) / 2, 0.3],
        [0.0, 0.8],
        [0.2 * sqrt(3) / 2, 0.9],
        [0.2 * sqrt(3) / 2, 0.7],
        [-0.2 * sqrt(3) / 2, 0.9],
        [-0.2 * sqrt(3) / 2, 0.7],
        [2 * 0.2 * sqrt(3) / 2, 0.6],
        [3 * 0.2 * sqrt(3) / 2, 0.7],
    ]...)

    angle_deg = -120.0
    rotation = [cosd(angle_deg) -sind(angle_deg); sind(angle_deg) cosd(angle_deg)]
    once = rotation * hourglass_points
    twice = rotation * once
    all_points = [tuple(hcat(hourglass_points, once, twice)[:, i]...) for i in 1:36]

    first_idxs = [36, 25, 17, 8, 30, 15, 24, 6, 28, 7, 23, 16]
    second_idxs = [19, 9, 31, 33, 12, 1, 21, 4, 10, 26, 14, 3]
    third_idxs = [11, 34, 32, 27, 13, 18, 20, 35, 22, 29, 5, 2]
    return all_points[sortperm(vcat(first_idxs, second_idxs, third_idxs))]
end

function get_circle_points()
    points = [(0.0, 0.0) for _ in 1:36]

    for (idxs, radius, phase) in (
        ([25, 36, 30, 5, 35, 34, 11, 13, 14, 4, 9, 19, 12, 23, 6], 1.0, π / 2),
        ([7, 8, 2, 20, 18, 29, 27, 3, 21, 1, 26, 33, 16, 24, 15], 1.3, π / 15 + π / 2),
        ([17, 32, 31], 0.7, π / 15 + π / 2),
        ([28, 22, 10], 1.55, π / 2),
    )
        for i in eachindex(idxs)
            theta = phase + 2π * (i - 1) / length(idxs)
            points[idxs[i]] = (radius * cos(theta), radius * sin(theta))
        end
    end

    return points
end

function spline_path(points; n=300)
    s_ctrl = range(0, 1, length=length(points))
    xs = [point[1] for point in points]
    ys = [point[2] for point in points]
    itp_x = CubicSplineInterpolation(s_ctrl, xs)
    itp_y = CubicSplineInterpolation(s_ctrl, ys)
    s = range(0, 1, length=n)
    return itp_x.(s), itp_y.(s), s
end
