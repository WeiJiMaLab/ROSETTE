function fit_main(initial_params, dataset::FastDataset, fitting_infos::Vector{FastFittingInfo}; iterations::Int=200, K::Int=10_000_000, show_trace::Bool=false)
    objective = params -> nll_main(params, dataset, fitting_infos; K=K)
    return optimize(objective, Float64.(initial_params); iterations=iterations, show_trace=show_trace)
end

function fit_main(initial_params, state_actions, fitting_infos::Vector{FastFittingInfo}; kwargs...)
    return fit_main(initial_params, prepare_dataset(state_actions), fitting_infos; kwargs...)
end
