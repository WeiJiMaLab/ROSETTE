module FastROSETTE

using LinearAlgebra
using Optim

export Action, FastFittingInfo, FastDataset
export prepare_fitting_info, prepare_fitting_infos, prepare_dataset
export probability_params, nll_main, fit_main

include("types.jl")
include("unified_model.jl")
include("fit.jl")

end
