module ROSETTE

using CSV
using DataFrames
using DataStructures
using Interpolations
using JLD2
using JSON
using LinearAlgebra
using Optim
using Plots
using ProgressBars
using Random
using SparseArrays
using StaticArrays
using Statistics
using StatsBase

export AbstractGame, Car, RushHour, TowerOfLondon, FittingStruct
export Action, RushHourState, RushHourFreeState, RushHourFixedState
export get_base, start_state, goal_state, check_digit, is_terminal, search
export load_prb, load_prb_, int_to_s, s_to_int, board_to_int32
export make_move, make_move!, neighbours, neighbours_actions, get_all_moves, draw_state
export possible_moves!, move_blocked_by!, unblocking_moves!, get_QR_matrices
export get_and_or_tree, propagate_ps, get_and_or_dicts
export get_fitting_infos, get_analysis_info, get_tol_fitting_infos, get_tol_analysis_info
export get_stopping_prob, get_stopping_probabilities!, apply_γp, p_s
export update_QR_matrices_and_or!, update_QR_matrices_tol!
export main_probability_params, get_ll_main, get_p_main, get_nll_main
export get_ll, get_p, get_nll, get_ll_new, get_p_new, get_nll_new
export get_ll_progress_rh, get_p_progress_rh, get_nll_progress_rh
export get_p_full_rh, get_p_full_tol, get_p_reduced_rh, get_p_reduced_tol
export get_ll_classical, get_nll_classical
export load_raw_data, filter_subjects, pre_process, get_filtered_df, load_rh_data
export games_to_idx, get_ss, load_rh_parameters, load_rh_parameters_reduced
export load_rh_lesion_parameters, load_all_ps
export load_tol_data, load_tol_data_with_rts, prepare_tol_dataframe, state_actions_tol
export load_tol_parameters, load_tol_parameters_reduced, load_tol_parameters_super_reduced
export load_tol_lesion_parameters
export calculate_summary_statistics, bin_stats, bin_stats_subj
export load_results, load_newest_results, load_perstate_results, load_classical_results
export load_ps!, load_simulated_games, puzzle_counts, get_puzzle_stats, get_binned_results
export get_state_mappings, get_tol_points, get_circle_points, spline_path

include("types.jl")
include("shared_game.jl")
include("rush_hour.jl")
include("tower_of_london.jl")
include("markov.jl")
include("and_or_trees.jl")
include("data_loading.jl")
include("models.jl")
include("classical_search_models.jl")
include("summary_statistics.jl")
include("analysis_io.jl")
include("analysis_helpers.jl")
include("state_space_helpers.jl")
include("plotting.jl")
include("workflows.jl")

end # module ROSETTE
