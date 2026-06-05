# ROSETTE

ROSETTE stands for **Rollouts with Stochastic Early Termination and Tractable Estimation**. This repository contains Julia code for modeling human planning behavior in Rush Hour and Tower of London tasks, fitting several model variants, generating simulations, and producing summary statistics and figures.

The original implementation lives in `src/`. It is preserved as legacy code. The readable recreation lives in `cleaned_code/` and is the recommended entrypoint for understanding and extending the project.

## Folder Map

- `src/`: legacy scripts and functions. Some files are reusable libraries, while others execute analyses at top level when included.
- `cleaned_code/`: cleaned Julia module-style recreation. It separates game logic, data loading, model likelihoods, fitting workflows, plotting helpers, and analysis readers.
- `fast_code/`: fitting-only optimized implementations. The current Julia backend lives in `fast_code/julia/`.
- `data/`: raw, processed, cached, fitted, simulated, and lesion-analysis data. Many files are large `.jld2` caches.
- `figures/` and `tol_trajs/`: generated figures and trajectory visualizations.
- `resources/`: manuscript and journal-submission resources.
- `Project.toml`: dependency metadata for running the cleaned code and tests.
- `test_cleaned_code.jl`: equivalence checks comparing cleaned code against safe legacy functions.

## Using the Cleaned Code

From the repository root:

```julia
include("cleaned_code/ROSETTE.jl")
using .ROSETTE
```

The cleaned code is intentionally not installed as a package yet; it is a module-like source tree that can be included from the project root. It assumes relative paths such as `data/problems/...`, so run scripts from the repository root unless you adapt paths.

Run the equivalence checks with:

```bash
julia --project=. test_cleaned_code.jl
```

On this machine Julia was available at `/usr/local/bin/julia`, and the verified command was:

```bash
/usr/local/bin/julia --project=. test_cleaned_code.jl
```

## Data Flow

Rush Hour raw/processed flow:

1. `load_raw_data()` reads `data/all_subjects.csv`.
2. `filter_subjects(df)` applies the original exclusion criteria.
3. `pre_process(df, d_goals_prbs)` converts raw events into subject, puzzle, state, action, reaction-time, and distance-to-goal rows.
4. `load_rh_data()` reads `data/rh_data.csv` and builds nested state-action trajectories.
5. `games_to_idx(state_actions, analysis_info)` converts integer game states to transient/absorbing matrix indices.

Tower of London flow:

1. `load_tol_data()` reads subject CSVs under `data/tower_of_london/subject_data`.
2. `state_actions_tol(problems, data)` converts pole-level moves into slot-level state-action trajectories.
3. `prepare_tol_dataframe(state_actions, problems[, rts])` builds the analysis DataFrame and distance dictionaries.
4. `get_tol_fitting_infos(problems)` and `get_tol_analysis_info(problems)` build Markov matrices and heuristic caches.

Cached objects in `data/processed_data/` and `data/tower_of_london/` are used by fitting and plotting workflows to avoid recomputing state spaces.

## Game Representations

Rush Hour:

- A `RushHour` game stores a mixed-radix base vector, zero-based initial car positions, and fixed car descriptors.
- A car is `Car(dim2, dir, len)`, where `dir` is `:x` or `:y`.
- Integer states encode each car’s free coordinate.
- `int_to_s`, `s_to_int`, and `board_to_int32` bridge between integer states, mutable state vectors, and compact dataset identifiers.
- `neighbours`, `get_all_moves`, `draw_state`, and `possible_moves!` implement legal move generation.

Tower of London:

- A `TowerOfLondon` game stores initial and goal vectors over six slots.
- Integer states encode which bead occupies each slot.
- `neighbours_actions` returns both next states and the slot-level actions that generated them.
- `get_distances` computes optimal distances to the goal by repeated search.

The cleaned code avoids the legacy collision where `s_type` meant “Rush Hour mutable state” in one file and `Int` in another. It uses explicit names such as `RushHourState`, `RushHourFreeState`, `TOLState`, and `Action`, while keeping compatibility aliases where they do not create import-order hazards.

## Model Logic

The central fitting object is:

```julia
FittingStruct(all_dicts, all_moves, all_idxs, Q, R, P, all_sizes)
```

- `Q`: transient-to-transient transition probabilities.
- `R`: transient-to-absorbing transition probabilities.
- `P`: per-state stopping/restart/progress feature matrix.
- `all_dicts`: Rush Hour AND/OR tree policy dictionaries.
- `all_moves` and `all_idxs`: legal action and indexed-neighbor caches.
- `all_sizes`: heuristic values or tree sizes, depending on model/domain.

Important model families:

- Main three-parameter restart model: `get_p_main`, `get_ll_main`, and `get_nll_main`.
  - Rush Hour fitting parameters are `[nlog_gamma_ao, nlog_gamma_r, nlog_gamma_p]`.
  - Tower of London fitting parameters are `[beta, nlog_gamma_r, nlog_gamma_p]`.
  - Single-step probability calls use probability-scale parameters: `[gamma_ao, gamma_r, gamma_p]` for Rush Hour and `[beta, gamma_r, gamma_p]` for Tower of London.
- Default aliases: `get_p`, `get_ll`, `get_nll`, `get_p_new`, `get_ll_new`, and `get_nll_new` now route to the main three-parameter model unless an explicit alternative `model` keyword is provided.
- Alternative/full variants retained for reference: `get_p_full_rh`, `get_p_full_tol`, `get_p_reduced_rh`, `get_p_reduced_tol`, and the legacy progress/alpha Rush Hour helpers `get_p_progress_rh`, `get_ll_progress_rh`, and `get_nll_progress_rh`.
- Classical search baselines: BFS, DFS, best-first, A*, IDDFS, and IDA* through `get_ll_classical` and `get_nll_classical`.

The cleaned module keeps mutating updates such as `update_QR_matrices_and_or!` and `update_QR_matrices_tol!` explicit. When comparing likelihoods or running repeated fits, use `deepcopy(fi)` if you need an unmodified fitting cache for each evaluation.

## Fitting and Simulation

Legacy fitting scripts mix setup, optimization, saving, and plotting. The cleaned code moves the reusable pieces into `cleaned_code/workflows.jl`:

- `fit_rh_subject(subject_index, N_prbs; iters=200)` fits the main three-parameter Rush Hour objective.
- `fit_tol_subject_batch(batch_index; iters=200, batch_size=10)` fits the main three-parameter Tower of London objective.
- `simulate_reduced_rh_games(...)` simulates trajectories from cleaned reduced-model probabilities.

These functions do not save files automatically. Return values can be saved explicitly with JLD2 when needed.

## Summary Statistics and Plots

`calculate_summary_statistics` computes the same move-level statistics as the legacy code:

- distance-to-goal predictors,
- undo/same-car or undo/same-bead indicators,
- worse/same/better move categories,
- model probability assigned to the observed move.

`bin_stats` and `bin_stats_subj` reproduce the binned summary tables used by the plotting scripts. The plotting file includes compact versions of the legacy figure helpers and NLL lesion plotting.

## Current-State Code Review Notes

The legacy `src/` folder has several important risks:

- Some files are libraries, while others execute fitting, plotting, simulation, or data loading at top level when included.
- `s_type`, `a_type`, `or_type`, and `and_type` are redefined across domains, so behavior depends on include order.
- `get_fitting_infos`, `get_analysis_info`, `get_p_new`, `get_ll_new`, `get_nll_new`, `apply_γp`, and `inf_sum` are duplicated with incompatible meanings across files.
- Several cluster and analysis scripts hard-code data paths, subject counts, problem orders, and output directories.
- Fitting functions mutate `FittingStruct.Q`, `FittingStruct.R`, and `FittingStruct.P`, which can leak state across evaluations.
- Some exploratory files contain useful functions mixed with notebook-like experiments and figure-generation code.
- There is no original `Project.toml`, so dependency versions were implicit in the user’s Julia environment.

The cleaned code addresses these by separating domains, avoiding top-level work, giving conflicting concepts explicit names, and routing duplicated likelihood variants through named functions.

## Tests

`test_cleaned_code.jl` compares cleaned code with safe legacy code on:

- Rush Hour parsing, state encoding, board drawing, legal moves, and neighbors for `prb1707_7`.
- Tower of London state transitions and terminal checks on a tiny constructed problem.
- Tower of London main-model QR updates, probabilities, log-likelihoods, invalid parameter handling, and repeated scoring behavior.
- Rush Hour QR transition semantics, AND/OR propagated policies, legacy progress-model parity, and main-model probability/log-likelihood/NLL parity.
- Data loading, basic preprocessing, and a restart/duplicate-drag preprocessing fixture.
- Summary-statistic and binned-summary output on a tiny synthetic DataFrame.

The tests intentionally avoid legacy files that trigger expensive top-level fitting or plotting.
