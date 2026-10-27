# ROSETTE: paper reproducibility release

This is the compact Julia source-and-data release for *Rollout Models of Human Planning*. It contains the model implementation, anonymous indexed human choices, fitted model results, analysis-ready figure tables, the final figure files, and the source artwork for conceptual figures. The exploratory `cleaned_code/`, `fast_code/`, legacy `src/`, and large fitting/policy caches from earlier versions are available in Git history, but are not needed to rerun the released figure analyses.

## Quick start

From a fresh clone, with Julia 1.8 or later and `rsvg-convert` installed:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. test/runtests.jl
julia --project=. scripts/make_figures.jl
```

The last command writes regenerated vector panels, tables, and the second-round shortest-path analysis to `paper/generated/`. It does not overwrite the final reviewed figure files in `paper/figures/`. The figure scripts use SVG rather than the platform-dependent Plots/Qt backend. `rsvg-convert` is needed only for the PDF copies; the plotting data are plain CSV.

## Repository map

- `src/`: the single reference Julia implementation of game states, AND/OR trees, expected-evidence policy, likelihoods, and the classical-search baselines.
- `data/problems/`: 70 Rush Hour puzzle specifications, in the puzzle order indexed by `data/published/rh_puzzles.csv`.
- `data/published/`: anonymous participant-level choices, fitted results, and the analysis-ready numeric inputs for figures. No raw event logs or source participant IDs are included.
- `scripts/`: one-time export scripts (requiring the private original archive), portable public-data figure scripts, and a fitting-input reconstruction script.
- `paper/figures/`: the final reviewed main and supplementary artwork as included in the manuscript.
- `paper/figure_sources/`: editable Illustrator source for the conceptual Figures 1–5.
- `test/`: data-integrity and core-model smoke tests.

## Figure provenance

The published multi-panel artwork includes manual panel letters, board illustrations, and layout refinements. The scripts reproduce the numerical content; `paper/figures/` contains the exact reviewed composites.

| Paper figure | Released source |
| --- | --- |
| 1–5 | Final PDF and editable artwork in `paper/figure_sources/`; these are conceptual/task illustrations, not statistical plots. |
| 6 | `scripts/figure06_bic.jl` from `fit_nll.csv`; writes SVG/PDF. |
| 7 | `scripts/figure07_summary.jl` from anonymous participant-bin summaries; also writes S1–S2. |
| 8 | `scripts/figure08_counterintuitive.jl` from the **final** move definitions and their participant-bin summaries; writes panels C–D and S7. Board examples A–B are part of the final composite. |
| 9 | `scripts/figure09_excess_length.jl` from `excess_length_by_puzzle.csv`; `scripts/figure09_shortest_path_control.jl` reproduces the second-revision residual analysis. |
| 10 | Final state-space composite in `paper/figures/`; `scripts/figure10_transition_counts.jl` reconstructs the plotted directed-transition frequencies from indexed human choices and selected model simulations. |
| S3–S4 | Final trajectory graphics in `paper/figures/`; the illustrated simulation paths and transition counts are released. |
| S5–S6 | Final case-study composites in `paper/figures/`. These are illustrative diagrams, not new fitted summaries. |

Exact Illustrator placement and random spline jitter in Figures 8 and 10 and S3–S4 are not recreated by `make_figures.jl`. The final artwork is included so manuscript reproduction is exact, while the plotted quantitative values and selected underlying paths are inspectable separately.

## Public tables

| File | Content |
| --- | --- |
| `choices.csv` | One row per indexed human step (including terminal markers): task, anonymous subject index, puzzle, attempt, step, state, action, terminal flag. |
| `rh_puzzles.csv`, `tol_puzzles.csv` | Puzzle index to specification. Tower of London states are six-slot vectors. |
| `fit_nll.csv` | Negative log likelihood and parameter count for every participant/model fit; used by Figure 6. |
| `fit_parameters.csv` | Fitted parameter values for the full, lesion, and single-component models. |
| `move_summary_subject_bins.csv` | Anonymous participant-bin means and within-participant SEs for the better/same/worse move plots (Figure 7, S1–S2). |
| `counterintuitive_subject_bins.csv`, `counterintuitive_summary.csv` | Final Rush Hour “worsening indirect preparatory” and Tower of London “temporarily misplaced bead” move checks (Figure 8, S7). |
| `excess_length_by_puzzle.csv`, `puzzle_metadata.csv` | Puzzle-level human/model excess length and shortest-path metadata (Figure 9 and revision analysis). |
| `illustrated_simulations.csv` | Selected simulation trajectories used for Figure 10 and S3–S4: Rush Hour puzzle 52, four displayed models, ten repeats; full-model Tower of London trajectories, repeat 8, all 25 puzzles. |

The participant index is stable within a task and across the exported tables. Source identifiers were replaced before release. The figure tables are a compact sufficient representation of the displayed summaries. Recomputing all fitted parameters or every model-probability column from scratch is more expensive than regenerating the plots, so the fitted estimates and summary inputs are versioned as scientific data.

## Rebuilding model inputs

The core code can rebuild per-puzzle fitting objects and nested indexed choices from the released puzzle and choice tables. For a quick check:

```sh
julia --project=. scripts/rebuild_model_inputs.jl TOL 1
julia --project=. scripts/rebuild_model_inputs.jl RH 1
```

Use `TOL 25` or `RH 70` for the complete tasks. The resulting JLD2 caches go into ignored `data/cache/`. Full fitting is computationally costly; the released `fit_nll.csv` and `fit_parameters.csv` are the fitted outputs used in the paper.

The `scripts/export_*.jl` programs document how the compact public tables were derived from the original archive. They are one-time provenance scripts, not required for a fresh-clone reproduction. They do not write raw identifiers to the release files.

## License and citation

See `LICENSE`. Please cite the paper and this repository if you reuse the model or data.
