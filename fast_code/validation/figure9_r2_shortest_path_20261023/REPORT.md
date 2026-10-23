# Figure 9 shortest-path adjustment for the second PLOS revision

## Question and data

Reviewer 1 asked whether the puzzle-level association between human and ROSETTE excess solution length in Figure 9 remains after accounting for shortest-path length, $L^*$. This analysis uses the exact 25 Tower of London and 35 Rush Hour puzzle-level means plotted in Figure 9. Each puzzle has equal weight, as in the published correlations. The values of $L^*$ come from the previously validated puzzle metadata. No model was refitted, no trajectories were resimulated, and no figure was changed.

## Method

For each task separately, regress mean human excess length, $H=L_{human}-L^*$, and mean model excess length, $M=L_{model}-L^*$, on $L^*$ with an intercept. Correlate the two sets of residuals. This is the partial Pearson correlation between $H$ and $M$ given $L^*$. Its two-sided $t$ test has $n-3$ degrees of freedom. The same $p$ value results from testing the model coefficient in $H\sim L^*+M$.

For comparison with Wei Ji's phrasing, also correlate the human residuals with the unadjusted model excess length. This one-sided, semi-partial correlation has the same sign but a different magnitude from the full partial correlation. The primary test conditions both measures on $L^*$.

## Results

| Task | Puzzles | Figure 9 raw $r$ | Human residual vs. model $r$ | Partial $r$ controlling $L^*$ | $p$ | Approx. 95% CI | Additional $R^2$ from model |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tower of London | 25 | .534 | -.068 | -.174 | .415 | [-.540, .246] | .018 |
| Rush Hour | 35 | .527 | .241 | .280 | .109 | [-.064, .565] | .040 |

The raw correlations reproduce the manuscript values and their original $p$ values. The confidence intervals use the conventional Fisher-$z$ approximation for one controlled variable. The adjusted relationships are not statistically clear in either task. Tower of London shows no positive residual association. Rush Hour retains a positive estimate, but its interval includes zero. The model adds approximately 1.8% and 4.0% to the variance explained in human excess length after $L^*$ in Tower of London and Rush Hour, respectively.

## Checks

- Permuting the model residuals 50,000 times gave two-sided $p=.411$ for Tower of London and $p=.112$ for Rush Hour, close to the conventional tests.
- Leaving out any one puzzle gave partial $r$ values from -.247 to -.032 in Tower of London and .245 to .417 in Rush Hour. This preserves the direction of each point estimate, though the Rush Hour result remains uncertain.
- The raw correlations and their $p$ values match the saved Figure 9 analysis to numerical precision. The coefficient test in the multiple regression matches the partial-correlation test algebraically.
- The August analysis of **log total solution length** asked a different question. Its partial correlations must not be substituted for these Figure 9 results.

## Interpretation for the revision

The Figure 9 correlation alone should not be described as evidence that ROSETTE predicts which puzzles generate more errors after shortest-path difficulty is accounted for. A concise addition to the Results and a short, direct reviewer response can report the adjusted values above. Figure 9 itself can stay as it is. The caption already identifies the plotted values as excess length and the text should distinguish the descriptive raw correlation from the adjusted check.

## Reproduction

From the repository root:

```bash
julia --project=. fast_code/validation/figure9_r2_shortest_path_20261023/analyze.jl
```

The script reads `paper_excess_length_scatter.csv` and `puzzle_metadata.csv` beside this report. These input files are unchanged copies from the validated Figure 9 analysis in `fast_code/validation/results_model_evidence_20260808/`. It writes `figure9_shortest_path_summary.csv` and `figure9_shortest_path_puzzles.csv` to the same directory. The latter contains all 60 puzzle-level inputs and fitted residuals.
