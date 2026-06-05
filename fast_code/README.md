# Fast Fitting Code

This folder is for fitting-only implementations. It should not contain analysis,
plotting, preprocessing, or exploratory scripts.

## Julia Backend

The current backend lives in `julia/` and exposes `FastROSETTE`.

```julia
include("fast_code/julia/FastROSETTE.jl")
using .FastROSETTE
```

The fast path implements one unified three-parameter main model:

- Rush Hour fitting parameters: `[nlog_gamma_ao, nlog_gamma_r, nlog_gamma_p]`
- Tower of London fitting parameters: `[beta, nlog_gamma_r, nlog_gamma_p]`

Domain-specific code only builds the heuristic policy:

- Rush Hour applies `gamma_ao` to AND/OR tree policy dictionaries.
- Tower of London applies `beta` to a softmax over heuristic values.

The restart and eventual action-selection logic controlled by `gamma_r` and
`gamma_p` is shared.

Run the speed/parity smoke test from the repository root:

```bash
/usr/local/bin/julia --project=. fast_code/test_fast_julia.jl
```

The test compares fast NLL values with `cleaned_code`, times repeated NLL
evaluation, and runs a short Optim fit.
