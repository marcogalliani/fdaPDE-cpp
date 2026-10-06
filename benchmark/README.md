# fdaPDE benchmarks

Measurements of *behaviour*, as opposed to the invariants pinned by `../test`. Each benchmark prints its
numbers and then checks a handful of named claims about their shape — an ordering between methods, a
convergence rate — so a change is reported rather than left to be noticed in a table. The process exits
non-zero if any claim fails, which makes the suite usable in CI alongside the tests.

## Running

```sh
./run_benchmarks.sh                      # build and run everything (~40 s)
./run_benchmarks.sh --list               # what is available
./run_benchmarks.sh grid_refinement
./run_benchmarks.sh --full --reps 12     # wider sweeps, more replicates
```

Or against an existing build: `cd build && make && ./fdapde_benchmark [options] [name ...]`.

## What is here

| benchmark | records |
|---|---|
| `grid_refinement` | What a coarser time grid costs, separating integration order from control resolution. |
| `stiff` | Stiff dynamics with and without an unresolved fast layer. |
| `stiff_control_sensitivity` | The control sensitivity `B_t` against the assumption the preconditioner makes about it. |

## The findings these pin

**Coarse grids are expensive because of integration order, not control resolution.** GL1 is
integration-limited on coarse grids; GL2 is not, and GL3 adds essentially nothing over GL2. Raising the
scheme order is what makes a coarse grid affordable.

*(Three benchmarks that stood here were removed with the integration-mesh refinement feature during core
development: `adaptive_refinement` and `refinement_vs_grid`, which justified separating the integration mesh
from the control mesh, and `defect_estimator`, which calibrated the collocation defect that drove the
refinement. Their source and their recorded numbers are in `INTEGRATION_REFINEMENT.md` at the repo root.)*

*(`control_space_noiseless` and `control_space_noisy`, which compared the stage-wise control with a
piecewise-constant restriction, were removed with that restriction. Their source, findings and last numbers are
in `CONTROL_SPACE.md` at the repo root: they found the restricted space the better estimate.)*

**Stiffness only costs something when a fast layer is unresolved.** On the slow manifold every Gauss scheme
is noise-limited even at `k*dt = 40`. Off it, errors rise by an order of magnitude and convergence order
collapses — Gauss schemes are A-stable but not L-stable, so the layer is propagated rather than damped.
That regime also drives the best `lambda` small, which is where the reduced solve is worst conditioned:
the preconditioner's scale assumes `B_t ~ dt*I`, while a stiff component is damped by `1/(1 + O(k*dt))`,
an anisotropy of over 100x at `k*dt = 40` that a single scalar per interval cannot represent.

## Conventions

Accuracy is always measured through the solver's continuous `eval()` on a **fixed** query grid, never at
the fit's own nodes — those move with `m`, so nodal errors are not comparable across resolutions. The
reference trajectory comes from GL3 with a tight Newton tolerance on a refinement containing every query
point exactly, so no interpolation of the truth enters. Noise is seeded by replicate index only, so two
methods compared at the same replicate see identical data; that pairing is what makes small differences
meaningful rather than Monte-Carlo scatter. Where `lambda` is swept, the reported figure is the best over
the sweep ("oracle lambda"), which isolates representational capability from selection and is therefore an
upper bound on what a GCV-driven fit would achieve.

All fixtures live in `src/bench_utils.h` and are shared, so a change in one benchmark's numbers cannot be
blamed on a different problem.
