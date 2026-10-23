# The refinement loop *(proposal)*

Back to the [design index](README.md).

One reduction gives a first model. The loop then improves it where the
reference says it is wrong. This is greedy sampling as in reduced-basis
methods, combined with error-driven marking as in adaptive mesh refinement.
Not implemented yet. Plans 005–006 cover a single reduce-then-fit pass.

## Site roles

| Role | Used for | Changes during the loop? |
|---|---|---|
| `fit` | targets for the reduction and fitting | grows: new sites each iteration |
| `probe` | measuring error to decide what to do next | resampled or extended each iteration |
| `test` | final score only | **locked** at the start, never seen by the loop |

Real synapses are split by axon between `fit` and `test`, so all synapses of
an axon stay on one side. Sampled sites fill the probe pool.

## One iteration

1. **Reduce** with the current fit sites.
2. **Export a temporary result** to `iter_k/` (model, errors), so it can be
   inspected, resumed or rolled back.
3. **Compare** on fit and probe sites: port impedances at a few
   frequencies, single-synapse EPSPs, and pattern summation. Each
   comparison costs one FEM back-substitution plus a small model solve.
4. **Build error maps:** per site, interpolated over the membrane, and
   aggregated per compartment and region.
5. **Informed update:**
   - sample new sites where error is high (the `weighted` sampler with the
     error field as feature, with a minimum spacing);
   - mark and refine compartments carrying the largest share of the error
     (Dörfler-style), for example by splitting, raising a region's rank, or
     reclassifying a region as 3D;
   - optionally coarsen compartments whose error is far below tolerance;
   - **fit** against FEM targets (a core step every iteration).
6. **Stop** at tolerance, budget, iteration cap, or stagnation.

Repeating at several compartment budgets traces the Pareto front. The final
model is scored once on the locked `test` set and exported with its history.
A large probe/test gap flags overfitting.

## Modes

- `run(...)` runs automatically to a stopping criterion.
- `step()` runs one iteration and returns the result and error maps. The
  user can inject sites or constraints before the next step.
