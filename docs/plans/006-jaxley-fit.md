# 006: Fit the reduced TS1 model and export it to jaxley

**Status:** done
**Design sections:** [Goal](../design/README.md#goal) (reference vs reduced
model), [Reduction](../design/reduction.md), open questions 6–8 in [open-questions.md](../design/open-questions.md)

## Goal

Turn plan 005's tree into a fitted reduced model and run it in jaxley:
1. **fit** branch resistances and membrane areas to the FEM port matrices
   at DC, 100 Hz and 1 kHz by gradient descent (JAX);
2. **export** the fitted tree as a jaxley `Cell` and verify that jaxley
   reproduces it;
3. repeat for smaller trees to get a **fitted** error-vs-size curve.

## Approach

- **Fitting engine:** a small differentiable port-impedance solver in JAX
  for tree models (dense, at most about 50 nodes). It is exact for the
  frequency-domain targets. jaxley simulates in the time domain and has no
  impedance output, so fitting there would need time-domain FEM targets,
  which we don't have yet.
- **Parameters:**
  - log branch resistances;
  - membrane areas as total × softmax, which conserves total membrane
    exactly (it matters for later channel densities).
- **Loss:** relative error of the neck-clamped port matrix (the spine's
  internal structure), averaged over frequencies, plus a small term on the
  isolated matrix.
- **Export:** each tree node becomes a one-compartment jaxley branch,
  rooted at the neck.
  - A branch's `resistive_load_in` carries its edge resistance to the parent
    (load = R / (r_L × 10⁴)).
  - `resistive_load_out` is set near zero, so the branch point coincides
    electrically with the parent compartment.
  - Membrane is set through `area`, with a tiny area for branch points that
    carry no membrane.
- **Compartment counting:** each synapse keeps a leaf compartment (the
  default; see plan 005 notes).

## Steps

- [x] Prototype the jaxley mapping on a 5-node tree: DC agreement 0.03%.
- [x] `camovox/fit.py`: `port_impedance_jax`, `fit_tree_model`, `FitResult`
      (Adam via optax; membrane as total × softmax).
- [x] `camovox/jaxley_io.py`: `to_jaxley(tree, areas, params,
      root_shunt=...)` → `JaxleyModel`.
- [x] `circuit.simulate_current`: backward-Euler transients for any network,
      including the full FEM (about 7 s for 600 steps on TS1).
- [x] Fit the full TS1 tree and contracted trees, before vs after.
- [x] Notebook `04_ts1_fit_jaxley.py` (about 65 s).
- [x] Tests (`tests/test_fit.py`):
  - JAX impedance = camovox solver;
  - the fit recovers perturbed parameters with total membrane conserved;
  - the jaxley export matches the camovox transient (slow).

## Done when

- Notebook 04 shows fitted error against size, and a jaxley simulation of the
  fitted TS1 model that matches the camovox solver.

## Results (TS1)

- **Full tree (50 compartments):** clamped error 2.83% → 2.77% after
  fitting. More steps or larger steps don't change this, so it is a floor
  set by the loop-free tree shape: the electrical cost of dropping TS1's
  loops.
- **Merged, then fitted:** 40 compartments 5.7 → 3.9%; 33: 10.0 → 5.8%;
  26: 11.3 → 6.5%; 21: 12.0 → 7.2%; 16: 29.7 → 22.2%; 11: 39.9 → 23.7%.
- **EPSPs** (α current, 50 pA, τ 0.5 ms; neck tied to rest through 1 kΩ),
  fitted full tree against FEM:
  - AZ 16 (closest to neck): 0.386 vs 0.386 mV (−0.2%);
  - AZ 4 (most isolated): 1.526 vs 1.527 mV (−0.0%);
  - AZ 13 (other lobe): 1.146 vs 1.148 mV (−0.2%);
  - jaxley identical to camovox.

  The whole-matrix 2.8% is mostly synapse-to-synapse cross-talk.

## Notes

- **jaxley export gotcha:** `resistive_load_out` = 1e-12 loses precision
  (+2.5% with `jaxley.dhs`; `jax.sparse` blows up). 1e-6 (about 1 Ω) is
  exact. This is now the default `branchpoint_load`.
- jaxley's recorded traces include t = 0, so drop the first sample to align
  with `simulate_current`.
- Next: smarter small models. Pick which synapses share a compartment
  (`electrical_groups`), keep per-synapse series resistances, then fit.
  Notebook 03's bound suggests about 8 shared compartments could reach about
  1%. Then the other spines.
