# 003: TS1 electrical portrait

**Status:** done (first pass; follow-ups below)
**Design sections:** [Ports and complexity](../design/ports-and-complexity.md),
[Reduction](../design/reduction.md),
[Sites and scenarios](../design/sites-and-scenarios.md)

## Goal

Build the FEM reference model of TS1 with its 25 real active zones (AZs) and
its neck. Compute the port impedance matrix and derive from it:
- the **rank lower bound** on compartment count;
- the **loopiness index**: how far the ports are from a tree metric;
- **placement sensitivity**: real AZ positions against uniform re-draws.

## Steps

- [x] Mesh TS1 (0.005 µm/px): 134,946 tets, 34,128 nodes, area −0.80%,
      volume −0.14%, 4.8% of |κ| on negative couplings. Edge 0.2 µm, envelope
      0.005 µm, about 2 min. Cached with `tetrahedralize_cached`, then 2–3 s.
- [x] `io.load_points`, `io.load_axon_groups`.
- [x] `sites.py`: `MembraneSurface` (snap, geodesic patches via Dijkstra on
      surface edges), `Sites`, `sites_from_points`, `sample_uniform`.
- [x] `circuit.py`: `g_ground`/`c_ground` shunts, `attach_sink`,
      `port_impedance`.
- [x] `ports.py`: `resistance_distances`, `four_point_violation`,
      `loopiness_index`, `clamp_port` (Schur complement), `rank_bound`.
- [x] Notebook `notebooks/02_ts1_portrait.py` (about 27 s with the cached mesh).
- [x] Tests (`tests/test_ports.py`): resistance distances on a star equal
      path sums (loopiness 0); a 4-port ring gives 0.25; rank-bound
      endpoints.

## Done when

- The notebook runs and reports, for TS1, the rank-bound curve, the
  loopiness index for the real AZs, and its spread over uniform re-draws.
  **Met.**

## Results (TS1, high-conductance membrane, PLACEHOLDER parent 80 MΩ / 100 pF, PSD area 0.05 µm²)

- AZs snap 0.025–0.100 µm onto the membrane.
- Input resistance at AZs is 87–111 MΩ. Spine-local resistance (neck
  clamped) is 8.7–32.5 MΩ, a 4× spread across the 25 synapses.
- **Loopiness is very low.**
  - Real AZs: median 0.0, 90th percentile 0.011, maximum 0.098.
  - 20 uniform re-draws of 25 sites: median about 1e-4, 90th percentile
    0.015 (0.009–0.039), maximum 0.136 (0.108–0.180).

  The real AZs are slightly more tree-like than random sites. Electrically,
  TS1's ports are close to a tree metric despite about 7 holes, which
  supports tree realization.
- **The rank bound is high** (neck clamped, DC): 10% error needs rank ≥ 6,
  5% needs ≥ 12, and 1% needs ≥ 24 of 25.

## Revision: stronger nulls and two violation measures (2026-10-23)

- `circuit.PortSolver` and `ports.ResistanceMetric` factorize once, so each
  null draw costs only back-substitutions (100 uniform draws in about 12 s).
- `sites.jitter_sites` moves real AZs within a geodesic radius.
- `ports.four_point_violation(d, normalize=...)`:
  - `"max"`: relative, (a−b)/a, the accuracy a tree must accept;
  - `"gap"`: intrinsic, (a−b)/(a−c), invariant to per-port leaf resistance;
  - `"none"`: absolute gap a−b.

  Patch spreading resistance dilutes the relative measure: the torus demo
  gives 0.159 relative but 0.996 intrinsic for four evenly spaced ports.
- **Revised TS1 interpretation:**
  - The holes create real loop structure (intrinsic p90 0.21, max 0.91),
    but it is electrically weak: absolute gaps are at most 4.5 MΩ against
    median port distances of 28 MΩ. A tree should reproduce the distances to
    about 1% typically and about 10% for the worst quadruple.
  - Against 100 uniform draws, real AZs sit at the 8th–15th percentile
    (relative) and the 20th–34th (intrinsic mean and p90). The earlier
    "almost all random draws are higher" was 20-draw noise.
  - Jitter percentiles are mostly 13–70%: not a fine-tuned optimum.
  - **Lead:** the intrinsic *max* sits at the 4th percentile of uniform
    draws and 0–13 under jitter. The real AZs avoid four ports evenly spread
    around one hole. This is fragile (a maximum of an unstable ratio). It
    needs a matched null (geodesic spacing, per-axon clustering) and the
    other spines.
- Notebooks 01 and 02 now explain each section and how to read it.

## Revision: two lobes, and notebooks for collaborators (2026-10-23)

- **TS1 has two electrical lobes.** `ports.coupling_coefficients`
  (Z_ij/√(Z_ii Z_jj), neck clamped) and `ports.electrical_groups`
  (average-linkage clustering) split the AZs into AZs 0–12 and 13–24.
  - Coupling within a lobe: median 0.45 (min 0.21). Across lobes: median
    0.05 (max 0.12).
  - **No axon has synapses in both lobes**: axons {1, 3, 4, 6, 7} vs
    {0, 2, 5, 8, 9}.
  - A natural reduced-model starting point is the neck plus one sub-tree
    per lobe.
- The most loop-like quadruple is the neck plus AZs 6, 11 and 12, all in
  lobe 0.
- Notebooks 01 and 02 were revised for collaborators:
  - a glossary;
  - a single parameter table with sources and placeholders;
  - descriptive variable names, and every number and choice explained;
  - "what to look for" for each figure;
  - summaries.
- Notebooks are exempt from ruff E501, because markdown tables in comments
  can't wrap. Code cells are still formatted by `ruff format`.

## Notes and follow-ups

- **The rank bound is inflated by per-synapse self-terms.** Each patch has its
  own local spreading resistance (about r_L/(4a), roughly 2 MΩ at 0.05 µm²).
  This adds a nearly diagonal, full-rank component. A tree captures it
  cheaply with one leaf edge per synapse, and so does a per-synapse access
  resistance. A more meaningful bound is **diagonal + low rank**
  (factor-analysis style, Z ≈ D + LLᵀ). Next step: add `ports.diag_lowrank_bound`
  and compare. This also bears on how compartments are counted (leaf edges
  vs compartments).
- The loopiness comparison uses `sample_uniform` with Euclidean spacing
  (placeholder) and a 1 µm exclusion around the neck.
- **Performance gotcha:** PyVista `threshold` on the TS1 surface took about
  400 s. `viz.plot_sites` now selects by point index instead (about 1 s).
  One full run once took 5 minutes. A rerun took 27 s, so the slow run
  looks like one-off cache warm-up.
- `ruff format` also formats `.ipynb` files. After formatting, re-run
  `jupytext --sync` and check that both files carry the change.
- The parent sink values are placeholders (open question 4). Clamped results
  and loopiness don't depend on them. Transfer ratios do.
