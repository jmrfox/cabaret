# 003: TS1 electrical portrait

**Status:** done (first pass; follow-ups below)
**Design sections:** [Effective complexity of a toric spine](../design.md#effective-complexity-of-a-toric-spine-proposal),
[Loops without loops](../design.md#loops-without-loops-tree-realization-proposal),
[Sites, sampling and scenarios](../design.md#sites-sampling-and-scenarios-proposal)

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
