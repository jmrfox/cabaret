# 005: First reduced model of TS1 (tree realization)

**Status:** done
**Design sections:** [Reduction](../design/reduction.md),
[Ports and complexity](../design/ports-and-complexity.md)

## Goal

Build a reduced passive model of TS1, a tree of resistors with membrane on
its nodes, from the FEM reference model. Measure how well it reproduces the
FEM port behaviour across frequencies, and trace error against model size.
Everything runs in camovox's own solver. Export to jaxley and gradient
fitting are plan 006.

## Steps

- [x] `ports.diag_lowrank_bound(z)`: a lower bound with a private series
      resistance per port (D) plus m shared compartments (L Lᵀ), from
      alternating principal-axis factoring. It estimates a local optimum and
      is not certified.
- [x] `camovox/reduce.py`:
  - `PortTree`;
  - `fit_tree`: Biopython neighbor-joining topology, then `scipy.optimize.nnls`
    branch resistances over all port pairs;
  - `contract_tree`: merge branches below a threshold;
  - `assign_membrane`: fingerprint matching with the neck held at rest;
  - `tree_to_graph`: κ = r_L / (R · 1e-4), so resistances survive solving
    with the same r_L;
  - `port_matrix_error`.
- [x] `graft.load_spine` / `TSSpine`: notebook 02's setup in one call (one
      electrode per neck).
- [x] `viz.plot_port_tree`: radial layout, radius = resistance from the
      root. `viz.plot_surface(categories=True)`.
- [x] Notebook `03_ts1_reduced_model.py` (about 30 s).
- [x] Tests (`tests/test_reduce.py`):
  - the tree fit recovers an exact tree metric;
  - contraction reduces node count;
  - the diagonal + low-rank bound is about 0 for D + rank 2, and never above
    the rank bound.

## Done when

- Notebook 03 shows a reduced TS1 tree, its error against the FEM at three
  frequencies, and an error-vs-size curve. **Met.**

## Results (TS1, high-conductance membrane, PSD placeholder 0.05 µm²)

- **Tree:** 50 nodes (26 ports + 24 branch points). It reproduces the
  port-to-port resistances to 1.35%. Right at the neck it splits into one
  subtree per lobe.
- **Membrane** from fingerprint assignment: 8.8 µm² on the neck node, and
  0.17–6.07 µm² per synapse node (71.3 µm² total, conserved).
- **Against the FEM:**
  - neck-clamped port matrix error **2.83%** at DC, 100 Hz and 1 kHz;
  - isolated-spine error 0.01–0.11%;
  - spine-local input resistance error per synapse: median 0.29%, maximum
    7.6% (AZ 7).
- **Merging short branches:** 40 nodes give 5.7%, 26 nodes 11%, 21 nodes
  12%. That is crude.
- **Diagonal + low-rank bound** (neck clamped, DC): m = 2: 18%, 4: 9.6%,
  8: 1.3%, 12: 0.5%. The rank bound at m = 8 is 8%. So about 8 shared
  compartments plus a private series resistance per synapse could reach
  about 1%.

## Notes

- **Open modelling question for plan 006:** how to count "compartments".
  - A per-synapse private resistance can be a leaf compartment (as now), or
    a series resistance inside the synapse model (no compartment).
  - The second makes models of about 8–12 compartments plausible for 1%.
- Branch merging is far from optimal. Better routes:
  - choose which synapses share a compartment (e.g. `electrical_groups` at
    k groups, plus private leaf resistors);
  - then fit resistances and membrane by gradient descent against FEM
    targets.
- Evaluation is frequency-domain Z(f), which fully characterizes a linear
  passive network. Conductance-synapse transients need a FEM transient
  solver (not yet built).
