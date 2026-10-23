# Toward a generative model of toric spines *(proposal, long-term)*

Back to the [design index](README.md).

## Motivation

The long-term network model needs many space-specific cells (100+ neurons).
High-resolution morphology is scarce: one whole-cell EM mesh
(`data/cell.obj`, with all 76 toric spines) and the 9 separately traced spine
meshes ([spines-and-cell.md](spines-and-cell.md)). More imaging may come, but
nothing else is likely to resolve spine-scale structure. So the plan is a
**generative model for space-specific cells** that contains a **generative
model for toric spines**, fitted on the reconstructed spines and used to make
fully synthetic cells.

## Generate port models, not geometry

If a spine's role in a simulation reduces to its ports related by an
impedance (or resistance) matrix, then generating spines means generating
**valid port models**. Tree realization ([reduction.md](reduction.md)) gives
a compact, always-valid parameterization:

| Component | What is generated | Fitted from |
|---|---|---|
| Size | spine membrane area, volume | 76 spines in `cell.obj` (neuromesh-cgal parts) |
| Port count | number of synapses (AZs), given size | Sanculi et al.: AZs vs area r² = 0.73; about 0.81 AZ/µm² |
| Axon structure | axons per spine, synapses per axon | TS1 counts (`data/TS1_syn_axon_dist.txt`), paper statistics, synapse tables |
| Tree topology | the port tree's shape | trees fitted to reconstructed spines. Phylogenetic tree priors (Yule / coalescent-type processes) are a natural model family |
| Branch resistances, membrane per node | edge resistances and node areas | fitted trees, conditioned on size and port count |
| Residual loopiness | literal loops where trees fail | loopiness index distribution |
| Neck | the neck port's coupling | fitted neck resistance and area |

**Why trees:** any sampled tree with non-negative branch resistances and
positive membrane is a **physically valid** passive network. Sampling raw
impedance matrices instead would need constraints to stay realizable (a
symmetric reduced Laplacian with non-positive off-diagonals, or a metric
satisfying the four-point condition). Generating in tree space gets validity
for free, and the result is jaxley-ready.

TS1's structure (two electrically separate lobes, each fed by its own axons;
[ports-and-complexity.md](ports-and-complexity.md)) hints that lobe count may
be a useful generator variable.

## Validation of the generator

Leave-one-out over reconstructed spines. Compare distributions of input
resistance, spine-to-neck transfer, cross-talk between synapses, summation
index and loopiness between synthetic and real spines. Downstream, compare
cell-level response distributions.

## Prerequisites in camovox

- spine segmentation of the whole-cell mesh: **available** (neuromesh-cgal
  parts, read by `graft.load_parts`);
- a port-model export format for one spine (ports, tree, Z(f), metadata):
  not yet defined. `PortTree` and `FitResult` hold the content;
- placement-sensitivity results ([reduction.md](reduction.md), "Dependence
  on synapse locations"): partly done for TS1.

The generator itself is a later plan. The design choice now is that every
fitted spine is saved as a **port model** that the generator can learn from.
