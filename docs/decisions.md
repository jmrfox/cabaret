# Decisions

A short log of design choices, new dependencies and changes of direction. Newest first.

## 2026-10-23: Fit in our own JAX solver, export to jaxley (plan 006)
**Context:** The reduced model must run in jaxley, and it has to be fitted
to FEM targets in the frequency domain (port impedance at DC, 100 Hz and
1 kHz). jaxley simulates in time and has no impedance output, and there are
no FEM time-domain targets beyond what we generate ourselves.
**Decision:**
- Fit with a small dense differentiable port-impedance solver in JAX
  (`camovox.fit`). Add `optax` for Adam.
- Export the fitted tree to jaxley (`camovox.jaxley_io`): one compartment
  per node; edge resistance in `resistive_load_in`; branch-point load 1e-6;
  membrane via `area`.
- Verify the export against camovox transients (`simulate_current`).

Rejected: fitting through jaxley's time-domain simulation (slower; needs
transient targets, and frequency-domain targets are complete for linear
networks).

## 2026-10-23: PSD area is a literature-based guess; biology deferred
**Context:** Real PSD areas won't be available (the DAB fill hid them). The
user chose not to ask the experts for now, and to keep the focus on the core
modelling goals.
**Decision:**
- Keep 0.05 µm² per synapse patch as a literature-based guess (mammalian
  spine synapses average about 0.04 µm²; Schikorski & Stevens 1997).
- Do uncertainty quantification later (`lognormal`, `area_sweep`).
- Expert questions are pinned in `docs/questions-for-experts.md`.
- Synapse biology (AMPA/NMDA, inhibition, ephys calibration) is deferred.
- Recorded: neck points are approximate.

## 2026-10-23: Tree fitting with Biopython neighbor-joining + NNLS (plan 005)
**Context:** Tree realization needs a tree topology from port distances,
plus non-negative branch resistances.
**Decision:**
- Add `biopython` for `DistanceTreeConstructor().nj` (topology only). It is
  pure Python and standard.
- Refit branch resistances with `scipy.optimize.nnls` over all port pairs,
  because neighbor-joining can return negative branch lengths.

Rejected: scikit-bio (heavier compiled dependencies for the same algorithm);
hand-written neighbor-joining.

## 2026-10-23: Neck ports from contact with the cell mesh
**Context:** The TS meshes have no cut face at their neck points: the
points sit on the spine surface, and plane cross-sections were unstable.
The user chose to define the attachment from the whole-cell mesh.
**Decision:**
- The neck port is the TS spine region inside a parent part of the
  segmented cell, or within 0.2 µm of one (tracing-mismatch tolerance),
  within 3 µm of a neck point.
- Its FEM nodes are tied into one electrode (`add_electrode`, couplings of
  1e4 µm), with their membrane removed.
- Spine parts are those at least 50% covered by the TS mesh. Vertex-vote
  rules misclassified parents.
- Added `rtree` (trimesh closest-point and polygon queries).

Rejected: plane cuts through neck points (unstable); a placeholder neck patch
(it produced a spurious "intrinsic maximum" effect in plan 003); `shapely`
and `mapbox-earcut` (only needed for capping plane cuts).

## 2026-10-23: No sink; isolated-spine multiports; neuromesh outputs
**Context:** The Arbor suite attached a large sink to each spine so the
sink's voltage could be read under many inputs. camovox's goal is a reduced
spine that behaves like the real one wherever it is attached. The TS
meshes were traced separately from `cell.obj`. The user's neuromesh-cgal
(C++/CGAL) already segments the cell into 91 capped, watertight parts.
**Decision:**
- Spines are fitted as **isolated multiports**: synapse ports plus neck
  port(s) at the cut face, with the cap membrane removed. There is no sink
  and no parent placeholder. Multiport composition makes this exact for any
  parent.
- The parent is attached at grafting time, with the junction spreading
  resistance on the parent side.
- In-situ validation swaps the cell's own version of a spine for the TS
  multiport at a neck cut.
- The two tracings of each spine give a tracing-variability measure.
- camovox reads neuromesh-cgal outputs (sidecars and part meshes), copied
  into `data/neuromesh/`, instead of reimplementing or wrapping the C++.

Rejected: keeping a parameterized sink, whose placeholder values leak into
results; fitting against the cell's coarser spine versions only;
reimplementing SDF segmentation in Python.

## 2026-10-23: Terminology, placement sensitivity, generative spine model
**Context:** The user asked for unambiguous terms instead of
"teacher/student". Tree realization assumes known PSD locations, which the
user expects to matter. High-resolution morphology is scarce (one whole cell
with 76 spines, plus 9 processed spines), and the long-term network model
needs synthetic cells.
**Decision:**
- Terminology: **reference model** (FEM) and **reduced model** (the jaxley
  cable model fitted to it), used throughout `design.md`. Earlier log
  entries are reworded to match.
- Placement dependence is measured, not assumed away, in three ways:
  impedance spread under jittered and re-drawn synapses; the transfer error
  of a fixed reduced model on new sites; and convergence of a
  placement-agnostic port basis.
- Every fitted spine is saved as a **port model** (ports, tree, Z(f),
  metadata), so that a later generative model can sample spines in tree
  space, which is valid by construction. The generator itself is deferred.

## 2026-10-23: Visualization stack (plan 002)
**Context:** FEM solutions on odd shapes like TS1 need to be seen, not just
summarized.
**Decision:** Use PyVista for 3D. It is already installed through
pytetwild, it is VTK-based, and it handles slices, isosurfaces, streamlines
and off-screen rendering. Add `matplotlib` for profiles and traces. Fields
export to `.vtu` for ParaView. Rejected: plotly (slow on large meshes, no
volume filters); matplotlib 3D (slow, poor depth sorting); napari
(voxel-oriented).

## 2026-10-23: Tree realization of loops by default
**Context:** jaxley loops need a private-API workaround and give up the fast
tree solver. But the goal is electrical equivalence, not geometry. Inside an
electrically small spine, behaviour at the ports is set by port resistance
distances. A tree with membrane-less internal nodes reproduces those exactly
when they satisfy the four-point condition, and always for three ports.
**Decision:** By default, represent loops by the best-fitting tree with
internal nodes (neighbor-joining, then non-negative least squares) on FEM
port resistance distances. A loopiness index (four-point violation) decides
where literal loops are needed. Those fall back to the extra-edge workaround.
The complexity analysis reports tree-only and looped fronts, and their gap
measures the electrical value of loops. Rejected: always using literal loops
(private API, slower solver); breaking cycles MASCAF-style without
compensation (loses the coupling).

## 2026-10-23: Add jaxley; close loops by extra compartment edges
**Context:** The reduced model must be a jaxley model, and toric spines need
loops. Probing jaxley 0.14.0 showed three things. Gap-junction synapses
can't carry axial-strength couplings (explicit, unstable). Extra
compartment edges solved with `jax.sparse` are exact. Membrane area can be
set independently of radius.
**Decision:** Add `jaxley` (pinned `>=0.14,<0.15`, since the loop workaround
touches the private `_comp_edges`). Loops are closed by appending
compartment-to-compartment edges and using the `jax.sparse` voltage solver.
This is wrapped in one camovox function so it can be swapped out if
jaxley adds official support. Rejected: gap-junction synapses (unstable);
our own JAX solver as the reduced-model simulator (it would lose jaxley's channel library
and community).

## 2026-10-23: FEM reference model and jaxley reduced model; jaxley is the target simulator
**Context:** The long-term goal is population simulations (100+ neurons) of
barn owl neurons with toric spines. FEM is too expensive at that scale. TS1
alone has 25 synapses from 10 axons, so spine structure plausibly matters,
but how much is unknown. The earlier Arbor suite (jmrfox/toric_spines_sim)
was long and quirky. jaxley is used in the user's group and gives gradients.
**Decision:** The FEM model is the reference, and a jaxley cable model is the reduced model fitted to it.
The field-based reduction provides topology and initialization. Gradient
fitting through jaxley against FEM targets is a core step of every loop
iteration (open question 6 answered). The output is a Pareto front of error
against compartment count. A dedicated "effective complexity" analysis
(rank of the FEM site-impedance matrix) measures how many compartments a
spine needs for each target behaviour. jaxley expressiveness becomes a hard
constraint. Rejected: parameters taken purely from solved fields with no
optimization; one fixed-size model per cell.

## 2026-10-21: Dependencies for the FEM reference (plan 001)
**Context:** Plan 001 needs mesh I/O, tetrahedralization, P1 assembly and
sparse solves.
**Decision:**
- `trimesh` for mesh I/O and test shapes;
- `scikit-fem` for P1 stiffness assembly (BSD), instead of hand-rolled
  assembly;
- `scipy` for sparse LU and ARPACK;
- `pytetwild` (fTetWild, MPL-2.0) for tetrahedralization, robust to messy
  input;
- `pyvista` only because pytetwild imports it unconditionally (it brings in
  vtk).

Rejected for now: TetGen (AGPL, fragile on bad input). Revisit making the
mesher an optional extra if the vtk dependency becomes a problem.

## 2026-10-21: Iterative compare-and-refine loop
**Context:** One reduction pass may be wrong in places only a comparison
reveals. The user proposed iterating: sample sites, reduce, export a
temporary result, compare, then use what was learned to sample and reduce
again.
**Decision:** Add a `Refiner`, combining greedy sampling (as in reduced-basis
methods) with error-driven marking (as in adaptive mesh refinement). Sites
have roles: `fit` (grows), `probe` (measures error) and `test` (locked at the
start, used once at the end). Each iteration exports to `iter_k/`. The loop
supports automatic `run()` and interactive `step()` with user-injected sites.
It stops on tolerance, budget or stagnation. Rejected: a single pass with no
feedback; scoring on the sites the loop selected (optimistic).

## 2026-10-21: Tetrahedral FEM as the reference; voxels only as input
**Context:** Voxels were chosen on the assumption that a 3D reference would
be easier on voxels. In fact, the toric-spine data is meshes, voxelization
loses thin necks (BF), and tetrahedral FEM was already BF's ground truth.
P1 FEM with lumped membrane mass is itself an electrical network on mesh
vertices.
**Decision:** The pipeline is load → closed surface mesh → tetrahedralize →
FEM reference (a `CompartmentGraph` on vertices) → reduce. Voxel input is
converted to a surface (marching cubes + smoothing). The voxel finite-volume
model is dropped from the core. Sites become patches of surface triangles.
fTetWild (MPL-2.0) is the default mesher, and TetGen (AGPL) is optional.
Rejected: keeping two reference models; voxel-only processing. New risks
(mesher fidelity at necks, element quality) are tracked in `design.md`.

## 2026-10-21: Activation patterns shared across backends
**Context:** Validation needs standard stimuli anyway. Toric-spine questions
(e.g. synchronous input from one axon) need conductance synapses, whose
summation is nonlinear. The strongest test of an effective model is the
same inputs driving the reference and the reduction.
**Decision:** Add a thin, data-first `ActivationPattern` (site, time, weight,
plus a `SynapseModel`), with samplers using `rng` and placeholder kinetics
labelled as such. Patterns are defined on sites and applied unchanged to
jaxley, the camovox solver, the voxel reference and FEM. Reference transients
reuse one factorization with low-rank (Woodbury) synapse updates. Rejected:
(a) leaving all stimulation to user code, since references and reductions
would then not share inputs; (c) a full protocol and analysis suite, which
is scope creep into a simulation framework.

## 2026-10-21: Placeholder PSD area
**Context:** Synthetic sites need an area until real PSD areas arrive. The
area affects local input resistance (spreading term), resolution, packing and,
possibly, synaptic strength.
**Decision:** The default is `fixed(0.05 µm²)`, labelled as a placeholder
everywhere. `lognormal` is opt-in with user-supplied parameters, and an
`area_sweep` scenario measures sensitivity. Rejected: shipping rodent
literature lognormal parameters as defaults, since toric spines may differ
and the defaults would look like data.

## 2026-10-21: Sites as first-class objects, with a sampling system
**Context:** The truly 3D parts (toric spine heads and loops) are the main
interest, and a small model there can only be exact with respect to chosen
inputs. Real synapse data (positions, PSD areas, presynaptic axon IDs, in the
mesh frame) exists but must be requested. Synapse placement may correlate
with morphology.
**Decision:** Add a `SiteSet` (membrane patches with area and labels) that
real data and samplers both produce. Samplers: uniform, Poisson-disk,
feature-weighted, clustered by axon, and landmarks, with an empirical sampler
later. Reproducible TOML `Scenario`s serve both development and user
exploration. 3D regions are reduced by port-based Kron reduction, with model
size chosen from randomized probing. Placement analysis runs on the reference
model against sampled null models, never on a reduced model fitted to the
same sites. Alternative rejected: waiting for real data before building the
site-dependent stages.

## 2026-10-21: Effective (electrical) cable models, not geometric fitting
**Context:** The user clarified the goal. camovox should produce cable models
that are electrotonically equivalent to the voxel data, not geometrically
close to it. A geometric route (skeleton + radius fitting) would rebuild
MASCAF. The BF voxel study showed that a voxel finite-volume model is an
accurate electrical reference for thick parts and fails at thin necks. BF's
comparisons against MASCAF are set aside: the user suspects errors in that
evaluation, and we will re-evaluate MASCAF ourselves.
**Decision:** This replaces the first draft below. TEASAR/kimimaro
skeletonization is dropped from the core. The voxel finite-volume model is
the electrical reference. The cable model (`MorphologyGraph`, loops allowed)
is obtained by coarse-graining that model: a potential-field partition, with
conductances derived from flux. jaxley is used for simulation, validation and
differentiable refinement. SWC export keeps MASCAF's cycle-breaking header
convention. The proposed method and alternatives (pore-network partition,
Kron reduction) are in `design.md`.

## 2026-10-21: First draft of the design (superseded)
**Context:** The design template was empty. The user described the pipeline
as load → voxelize → cable model → SWC.
**Decision:** Every input goes through a common `VoxelGrid` (binary volume
with voxel size and origin, in µm). The cable model is a `networkx` graph
with node radii. SWC writing is implemented in camovox, with navis/NeuroM
used only to validate the output. Skeletonization is delegated to `kimimaro`
(TEASAR), with `scikit-image` + `skan` as the simple fallback. Rejected for
now: mesh-native skeletonization (skeletor/CGAL), because it would bypass the
voxel stage that the other input types need.

<!--
## YYYY-MM-DD: <decision>
**Context:** why this came up.
**Decision:** what we chose, and which alternatives we rejected.
-->
