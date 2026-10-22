# Decisions

A short log of design choices, new dependencies and changes of direction. Newest first.

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
