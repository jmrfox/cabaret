# camovox: design

> This is a living document. Update it when the design changes, and log the reason in `decisions.md`.
> Status: **draft for discussion**. Sections marked *(proposal)* are not settled.

## Goal

camovox builds **effective cable models** of neurons from 3D reconstructions:
surface meshes, voxel segmentations or masks. An effective cable model need
not match the cell's geometry. It must match the cell's **electrotonic
behaviour**: input and transfer impedances, attenuation, time constants, and
responses to synaptic input, as a multicompartmental simulator computes them.

The reference for "behaviour" is a **3D finite-element (FEM) model** of the
cell on a tetrahedral mesh. camovox reduces that reference to a
`MorphologyGraph`: a cable model (a 3D graph with vertex radii) that allows
loops, so it can represent toric spines and other non-tree structures. The
reduction is **iterative**. Each candidate model is compared against the
reference, and the comparison decides where to sample and refine next.

Output: SWC (a valid tree, with the edges cut to break cycles recorded in
the header, following the MASCAF convention) plus an exact sidecar.
[jaxley](https://github.com/jaxleyverse/jaxley) is used to simulate and, being
differentiable, to fit models.

Users: computational neuroscientists who have EM/LM reconstructions and need
simulation-ready morphologies they can trust electrically. The first
application is the toric spines of barn owl space-specific neurons.

## Background

### Cable models in simulators

NEURON, Arbor, jaxley and similar simulators solve the cable equation on a
graph of compartments:

$$
c_\mu \frac{dV_\mu}{dt} = i_\mu + \frac{I^{\mathrm{ext}}_\mu}{A_\mu} + \sum_\nu g_{\mu\nu}\,(V_\nu - V_\mu)
$$

The only things the morphology contributes are **membrane area per
compartment** ($A_\mu$) and **axial conductance between compartments**
($g_{\mu\nu}$). An SWC file is a geometric way to encode those numbers. If two
morphologies give the same $\{A_\mu, g_{\mu\nu}\}$, a simulator cannot tell them
apart.

### What MASCAF taught us

MASCAF (Fox, Fischer, DeBello, Peña 2026; `design/mascaf_revision.pdf`)
skeletonizes the mesh by mean curvature flow and fits radii from mesh
cross-sections ($r = \sqrt{A/\pi}$). It handles loops by breaking cycles and
writing reconnection directives to the SWC header. Its validation is
geometric: volume and surface area against the mesh.

The problem: geometric fidelity is hard to achieve (centring, spurious
branches, overlap at junctions, choice of edge length) and is **not the
target**.

### What the voxel study found

The voxel study ("BF", `design/BF - Voxel-based compartmental models of toric
spines methods and validation.pdf`) built voxel finite-volume models and
validated them against analytic solutions and tetrahedral FEM (scikit-fem,
P1, TetGen). Relevant findings:

1. **Tetrahedral FEM worked well as ground truth.** It converged (coarsening
   from 850k to 532k tetrahedra changed spine-head input resistances by less
   than 0.4%), and it agreed with analytic solutions.
2. **Voxels fail at thin necks.** Necks of radius 0.06–0.13 µm are only 1–5
   voxels across even at h = 0.05 µm. Their cross-section is misjudged by
   about ±50%, and the error does not shrink steadily as h decreases.
   Rasterization also fuses nearby surfaces and cuts off necks thinner than h.
   FEM on the surface has neither problem.
3. **Reduced models work.** A soma collapsed to one compartment, plus access
   resistances measured in the 3D model, matched the full model to within
   0.1–0.6%.
4. **Neck-to-soma junction:** a spreading-resistance term $r_L/(4a)$
   (Maxwell/Holm) is needed.

BF also compares against MASCAF cables. **We are not using those comparisons
for now**: the user suspects mistakes in that evaluation. The theoretical
point stands on its own. Axial resistance $\int dl/A$ is dominated by the
narrowest stretches, so matching total area and volume does not guarantee
matching resistance. That is a reason to target electrical quantities
directly.

Conclusion: make the **tetrahedral FEM model** the reference, and reduce it
directly. Voxels are only an input format.

### Prior work on electrical reduction (to build on, not reinvent)

- **Morphoelectrotonic transform** (Zador, Agmon-Snir & Segev 1995):
  electrotonic distance as the natural coordinate for an effective model.
- **Impedance-matching reduction**: Wybo et al. 2021 (eLife, NEAT toolbox)
  fit reduced models to impedance matrices at chosen sites. Amsalem et al.
  2020 (neuron_reduce) preserve transfer impedance. Kellems et al. 2010 use
  balanced truncation and Krylov model-order reduction.
- **Rall equivalent cylinders and the 3/2 rule**: closed-form sanity checks.

### Ideas from other fields *(proposal)*

| Field | Idea | Use in camovox |
|---|---|---|
| Power systems / circuit theory | **Kron reduction** (Schur complement of the Laplacian). Exactly preserves effective resistances between the nodes kept (Dörfler & Bullo 2013) | Exact DC reduction of the FEM network onto ports |
| Spectral graph theory | **Effective-resistance sparsification** (Spielman & Srivastava 2011). Laplacian eigenmodes | Sparsify Kron-reduced networks. Match the slow passive modes (τ₀, τ₁, …) |
| Topological data analysis | **Reeb graphs and Mapper**: level sets of a scalar field, split into connected components, give a graph that keeps loops | Partition the tetrahedral mesh by level sets of an electrical potential. Branches and loops appear without skeletonization |
| Numerical linear algebra | **Randomized range finding** (Halko, Martinsson & Tropp 2011). Algebraic multigrid | Probe 3D regions to find their electrical rank. A cable model is a coarse level of the FEM operator |
| Reduced-basis methods | **Greedy sampling**: add the parameter or input sample where the current reduced model's error is largest | The refinement loop adds sites where the cable model is worst |
| Adaptive mesh refinement | **A posteriori error estimation and marking** (e.g. Dörfler marking: refine the smallest set of elements that carries a fixed fraction of the total error) | Decide which compartments to split, or which 3D regions to give more rank, each iteration |
| Digital rock physics | **Pore network extraction** (PoreSpy SNOW, OpenPNM): a network with conductances calibrated to direct simulation | The same idea in another field. The calibration logic transfers, while the voxel-based partition does not |
| Differentiable simulation | Gradient-based fitting through the simulator (jaxley, or our own solver in JAX) | Fine-tune effective parameters against FEM responses |

## Principles

1. **Electrical equivalence is the objective.** Geometric error is reported,
   not minimized.
2. **Tetrahedral FEM is the reference.** Every effective model is judged
   against a FEM solve of the same closed surface. FEM is itself checked
   against analytic solutions and by mesh refinement.
3. **Derive parameters from solved fields, not measured shapes.** Take
   conductances from flux divided by potential drop, not from radius
   estimates.
4. **Discretize in electrotonic units** in cable-like regions (a
   d_lambda-style rule). In 3D regions, size the model by its measured
   electrical rank.
5. **Loops are first-class.** Cycles are broken only at export.
6. **Report fidelity limits**: surface changes introduced by repair and
   tetrahedralization, and structures that the input (especially binary
   voxels) cannot resolve. Never return a silently wrong number.
7. **Effective with respect to declared sites.** Every model is built and
   validated against an explicit set of membrane **sites**, and says so in its
   metadata.
8. **Real data and sampled data are interchangeable.** Synapse tables from
   EM and synthetic samples produce the same `SiteSet` object.
9. **Iterate against the reference, and score once at the end.** The refine
   loop may use fit and probe sites freely. A **locked test set** is never
   seen by the loop and gives the final, honest score.

## Scope

**In scope:**

1. **Load** morphology (mesh, voxel array, segmented mask stack) into a
   **closed surface mesh**: repair it, or, for voxel input, extract it with
   marching cubes plus smoothing.
2. **Tetrahedralize** the surface and **build the reference FEM model**, with
   quality and fidelity diagnostics.
3. **Sites and scenarios**: load synapse tables (position, PSD area,
   presynaptic axon ID), or sample synthetic sites under named, reproducible
   scenarios. Sites are patches of surface triangles.
4. **Reduce** the reference to an effective `MorphologyGraph` (the hard
   problem; see below).
5. **Refine iteratively**: compare, sample new sites where the error is
   large, refine where needed, and reduce again until a tolerance or budget
   is met.
6. **Export** SWC with header metadata and cycle-reconnection directives
   (MASCAF-compatible), plus an exact sidecar. Intermediate results are
   exported per iteration.
7. **Simulate and compare**: FEM against the cable model (jaxley or the
   camovox solver), driven by shared sites and activation patterns. The
   membrane is passive, and synapses are conductance-based.

**Out of scope (for now):**

- Segmentation of raw intensity images (we take masks).
- Active-channel fitting. We match passive electrotonic structure. The user
  places active conductances on the resulting model.
- Extracellular or electrodiffusion effects (extracellular space grounded,
  Ohmic models).
- A voxel finite-volume solver. BF showed it agrees with FEM where it can
  resolve the geometry. It can be added later behind the same interface if
  a cross-check is wanted.

## Core concepts and data model

Units: lengths in µm, areas in µm², volumes in µm³. Geometric conductance
factors $\kappa$ in µm, so the axial conductance is $g = \kappa / r_L$.
Biophysical parameters ($r_L$, $c_m$, $g_\mathrm{pas}$) are kept **separate**
from morphology, so one model serves any parameter set.

| Object | What it is | Contents |
|---|---|---|
| surface mesh | the cell's membrane | `trimesh.Trimesh`, closed and manifold after repair; vertices in µm |
| `VoxelGrid` | input container only | `occupancy (I,J,K) bool`, `voxel_size (3,)` µm, `origin (3,)` µm. `to_surface()` gives a surface mesh (marching cubes + smoothing) |
| `TetMesh` | volume mesh | `points (N,3)` µm, `tets (M,4)`, boundary `faces (F,3)` mapped to surface triangles; quality and fidelity report |
| `CompartmentGraph` | any passive electrical network | node arrays `area` (µm², membrane), `position` (µm); sparse edges with `kappa` (µm). Provides $C$, $G_m$, $L$ for given parameters |
| reference model | the FEM model as a `CompartmentGraph` on mesh vertices | P1 stiffness gives the edges ($\kappa_{ij} = -r_L K_{ij}$); lumped boundary mass gives the vertex membrane area. Keeps a cached factorization. A consistent-mass variant is available for checking |
| `MorphologyGraph` | effective cable model, loops allowed | `networkx.Graph`. Nodes: `pos`, `radius`, `type`, `area`. Edges: `kappa`, `length`. Also `root`, `metadata`. It is also a `CompartmentGraph` |
| `Correspondence` | fine-to-coarse map | mesh vertex → compartment (or weights, in 3D regions). Used for synapse placement, diagnostics and colouring |
| `SiteSet` | membrane input/output sites | pandas table: `site_id`, `x,y,z`, `area` (PSD, µm²), `kind`, `axon_id`, `spine_id`, `source`, `role` (`fit` / `probe` / `test`), `iteration` added; provenance. Resolved into **patches** (surface triangles with area weights) |
| `Scenario` | reproducible recipe | samplers + parameters + seed (TOML). Covers sites and, optionally, activation |
| `SynapseModel` | synaptic kinetics | α ($\tau$) or double-exponential ($\tau_\mathrm{rise}, \tau_\mathrm{decay}$) in ms; $E_\mathrm{syn}$ in mV; $g_\mathrm{max}$ in nS, fixed or per µm² of PSD |
| `ActivationPattern` | which sites fire, when | table: `site_id`, `t` (ms), `weight`, plus a `SynapseModel`. Defined on sites, so it applies to every backend |
| `RefineHistory` | record of the loop | per iteration: model summary, sites added, errors per site and metric, refinements made, export paths |

One solver interface covers every `CompartmentGraph`, whether it is the FEM
reference or a MorphologyGraph. It provides DC, AC impedance Z(ω),
eigenmodes, and transients with conductance synapses on patches. This keeps
comparisons fair.

**A note on FEM as a network.** P1 stiffness entries are conductances only
when the mesh is well shaped. Obtuse dihedral angles produce positive
off-diagonals, which act as negative conductances. Kron reduction and the
"conductance from flux" logic assume positive couplings, so tetrahedralization
must meet a quality criterion, and the reference model checks it.

## Pipeline

```
load (mesh / voxels / masks)
  → closed surface mesh (repair, or marching cubes + smoothing)
  → tetrahedralize + fidelity and quality report
  → reference FEM model (factorized once)
  → initial sites (data and/or scenario) → split into fit / probe / locked test
  → ┌─ reduce → MorphologyGraph_k
    │  export temporary result k
    │  compare on fit + probe sites (FEM vs cable)
    │  informed update: new sites where error is high, refine where marked
    └─ repeat until tolerance, budget or stagnation
  → final score on the locked test set
  → export (SWC + sidecar + history)
```

## The reduction: FEM → effective MorphologyGraph *(proposal)*

### Cable-like regions: potential-field coarse-graining

1. **Root**: the soma (largest inscribed sphere, or given by the user), or a
   spine's neck base for an isolated spine.
2. **Field**: solve the reference with current injected at the root and
   withdrawn through the membrane in proportion to area. This is the
   $\lambda \to \infty$ limit, and it is parameter-free. The potential $\phi$
   measures distance from the root as current sees it. The screened variant
   depends only on $r_L\, g_\mathrm{pas}$.
3. **Partition (Reeb/Mapper)**: cut $\phi$ into bands and split each band into
   connected components on the vertex graph. Each component is a
   compartment, and adjacency gives the edges. Branches and loops come from
   the topology of the level sets. Band width follows an electrotonic rule
   with a maximum in µm.
4. **Effective parameters**: membrane area = Σ lumped vertex area (exact).
   Edge $\kappa = I_\mathrm{interface}\, r_L / \Delta\bar\phi$, from the stiffness
   couplings cut by the interface. A κ can instead be fitted to fluxes from
   several source fields (open question 1).
5. **Embedding**: node position = compartment centroid (inside the volume).
   Radius and length are derived from ($\kappa$, area). See open question 5.

### Truly 3D regions: port-based reduction

Toric spine heads, loops, junctions and the soma are where the interest
lies, and where a single field or a 1D cable is not enough. A small model can
only be exact on chosen **ports**.

Two facts make this tractable:
- **Spines are electrically small.** For a neck with $a = 0.1$ µm,
  $r_L = 100$ Ω·cm and $g_\mathrm{pas} = 2.6\times10^{-4}$ S/cm²,
  $\lambda_\mathrm{DC} \approx 140$ µm and $\lambda_\mathrm{AC}(1\,\mathrm{kHz}) \approx 40$ µm, while a spine is a few µm
  across. Current spread inside a spine is almost purely resistive, so DC
  Kron reduction onto ports should be close to exact across synaptic
  frequencies (to verify).
- **Sampling is cheap.** With the reference factorized once, each extra
  source costs one back-substitution.

Method:
1. **Classify regions** with the field-sensitivity diagnostic. Regions where
   $\kappa$ shifts by more than a few percent across source fields, or between
   resting and high-conductance $\lambda$, are 3D.
2. **Ports** = the region's sites plus its interfaces to neighbouring
   regions.
3. **Size from data**: the singular-value decay of responses to random
   current patterns over the region's membrane gives its electrical rank.
4. **Reduce**: Kron-reduce onto the ports, then sparsify into a small loopy
   network with $\kappa$ fitted to the sampled responses. In these regions the
   MorphologyGraph need not look like a cable.

A site is always a **patch with an area** (PSD area), because point sources
in 3D have divergent input resistance.

### Alternatives to keep in view

- **Pure Kron reduction plus sparsification** everywhere, onto sites, branch
  points and loop arms.
- **Hybrid soma**: one compartment with access resistances measured in FEM
  (BF's validated reduction).

## The refinement loop *(proposal)*

One reduction gives a first model. The loop then improves it where the
reference says it is wrong. This is greedy sampling as in reduced-basis
methods, combined with error-driven marking as in adaptive mesh refinement.

### Site roles

| Role | Used for | Changes during the loop? |
|---|---|---|
| `fit` | targets for the reduction and $\kappa$ fitting | grows: new sites are added each iteration |
| `probe` | measuring error to decide what to do next | resampled or extended each iteration |
| `test` | final score only | **locked** at the start, never seen by the loop |

Real synapses are split by axon between `fit` and `test`, so all synapses of
an axon stay on one side. Sampled sites fill the probe pool.

### One iteration

1. **Reduce** with the current fit sites → `MorphologyGraph_k`.
2. **Export temporary result** to the work directory (`iter_k/`: SWC, sidecar,
   errors). These are inspectable, and they let you resume or roll back.
3. **Compare** on fit and probe sites, for each metric: input and transfer
   impedance at a few frequencies, single-synapse EPSPs, and the summation
   index for patterns. Each comparison costs one FEM back-substitution plus a
   small cable solve.
4. **Build error maps**: error per site, interpolated over the membrane as
   an error field, and aggregated per compartment and per region.
5. **Informed update**:
   - **sample new sites** where error is high: the `weighted` sampler with the
     error field as its feature (greedy, with a minimum spacing so sites don't
     pile up);
   - **mark and refine** compartments carrying the largest share of the error
     (Dörfler-style marking): split bands, raise the rank of a 3D region, or
     reclassify a cable region as 3D;
   - **coarsen** compartments whose error is far below tolerance (optional,
     to keep the model small);
   - optionally run a **gradient refinement** of $\kappa$ and areas
     (jaxley/JAX).
6. **Stop** when the probe error is below tolerance, the compartment budget
   or iteration cap is reached, or the error stops improving.

Then score once on the locked `test` set, and export the final model with
its `RefineHistory`, tolerances and scenario in the metadata. A large gap
between probe and test error flags overfitting to the probe distribution.

### Modes

- `refine.run(...)` runs automatically to a stopping criterion.
- `refine.step()` runs one iteration and returns the temporary result and
  error maps. The user can inspect them and inject sites or constraints
  ("informed input") before the next step, for example new real synapses,
  regions to protect, or a tighter tolerance in one spine.

## Sites, sampling and scenarios *(proposal)*

Real synapse data (positions, PSD areas, presynaptic axon IDs, in the mesh
coordinate frame) exists but has to be requested. Until then, samplers stand
in. They serve **development**, since every site-dependent stage runs now and
switches to real data with no code change, and **users** exploring "what if"
scenarios.

### Loading real data

`load_sites(path, ref)` reads CSV or Parquet with columns `x, y, z` (µm),
`psd_area` (µm²), optional `axon_id`, `spine_id`, `site_id`. It snaps each
site to the surface and builds its patch. Sites farther than a tolerance
from the membrane are reported, not silently moved.

### Patches

A patch is a geodesic disk of surface triangles grown from the snapped point
until its area reaches the site's area (heat-method geodesics on the
surface). Current is injected or recorded in proportion to area. If the area
is smaller than the local triangles, the patch is flagged, and the surface
can be locally remeshed before tetrahedralization.

### Samplers

All samplers take `rng: np.random.Generator` and return a `SiteSet` with
provenance. They compose: a location sampler, an optional cluster (axon)
process, and an area sampler.

| Sampler | Locations | Typical scenario |
|---|---|---|
| `uniform` | area-weighted uniform over the membrane | null model; generic coverage; probe pool |
| `poisson_disk` | uniform with a minimum geodesic spacing | non-overlapping PSDs; even coverage |
| `weighted` | density ∝ $w(\text{features})$ | "synapses prefer thick heads / tips / near the neck"; **error-driven sampling in the loop** |
| `clustered` | Thomas/Neyman–Scott on the surface: axon parents, then synapses around them; synthetic `axon_id`s | many synapses per axon (e.g. 49 from 11) |
| `empirical` *(later)* | density learned from real synapses in feature space | transfer placement statistics to other spines |
| `landmarks` | deterministic: tips, branch points, neck bases, loop arms | standard ports |

**Feature fields**, per surface vertex. A user can write any weight
function over them:
- geometric: local thickness (shape-diameter function or inscribed sphere),
  mean curvature, geodesic distance to the neck base or root, region label;
- electrical, from the reference: local input resistance, transfer to the
  neck base and soma, electrotonic distance;
- loop-internal: the current error field.

**Area samplers** (PSD area, µm²):

| Sampler | Role |
|---|---|
| `fixed(area=0.05)` | **default placeholder**. 0.05 µm² is a mid-range order-of-magnitude value, not owl data |
| `lognormal(median, sigma)` | opt-in, with user-supplied parameters. No literature defaults are shipped |
| `morph_scaled(feature, ...)` | opt-in hypothesis scenario |
| `empirical` *(when data arrives)* | resample measured PSD areas |

Placeholders are labelled as such in code, output metadata and reports.

Why the area matters:
- **local input resistance**, through a spreading term $\approx r_L/(4a)$ with
  $a = \sqrt{A/\pi}$: about 4.4, 2.0 and 1.0 MΩ for 0.01, 0.05 and 0.2 µm² at
  $r_L = 100$ Ω·cm. That is a percent-level effect against spine-head local
  resistances of tens of MΩ;
- **surface resolution**: triangles must be small compared with the PSD;
- **packing**: how many synapses fit on one spine;
- **synaptic strength**, if $g_\mathrm{max} \propto$ area.

**`area_sweep`** recomputes site metrics over a range of areas (default
0.01–0.3 µm²) to measure how much conclusions depend on the placeholder.

### Scenarios

```toml
name = "clustered_axons"
seed = 7
[locations]
sampler = "clustered"
n_axons = 11
synapses_per_axon = { dist = "poisson", mean = 4.5 }
cluster_scale_um = 0.5
[areas]
sampler = "fixed"       # placeholder until PSD data arrives
area_um2 = 0.05
```

Built-in scenarios: `uniform`, `clustered_axons`, `distal_bias`,
`neck_bias`, `isolated_bias`, `landmarks`, `area_sweep`. The scenario and
seed go into every output's metadata.

### Synapse placement analysis

This asks whether morphology is related to synapse count and placement. It
is run on the **FEM reference**, never on a reduced model fitted to those
sites.
- Per-site electrical features: input resistance, transfer to the neck base
  and soma, cross-transfer between sites, loop arm.
- Compare against null `SiteSet`s on the **same spine** (uniform, or matched
  for area), using permutation tests.
- Axon-level questions: are same-axon synapses electrically closer than
  chance? Does synapse count scale with spine-level electrical properties?
- With about 9 spines, power is modest. Report effect sizes and intervals.
  Before the real data arrives, check detectability on planted-effect
  scenarios.

### Activation patterns, shared across backends

The same sites and the same `ActivationPattern` drive FEM and the cable
model. Conductance synapses sum nonlinearly (driving force, shunting), and
that regime matters most for toric spines.

| Pattern sampler | What fires |
|---|---|
| `single_site` | one site at a time (validation) |
| `axon_synchronous` | all synapses of an axon together; optional jitter and release probability |
| `poisson_axons` | each axon as a Poisson process |
| `random_subset` | n random sites at once |
| planted patterns | known structure, for testing analyses |

Synaptic kinetics and $g_\mathrm{max}$ defaults are labelled placeholders.

| Backend | Synapse placement |
|---|---|
| FEM reference | a boundary term over the patch's surface triangles |
| camovox solver (any `CompartmentGraph`) | through the `Correspondence`, by membrane share |
| jaxley | the same mapping, onto jaxley compartments |

FEM transients use backward Euler with one factorization. Synapses touch only
the patch rows, so each step is a low-rank (Woodbury) update. Readouts are
patch-averaged voltages at sites, the neck base and the soma. Metrics: peak,
time-to-peak, half-width, integral, and the **summation index** (pattern
response divided by the sum of single-site responses).

## API sketch *(proposal)*

```python
import camovox as cv

mesh = cv.load_mesh(path)  # or cv.load_voxels(p, voxel_size).to_surface()
mesh, rep = cv.repair(mesh)  # report: what changed
tets = cv.tetrahedralize(mesh, envelope=0.005, max_edge=0.2)  # µm; fidelity report
ref = cv.fem_model(tets)  # CompartmentGraph on vertices, factorized

rng = np.random.default_rng(0)
sites = cv.sample_sites(ref, cv.Scenario.builtin("clustered_axons"), rng=rng)
# later: sites = cv.load_sites("synapses.csv", ref)
sites = cv.assign_roles(sites, test_frac=0.2, by="axon", rng=rng)  # locks the test set

params = cv.Passive(r_L=100, c_m=1, g_pas=2.5e-5)
refiner = cv.Refiner(
    ref,
    sites,
    params=params,
    tolerance=0.05,
    budget=500,
    workdir=out_dir / "work",
    rng=rng,
)
result = refiner.step()  # one iteration: model, error maps, export
refiner.add_sites(more_sites)  # informed input
mg, history = refiner.run()  # to a stopping criterion

score = cv.evaluate(mg, ref, sites[sites.role == "test"], params)  # once
cv.write_swc(mg, out_dir / "cell.swc", metadata=history.metadata())
cell = cv.to_jaxley(mg)
```

Module layout (draft):

```
camovox/
  io/          load_mesh, load_voxels, load_sites, read_swc, write_swc
  surface.py   repair, marching cubes + smoothing, surface features, geodesics
  tet.py       tetrahedralize, quality and fidelity diagnostics
  circuit.py   CompartmentGraph, solver (DC, Z(ω), modes, transients + synapses)
  fem.py       P1 assembly, lumped/consistent membrane, reference model
  coarse.py    potential fields, Reeb partition, effective parameters, Correspondence
  ports.py     region classification, randomized probing, Kron reduction
  refine.py    Refiner, error maps, marking, stopping, RefineHistory
  sites.py     SiteSet, patches, samplers, Scenario, roles
  activation.py SynapseModel, ActivationPattern, pattern samplers
  placement.py synapse placement statistics
  morph.py     MorphologyGraph, embedding, cycle breaking
  sim.py       jaxley conversion, simulation, gradient refinement
  validate.py  analytic references, metrics, reports
```

## Existing libraries to build on

| Need | Package | Notes |
|---|---|---|
| Mesh I/O, queries | `trimesh` | load, containment, proximity, ray tests |
| Mesh repair | `pymeshfix` | self-intersections and holes (used by BF) |
| Voxels → surface | `scikit-image` | `measure.marching_cubes`; smoothing via trimesh (Taubin/Laplacian) |
| Tetrahedralization | fTetWild (e.g. via `pytetwild`) by default; TetGen optional | fTetWild is MPL-2.0 and robust to messy input, but approximates within an envelope. TetGen is AGPL and stays an optional extra |
| FEM assembly | `scikit-fem` | P1 stiffness and boundary mass (BSD). Or a thin numpy assembly |
| Sparse solves | `scipy.sparse`, `pyamg` | SuperLU / Cholesky; AMG-CG for large meshes |
| Surface geodesics | `potpourri3d` | heat method; patches, clustered sampling, features |
| Graphs | `networkx` | MorphologyGraph, cycle basis, partition adjacency |
| Randomized range finding | `scipy.sparse.linalg`, `sklearn.utils.extmath` | probing 3D regions |
| Site tables | `pandas` (+ `pyarrow`) | `SiteSet` I/O |
| Point processes on surfaces | thin custom code over numpy | check existing packages first |
| Scenario files | `tomllib` (stdlib), `tomli-w` | read and write TOML |
| Simulation, gradients | `jaxley`, `jax`, `optax` | cable sims and fitting |
| Reduction references | NEAT (Wybo) | comparison |
| Baseline | `mascaf` | geometric baseline; SWC header convention |
| SWC sanity checks (dev) | `navis` or `MorphIO` | parse written files |

Add each when its stage is built, and log it in `decisions.md`.

## Validation

Reference ladder:

1. **Analytic**: isopotential sphere ($R_\mathrm{in}$, $\tau$); sealed cylinder
   ($\cosh$ attenuation, $R_\mathrm{in} = r_a \lambda \coth(L/\lambda)$);
   ball-and-stick; Rall equivalent cylinders.
2. **FEM reference**: agrees with level 1, and converges under mesh
   refinement (global and local, at necks).
3. **Effective MorphologyGraph**: checked against level 2 on the **locked
   test set**, plus fresh uniform samples and `landmarks`.

Metrics: input and transfer resistance; |Z(f)| and phase for 1–1000 Hz; the
slowest passive time constants; single-synapse EPSP peak, time-to-peak and
half-width; multi-synapse pattern responses and the summation index. Each is
evaluated in at least two membrane states: resting
(2.5×10⁻⁵ S/cm²) and high-conductance (2.63×10⁻⁴ S/cm²).

Loop diagnostics: error against iteration and against compartment count,
the probe/test gap, and where sites were added.

Fidelity diagnostics: surface area, volume, genus and local deviation before
and after repair and tetrahedralization; thinnest-neck preservation; element
quality (no negative couplings).

Test shapes: sphere, cylinder, ball-and-stick, a spine on a sphere (stubby,
with a ring), a torus, the 9 toric spines from MASCAF
(`mascaf/data/mesh/processed`), and BF's reconstructed cell.

Baselines: MASCAF cable models (evaluated by us), and BF's published
voxel-model numbers as a consistency check.

## Open questions

Numbered for reference. "Proposed" means a recommendation exists but has not
been confirmed.

### Method

1. **Partition field.** *Proposed:* partition with the parameter-free
   $\lambda \to \infty$ field. Fit $\kappa$ by least squares to fields from several
   sources. Use the screened field at resting and high-conductance $\lambda$ as
   the 3D-region diagnostic. Needs confirmation.
2. **Fitting targets in 3D regions**: synapse sites only, synapses plus
   uniform coverage, or a weighted mix? The loop softens this, since coverage
   can be added where error appears.
3. **Soma**: BF's one compartment + access resistances as the default, or
   the port-based method like any other 3D region?
4. **Isolated spines**: the boundary condition at the neck base for the 9
   MASCAF spine meshes:
   - (a) a parameterized sink (one-compartment soma with $R_\mathrm{in}$, $C$);
   - (b) a semi-infinite cable;
   - (c) grafting onto a whole-cell model.

   The spine model is exported with a named port at its base either way.

### Refinement loop

5. **Stopping and budget**: which metric or metrics define "tolerance"
   (worst-case or median relative error, which frequencies), and how should
   model size trade against accuracy? Is a fixed compartment budget, or a
   Pareto front of error against size, more useful?
6. **What the loop may change**: only sites and partition, or also run
   gradient refinement of $\kappa$ and areas every iteration? Gradient steps are
   powerful but blur the "parameters from solved fields" principle.

### Representation and export

7. **What goes into SWC?** An effective edge has two numbers ($\kappa$ and a
   share of membrane area), but SWC gives one radius per node, and positions
   fix the lengths. Options:
   - (a) geometric positions, radius matches $\kappa$, area scale factors in a
     sidecar;
   - (b) an effective length and radius that match both
     ($r = (\kappa A / 2\pi^2)^{1/3}$, $l = A/(2\pi r)$), moving nodes along the
     local direction;
   - (c) radius matches area, with per-segment $r_L$ scaling in a sidecar.

   Leaning towards (b), with an exact sidecar always written. Since 3D
   regions produce small loopy networks, is SWC the primary deliverable or a
   compatibility export beside an exact graph format (JSON or NeuroML-like)?
8. **What can jaxley express?** Loops, per-edge conductances that don't follow
   from radii, and non-cable networks. Fallback: the camovox solver in JAX,
   which is also differentiable. **Check first**: this decides where fitting
   and comparison run.
9. **Compatibility with MASCAF**: reuse its SWC writer and cycle-header code,
   or reimplement the same format?

### Meshing

10. **Fidelity tolerance**: fTetWild's envelope must sit well below the
    thinnest neck radius (0.06 µm). What maximum surface deviation is
    acceptable, and is local refinement near necks automatic?
11. **Voxel input surface**: which smoothing after marching cubes (Taubin,
    constrained so it never moves more than half a voxel, ...), and which
    voxel connectivity defines "connected"?
12. **Membrane mass**: lumped (diagonal, a clean network) or consistent (more
    accurate transients)? *Proposed:* lumped for the reference network, with
    consistent mass as a check.

### Simulation

13. **Synapse models**: fixed-kinetics conductance (AMPA-like) only, or also
    NMDA with Mg²⁺ block (nonlinear at patch nodes, Newton steps)?
14. **Placeholder synaptic parameters**: which kinetics and $g_\mathrm{max}$, and
    whether $g_\mathrm{max} \propto$ PSD area is the default.

### Data, dependencies and interface

15. **Data access**: do we have BF's whole-cell mesh
    (`cell_wrapped_simplified_1cc.obj`), and when are the synapse tables
    expected?
16. **Your MCF skeleton code**: optional baseline or diagnostic?
17. **Interface**: library + notebooks only, or also a CLI for batch runs?
18. **Package name**: "cable morphologies from voxels" now describes one
    input type, not the method. Keep it?

### Decided (kept here for context)

- **Reference model**: tetrahedral FEM on the surface mesh. Voxels are an
  input format only.
- **Iterative refinement** with fit / probe / locked-test site roles and
  per-iteration temporary exports.
- **Placeholder PSD area**: `fixed(0.05 µm²)`, `lognormal` opt-in, and
  `area_sweep`. To do: check Sanculi et al. 2020 for toric-spine PSD sizes.
- **Activation patterns**: a thin, data-first layer defined on sites and
  shared across backends.

## Known limits and risks

- **Tetrahedralization is the fragile step.** EM meshes can be non-manifold
  or self-intersecting. Robust meshers approximate the surface, and too loose
  a tolerance can pinch necks shut. Fidelity diagnostics are required.
- **Element quality**: obtuse tetrahedra create negative couplings that break
  network assumptions.
- **Binary voxel input** cannot resolve sub-voxel necks. Meshing doesn't add
  information, so those structures are flagged, not fixed.
- **A partition from a single field** is exact for one loading condition. The
  loop and multi-source fits must cover others.
- **Refinement-loop overfitting** to the probe distribution. The locked test
  set and the probe/test gap guard against it.
- **Woodbury cost** grows with the number of active patch nodes. Hundreds of
  simultaneous synapses may need refactorization or an iterative solver.
- **Placeholder defaults** (PSD areas, kinetics, cluster sizes) could leak
  into conclusions. Outputs carry their scenario, and placeholders are
  labelled.
- **Placement statistics** on about 9 spines have limited power.
