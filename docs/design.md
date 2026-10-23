# camovox: design

> This is a living document. Update it when the design changes, and log the reason in `decisions.md`.
> Status: **draft for discussion**. Sections marked *(proposal)* are not settled.

## Goal

camovox builds **effective cable models** of neurons from 3D reconstructions:
surface meshes, voxel segmentations or masks. An effective cable model need
not match the cell's geometry. It must match the cell's **electrotonic
behaviour**: input and transfer impedances, attenuation, time constants, and
responses to synaptic input, as a multicompartmental simulator computes them.

**Terminology.** Two models appear throughout:
- the **reference model**: a 3D finite-element (FEM) model of the cell on a
  tetrahedral mesh. It is expensive, high-resolution and treated as ground
  truth;
- the **reduced model**: a [jaxley](https://github.com/jaxleyverse/jaxley)
  cable model fitted to the reference model. It is cheap, and it is the
  model that will actually be simulated, eventually in **populations of
  100+ neurons**, where FEM is not affordable.

camovox makes the reduced model reproduce the reference model's behaviour at
the lowest compartment count:

1. a physics-based reduction of the reference model gives the reduced
   model's topology and initial parameters;
2. gradient-based optimization through jaxley (JAX) fits the reduced model's
   parameters to reference-model targets;
3. an iterative loop compares the two and refines where the reduced model
   is wrong.

The reduced model is a `MorphologyGraph`: a cable model (a 3D graph with
vertex radii), with loops allowed where needed, that must be expressible in
jaxley.

Output: the jaxley model plus SWC (a valid tree, with the edges cut to break
cycles recorded in the header, following the MASCAF convention) and an exact
sidecar. Because model size matters at population scale, the primary result
of a fit is a **Pareto front of error against compartment count**, not one
model.

Users: computational neuroscientists who have EM/LM reconstructions and need
simulation-ready morphologies they can trust electrically. The first
application is the toric spines of barn owl space-specific neurons.

## Background

### Project context

- High-resolution EM morphology exists for space-specific neurons of the
  barn owl auditory midbrain. Their **toric spines** are highly atypical:
  large, with multiple necks and topological holes. Standard cable-fitting
  tools failed on them.
- **MASCAF** made cable fitting robust, but it remains hard, and its models
  were validated only geometrically. No high-resolution simulation was ever
  used to check them. camovox adds that simulation (the FEM reference model).
- Synaptic input is rich, which is the reason for this effort: **TS1 has 25
  synapses from 10 presynaptic axons** and about 7 holes. With a single
  synapse per spine, a detailed model would hardly be needed. The per-axon
  synapse distribution will be supplied by the user. Until then, the input
  generator samples it.
- Facts from the original description (Sanculi et al. 2020, "Toric Spines at
  a Site of Learning", *eNeuro* 7(1), ENEURO.0197-19.2019,
  [PMC6944481](https://pmc.ncbi.nlm.nih.gov/articles/PMC6944481/)):
  - **Imaging:** SBEM at 5 nm × 5 nm × 70 nm.
  - **Spines:** 76 toric spines on one space-specific neuron (52 dendritic,
    21 somatic, 3 on the thin process). 37% have holes, up to 7 per spine,
    and larger spines have more holes.
  - **Size:** mean volume about 1.99 µm³.
  - **Active zones (AZs):** 1–49 per spine (622 in total), about
    0.81 AZ/µm² of spine surface.
  - **Inputs:** most spines receive several axons (24 of 27 analysed), up to
    11. Examples: TS1 has 10 axons and 25 AZs; TS7 has 10 axons and 23 AZs.
    The distribution of AZs across axons is **non-random** (bootstrap
    test), and most inputs are weak (mean input fraction 0.20).
  - **No PSD sizes** were measured (DAB obscured them), and there were no
    spine electrical measurements. Earlier compartmental simulations of the
    whole cell were "unsuccessful" for computational reasons.
- The earlier simulation suite used **Arbor**
  ([jmrfox/toric_spines_sim](https://github.com/jmrfox/toric_spines_sim)).
  camovox uses jaxley instead because:
  1. it is used in the user's current research group;
  2. Arbor code for full-control simulation tends to be long and quirky, and
     we want to see whether jaxley is more concise;
  3. JAX gives gradients, which suits fitting the reduced model to the reference model.
- **Open scientific question:** for a given accuracy, how many compartments,
  and in what configuration, does a cable model of a toric spine need to
  reproduce the FEM? The answer depends on which behaviour must be
  reproduced (see "Effective
  complexity of a toric spine").

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
| Phylogenetics / metric geometry | **Tree metrics**: the four-point condition (Buneman 1974); neighbor-joining (Saitou & Nei 1987) | Replace geometric loops by trees with membrane-less internal nodes that reproduce port resistances. A loopiness index measures when that fails |
| Network synthesis | **Y-Δ (star-mesh) transform**; **Cauer ladders** realize any RC one-port as a chain | Exact loop removal for three ports; chains suffice for somatic-only targets |
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
3. **Initialize from solved fields, then optimize against the reference model.** The
   reduction takes conductances from flux divided by potential drop, not from
   radius estimates. Gradient fitting through jaxley then tunes them to FEM
   targets. Membrane area per compartment stays physically meaningful,
   because active channel densities will later be placed on it.
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
10. **Cost is an objective.** Compartment count is traded explicitly against
    error (a Pareto front), because the reduced model will be simulated in
    populations.

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

## Effective complexity of a toric spine *(proposal)*

**The question:** given a target behaviour and a tolerance ε, what is the
smallest cable model that reproduces the FEM to within ε? That means both
its **compartment count** and its **configuration**: the topology (chain,
tree, which loops are kept), where each synapse attaches, and how the spine
joins its parent. This is a scientific result in its own right, and it sets
the reduced model's compartment budget.

The answer comes in two halves, which bracket the truth.

**Lower bound on the count, from the FEM alone** (no reduction needed):
1. Attach the spine to its parent through the neck-base boundary condition
   (open question 4).
2. Place sites: real synapses when available, otherwise a scenario. TS1 has
   25 synapses from 10 axons.
3. Solve FEM once per site (one back-substitution each). This gives the site
   impedance matrix $Z_{ij}(f)$ among all sites plus the neck base, at a few
   frequencies.
4. If every site attaches to one of m spine compartments, the reduced model's site
   impedances are $P Z_\mathrm{comp} P^\top$, with rank at most m (plus the base).
   So the best rank-m approximation error of the FEM $Z$ (Eckart–Young)
   **lower-bounds the error of any reduced model with m spine compartments**,
   whatever its configuration. The smallest m whose bound is below ε is a
   floor on the count.

**Achieved models, from reduction plus fitting** (the configuration):
5. Build candidate configurations: the field-based partition at several
   resolutions; variants with individual loops kept or pruned; and
   alternative assignments of synapses to compartments.
6. Fit each through jaxley against the FEM targets and record its error.
7. The best error at each count traces the **achieved front**. Where it sits
   close to the lower bound, configuration is not the limitation. A large gap
   means the cable topology itself is the bottleneck.
8. Build two achieved fronts: **tree-only** reduced models (see "Loops without
   loops") and **looped** reduced models. The gap between them is the electrical
   value of the loops.

Report, per spine and per target behaviour, the lower-bound curve, the
achieved front, and the configuration at each point:

| Target behaviour | What must match | Expected complexity |
|---|---|---|
| Somatic response to each synapse | $Z_{i,\mathrm{base}}$ | low: neck-dominated |
| Local voltage at each synapse (NMDA, plasticity) | $Z_{ii}$ | higher |
| Interactions between synapses (shunting, nonlinear summation, cross-talk between loop arms) | all of $Z_{ij}$, plus conductance-synapse transients | highest |

Physical intuition: spines are tiny compared with λ, so their membrane is
nearly isopotential in the cable-theory sense. But toric loops are thin tubes
whose axial resistance between synapse sites may rival the neck resistance.
That is where few-compartment models would fail and where loop topology could
matter functionally. This analysis makes the hypothesis testable.

## Loops without loops: tree realization *(proposal)*

The reduced model must reproduce the electrical relationships between the points
that matter, not the wiring. So a geometric loop does not have to become a
loop in the cable model.

**Why this can work.** Inside a spine, membrane leak is tiny compared with
the internal axial resistances (the spine is electrically small). The spine
behaves as a resistive network with membrane lumped onto it. Everything a
simulator sees at the **ports** (synapse patches and the neck base) is then
determined by the matrix of **resistance distances** between ports. That
holds even with nonlinear conductance synapses, because they sit at the
ports and everything between them is linear. Any network with the same port
resistances is indistinguishable at those ports.

**When a tree is exact:**
1. **A loop carrying no ports** is series-parallel, and it collapses exactly
   to a single edge.
2. **Any three ports** can always be joined by a tree: a star with one
   membrane-less internal node (the Y-Δ / star-mesh transform). jaxley
   branchpoints are exactly such membrane-less nodes.
3. **In general**, port resistances are realizable by a tree with internal
   (Steiner) nodes **if and only if** they satisfy the **four-point condition**
   (Buneman 1974). This is the same condition that decides whether
   distances fit a phylogenetic tree. Resistance distance is a metric (Klein
   & Randić 1993). A ring with four equally spaced ports violates the
   condition, so some loops cannot be replaced exactly, but the violation
   is measurable.
4. **Somatic output only** (one output port): any passive RC driving-point
   impedance is realizable as a **Cauer ladder**, which is a chain of
   compartments (classical network synthesis).

**Method:**
1. From the FEM, compute the port resistance-distance matrix: one solve per
   port, shared with the complexity analysis. Use internal resistances with
   membrane removed, plus the neck base as a port.
2. Compute the **loopiness index**: the normalized four-point violation over
   port quadruples, per spine and per loop region. It says how un-tree-like
   the spine is electrically, for these ports.
3. Fit the **best tree with internal nodes**: neighbor-joining (Saitou & Nei
   1987) as a start, then constrained least squares so branch lengths
   (resistances) are non-negative. Branch lengths become compartments, or
   half-resistances, in jaxley. Membrane is lumped onto port nodes in
   proportion to the FEM membrane nearest each port.
4. Check the tree-realized reduced model against FEM across frequencies and with
   conductance-synapse transients, then gradient-fit it like any other
   reduced model.

**Default policy:** use **tree realization** by default, and fall back to
literal loops (the jaxley private-API workaround, open question 8) only
where the loopiness index says a tree cannot meet ε. This keeps jaxley's
fast tree solver and its public API for most models.

**What it tells us scientifically:**
- If the loopiness index is small, the holes affect function only through
  the resistances they create, not through topology. For the given synapse
  positions, a loopless model reproduces the FEM within ε.
- If it is large, loops genuinely shape synaptic cross-talk in ways no tree
  can express.

**Caveats:**
- Real membrane is distributed along loop arms, so lumping it onto ports
  introduces a small error. The FEM check measures it.
- The tree is exact only for the **declared ports**. New synapse locations
  need a refit, consistent with principle 7. See "Dependence on synapse
  (PSD) locations" below.
- Active channels distributed in spine membrane break the "linear between
  ports" argument (open question 20).
- Neighbor-joining can return negative branch lengths, so they must be
  constrained.

### Dependence on synapse (PSD) locations

Tree realization, like any port-based reduction, **assumes the port
locations are known**: the ports are the PSD patches. It is exact only for
those ports. Synapses are expected to matter (the user's guess, and
plausible given thin loop arms). So sensitivity to placement is measured,
not assumed away. Three quantities, all computed on the reference model with
`SiteSet` samplers:

1. **Placement sensitivity of the spine itself.** How much do port
   impedances ($Z_{ii}$, $Z_{i,\mathrm{base}}$, $Z_{ij}$) and the loopiness index change
   when synapses move? Compare real positions against jittered positions
   (geodesic jitter of a given radius), against `uniform` re-draws with the
   same count and areas, and against `clustered` re-draws by axon. A large
   spread means electrical behaviour depends strongly on exactly where the
   synapses are.
2. **Transfer error of a fixed reduced model.** Fit a tree to one site set,
   then evaluate it on another (synapses mapped to the nearest port). This
   measures how wrong a model becomes when synapse positions are unknown or
   change, which is the situation for spines without synapse data.
3. **Port-basis convergence.** Instead of synapse ports, use a dense,
   placement-agnostic **port basis**: K evenly spread patches (Poisson-disk).
   Any synapse is then mapped to its nearest basis port. Error against K
   shows how many ports make a model placement-agnostic to within ε. This
   is the fallback when PSD locations are unknown.

These results also feed the generative model below. If placement matters
little, spines can be generated without modelling synapse positions in
detail. If it matters a lot, the generative model must produce ports and
synapses jointly.

## Toward a generative model of toric spines *(proposal, long-term)*

**Motivation.** The long-term network model needs many space-specific cells.
High-resolution morphology is scarce: one whole-cell EM mesh (`data/cell.obj`,
with all 76 toric spines) and the 9 separately processed spine meshes. More
imaging may come, but nothing else is likely to resolve spine-scale
structure. So the plan is a **generative model for space-specific cells**
that contains a **generative model for toric spines**. It is fitted on the
reconstructed spines and used to make fully synthetic cells.

**Key idea: generate electrical port models, not geometry.** If a spine's
role in a simulation reduces to its ports related by an impedance (or
resistance) matrix, then generating spines means generating **valid port
models**. Tree realization gives a compact, always-valid parameterization:

| Component | What is generated | Fitted from |
|---|---|---|
| Size | spine membrane area, volume | 76 spines in `cell.obj` (after spine segmentation) |
| Port count | number of synapses (AZs), given size | Sanculi et al.: AZs vs area r² = 0.73; about 0.81 AZ/µm² |
| Axon structure | axons per spine, synapses per axon | TS1 counts (`data/TS1_syn_axon_dist.txt`), paper statistics, synapse tables |
| Tree topology | the port tree's shape | trees fitted to reconstructed spines. Phylogenetic tree priors (Yule / coalescent-type processes) are a natural model family |
| Branch resistances, membrane per node | edge κ and node areas | fitted trees, conditioned on size and port count |
| Residual loopiness | literal loops where trees fail | loopiness index distribution |
| Neck | the base port's coupling to the parent | fitted neck resistance and area |

Why trees: any sampled tree with non-negative branch resistances and
positive membrane is a **physically valid** passive network. Sampling raw
impedance matrices instead would need constraints to stay realizable: a
symmetric reduced Laplacian with non-positive off-diagonals, or a metric
satisfying the four-point condition. Generating in tree space gets validity
for free, and the reduced model is jaxley-ready.

**Validation of the generator:** held-out reconstructed spines (leave one
out). Compare distributions of input resistance, spine-to-base transfer,
cross-talk between synapses, summation index and loopiness between synthetic
and real spines. Downstream, compare cell-level response distributions.

**Prerequisites in camovox:**
- a spine segmentation of the whole-cell mesh (the user has SDF-based
  partition code, to be integrated);
- the port-model export format for one spine: ports, tree, Z(f), metadata;
- placement-sensitivity results (above).

The generator itself is a later plan. The design choice now is that every
fitted spine is saved as a **port model** that the generator can learn from.

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
   - run **gradient fitting** of $\kappa$ and areas through jaxley/JAX against
     FEM targets on the fit sites. This is a core step every iteration, not an
     option.
6. **Stop** when the probe error is below tolerance, the compartment budget
   or iteration cap is reached, or the error stops improving.

Repeating the loop at several compartment budgets traces the **Pareto front**
of error against size. The user picks a point on it for population
simulations.

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

**Data available now** (see `data/README.md`). All files are in pixel
coordinates, at 0.005 µm per pixel.
- `TS*_AZ.txt`: active-zone (= PSD) points for 7 of the 9 spines (1 `x y z`
  per line). They sit 0.03–0.16 µm off the spine surface (presynaptic side),
  so the snapping tolerance must be about 0.2 µm. They carry no PSD areas,
  so the area sampler stays a placeholder. `load_az(path, ref, scale=0.005)`
  wraps `load_sites` for this format.
- `TS1_syn_axon_dist.txt`: AZ indices grouped by axon. `load_az` attaches
  `axon_id` from it when present.
- `TS*_neckpoint.txt`: one or more neck points per spine (TS3 has 3, TS21
  has 2). Each becomes a **base port** (`kind = "neck"`), and with several
  necks a spine has several base ports to its parent.

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

The **input generator** for multi-axon spines is the `clustered` sampler.
The number of synapses per axon can be given three ways:
- as a distribution (`poisson`, `negative_binomial`, ...);
- as an **explicit list of counts**, e.g. TS1's 25 synapses over 10 axons,
  once the user supplies the real distribution. Axon order and locations
  are then randomized;
- as an `empirical` distribution to resample from.

The total can be fixed, as in "exactly 25 synapses", by conditioning.

```toml
name = "clustered_axons"
seed = 7
[locations]
sampler = "clustered"
n_axons = 10
synapses_per_axon = { counts = [8, 4, 4, 2, 2, 1, 1, 1, 1, 1] }  # TS1 (data/TS1_syn_axon_dist.txt)
# synapses_per_axon = { dist = "poisson", mean = 2.5 }  # distribution form
total = 25
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
| Tree fitting from distances | `scikit-bio` (`nj`) or Biopython (`Bio.Phylo.TreeConstruction`); `scipy.optimize.nnls` | neighbor-joining start, then non-negative least-squares branch lengths |
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

   The spine model is exported with a named port at each neck either way.
   Spines with **several necks** (TS3: 3, TS21: 2, from the neck-point
   files) need a boundary condition that couples the necks through the
   parent, e.g. all necks attached to one shared sink, or to points along a
   parent cable. Which one is right depends on where the necks land on the
   real dendrite or soma. `cell.obj` can tell us.

### Refinement loop

5. **Stopping and tolerance** *(partly answered: the output is a Pareto
   front of error against compartment count)*: which metrics define the error
   axis (worst-case or median relative error, which frequencies, which target
   behaviours)?
6. *Answered:* the loop runs gradient fitting through jaxley every iteration.
   The field-based reduction provides initialization and topology.

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
8. **What can jaxley express?** *(probed 2026-10-23 with jaxley 0.14.0. The
   reduced model must be a jaxley model. Literal loops are now the **fallback**:
   tree realization is the default, see "Loops without loops".)*
   - **Membrane area is independent of geometry.** `set("area", ...)` changes
     membrane conductance and capacitance without touching axial coupling.
     Area and radius can therefore be decoupled, so open question 7's
     two-number problem is solvable inside jaxley.
   - **Axial coupling** comes from per-compartment `axial_resistivity` and
     half-compartment `resistive_load_in/out`. On a tree, per-compartment
     values can realise any per-edge κ.
   - **Gradients** flow through `area` and `axial_resistivity`
     (`make_trainable`, then `jax.grad`).
   - **Loops: not through the public API.**
     - A gap-junction-style synapse pair blows up: synaptic currents are
       explicit, and an axial-strength coupling is stiff (time constant
       about 1e-4 ms against dt = 0.025 ms).
     - Appending compartment-to-compartment rows to the private
       `cell._comp_edges`, re-running `_init_solvers()`, and integrating with
       `voltage_solver="jax.sparse"` reproduces a ring **exactly** (error
       3e-13 against the camovox solver).
     - The default `jaxley.dhs` solver assumes a tree.

   Remaining questions:
   - is `jax.sparse` fast enough at population scale (it is not the GPU tree
     solver)?
   - should we ask jaxley upstream for supported loop edges?
   - camovox wraps the private-API workaround in one function, and jaxley is
     pinned to `>=0.14,<0.15` until then.
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

15. **Data** *(partly answered; inventory in `data/README.md`)*:
    `data/cell.obj` (whole cell, 330k faces, likely BF's
    `cell_wrapped_simplified.obj`), the 9 TS meshes, and TS1's synapse-to-axon
    grouping are present. Still to confirm:
    - mesh units (5 nm pixels?) and provenance;
    - whether `cell.obj` is the version where soma and spines form one
      connected surface;
    - when the synapse tables (positions, PSD areas, axon IDs) are expected,
      and for which spines.
16. **Your MCF skeleton code**: optional baseline or diagnostic?
17. **Interface**: library + notebooks only, or also a CLI for batch runs?

### Population goal (new)

18. **Population morphology**: will each of the 100+ simulated neurons be a
    high-resolution reconstruction with its own spines? Or will population
    cells receive spines from a **library** of effective spine models fitted
    on the reconstructed ones? This decides whether camovox fits per cell or
    builds a spine library.
19. **Target behaviours**: somatic output only, or also local spine
    voltages and synapse interactions? This sets the fitting targets and
    largely decides compartment count.
20. **Active conductances in spines** (NMDA, voltage-gated channels) in the
    population models? If yes, local fidelity matters more, and the fit must
    stay valid once channels are added.
21. **Negative couplings** (found in plan 001: 4.5–6.4% of |κ| on fTetWild
    meshes): tolerate them in the reduction, require better-shaped meshes,
    or both, with a diagnostic? To be settled by experiment in the reduction
    plan.

### Generative model (new)

22. **Spine segmentation of `cell.obj`**: integrate the user's SDF-based
    partition code into camovox (as a dependency, or ported into
    `surface.py`)? Where does it live?
23. **Synapse data beyond TS1**: are synapse positions available for spines
    other than the 9 processed ones? If not, the generator's port statistics
    rest on TS1/TS7-type tables plus paper-level counts.
24. **Generator scope**: spines only, or the whole cell (soma, dendrite
    branching, spine placement along dendrites)? Dendrites are presumably
    lower-risk, since cable-like.

### Decided (kept here for context)

- **Reference model**: tetrahedral FEM on the surface mesh. Voxels are an
  input format only.
- **Iterative refinement** with fit / probe / locked-test site roles and
  per-iteration temporary exports.
- **Placeholder PSD area**: `fixed(0.05 µm²)`, `lognormal` opt-in, and
  `area_sweep`. Sanculi et al. 2020 did not measure PSD sizes (DAB obscured
  them), so real areas must come from the user's synapse tables.
- **Activation patterns**: a thin, data-first layer defined on sites and
  shared across backends.
- **Package name**: keep `camovox` during development. It may be renamed
  later, since voxels are now only one input type.

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
