# camovox design

> A living set of documents. Update the relevant chapter when the design
> changes, and log the reason in [`../decisions.md`](../decisions.md).
> Sections marked *(proposal)* are not settled. Sections marked
> *(implemented, plan NNN)* describe working code.

## Chapters

| Chapter | What it covers |
|---|---|
| [background.md](background.md) | Project context, cable models, MASCAF, the voxel study (BF), prior work, ideas from other fields |
| [fem-reference.md](fem-reference.md) | **The reference model**: physics, equations, FEM discretization, the network view, solvers, ports and electrodes, meshing, validation, limitations |
| [ports-and-complexity.md](ports-and-complexity.md) | Port impedance matrices, neck clamping, compartment-count bounds, loopiness (four-point condition), coupling and lobes |
| [reduction.md](reduction.md) | From the reference to a reduced model: tree realization, fitting, jaxley export, and the proposed field-based and port-based reductions |
| [refinement-loop.md](refinement-loop.md) | The proposed iterative compare-and-refine loop |
| [sites-and-scenarios.md](sites-and-scenarios.md) | Synapse sites, patches, samplers, PSD area, scenarios, placement analysis, activation patterns |
| [spines-and-cell.md](spines-and-cell.md) | The data, the two tracings of each spine, neck ports from the cell mesh, grafting |
| [generative-model.md](generative-model.md) | The long-term generative model of toric spines |
| [validation.md](validation.md) | The reference ladder, metrics and test shapes |
| [implementation.md](implementation.md) | Data model, module layout, API, libraries |
| [open-questions.md](open-questions.md) | Open questions (numbered), decided items, known limits and risks |

Pinned questions for the experts are in
[`../questions-for-experts.md`](../questions-for-experts.md).

## Goal

camovox builds **effective cable models** of neurons from 3D reconstructions:
surface meshes, voxel segmentations or masks. An effective cable model need
not match the cell's geometry. It must match the cell's **electrotonic
behaviour**: input and transfer impedances, attenuation, time constants and
responses to synaptic input, as a multicompartmental simulator computes them.

**Terminology.** Two models appear throughout:
- the **reference model**: a 3D finite-element (FEM) model of the cell's
  interior on a tetrahedral mesh ([fem-reference.md](fem-reference.md)). It
  is expensive, high-resolution, and treated as ground truth;
- the **reduced model**: a [jaxley](https://github.com/jaxleyverse/jaxley)
  cable model fitted to the reference model. It is cheap, and it is the
  model that will actually be simulated, eventually in **populations of
  100+ neurons**, where FEM is not affordable.

camovox makes the reduced model reproduce the reference model's behaviour at
the lowest compartment count:
1. a physics-based reduction of the reference model gives the reduced
   model's topology and initial parameters;
2. gradient-based optimization (JAX) fits the reduced model's parameters to
   reference-model targets;
3. an iterative loop compares the two and refines where the reduced model is
   wrong (proposal).

Because model size matters at population scale, the primary result of a fit
is a **Pareto front of error against compartment count**, not one model.

Users: computational neuroscientists who have EM/LM reconstructions and need
simulation-ready morphologies they can trust electrically. The first
application is the toric spines of barn owl space-specific neurons.

## Principles

1. **Electrical equivalence is the objective.** Geometric error is reported,
   not minimized.
2. **Tetrahedral FEM is the reference.** Every reduced model is judged
   against a FEM solve of the same closed surface. FEM is itself checked
   against analytic solutions and by mesh refinement.
3. **Initialize from solved fields, then optimize against the reference.**
   Parameters come from solved electrical quantities (resistances, fluxes),
   not from radius estimates, and are then fitted to FEM targets. Membrane
   area per compartment stays physically meaningful, because channel
   densities will later be placed on it.
4. **Discretize in electrotonic units** in cable-like regions (a
   d_lambda-style rule). In 3D regions, size the model by its measured
   electrical rank.
5. **Loops are first-class** in the analysis, but a loop-free tree is used
   wherever it is accurate enough ([reduction.md](reduction.md)).
6. **Report fidelity limits**: surface changes from meshing, and structures
   the input cannot resolve. Never return a silently wrong number.
7. **Effective with respect to declared sites.** Every model is built and
   validated against an explicit set of membrane **sites**, and says so in
   its metadata.
8. **Real data and sampled data are interchangeable.** Synapse tables from
   EM and synthetic samples produce the same site objects.
9. **Iterate against the reference, and score once at the end** on a locked
   test set.
10. **Cost is an objective.** Compartment count is traded explicitly against
    error.

## Scope

**In scope:**
1. **Load** morphology (mesh, voxel array, segmented masks) into a **closed
   surface mesh**.
2. **Tetrahedralize** it and **build the FEM reference model**, with quality
   and fidelity diagnostics.
3. **Sites and scenarios**: real synapse locations or sampled ones, as
   membrane patches.
4. **Reduce** the reference to a small cable model, and **fit** it.
5. **Refine iteratively** (proposal).
6. **Export** to jaxley, and to SWC with an exact sidecar (SWC is proposal).
7. **Simulate and compare** the reference and reduced models on shared
   inputs.

**Out of scope (for now):**
- segmentation of raw intensity images (we take masks);
- active-channel fitting. We match passive electrotonic structure; active
  conductances are placed on the result by the user;
- extracellular and electrodiffusion effects (extracellular space grounded,
  Ohmic cytoplasm);
- a voxel finite-volume solver. BF showed it agrees with FEM where it
  resolves the geometry. It could be added later behind the same interface.

## Pipeline and implementation status

```
load (mesh / voxels / masks)
  → closed surface mesh (repair, or marching cubes + smoothing)
  → tetrahedralize + fidelity and quality report
  → reference FEM model (factorized once)
  → sites (data and/or scenario), neck ports
  → port matrices, bounds, loopiness            (ports-and-complexity.md)
  → reduce (tree) → fit (JAX) → export (jaxley) (reduction.md)
  → refinement loop and locked-test score        (refinement-loop.md, proposal)
```

| Stage | Status | Plan |
|---|---|---|
| Mesh loading, tetrahedralization (fTetWild, cached), fidelity report | implemented | 001, 003 |
| FEM reference model, DC/AC/eigen solvers | implemented | 001 |
| Transient solver (backward Euler, current injection) | implemented | 006 |
| Visualization (PyVista, ParaView export) | implemented | 002 |
| Sites: AZ loading, geodesic patches, uniform and jitter samplers | implemented | 003 |
| Port matrices, clamping, rank and diagonal + low-rank bounds, loopiness, lobes | implemented | 003, 005 |
| Neck ports from contact with the segmented cell mesh | implemented | 004 |
| Tree realization, membrane assignment, branch merging | implemented | 005 |
| Fitting (JAX) and jaxley export | implemented | 006 |
| Surface repair, voxel input | not started | – |
| Other samplers, scenarios, activation patterns, conductance synapses | not started | – |
| Field-based partition, port-based (Kron) reduction of 3D regions | proposal | – |
| Refinement loop, SWC export | proposal | – |
| Tracing comparison, all-spine portraits, grafting validation | planned | 004 |
| Generative spine model | long-term | – |
