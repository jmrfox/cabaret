# Open questions, decisions and risks

Back to the [design index](README.md).

Numbered for reference; the numbers are stable, so plans and
`decisions.md` can cite them. "Proposed" means a recommendation exists but
has not been confirmed. Questions for outside experts are pinned separately
in [`../questions-for-experts.md`](../questions-for-experts.md).

## Method

1. **Partition field.** *Proposed:* partition with the parameter-free
   $\lambda \to \infty$ field, fit $\kappa$ by least squares to fields from
   several sources, and use the screened field at resting and
   high-conductance $\lambda$ as the 3D-region diagnostic
   ([reduction.md](reduction.md)). Needs confirmation.
2. **Fitting targets in 3D regions:** synapse sites only, synapses plus
   uniform coverage, or a weighted mix? The loop softens this, since coverage
   can be added where error appears.
3. **Soma:** BF's one compartment + access resistances as the default, or
   the port-based method like any other 3D region?
4. *Answered (2026-10-23):* **no sink for fitting.** Spines are isolated
   multiports with neck ports, and the parent is attached at grafting time.
   See [spines-and-cell.md](spines-and-cell.md).

   4b. *Answered (option a):* the neck port is the TS surface inside the
   cell's parent part or within 0.2 µm of it, within 3 µm of a neck point.
   See [spines-and-cell.md](spines-and-cell.md) and plan 004.

## Refinement loop

5. **Stopping and tolerance** *(partly answered: the output is a Pareto
   front of error against compartment count)*: which metrics define the
   error axis (worst-case or median relative error, which frequencies, which
   target behaviours)?
6. *Answered:* the loop fits by gradient every iteration. The tree or field
   reduction provides initialization and topology. Fitting uses cabaret's
   own JAX port solver, and jaxley is the export and simulation target
   ([reduction.md](reduction.md)).

## Representation and export

7. **What goes into SWC?** An effective edge has two numbers (κ and a share
   of membrane area), but SWC gives one radius per node, and positions fix
   the lengths. Options:
   - (a) geometric positions, radius matches κ, area scale factors in a
     sidecar;
   - (b) an effective length and radius that match both
     ($r = (\kappa A / 2\pi^2)^{1/3}$, $l = A/(2\pi r)$), moving nodes along
     the local direction;
   - (c) radius matches area, with per-segment r_L scaling in a sidecar.

   Leaning towards (b), with an exact sidecar always written. Is SWC the
   primary deliverable, or a compatibility export beside an exact graph
   format? (jaxley export sets area and loads directly, so it doesn't need
   SWC.)
8. **Loops in jaxley** *(probed 2026-10-23 with jaxley 0.14.0; now a
   fallback, since tree realization is the default)*:
   - membrane area is independent of geometry (`set("area")`), and axial
     coupling is per compartment, with gradients through both;
   - loops are **not** in the public API. A gap-junction-style synapse pair
     is too stiff (time constant about 1e-4 ms against dt = 0.025 ms).
     Appending rows to the private `cell._comp_edges`, re-running
     `_init_solvers()` and using `voltage_solver="jax.sparse"` reproduces a
     ring exactly (3e-13). The default `jaxley.dhs` solver assumes a tree.

   Remaining: is `jax.sparse` fast enough at population scale? Ask jaxley
   upstream for supported loop edges? cabaret would wrap the workaround in
   one function, with jaxley pinned until then.
9. **Compatibility with MASCAF:** reuse its SWC writer and cycle-header
   code, or reimplement the format?

## Meshing

10. **Fidelity tolerance:** fTetWild's envelope must sit well below the
    thinnest neck radius (0.06 µm). What maximum surface deviation is
    acceptable, and is local refinement near necks automatic?
11. **Voxel input surface:** which smoothing after marching cubes, and which
    voxel connectivity defines "connected"?
12. **Membrane mass:** *Proposed and in use:* lumped mass for the reference
    network, with consistent mass as a check (not built yet).

## Simulation

13. **Synapse models** *(deferred: biology comes after the core goals)*.
    Known: spine synapses are expected to be excitatory; the AMPA:NMDA ratio
    is unknown; inhibition was previously modelled as effective inputs to
    the sink/soma; ICx electrophysiology exists for later calibration. Open:
    AMPA-like fixed kinetics only, or NMDA with Mg²⁺ block too?
14. **Placeholder synaptic parameters:** which kinetics and $g_\mathrm{max}$,
    and is $g_\mathrm{max} \propto$ PSD area the default?

## Data, dependencies and interface

15. **Data** *(mostly answered)*: see [spines-and-cell.md](spines-and-cell.md)
    and `data/README.md`. Still open: synapse data for spines beyond the 7.
16. **The user's MCF skeleton code:** optional baseline or diagnostic?
17. **Interface:** library + notebooks only, or also a CLI for batch runs?

## Population goal

18. **Population morphology:** will each of the 100+ simulated neurons be a
    reconstruction with its own spines, or receive spines from a **library**
    of effective spine models? This decides whether cabaret fits per cell or
    builds a spine library.
19. **Target behaviours:** somatic output only, or also local spine voltages
    and synapse interactions? This sets the fitting targets and largely
    decides compartment count ([ports-and-complexity.md](ports-and-complexity.md)).
20. **Active conductances in spines** (NMDA, voltage-gated channels)? If
    yes, local fidelity matters more, and the fit must stay valid once
    channels are added. (Fitting conserves total membrane for this reason.)
21. **Negative couplings** (4.5–6.4% of |κ| on fTetWild meshes): tolerate
    them, require better-shaped meshes, or both with a diagnostic? So far
    the reductions work with them in place
    ([fem-reference.md](fem-reference.md)).

## Generative model

22. *Answered:* spine segmentation of `cell.obj` comes from neuromesh-cgal,
    whose outputs cabaret reads ([spines-and-cell.md](spines-and-cell.md)).
23. **Synapse data beyond the 7 spines:** if none, the generator's port
    statistics rest on TS tables plus paper-level counts.
24. **Generator scope:** spines only, or the whole cell (soma, dendrite
    branching, spine placement)? Dendrites are presumably lower-risk, since
    cable-like.

## Decided

- **Reference model:** tetrahedral FEM on the surface mesh. Voxels are an
  input format only.
- **Terminology:** "reference model" (FEM) and "reduced model" (jaxley).
- **Loops without loops:** tree realization is the default; literal loops
  are a fallback.
- **Iterative refinement** with fit / probe / locked-test site roles and
  per-iteration exports (proposal, not built).
- **PSD area:** `fixed(0.05 µm²)`, a literature-based guess; `lognormal`
  and `area_sweep` later for UQ. Real PSD areas will not be available
  (the DAB fill hid them).
- **Activation patterns:** a thin, data-first layer defined on sites and
  shared across models.
- **Package name:** `cabaret` (Cable Approximation by REsistor Trees), renamed
  from camovox. The PyPI/distribution name is `cabaret-neuro`, because
  `cabaret` is taken on PyPI.

## Known limits and risks

- **Tetrahedralization is the fragile step.** EM meshes can be non-manifold
  or self-intersecting, robust meshers approximate the surface, and too
  loose a tolerance can pinch necks shut. Fidelity diagnostics are required.
- **Element quality:** obtuse tetrahedra create negative couplings.
- **Binary voxel input** cannot resolve sub-voxel necks; those are flagged,
  not fixed.
- **A partition from a single field** is exact for one loading condition.
- **Loop-free floor:** a tree can't go below the error set by the spine's
  loops (TS1: about 2.8%).
- **Refinement-loop overfitting** to the probe distribution; the locked test
  set guards against it.
- **Woodbury cost** grows with the number of active patch nodes.
- **Placeholder defaults** (PSD area, kinetics, cluster sizes) could leak
  into conclusions. Outputs carry their scenario, and placeholders are
  labelled.
- **Placement statistics** on about 9 spines have limited power.
