# Background

Back to the [design index](README.md).

## Project context

- High-resolution EM morphology exists for space-specific neurons of the
  barn owl auditory midbrain. Their **toric spines** are highly atypical:
  large, with multiple necks and topological holes. Standard cable-fitting
  tools failed on them.
- **MASCAF** made cable fitting robust, but it remains hard, and its models
  were validated only geometrically. No high-resolution simulation was ever
  used to check them. camovox adds that simulation (the FEM reference model).
- Synaptic input is rich, which is the reason for this effort: **TS1 has 25
  synapses from 10 presynaptic axons** and about 7 holes. With a single
  synapse per spine, a detailed model would hardly be needed.
- Facts from the original description (Sanculi et al. 2020, "Toric Spines at
  a Site of Learning", *eNeuro* 7(1), ENEURO.0197-19.2019,
  [PMC6944481](https://pmc.ncbi.nlm.nih.gov/articles/PMC6944481/)):
  - **Imaging:** SBEM at 5 nm × 5 nm × 70 nm. Only the postsynaptic neuron
    was DAB-filled (retrograde tracer from the optic tectum), so PSDs were
    obscured. Presynaptic axons were unlabeled.
  - **Spines:** 76 toric spines on one space-specific neuron (52 dendritic,
    21 somatic, 3 on the thin process). 37% have holes, up to 7 per spine,
    and larger spines have more holes.
  - **Size:** mean volume about 1.99 µm³.
  - **Active zones (AZs):** 1–49 per spine (622 in total), about 0.81 AZ/µm²
    of spine surface. AZs were scored from the presynaptic side (≥5
    vesicles, apposed membranes, seen in ≥2 sections).
  - **Inputs:** most spines receive several axons (24 of 27 analysed), up to
    11. Examples: TS1 has 10 axons and 25 AZs; TS7 has 10 axons and 23 AZs.
    The distribution of AZs across axons is **non-random** (bootstrap test),
    and most inputs are weak (mean input fraction 0.20).
  - **No PSD sizes** were measured, and there were no spine electrical
    measurements. Earlier compartmental simulations of the whole cell were
    "unsuccessful" for computational reasons.
- The earlier simulation suite used **Arbor**
  ([jmrfox/toric_spines_sim](https://github.com/jmrfox/toric_spines_sim)).
  camovox uses jaxley instead because:
  1. it is used in the user's current research group;
  2. Arbor code for full-control simulation tends to be long and quirky, and
     we want to see whether jaxley is more concise;
  3. JAX gives gradients, which suits fitting the reduced model to the
     reference model.
- **Open scientific question:** for a given accuracy, how many compartments,
  and in what configuration, does a cable model of a toric spine need to
  reproduce the FEM? The answer depends on which behaviour must be
  reproduced ([ports-and-complexity.md](ports-and-complexity.md)).

## Cable models in simulators

NEURON, Arbor, jaxley and similar simulators solve the cable equation on a
graph of compartments:

$$
c_\mu \frac{dV_\mu}{dt} = i_\mu + \frac{I^{\mathrm{ext}}_\mu}{A_\mu} + \sum_\nu g_{\mu\nu}\,(V_\nu - V_\mu)
$$

The only things the morphology contributes are **membrane area per
compartment** ($A_\mu$) and **axial conductance between compartments**
($g_{\mu\nu}$). An SWC file is a geometric way to encode those numbers. If two
morphologies give the same $\{A_\mu, g_{\mu\nu}\}$, a simulator cannot tell them
apart. The FEM reference model has exactly this form too, with one node per
mesh vertex ([fem-reference.md](fem-reference.md), "The FEM is a resistor
network").

## What MASCAF taught us

MASCAF (Fox, Fischer, DeBello, Peña 2026; `design/mascaf_revision.pdf`)
skeletonizes the mesh by mean curvature flow and fits radii from mesh
cross-sections ($r = \sqrt{A/\pi}$). It handles loops by breaking cycles and
writing reconnection directives to the SWC header. Its validation is
geometric: volume and surface area against the mesh.

The problem: geometric fidelity is hard to achieve (centring, spurious
branches, overlap at junctions, choice of edge length), and it is **not the
target**.

## What the voxel study found

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
   Rasterization also fuses nearby surfaces and cuts off necks thinner than
   h. FEM on the surface has neither problem.
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

## Prior work on electrical reduction (to build on, not reinvent)

- **Morphoelectrotonic transform** (Zador, Agmon-Snir & Segev 1995):
  electrotonic distance as the natural coordinate for an effective model.
- **Impedance-matching reduction**: Wybo et al. 2021 (eLife, NEAT toolbox)
  fit reduced models to impedance matrices at chosen sites. Amsalem et al.
  2020 (neuron_reduce) preserve transfer impedance. Kellems et al. 2010 use
  balanced truncation and Krylov model-order reduction.
- **Rall equivalent cylinders and the 3/2 rule**: closed-form sanity checks.
- **Beyond the cable equation:** Lindsay, Rosenberg & Tucker 2004 derive the
  cable equation as the first of a hierarchy of 1D models of the 3D problem.
  The EMI model (Tveito et al. 2017) also resolves the extracellular space.

## Ideas from other fields

| Field | Idea | Use in camovox |
|---|---|---|
| Power systems / circuit theory | **Kron reduction** (Schur complement of the Laplacian). Exactly preserves effective resistances between the nodes kept (Dörfler & Bullo 2013) | Exact DC reduction onto ports; neck clamping is a Schur complement |
| Spectral graph theory | **Effective-resistance sparsification** (Spielman & Srivastava 2011); Laplacian eigenmodes | Sparsify reduced networks; match slow passive modes |
| Phylogenetics / metric geometry | **Tree metrics**: the four-point condition (Buneman 1974); neighbor-joining (Saitou & Nei 1987) | Replace geometric loops by trees with internal nodes that reproduce port resistances (implemented). A loopiness index measures when that fails |
| Network synthesis | **Y-Δ (star-mesh) transform**; **Cauer ladders** realize any RC one-port as a chain | Exact loop removal for three ports; chains suffice for somatic-only targets |
| Factor analysis | **Diagonal + low-rank** decomposition of a covariance-like matrix | A fairer compartment-count bound: a private resistance per synapse plus m shared compartments (implemented) |
| Topological data analysis | **Reeb graphs and Mapper**: level sets of a scalar field, split into connected components, give a graph that keeps loops | Partition the mesh by level sets of an electrical potential (proposal) |
| Numerical linear algebra | **Randomized range finding** (Halko, Martinsson & Tropp 2011); algebraic multigrid | Probe 3D regions to find their electrical rank (proposal) |
| Reduced-basis methods | **Greedy sampling**: add the sample where the reduced model's error is largest | The refinement loop (proposal) |
| Adaptive mesh refinement | **A posteriori error estimation and marking** (Dörfler marking) | Decide which compartments to refine (proposal) |
| Digital rock physics | **Pore network extraction** (PoreSpy SNOW, OpenPNM) | A network calibrated to direct simulation: the same idea in another field |
| Differentiable simulation | Gradient-based fitting through a simulator | Fit reduced-model parameters to FEM responses (implemented, in JAX) |
