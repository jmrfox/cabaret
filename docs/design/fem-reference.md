# The FEM reference model *(implemented, plans 001–006)*

Back to the [design index](README.md).

The reference model is a 3D finite-element (FEM) simulation of current flow
inside a reconstructed piece of neuron, here a toric spine, bounded by its
membrane. Every reduced model is judged against it. This chapter states the
physics and its assumptions, how it is discretized, why the result is a
resistor network, what is solved and how, and how sites and the neck enter.

Code: `cabaret/tet.py` (meshing), `cabaret/fem.py` (assembly),
`cabaret/circuit.py` (networks and solvers), `cabaret/sites.py` (patches),
`cabaret/graft.py` (neck electrodes).

## 1. Physical model

Let Ω be the intracellular volume and ∂Ω the membrane, with outward normal
**n**. V(**x**, t) is the intracellular potential relative to rest.

**Assumptions:**
1. **Extracellular space is isopotential and grounded** (V_e = 0), so V on
   ∂Ω is the membrane potential. This neglects extracellular resistance,
   which the EMI model (Tveito et al. 2017) resolves. For a single spine in
   tissue, the extracellular voltage drop is small compared with the
   intracellular one.
2. **Cytoplasm is an Ohmic, homogeneous, isotropic conductor** with
   conductivity σ = 1/r_L. Current density is **J** = −σ∇V.
3. **Quasi-static:** no charge accumulates in the cytoplasm, so ∇·**J** = 0
   in Ω. Charge storage happens only at the membrane (its capacitance).
4. **Passive membrane:** capacitance c_m and leak conductance g_pas, uniform
   over the surface. Leak reversal is at rest (V = 0). Synapses, when
   present, are current or conductance sources on membrane patches.
5. **No electrodiffusion:** ion concentrations are constant, and the
   membrane and cytoplasm are linear.

**Governing equations:**

$$
\nabla \cdot (\sigma \nabla V) = 0 \quad \text{in } \Omega
$$

$$
-\sigma \nabla V \cdot \mathbf{n} \;=\; c_m \frac{\partial V}{\partial t} + g_\mathrm{pas} V - j_\mathrm{inj}
\quad \text{on } \partial\Omega
$$

The boundary condition is current conservation at the membrane. Axial
current arriving at the membrane from inside equals capacitive plus leak
current leaving through it, minus any current injected into the cell there
($j_\mathrm{inj}$, A/cm², e.g. a synapse). This is a Robin boundary condition.
The cable equation is the 1D limit of this problem for thin processes
(Lindsay, Rosenberg & Tucker 2004). The FEM solves it without that limit,
which matters in spine heads, junctions and loops.

**Parameter values** used so far:
- r_L = 100 Ω·cm and c_m = 1 µF/cm²;
- g_pas = 2.63×10⁻⁴ S/cm² (high-conductance, in vivo) or 2.5×10⁻⁵ S/cm²
  (resting, slice), as in the earlier voxel study (BF).

Biophysical parameters are kept separate from geometry (`cabaret.Passive`),
so one mesh serves any parameter set.

## 2. Weak form and discretization

Multiply by a test function φ, integrate over Ω, and use the divergence
theorem. The boundary flux is then replaced by the membrane condition:

$$
\int_\Omega \sigma\, \nabla V \cdot \nabla \phi \, dx
\;+\; \int_{\partial\Omega} \Big( c_m \frac{\partial V}{\partial t} + g_\mathrm{pas} V \Big) \phi \, ds
\;=\; \int_{\partial\Omega} j_\mathrm{inj}\, \phi \, ds .
$$

**Elements:** linear (P1) Lagrange elements on tetrahedra. V is approximated
as $\sum_i V_i \varphi_i$, with one hat function $\varphi_i$ per mesh vertex.
This gives two matrices:

- **stiffness** $K_{ij} = \int_\Omega \nabla\varphi_i \cdot \nabla\varphi_j \, dx$,
  assembled with unit conductivity (units: µm). Assembly uses scikit-fem
  (`laplace` on `ElementTetP1`);
- **boundary mass**, **lumped**: each surface vertex gets one third of the
  area of every boundary triangle it belongs to,
  $m_i = \sum_{\triangle \ni i} |\triangle| / 3$ (µm²). Interior vertices get 0.
  Lumping makes the membrane terms diagonal (open question 12 discusses the
  consistent-mass alternative).

The discrete system is a set of ODEs for the vertex voltages:

$$
C\, \dot{V} \;=\; -\big(L + G_m\big)\, V \;+\; I(t),
\qquad L = \sigma K,\quad G_m = g_\mathrm{pas}\, \mathrm{diag}(m),\quad C = c_m\, \mathrm{diag}(m).
$$

Here I is the vector of injected currents (A). Interior vertices have no
capacitance or leak, so their equations are algebraic: they enforce
current conservation at every instant.

## 3. The FEM is a resistor network

K is symmetric, and its rows sum to zero (a constant potential carries no
current). So L = σK is the **Laplacian of a weighted graph** on the mesh
vertices, with coupling conductances

$$
g_{ij} = \sigma\, \kappa_{ij}, \qquad \kappa_{ij} = -K_{ij}\ (\text{µm}).
$$

For P1 tetrahedra there is a closed form (the 3D cotangent formula). Summing
over the tetrahedra T that contain edge ij,
$\kappa_{ij} = \tfrac{1}{6} \sum_{T} \ell_{kl} \cot\theta_{kl}$, where kl is the
edge of T opposite ij, $\ell_{kl}$ its length, and $\theta_{kl}$ the dihedral angle
at kl.

The discrete reference model is therefore exactly a compartmental network:
- one "compartment" per mesh vertex;
- membrane area $m_i$ on surface vertices;
- axial conductance $g_{ij}$ between neighbours.

This is the same form as a cable model ([background.md](background.md),
"Cable models in simulators"). So the FEM and every reduced model are the
same kind of object (`CompartmentGraph`: `area`, `position`, `edges`,
`kappa`), solved by the same code. Comparisons between them are
like-for-like.

**Units** (`cabaret.params`):
- axial conductance g = κ · 10⁻⁴ / r_L (S), with κ in µm and r_L in Ω·cm;
- membrane conductance g_pas · A · 10⁻⁸ (S), with A in µm²;
- capacitance c_m · 10⁻⁶ · A · 10⁻⁸ (F).

**Negative couplings.** κ_ij is negative when a dihedral angle opposite the
edge exceeds 90° (obtuse tetrahedra). The FEM solution is still correct,
since it is a consistent discretization, but those entries act as negative
conductances. fTetWild meshes have about 4.5–6.4% of total |κ| on negative
couplings (sphere, cylinder, TS1). Reduction steps that assume positive
conductances must tolerate this, and `FemReport.negative_fraction` reports
it (open question 21).

## 4. What is solved

All solves use the admittance matrix $A(\omega) = L + G_m + i\omega C$, plus
any extra ground shunts (below). Sparse LU (SuperLU) is factorized once and
reused for many right-hand sides.

| Quantity | Equation | Code |
|---|---|---|
| Steady-state voltage for an injection I | $A(0)\,V = I$ | `solve_dc` |
| Impedance between patches $w_a$, $w_b$ at frequency f | $Z_{ab}(f) = w_a^\top A(2\pi f)^{-1} w_b$ (weights normalized) | `transfer_impedance`, `port_impedance`, `PortSolver` |
| Slowest time constants | $(L+G_m)\,v = \mu\, C v$, $\tau = 1/\mu$ (shift-invert on $A^{-1}C$; C is singular because interior nodes have none, and those modes have infinite μ) | `time_constants` |
| Transient response to an injected current I(t) | backward Euler: $(C/\Delta t + L + G_m)\,V^{n+1} = (C/\Delta t)\,V^n + w\, I^{n+1}$ | `simulate_current` |
| Current density | P1 gradient per tetrahedron, $\mathbf{J} = -\sigma \nabla V$ | `current_density`, `tet_gradient` |

The isolated spine (no parent attached) is nonsingular, because every surface
vertex leaks to ground through its membrane. A transient on the full TS1
model (34k vertices, 600 steps) takes about 7 s.

## 5. Sites, ports, electrodes and the neck

**Patches, not points.** A site (a synapse, or a recording or injection
point) is a patch of membrane. A point source in 3D has infinite input
resistance, so a point would make no sense. A patch is a geodesic disk of
surface vertices grown until its membrane area reaches the target (PSD)
area ([sites-and-scenarios.md](sites-and-scenarios.md)).
- Its **weights** are the vertices' lumped membrane areas.
- **Injection** spreads the current over the patch in proportion to area,
  which is uniform current density.
- **Readout** is the area-weighted average voltage over the patch.

A patch is not equipotential. Current must fan out from it, which adds a
local **spreading resistance**, about $r_L/(4a)$ for a disk of radius a (about
2 MΩ for 0.05 µm²).

**Electrodes (equipotential ports).** `add_electrode` adds one extra node,
joined to a set of vertices by very large couplings (κ = 10⁴ µm, against
typical mesh couplings below 1 µm). That set then behaves as a perfect
conductor. Optionally, their membrane is removed.

**The neck port.** A spine joins its parent through its neck. The TS meshes
were traced separately from the whole cell, and their bases reach into the
parent. Every TS vertex lying inside the cell's parent part, or within
0.2 µm of it near the neck point, is parent cytoplasm. Those vertices
(interior ones included) are tied into one electrode, and their membrane is
removed ([spines-and-cell.md](spines-and-cell.md)). For TS1, that is 2,044
vertices and 6.3 µm² of membrane. The parent is nearly isopotential at
spine scale, so an equipotential electrode is a good stand-in.

**Ground shunts and clamping.**
- `attach_sink` adds a conductance (and optionally a capacitance) from a
  patch to ground. With a tiny resistance (1 Ω), that holds the neck at
  rest.
- The same "neck clamped" matrix is obtained exactly, with no shunt at all,
  as the Schur complement of the isolated port matrix:
  $Z_c = Z_{AA} - Z_{An} Z_{nA} / Z_{nn}$ (`ports.clamp_port`, n = neck).

**Resistance distances.** With the membrane removed and the neck grounded,
the resistance matrix R between ports gives the metric
$d_{ij} = R_{ii} + R_{jj} - 2R_{ij}$. Tree realization starts from it
(`ports.ResistanceMetric`).

## 6. Meshing

- **Input:** a closed (watertight) surface in µm. The TS data are in
  0.005 µm pixels.
- **Tetrahedralization:** fTetWild (`pytetwild`), with a target edge length
  and an **envelope**, the maximum distance the meshed surface may move
  from the input. The envelope must stay far below the thinnest neck radius
  (about 0.06 µm). TS1 uses edge 0.2 µm and envelope 0.005 µm: 135k
  tetrahedra, area −0.80%, volume −0.14%, about 2 minutes. Results are cached
  (`tetrahedralize_cached`).
- **Fidelity report:** area and volume before and after meshing, plus
  tetrahedron and vertex counts (`FidelityReport`).
- **Gotcha:** fTetWild writes `__tracked_surface.stl` into the working
  directory (gitignored).

## 7. Validation so far

| Test | Result |
|---|---|
| Isopotential sphere (r = 8 µm): input resistance | +0.2% (9.9k tets), +0.6% (3.8k tets) |
| Isopotential sphere: time constant | exact (40.00 ms) |
| Sealed cylinder (r = 1 µm, L = 200 µm): input resistance | −0.3% |
| Sealed cylinder: end-to-end attenuation | 0.9026 vs 0.9033 analytic |
| P1 gradient of a linear field | exact (test) |
| Torus | current splits around the hole as expected; four-point violation behaves as predicted (notebook 01) |
| Reduced TS1 model against FEM | EPSP peaks within 0.2% at three synapses (notebook 04) |
| TS1 mesh convergence, 0.2 µm vs 0.05 µm fTetWild mesh (notebook 05) | neck-clamped port matrix 1.1%; spine-local R_in 1.1% low on average, 1.0% median, 4.1% max; isolated R_in 0.5% |

**TS1 mesh convergence** (plan 008, notebook 05):
- fTetWild's target edge length is a weak knob: 0.2 → 0.1 µm adds only 38%
  more vertices (34k → 47k), since the 0.005 µm envelope already sets the
  surface resolution. 0.05 µm gives 154k vertices and takes about 40
  minutes to mesh.
- Against the 0.05 µm mesh, the 0.2 µm mesh reads spine-local R_in 1.1% low
  on average (4.1% max) and the neck-clamped matrix 1.1% off; 0.1 µm is
  0.7% low (2.8% max).
- The edge-length series alone does not converge cleanly (0.2 → 0.1 and
  0.1 → 0.05 each change 0.8% median), partly because each fTetWild run
  moves the surface (area −0.80% to −1.04%).
- Two independently built fine meshes, 0.05 µm and the 0.2 µm mesh refined
  uniformly (`TetMesh.refined`, 223k vertices), agree to 0.34% median and
  0.44% on the clamped matrix. So a fine mesh of either kind is good to a
  few tenths of a percent, and the 0.2 µm mesh is about 1% off.
- The other spines are still to do ([validation.md](validation.md)).

## 8. Limitations

- **Grounded extracellular space.** Extracellular resistance and ephaptic
  effects are ignored.
- **Lumped membrane mass** slightly smears membrane at the scale of one
  triangle. A consistent-mass check is open (question 12).
- **Negative couplings** from obtuse tetrahedra (section 3).
- **Geometry resolution:** the envelope must stay below the thinnest
  necks, and triangles must be small compared with the PSD patches.
- **Linear and passive:** active channels distributed over the spine
  membrane would require a nonlinear FEM (see
  [open-questions.md](open-questions.md), question 20).
- **Patch model:** synapses are uniform-current-density patches of a
  placeholder area (0.05 µm²). Their spreading resistance depends on that
  area.
