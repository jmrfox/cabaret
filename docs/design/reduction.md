# Reduction: from the reference to a reduced model

Back to the [design index](README.md).

Code: `cabaret/reduce.py` (trees), `cabaret/fit.py` (fitting),
`cabaret/jaxley_io.py` (export). Notebooks 03 and 04 (plans 005, 006).

## Loops without loops: tree realization *(implemented, plan 005)*

The reduced model must reproduce the electrical relationships between the
points that matter, not the wiring. So a geometric loop does not have to
become a loop in the cable model.

**Why this can work.** Inside a spine, membrane leak is small compared with
the internal axial resistances (the spine is electrically small), so the
spine behaves as a resistive network with membrane lumped onto it.
Everything a simulator sees at the **ports** is then set by the
port-to-port resistance distances. That holds even with nonlinear
conductance synapses, because they sit at the ports and everything between
them is linear.

**When a tree is exact:**
1. **A loop carrying no ports** is series-parallel, and it collapses to a
   single edge.
2. **Any three ports** can be joined by a star with one membrane-less
   internal node (Y-Δ transform).
3. **In general**, port distances are realizable by a tree with internal
   nodes **if and only if** they satisfy the four-point condition
   ([ports-and-complexity.md](ports-and-complexity.md)).
4. **A single output port:** any passive RC driving-point impedance is a
   **Cauer ladder** (a chain).

**Method (implemented):**
1. FEM port resistance distances, neck = port 0 (`ports.ResistanceMetric`).
2. **Topology** by neighbor-joining (Biopython), then **non-negative branch
   resistances** by least squares over all port pairs (`fit_tree`,
   `scipy.optimize.nnls`). Each port becomes a leaf, and internal nodes are
   branch points.
3. **Membrane assignment** by electrical fingerprint (`assign_membrane`).
   With the neck at rest, each membrane vertex's voltages across injections
   at every synapse form its fingerprint, and the vertex goes to the port
   with the nearest one (or to the neck if it barely responds). Total
   membrane is conserved.
4. **Network:** `tree_to_graph` gives a `CompartmentGraph` with
   κ = r_L / (R · 10⁻⁴), so the same solvers evaluate it.
5. **Smaller trees** by merging branches below a threshold resistance
   (`contract_tree`).

**Default policy:** tree realization is the default. Literal loops (the
jaxley private-API workaround, open question 8) are a fallback, only where
loopiness says a tree can't meet ε.

**TS1 results** (notebook 03):
- the tree has 50 nodes (26 ports + 24 branch points) and fits port
  distances to 1.35%;
- it splits at the neck into one subtree per lobe;
- neck-clamped port matrix error **2.83%** at DC, 100 Hz and 1 kHz;
- spine-local input resistance error per synapse: median 0.3%, max 7.6%;
- merging branches without fitting: 40 nodes 5.7%, 26 nodes 11%.

**Caveats:**
- real membrane is distributed along loop arms, so lumping it onto ports
  introduces a small error;
- the tree is exact only for the declared ports (principle 7);
- active channels distributed in spine membrane would break the "linear
  between ports" argument.

## Dependence on synapse locations

Tree realization, like any port-based reduction, assumes the port locations
are known. Sensitivity to placement is measured, not assumed away:
1. **Placement sensitivity of the spine itself:** port impedances and
   loopiness for real positions against jittered and re-drawn ones.
   Uniform and jitter are *implemented* (notebook 02). A clustered-by-axon
   null is *proposal*.
2. **Transfer error of a fixed reduced model:** fit to one site set,
   evaluate on another (*proposal*).
3. **Port-basis convergence:** K evenly spread ports as a
   placement-agnostic model, error against K (*proposal*). This is the
   fallback for spines without synapse data.

These results feed the generative model ([generative-model.md](generative-model.md)).

## Fitting *(implemented, plan 006)*

`fit.fit_tree_model` tunes each branch resistance and each node's membrane
area so the model's port matrices match the FEM's.
- **Solver:** a small, dense, differentiable port-impedance solver in JAX
  (`port_impedance_jax`). Exact gradients; cheap for tens of nodes.
- **Parameters:** log resistances, and areas as total × softmax, which
  conserves total membrane exactly. That matters because channel densities
  will later be specified per membrane area.
- **Loss:** mean over frequencies (DC, 100 Hz, 1 kHz) of the relative error
  of the neck-clamped matrix, plus 0.1 × that of the isolated matrix. Adam
  (optax).
- **Why our own solver, not jaxley:** the targets are frequency-domain, and
  jaxley simulates in time. For linear networks, frequency-domain targets
  are complete.

**TS1 results** (notebook 04):

| Compartments | Before fitting | After fitting |
|---|---|---|
| 50 (full tree) | 2.83% | 2.77% |
| 40 | 5.7% | 3.9% |
| 33 | 10.0% | 5.8% |
| 26 | 11.3% | 6.5% |
| 21 | 12.0% | 7.2% |
| 16 | 29.7% | 22.2% |
| 11 | 39.9% | 23.7% |

- About 2.8% is a **floor set by the loop-free tree shape**: the electrical
  cost of dropping TS1's loops. More or larger optimization steps don't
  change it.
- **Next:** choose which synapses share a compartment (e.g. by
  `electrical_groups`), keep a private series resistance per synapse, then
  fit. The diagonal + low-rank bound suggests about 8 shared compartments
  could reach about 1%.

## Export to jaxley *(implemented, plan 006)*

`jaxley_io.to_jaxley` builds a passive jaxley `Cell`:
- one one-compartment branch per tree node, rooted at the neck;
- `resistive_load_in` = R_edge / (r_L · 10⁴), giving the edge resistance to
  the parent;
- `resistive_load_out` = 1e-6 (about 1 Ω), so each branch point coincides
  electrically with its parent compartment. **At 1e-12, jaxley loses
  precision** (+2.5%), and the `jax.sparse` solver fails;
- membrane set via `area` (1e-4 µm² for nodes without membrane), `Leak`
  with reversal 0;
- an optional `root_shunt` ties the neck to rest, matching `attach_sink`.

Radius and length are set to dummy values *before* area and loads, because
setting them recomputes both. Never make them trainable on the exported
cell.

**Verification:** with an α current (50 pA, τ 0.5 ms) and the neck tied to
rest, FEM, reduced (cabaret) and reduced (jaxley) EPSPs agree within 0.2%
at three synapses. jaxley and cabaret are identical.

**What jaxley can express** (probed with jaxley 0.14):
- membrane area is independent of radius (`set("area")`);
- axial coupling is set per compartment (`axial_resistivity`,
  `resistive_load_in/out`);
- gradients flow through both;
- loops only through a private-API workaround (open question 8).

## Field-based partition of cable-like regions *(proposal)*

For dendrites and other cable-like parts:
1. **Root:** the soma, or a spine's neck.
2. **Field:** current injected at the root and withdrawn through the
   membrane in proportion to area. This is the parameter-free λ → ∞ limit
   (notebook 01, section 4, shows an approximation on a torus).
3. **Partition:** cut the potential φ into bands, then split each band into
   connected components (a Reeb/Mapper graph). Branches and loops come from
   the level sets, with no skeleton.
4. **Parameters:** membrane = Σ vertex area; edge κ = interface current ×
   r_L / Δφ. Alternatively, fit to several source fields (open question 1).
5. **Embedding:** node position = compartment centroid.

## Port-based reduction of truly 3D regions *(proposal)*

For spine heads, loops, junctions and the soma, when tree realization is not
enough:
1. **Classify regions:** κ that shifts across source fields, or between
   membrane states, marks a 3D region.
2. **Ports:** the region's sites plus its interfaces.
3. **Size from data:** randomized probing of the region's response gives
   its electrical rank.
4. **Reduce:** Kron-reduce onto the ports, then sparsify into a small,
   possibly loopy network fitted to sampled responses.

**Alternatives to keep in view:** pure Kron reduction plus sparsification
everywhere; a hybrid soma (one compartment with access resistances measured
in FEM, as BF validated).
