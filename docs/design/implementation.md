# Implementation: data model, modules, API, libraries

Back to the [design index](README.md).

Status labels: *implemented*, *proposal* (designed, not built).

## Units and parameters

Lengths in µm, areas in µm², volumes in µm³. Geometric conductance factors
κ in µm, so an axial conductance is g = κ · 10⁻⁴ / r_L in S, with r_L in
Ω·cm. Biophysical parameters (`Passive`: r_L, c_m, g_pas) are kept
**separate** from morphology, so one model serves any parameter set.

## Data model

| Object | Module | What it is | Status |
|---|---|---|---|
| surface mesh | `io` | `trimesh.Trimesh`, closed and manifold, vertices in µm (`load_mesh`) | implemented |
| `TetMesh` | `tet` | volume mesh: `points (N,3)` µm, `tets (M,4)`; with a `FidelityReport` (area and volume change) | implemented |
| `CompartmentGraph` | `circuit` | any passive network: node `area` (µm² membrane), `position` (µm), sparse edges with `kappa` (µm), optional `g_ground`/`c_ground`. Gives $C$, $G_m$, $L$ for given parameters | implemented |
| reference model | `fem` | `fem_model(tets)`: a `CompartmentGraph` on mesh vertices from P1 stiffness and lumped boundary mass, plus a `FemReport` | implemented |
| `PortSolver` | `circuit` | one factorization at a frequency, reused for many port sets | implemented |
| `MembraneSurface` | `sites` | the membrane triangles of a model, for snapping and geodesic patches | implemented |
| `Sites` | `sites` | table of sites (position, area, `axon_id`, ...) plus their patches | implemented |
| `TSSpine` | `graft` | a TS spine ready for analysis: mesh, tets, FEM, neck electrodes, membrane, AZ sites | implemented |
| `ResistanceMetric` | `ports` | membrane-removed network with the neck grounded; port resistance distances | implemented |
| `PortTree` | `reduce` | resistor tree with ports on nodes (`edges`, `resistance` Ω, `port_node`) | implemented |
| `FitResult` | `fit` | fitted tree, node areas, loss history | implemented |
| `JaxleyModel` | `jaxley_io` | jaxley `Cell` plus node ↔ branch maps | implemented |
| `VoxelGrid` | – | voxel input container, `to_surface()` by marching cubes | proposal |
| `MorphologyGraph` | – | `networkx` graph with loops allowed, positions and radii, for SWC export | proposal |
| `Correspondence` | – | mesh vertex → compartment map | proposal (`assign_membrane` gives one for trees) |
| `Scenario` | – | sampler recipe + seed (TOML) | proposal |
| `SynapseModel`, `ActivationPattern` | – | kinetics, and site/time/weight tables | proposal |
| `RefineHistory`, `Refiner` | – | the refinement loop | proposal |

One solver layer covers every `CompartmentGraph`, whether FEM or reduced:
DC, AC impedance, eigenmodes and transients
([fem-reference.md](fem-reference.md), section 4). This keeps comparisons
fair.

## Modules (as built)

```
camovox/
  params.py      Passive parameters, unit conversions
  io.py          load_mesh, load_points, load_axon_groups
  tet.py         TetMesh, tetrahedralize(_cached), fidelity report
  fem.py         P1 assembly, fem_model, gradients and current density
  circuit.py     CompartmentGraph, solvers, ports, electrodes, PortSolver
  sites.py       MembraneSurface, Sites, patches, samplers
  graft.py       cell parts, spine matching, neck contacts, load_spine
  ports.py       resistance distances, loopiness, bounds, coupling, lobes
  reduce.py      PortTree, fit_tree, contract_tree, assign_membrane
  fit.py         JAX port solver, fit_tree_model
  jaxley_io.py   to_jaxley
  viz.py         PyVista and matplotlib figures
  validate.py    analytic references
```

Planned additions: `refine.py` (loop), `scenario.py`/`activation.py`,
`coarse.py` (field partition), `morph.py` + SWC writer.

## Workflow today (TS1, notebooks 02–04)

```python
import camovox as cv
from camovox import fit, graft, jaxley_io, ports, reduce

params = cv.Passive(r_L=100, c_m=1, g_pas=2.63e-4)  # high-conductance state
parts = graft.load_parts(data / "neuromesh/parts", scale=0.005)
ts1 = graft.load_spine("TS1", data, parts)  # FEM, neck electrode, AZ sites
neck = ts1.neck_ports[0]
patches = [neck, *ts1.az.patches]  # port 0 = neck

# Characterize: port matrices, distances, bounds
z_fem = cv.port_impedance(ts1.spine, params, patches, freqs)
metric = ports.ResistanceMetric(ts1.spine, params, ground=neck)
d = metric.distances(ts1.az.patches)

# Reduce, fit, export
tree = reduce.fit_tree(d)
owner = reduce.assign_membrane(ts1.spine, params, ts1.az.patches, ground=neck)
result = fit.fit_tree_model(tree, start_area, z_fem, params, freqs)
model = jaxley_io.to_jaxley(result.tree, result.node_area, params)
```

(`start_area` sums each port's assigned membrane onto its tree node.)

## Planned API *(proposal)*

```python
rng = np.random.default_rng(0)
sites = cv.sample_sites(ref, cv.Scenario.builtin("clustered_axons"), rng=rng)
sites = cv.assign_roles(sites, test_frac=0.2, by="axon", rng=rng)  # locks test set
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
model, history = refiner.run()  # to a stopping criterion
score = cv.evaluate(model, ref, sites.test, params)  # once
```

## Libraries

In use (each logged in `decisions.md`):

| Need | Package |
|---|---|
| mesh I/O and queries | `trimesh` (+ `rtree`) |
| tetrahedralization | fTetWild via `pytetwild` (MPL-2.0; TetGen is AGPL and stays out) |
| FEM assembly | `scikit-fem` |
| sparse solves, NNLS, clustering | `scipy` |
| graphs | `networkx` |
| neighbor-joining | Biopython (`Bio.Phylo.TreeConstruction`) |
| gradients, optimization | `jax` (via jaxley), `optax` |
| cable simulation | `jaxley` (pinned `>=0.14,<0.15`) |
| 3D figures | `pyvista`; 2D figures `matplotlib` |

Candidates for later stages: `pymeshfix` (repair), `scikit-image` (marching
cubes), `pyamg` (large meshes), `potpourri3d` (heat-method geodesics),
`pandas`/`pyarrow` (site tables), `tomllib`/`tomli-w` (scenarios), NEAT
(reduction comparison), `navis` or `MorphIO` (SWC checks), `mascaf`
(baseline).
