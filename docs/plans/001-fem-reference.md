# 001: FEM reference model foundation

**Status:** active
**Design sections:** [Core concepts and data model](../design.md#core-concepts-and-data-model),
[Pipeline](../design.md#pipeline), [Validation](../design.md#validation)

## Goal

Build the electrical reference: load a closed surface mesh, tetrahedralize
it, assemble a P1 FEM model as a `CompartmentGraph` on mesh vertices, and
solve passive DC, AC and eigenmode problems. Validate the result against
analytic solutions for a sphere and a sealed cylinder.

## Steps

- [x] Add dependencies: `scipy`, `trimesh`, `scikit-fem`, `pytetwild`
      (fTetWild), and `pyvista`, which pytetwild imports unconditionally.
      Logged in `decisions.md`.
- [x] `camovox/params.py`: `Passive` ($r_L$ in Ω·cm, $c_m$ in µF/cm²,
      $g_\mathrm{pas}$ in S/cm²) with unit conversions to SI.
- [x] `camovox/circuit.py`: `CompartmentGraph`, Laplacian and system
      matrix, `transfer_impedance` (DC and AC), `input_resistance`,
      `time_constants` (shift-invert eigenproblem; singular $C$ handled).
- [x] `camovox/io.py`: `load_mesh(path, scale)`.
- [x] `camovox/tet.py`: `tetrahedralize(mesh, edge_length, envelope)` →
      `TetMesh` + `FidelityReport` (area and volume before and after).
- [x] `camovox/fem.py`: `fem_model(tets)` → `CompartmentGraph` (κ = −K from
      the P1 stiffness, lumped boundary area) + `FemReport` (negative
      couplings).
- [x] `camovox/sites.py` (placeholder): `ball_patch(graph, center, radius)`.
- [x] `camovox/validate.py`: analytic sphere and sealed-cylinder references.
- [x] Tests (`tests/test_fem.py`, marked slow): sphere $R_\mathrm{in}$ and $\tau$;
      sealed-cylinder $R_\mathrm{in}$ and end-to-end attenuation.
- [ ] Surface repair and checks (`surface.py`): watertightness, components,
      genus. Not started; see Notes.

## Done when

- `uv run pytest` passes, with the FEM sphere $R_\mathrm{in}$ within 1% and
  $\tau$ within 1% of analytic, and the cylinder $R_\mathrm{in}$ within a few
  percent. **Met** (suite takes about 47 s).

## Notes

Results from the first smoke run (not part of the test suite):
- Sphere, r = 8 µm, 9.9k tets: $R_\mathrm{in}$ +0.22%; $\tau_0$ exact
  (0.0400 s).
- Sealed cylinder, r = 1 µm, L = 200 µm, $g_\mathrm{pas}$ = 2.63×10⁻⁴ (BF's T2
  analogue), 197k tets: $R_\mathrm{in}$ −0.33%; $V(L)/V(0)$ 0.9026 against
  0.9033 analytic. Meshing took **214 s**, so the test uses a short, thick
  cylinder instead.

Findings to discuss:
- **Negative couplings are common.** fTetWild meshes gave 4.5% (sphere) and
  6.4% (cylinder) of total |κ| on negative couplings. The FEM solution is
  still accurate, but the "κ from flux" and Kron logic in the reduction must
  tolerate them, or the mesh must be made Delaunay-like. This bears on
  design open question 10/12.
- **Mesher speed** scales with tet count. Long thin processes at fine edge
  length are slow. A sizing field (fine at necks, coarse elsewhere; fTetWild
  supports background sizing) will matter for whole cells.
- `pytetwild` pulls in `pyvista` and `vtk` (large). It is imported lazily,
  but it is still a hard dependency. Possibly make it an optional extra later.

Placeholders left for discussion:
- `ball_patch` is a Euclidean ball, not a geodesic patch grown to a PSD area.
- No surface repair yet. `tetrahedralize` assumes a closed surface.
- `envelope` defaults to `edge_length / 50`, with no check against neck radius.
- `time_constants` uses non-symmetric ARPACK on $A^{-1}C$. This is fine for a
  few modes, but should be revisited for large meshes.
