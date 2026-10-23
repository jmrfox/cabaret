# 002: Visualization of FEM solutions

**Status:** done (pending the user's ParaView check)
**Design sections:** [Validation](../design/validation.md),
[FEM reference](../design/fem-reference.md)

## Goal

Make FEM solutions visible: voltages on the membrane, inside the volume,
equipotential surfaces, current paths, site patches and mesh-quality problems.
Export to ParaView. Demonstrate it all in a notebook on test shapes.

## Steps

- [x] `circuit.solve_dc(graph, params, injection)` returns the full voltage
      field (V) for a current injection (A).
- [x] `fem.tet_gradient(tets, values)`: P1 gradient per tetrahedron.
      `fem.current_density(tets, v, params)` gives $J = -\nabla V / r_L$ in A/µm².
- [x] `camovox/viz.py` (thin wrappers over PyVista and matplotlib):
      `to_pyvista`, `write_vtu`, `plot_surface`, `plot_slice`,
      `plot_isosurfaces`, `plot_streamlines`, `plot_sites`,
      `plot_mesh_quality`, `plot_profile`, `vertex_distance_along`.
- [x] Notebook `notebooks/01_fem_reference.py`: sphere and cylinder
      against analytic solutions, a torus with current streamlines around the
      hole, near-λ→∞ isosurfaces (a partition preview), sites, mesh quality,
      and `.vtu` export.
- [x] Add `matplotlib` and log it in `decisions.md`.
- [x] Test: the P1 gradient is exact for a linear field
      (`tests/test_gradient.py`).

## Done when

- The notebook runs top to bottom (verified headless as a script; about 40 s,
  with 12 s of meshing).
- `write_vtu` output opens in ParaView (manual check by the user, if they
  install it).

## Notes

- **Notebook speed.** The first version took more than 2 minutes, mostly in
  fTetWild with tight envelopes (0.005–0.01 µm). Demo meshes now use coarser
  edges and envelopes of 0.03–0.05 µm. Timings: sphere 1.5 s, cylinder 7.5 s,
  torus 3.1 s. With `optimize=False` fTetWild is 2–10× faster, but it can
  produce far more tets (the cylinder went from 9k to 55k). The first
  matplotlib import builds its font cache (about 27 s, once).
- The sphere at this coarse resolution gives $R_\mathrm{in}$ +0.6% (5003 against
  4974 MΩ). The torus has 5.4% of |κ| on negative couplings.
- On Windows, printing "Ω" from a script needs `PYTHONIOENCODING=utf-8`;
  notebooks are unaffected.
- PyVista's `extract_surface` is called with `algorithm="dataset_surface"`
  explicitly, to silence a future-default warning.
- fTetWild writes `__tracked_surface.stl` into the current directory (it is
  gitignored everywhere).

**Figure fixes (2026-10-23)**, found by extracting the notebooks' saved
PNGs:
- Colour bars choose tick precision from the data range, so ticks are no
  longer duplicated, and they are narrowed so end labels aren't clipped.
- `plot_sites` draws a sphere at each patch centre with a categorical bar.
  Real patches are smaller than the mesh triangles and were invisible.
- `plot_streamlines` integrates the unit direction field and colours by
  magnitude. Raw |J| of about 1e-12 had stalled the lines. Uses
  `max_length`, since `max_time` was removed in PyVista 0.49.
- `plot_isosurfaces` levels exclude the extremes. Notebook 01 plots
  φ − min(φ), and writes "phi" because the VTK font has no φ glyph.
- `plot_mesh_quality(fraction=0.2)` shows only the most negative couplings.
- The `off_screen` default is now `None` (follow `pyvista.OFF_SCREEN`). An
  explicit `False` had opened a blocking window in headless runs.
