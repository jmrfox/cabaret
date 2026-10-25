# 008: TS1 mesh convergence

**Status:** done
**Design sections:** [fem-reference.md](../design/fem-reference.md)
(sections 6–8, meshing, validation, limits), [validation.md](../design/validation.md)
(mesh refinement), open question 10 in [open-questions.md](../design/open-questions.md)

## Goal

Know how much the TS1 FEM reference changes with mesh resolution, for the
quantities the reduction is fitted to, and whether the 0.2 µm mesh used so
far is good enough.

## Why

At edge 0.2 µm the thinnest neck (about 0.12 µm across) may be only one
tetrahedron wide, and each 0.05 µm² PSD patch covers only one or two surface
vertices. Neck resistance and spine-local input resistance at each synapse
are exactly what the tree and the fit are matched to.

## Approach

- **Meshes:** edge 0.2, 0.14 and 0.1 µm, plus 0.2 µm with fTetWild's
  `coarsen` option. The envelope stays at 0.005 µm for all, so the geometry
  is the same and only the resolution changes. Meshes are cached
  (`data/cache/TS1_e<edge>_env0.005[_coarse].npz`) and built in the
  background.
- **Reference:** the 0.2 µm mesh refined uniformly (`TetMesh.refined`,
  same boundary). *Changed during the work:* the 0.1 µm fTetWild mesh was
  the planned reference, but it is barely finer than 0.2 µm (see notes).
- **Quantities:** at DC, 100 Hz and 1 kHz, with the neck port first:
  1. spine-local input resistance at each AZ (diagonal of the neck-clamped
     port matrix) and the isolated input resistance;
  2. the neck-clamped and isolated port matrices (relative Frobenius error,
     `port_matrix_error`);
  3. neck-to-AZ resistance distances (`ResistanceMetric`);
  4. for diagnosis: mesh size, area and volume change, negative-coupling
     share, surface vertices per AZ patch, and the neck electrode size.
- **Where:** notebook `05_ts1_mesh_convergence.py`. The package gains the
  `coarsen` option, `TetMesh.refined` and `load_spine(tets=...)`.

## Steps

- [x] `coarsen` option in `tetrahedralize`, `tetrahedralize_cached` and
      `graft.load_spine` (with its own cache name).
- [x] Build the four meshes in the background.
- [x] `TetMesh.refined` and `load_spine(..., tets=...)`; refined 0.2 µm mesh
      cached as `TS1_e0.2_env0.005_refined.npz`.
- [x] Notebook 05: load each with `load_spine`, compute the quantities,
      tabulate and plot errors against the refined mesh.
- [x] Record the outcome in fem-reference.md section 7. The mesh setting
      for notebooks 02–04 is unchanged (a decision for later), so there is
      no decisions.md entry.

## Done when

- Notebook 05 runs top to bottom from the cached meshes and states, for
  each quantity, the error of the 0.2 µm mesh against the reference.

## Notes

- fTetWild meshes (edge target → vertices): 0.2 coarse → 11.7k, 0.2 → 34.1k,
  0.14 → 37.6k, 0.1 → 47.0k; meshing took 3–6 minutes each. The refined
  0.2 µm mesh has 223k vertices (1.08M tets) and takes 3 s to build.
- AZ patches cover more vertices than first estimated: median 8 (min 3) at
  0.2 µm, because surface elements are much smaller than the edge target.
  Patch areas overshoot the 0.05 µm² target by up to 40%, but that hardly
  explains the per-AZ errors (correlation −0.17).
- The neck electrode removes 6.33–6.35 µm² of membrane on every mesh.
- Results: see notebook 05, section 4, and fem-reference.md section 7.
- Notebook 05 takes about 12 minutes, mostly the refined mesh's complex
  factorizations at 100 Hz and 1 kHz.
