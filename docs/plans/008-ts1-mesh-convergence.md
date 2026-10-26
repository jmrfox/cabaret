# 008: TS1 mesh convergence

**Status:** active (paused 2026-10-25: switching notebooks 02–04)
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

## Follow-up: edge-length series (2026-10-25)

At the user's request the check was redone by rerunning fTetWild at 0.2,
0.1 and 0.05 µm (geometry and resolution change together), with the 0.05
µm mesh as the reference and the uniformly refined 0.2 µm mesh as a
cross-check.

- 0.05 µm: 154k vertices, 754k tets, area −1.04%, 15.5k neck electrode
  nodes, 37 minutes to mesh.
- Against it: 0.2 µm is 1.1% low on spine-local R_in (1.0% median, 4.1%
  max at AZ 12), clamped matrix 1.1%; 0.1 µm 0.7% low (0.8% median, 2.8%
  max), clamped 0.85%.
- Successive steps 0.2 → 0.1 and 0.1 → 0.05 each change 0.8% (median), so
  the series by itself does not show convergence; isolated R_in shifts
  0.5%, which points to the surface moving between fTetWild runs.
- The refined 0.2 µm mesh agrees with the 0.05 µm mesh to 0.34% median,
  2.0% max (AZ 8), 0.44% clamped, +0.14% mean: two independently built
  fine meshes agree to a few tenths of a percent.
- At 100 Hz and 1 kHz the clamped errors equal the DC ones to 0.01%, so
  notebook 05 now computes DC only. With three frequencies the notebook
  took 58 minutes.

## Switching notebooks 02–04 to the 0.05 µm mesh (reverted)

Decided 2026-10-25, reverted 2026-10-26 (decisions.md): the demo notebooks
stay on the 0.2 µm mesh, and notebooks 03–04 say how large its error is and
point to notebook 05. Work done before the revert:

- [x] Notebooks 02–04 loaded the 0.05 µm mesh (`EDGE = 0.05`), never
      committed; now back on 0.2 µm.
- [x] Notebook 03 rerun headless (53 minutes): fit error 1.42% (was 1.35%),
      neck-clamped error 3.00% at all frequencies (was 2.83%), isolated
      0.24% at DC (was 0.01%), spine-local R_in median 0.31%, max 8.1% (was
      0.29%, 7.6%), diagonal + low-rank bound m=8 1.3%, m=12 0.4%. Its FEM
      node count in the text is updated; the "about 3%" wording still holds.
- [x] Notebook 04 rerun: dropped with the revert. Two attempts stalled before printing anything
      past the first cell; the first sat at ~0% CPU with 6.4 GB held. Not
      diagnosed. Suspects: `attachment_contacts` on the fine mesh (15.5k
      contact nodes; its containment tests are slow), or the headless
      plotting. Next step: time `load_toric_spine(edge_length=0.05)` alone,
      and consider caching the contacts with the mesh.
      Diagnosed 2026-10-26: not a hang. Cell 2 alone takes 42 minutes, and
      a headless run prints nothing until the cell finishes. Timings:
      `load_toric_spine` 96 s (contacts 80 s), `port_impedance` 2.3 min at
      DC but about 17 min each at 100 Hz and 1 kHz (complex LU),
      `ResistanceMetric` 2 min, `assign_membrane` 2 min. Peak memory 2.9 GB
      in a script. One sparse LU uses one core, which shows as about 6% CPU
      on this 16-core machine.
- [x] Notebook 02 rerun and TS1 number updates: dropped with the revert
      (the 0.2 µm numbers stay).
- [x] Notebooks 03–04 note the 0.2 µm mesh error and that the tree's
      roughly 3% error is the same against the 0.05 µm FEM.
- [x] Notebook 05 broke the 5-minute demo budget (40 min to mesh at
      0.05 µm, tens of minutes of solves). The meshing and solves now run in
      `scripts/ts1_mesh_convergence.py`, which saves
      `data/cache/ts1_mesh_convergence.npz`; the notebook loads it.
