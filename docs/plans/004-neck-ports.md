# 004: Neck ports from the cell mesh

**Status:** active (neck ports done; tracing comparison and grafting next)
**Design sections:** [Spines and the cell](../design/spines-and-cell.md); open questions 4, 4b, 15 and 22 in [open-questions.md](../design/open-questions.md)

## Goal

Characterize each TS spine **on its own**, as a multiport whose neck port is
where its separately traced mesh meets the whole-cell mesh. This removes the
parent placeholders. Then compare each spine's two tracings.

## Steps

- [x] Copy the neuromesh-cgal segmentation of `cell.obj` into
      `data/neuromesh/` (91 watertight parts with capped cuts; segment 2 =
      soma). `cell.obj` is byte-identical to `cell_wrapped_simplified_1cc.obj`.
- [x] Match TS meshes to cell parts by AZ and neck-point proximity.
- [x] Check the TS meshes at their neck points: no flat cut cap, and plane
      cross-sections are unstable (0.008–1.1 µm²). Neck points sit on the
      spine's side surface.
- [x] `camovox/graft.py`:
  - `load_parts`;
  - `spine_parts`: parts with at least 50% of their vertices within 0.3 µm
    of the TS mesh;
  - `neck_contacts`: TS FEM nodes inside a parent part, or within 0.2 µm of
    one, within 3 µm of a neck point, assigned to the nearest neck.
- [x] `circuit.add_electrode`: tie nodes into one equipotential port and
      remove their membrane. Plus `circuit.point_port`. Test:
      `test_electrode_makes_tied_nodes_equipotential`.
- [x] Notebook 02 uses the isolated spine with its neck electrode. No parent
      sink, and no `NECK_AREA` / `PARENT_R` / `PARENT_C` placeholders.
- [ ] Tracing variability: the same port analysis on the cell's own version
      of TS1 (parts 17 + 85), with AZs snapped to both tracings.
- [ ] Neck ports for all 9 spines, including the multi-neck TS3 and TS21.
- [ ] Grafting validation: TS multiport joined to the cell (with its own
      version of the spine removed), against a whole-cell FEM.

## Notes

Contact area at tolerance d = 0.1 / 0.2 / 0.3 / 0.5 µm, with the area inside
the parent (exploration, TS surface only):

| Spine | d = 0.1 | 0.2 | 0.3 | 0.5 µm | inside |
|---|---|---|---|---|---|
| TS1 | 5.49 | 6.39 | 7.13 | 8.94 µm² | 4.93 |
| TS2 | 0.38 | 0.50 | 0.58 | 0.90 | 0.20 |
| TS3 | 1.45 | 2.61 | 4.04 | 7.70 | 0.83 |
| TS4 | 0.05 | 0.19 | 0.55 | 1.18 | 0.03 |
| TS21 | 1.04 | 1.61 | 2.31 | 4.29 | 0.77 |

- TS1's traced base reaches into the parent (render checked). Its neck
  electrode ties 2,044 FEM nodes and removes 6.3 µm² (8%) of membrane.
- **Effect on TS1 results:**
  - spine-local resistance 7.7–30.5 MΩ (was 8.7–32.5);
  - lobes unchanged, with across-lobe coupling dropping to a median of 0.010
    (was 0.05);
  - rank bound at 10% is m ≥ 7.
- **The "intrinsic maximum" lead from plan 003 disappeared.** The real AZs
  are now at the 35th percentile. It was an artifact of the placeholder neck
  patch.
- Only the first neck is used in notebook 02 (TS1 has one). Multi-neck spines
  give one electrode per neck.
- `neck_contacts` queries containment in chunks of 2,000 points. Unchunked,
  trimesh's ray test tried to allocate 9.9 GB on TS24.
- On surface vertices, all remaining spines have contacts: TS24 12,169
  vertices, TS48 8,414, TS67 135, TS76 588.
- TS48 takes about 5.5 min (dense mesh: 273k faces). Speed up later if
  needed, e.g. a coarser candidate filter or the cell's SDF.
