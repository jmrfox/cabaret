# Spines and the cell: data, neck ports, grafting

Back to the [design index](README.md).

How the separately traced toric spine (TS) meshes relate to the whole-cell
mesh, and how a spine's neck port is defined. Code: `camovox/graft.py`,
`camovox/io.py`. Data inventory: `data/README.md`. Plan 004.

## The data

- **Units:** 5 nm pixels (0.005 µm/px). `load_mesh(path, scale)` converts
  to µm.
- **Whole cell:** `data/cell.obj`, byte-identical to
  `cell_wrapped_simplified_1cc.obj`. Soma and all 76 toric spines form one
  connected surface.
- **TS meshes:** 9 spines traced separately and in more detail.
- **AZ points** for 7 spines, and **neck points** for all 9
  (`TS*_neckpoint*.txt`; several per spine where it has several necks).
  Neck points are approximate: placed by eye on the postsynaptic membrane
  near or on the side of a neck.
- **Axon groups:** `TS1_syn_axon_dist.txt` (see
  [sites-and-scenarios.md](sites-and-scenarios.md)).
- **Segmentation:** the user's
  [neuromesh-cgal](https://github.com/jmrfox/neuromesh-cgal) (CGAL SDF,
  graph-cut, SDF-aware merging) splits `cell.obj` into 91 watertight parts
  with capped cuts. Segment 2 is the soma. camovox **reads its outputs**
  (capped part OBJs, `.seg`/`.sdf` sidecars, copied to `data/neuromesh/`)
  and doesn't run it.
- **Still missing:** real PSD areas (not coming; see the PSD guess), and
  synapse data for spines beyond the 7.

## Two tracings of each spine

Each of the 9 spines exists twice: as a detailed TS mesh, and as a coarser
part of `cell.obj`. The TS meshes are **not contained** in the cell mesh.

**Matching** (AZ and neck-point proximity; `spine_parts` keeps parts at
least 50% covered by the TS mesh within 0.3 µm):

| Spine | Cell part |
|---|---|
| TS1 | 17 |
| TS2 | 43 |
| TS3 | 21 (plus 1 AZ in part 13) |
| TS4 | 36 |
| TS48 | 57 |
| TS67 | 30 |
| TS76 | 39 |

Neck points often fall in the neighbouring (parent) part, so the
segmentation cut lies on the spine side of the traced neck point.

Two uses *(proposal)*:
- **Grafting validation:** a whole-cell FEM contains the cell's own version
  of each spine. To validate a reduced TS spine in situ, remove the cell's
  version (via its part), cut in a thin, cable-like stretch of the neck, and
  join the TS multiport to the parent multiport at the cut. This assumes the
  cut faces are equipotential.
- **Tracing variability:** port properties of the same spine in both
  tracings (the same AZs snapped to both) show how sensitive the conclusions
  are to reconstruction.

## No sink: spines as isolated multiports *(decided)*

A passive spine joined to its parent only through its neck(s) is a linear
multiport: synapse ports plus neck ports. A reduced spine that reproduces
that multiport is exact when connected to *any* parent. So:
- spines are characterized and fitted **in isolation**, with each neck as a
  port (the isolated-spine port matrix Z(f), nothing attached);
- the parent is attached at **grafting** time. The junction's spreading
  resistance (about $r_L/(4a)$, BF) belongs to the parent side: a whole-cell
  FEM has it automatically, and a reduced parent must add it;
- spines with **several necks** (TS3: 3, TS21: 2) are multiports with
  several neck ports. The coupling between their attachment points comes
  from the parent model;
- the large sink used before in toric_spines_sim (Arbor) answered "what
  does the parent see". Here grafting onto a whole-cell model answers it.

## Neck ports from contact with the cell *(implemented)*

The neck port is the part of the TS spine that lies **inside the cell's
parent part, or within 0.2 µm of it**, and within 3 µm of a neck point
(`neck_contacts`). The parent is every cell part except the spine's own.
- Its FEM nodes, interior included, are tied into one electrode
  (`add_electrode`, κ = 10⁴ µm per tie), and their membrane is removed:
  that membrane is really the parent's interior
  ([fem-reference.md](fem-reference.md), section 5).
- The rule tolerates approximate neck points, because the port is defined
  by where the meshes overlap. The point only sets the search region.
- Containment queries run in chunks of 2,000 points (a single query on TS24
  needed about 10 GB).
- `load_spine` puts it together: TS mesh → tetrahedra (cached) → FEM →
  one electrode per neck → AZ sites.

**TS1:** the neck electrode ties 2,044 nodes and removes 6.3 µm² of
membrane. The traced base reaches into the parent (4.9 µm² inside).

## Still to do (plan 004)

- the tracing comparison for TS1 (TS mesh against part 17);
- all spines through `load_spine`, with a summary table;
- grafting onto a reduced or FEM parent.
