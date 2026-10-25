# 009: A general reference-model builder

**Status:** done
**Design sections:** [README](../design/README.md) (terminology: reference
model), [spines-and-cell.md](../design/spines-and-cell.md),
[implementation.md](../design/implementation.md)

## Goal

cabaret's core builds a reference model from any closed surface, with no
toric-spine file conventions or vocabulary. Toric-spine specifics live in a
research module. Clean break: no aliases for the old names.

## Changes

- **New `cabaret/reference.py`:** `ReferenceModel` and
  `build_reference(mesh, attachments=..., site_points=..., ...)`: surface →
  tetrahedra (optionally cached or given) → FEM → one electrode per
  attachment → membrane surface → sites snapped onto it. `attachments` is a
  list of FEM node sets, or a function of the tetrahedral mesh that returns
  them (attachments can only be found once the mesh exists).
- **`graft.py`, generalized:** a region traced separately from a segmented
  cell. `spine_parts` → `region_parts`, `neck_contacts` →
  `attachment_contacts` (`neck_points` → `attachment_points`).
  `load_spine` and `TSSpine` leave the module.
- **New `cabaret/toric.py`:** `load_toric_spine(name, data_dir, parts)`
  reads the TS files (`<name>.obj`, `_AZ.txt`, `_neckpoint.txt`,
  `_syn_axon_dist.txt`) and calls `build_reference`. It returns a
  `ReferenceModel`.
- **Field renames:** `model` → `fem`, `spine` → `network`, `neck_nodes` /
  `neck_ports` / `contacts` → `attachment_nodes` / `attachment_ports` /
  `attachment_contacts`, `az` → `sites`; `name` and `own_parts` are dropped.

## Steps

- [x] `reference.py`, exported from `cabaret/__init__.py`.
- [x] `graft.py` renames; `toric.py`.
- [x] Notebooks 02–05 and the design docs use the new names.
- [x] Notebook 03's TS1 numbers are unchanged (fit error 1.35%, clamped
      error 2.83%).
- [x] decisions.md entry; ruff, pytest, jupytext sync.

## Done when

- No core module mentions toric spines, necks or AZs except as examples,
  and notebooks 03–05 give the same numbers as before.

## Notes

- Checked: notebook 03 run headless gives the same TS1 numbers (fit error
  1.35%; neck-clamped error 2.83% at DC, 100 Hz and 1 kHz; spine-local
  R_in median 0.29%, max 7.6%). The `coarsen`, `tets=` and notebook 02
  paths load as before (721 and 2,044 neck nodes).
- Core docstrings keep necks and AZs only as examples ("e.g. a spine's
  neck"). `sites_from_points` still lists `"neck"` among the site kinds,
  and `PLACEHOLDER_PSD_AREA` is synapse-specific; both are left as is.
- Notebooks 02 and 04–05 were synced but not rerun in full (02 and 05 take
  long); their calls were checked by the smoke tests above.
- Tests: `tests/test_reference.py`. A cylinder with both flat ends as
  attachments: one electrode each, their membrane removed, a site patch
  off the electrodes, and the end-to-end resistance equal to r_L·L/A to
  2.4e-4 (slow, ~7 s). Plus a check that an empty attachment raises.
