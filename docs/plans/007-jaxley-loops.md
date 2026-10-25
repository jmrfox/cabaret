# 007: Close loops in exported jaxley models

**Status:** done
**Design sections:** [Reduction](../design/reduction.md) (export to jaxley,
default policy), open question 8 in [open-questions.md](../design/open-questions.md),
decision "Add jaxley; close loops by extra compartment edges" in
[decisions.md](../decisions.md)

## Goal

`to_jaxley` can export a tree plus a few loop-closing resistances, and the
result is simulated correctly. The private-API workaround lives in one
function (`close_loop`), and a guard stops looped cells from running on a
solver that assumes a tree.

## Approach

- **Why a dummy compartment:** in jaxley 0.14 an extra compartment edge has
  no conductance of its own. Its resistance is
  `resistive_load_out(source) + resistive_load_in(sink)` (or the mirror,
  depending on `ordered`), and our export already uses `resistive_load_in`
  for each compartment's parent edge. So each loop a–b gets a dummy
  compartment D with no membrane, attached in the tree as a child of a. The
  closing edge D ↔ b is oriented so it uses `load_in(D) + load_out(b)`.
  With `load_in(D) = R/2`, the path a → D → b is R plus about 2 Ω of
  branch-point loads.
- **Guard:** `JaxleyModel` records whether it has loops. `integrate(jm, ...)`
  runs `jx.integrate` with `jax.sparse` when it does, and raises if another
  voltage solver is requested. `jaxley.dhs` returns wrong voltages for
  looped cells without warning.

## Steps

- [x] Check the dummy-compartment orientation numerically on a small ring.
- [x] `jaxley_io.close_loop(cell, dummy, other)`: append the two edge rows,
      re-initialize the solvers.
- [x] `to_jaxley(..., loops=[(node_a, node_b, R), ...])`: create the dummy
      branches, set their loads, close each loop.
- [x] `jaxley_io.integrate(jm, **kwargs)` with the solver guard.
- [x] Test (slow, jaxley compile): a ring network, exported as tree + one loop, matches
      `simulate_current`.
- [x] Test (slow): FEM torus with four electrode ports, Kron-reduced at DC
      to a 4-node network (tree + 3 loops); jaxley steady state matches the
      FEM port impedances, and dropping the loops does not.
- [x] Update reduction.md, open question 8, decisions.md.

## Done when

- Both tests pass; ruff and pytest are clean.

## Notes

- Measured: the ring export matches `simulate_current` to 3e-8 (relative
  to peak); the torus export matches the FEM DC port impedances to 2e-6,
  and the path tree alone is 18% off. Both tests are marked slow (~19 s).
- The orientation that works: rows `(other → dummy, ordered 1)` and
  `(dummy → other, ordered 0)`; both give `load_out(other) + load_in(dummy)`.
- `close_loop` must run after every other morphology change, since it edits
  the private `_comp_edges` table and re-initializes the solvers.
- Kron reduction of the FEM torus onto four rim electrodes gave only
  positive couplings (ring ≈ 245 nS, diagonals ≈ 55 nS), so no negative
  resistances had to be exported. That is not guaranteed in general.
