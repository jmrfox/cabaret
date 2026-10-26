# 010: Faster solves on fine meshes

**Status:** draft (pinned 2026-10-26; not started)
**Design sections:** [fem-reference.md](../design/fem-reference.md),
[implementation.md](../design/implementation.md)

## Goal

Make port-matrix solves on fine meshes (around 150k nodes and up) take
minutes, not tens of minutes, especially across several frequencies.
Notebooks on the 0.2 µm mesh are already fast (35–78 s), so this only
matters for fine-mesh studies (`scripts/`).

## Measurements (TS1, 0.05 µm mesh, 154k nodes, plan 008)

| Step | Time |
|---|---|
| `port_impedance` at DC (real `splu`) | 2.3 min |
| `port_impedance` at 100 Hz and 1 kHz (complex `splu`) | about 17 min each |
| `ResistanceMetric` (another factorization; memory 0.5 → 2.9 GB) | 2 min |
| `assign_membrane` (another factorization) | 2 min |
| `attachment_contacts` (trimesh ray containment, 2.2 ms per point) | 80 s |

All solves go through `scipy.sparse.linalg.splu` (SuperLU, single-threaded,
default `COLAMD` ordering), and each one factorizes from scratch.

## Priorities

- **Multi-frequency fine-mesh work:** AC dominates (34 of 42 minutes in
  notebook 04's first cell). The real fix is the Krylov reduction below.
- **The convergence script** is DC only, so faster DC factorization is what
  helps it. The Krylov reduction also builds on a factorization of G, so
  DC speedups carry over.

## Candidate approaches (untested)

The system is A(ω) = G + jωC. G (FEM Laplacian plus membrane leak) is
symmetric positive definite. C is diagonal and nonzero only on surface
nodes, so it is singular.

1. **Cholesky with nested-dissection ordering** (CHOLMOD via
   `scikit-sparse`). No pivoting needed, about half of LU's cost,
   multithreaded. Quick first check: `splu(a, permc_spec="MMD_AT_PLUS_A")`.
2. **Krylov model reduction across frequency** (block Lanczos/Arnoldi on
   G⁻¹C from G⁻¹B; PVL/PRIMA). Factor G once and match moments of
   Z(s) = Bᵀ(G + sC)⁻¹B around DC. Every frequency is then nearly free.
   PRIMA keeps the reduced model passive. Z barely changes up to 1 kHz
   (0.01%), so a few block iterations should be enough.
3. **Preconditioned iterative solves:** conjugate gradient with an
   algebraic multigrid preconditioner (`pyamg`) for the 26 right-hand
   sides, roughly O(N); COCG for the complex case. The singular C rules out
   simple shifted-system tricks.
4. **Reuse instead of refactoring:** Schur complements or Woodbury updates
   between the clamped, membrane-free and full systems. The neck sink node
   in `ResistanceMetric` links to about 15k nodes, which likely causes the
   memory jump. Eliminate it last, or apply a Dirichlet clamp instead.

## Steps

- [ ] Benchmark harness: time DC and one AC `port_impedance` on the 0.1 and
      0.05 µm TS1 meshes (in `scripts/`).
- [ ] Try `permc_spec` options in `splu`, then CHOLMOD (decide on the
      `scikit-sparse` dependency; note it in decisions.md).
- [ ] Prototype the block-Krylov frequency sweep and check it against direct
      solves at 0, 100 and 1000 Hz.
- [ ] Look at the neck sink node's fill and memory.

## Done when

- Port matrices at three frequencies on the 0.05 µm TS1 mesh take a few
  minutes and match the direct solves to within 0.01%.
