# Testing rules

The package is in active development and APIs change often, so keep the suite
**small**. A handful of meaningful tests is better than broad coverage that has to
be rewritten every week. You decide when a test is worth it, and during development
that should be rare.

## Add a test when

- You fix a bug. Add a regression test that would have caught it.
- Code is numerically subtle: edge cases, tolerances, degenerate inputs, unit conversions.
- There is a known answer to check against: an analytic solution, a published value,
  an invariant (conservation, symmetry, idempotence), or agreement with a reference library.
- Other parts of the package depend on the function and its API has stabilized.

## Don't add a test for

- Thin wrappers around library calls, trivial getters, or plotting.
- Code that is still exploratory or likely to be rewritten soon.
- Every parameter combination. One or two representative cases are enough.

## How

- Write plain pytest functions in `tests/test_<module>.py`. Use `pytest.mark.parametrize` sparingly.
- Compare floats with `np.testing.assert_allclose` and set `rtol`/`atol` deliberately. Never compare floats with `==`.
- Use the `rng` fixture from `conftest.py` for random inputs.
- Keep tests fast. Mark anything that takes more than a few seconds with `@pytest.mark.slow`.
- Put small fixture files in `tests/data/`. Never depend on the gitignored top-level `data/`.
- When you add a test, mention it in your summary so the user knows the suite changed.
