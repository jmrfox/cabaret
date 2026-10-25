# Sites, sampling and scenarios

Back to the [design index](README.md).

A **site** is a membrane patch where current enters or voltage is read
(synapses, probes, landmarks). Real synapse data and sampled sites produce
the same objects, so analyses don't change when data changes. Code:
`cabaret/sites.py` (`MembraneSurface`, `Sites`, `sites_from_points`,
`sample_uniform`, `jitter_sites`, `ball_patch`).

## Loading real data *(implemented)*

- `TS*_AZ.txt` has one active-zone point (`x y z`, in pixels at 0.005 µm/px)
  per row. AZs are indexed by row number.
- `TS1_syn_axon_dist.txt` lists those row numbers (1-based), one presynaptic
  axon per line. `load_axon_groups` turns them into 0-based groups.
- `sites_from_points` snaps each point to the nearest membrane vertex. AZs
  are presynaptic, so the points sit 0.03–0.16 µm off the spine; the snap
  tolerance is 0.25 µm, and farther points raise an error. It then builds
  the patches.
- `build_reference` does this for any region (`site_points`), after
  meshing and tying its attachments into electrodes;
  `toric.load_toric_spine` calls it with a TS spine's files.

The proposed general loader (`load_sites`: CSV or Parquet with `x, y, z`,
`psd_area`, `axon_id`, ...) is not built yet.

## Patches *(implemented)*

A patch is a geodesic disk on the membrane, grown from the snapped vertex
(Dijkstra over surface edges) until its membrane area reaches the site's
area. Weights are the vertices' lumped areas, so injection is
uniform-density and readout is area-weighted
([fem-reference.md](fem-reference.md), section 5). Vertices without
membrane (e.g. the removed neck region) are never part of a patch.

## Samplers

| Sampler | Locations | Status |
|---|---|---|
| `sample_uniform` | area-weighted uniform over the membrane, with a minimum Euclidean spacing (placeholder for geodesic) | implemented |
| `jitter_sites` | each site moved within a geodesic radius, keeping labels | implemented |
| `poisson_disk` | uniform with a minimum geodesic spacing | proposal |
| `weighted` | density ∝ w(features), e.g. thickness, curvature, electrical features, or the loop's error field | proposal |
| `clustered` | Thomas/Neyman–Scott on the surface: axon "parents", then synapses around them, with synthetic `axon_id`s. The input generator for multi-axon spines; counts per axon as a distribution or an explicit list, e.g. TS1's 8, 4, 4, 2, 2, 1, 1, 1, 1, 1 | proposal |
| `empirical` | density learned from real synapses in feature space | later |
| `landmarks` | tips, branch points, neck bases, loop arms | proposal |

All samplers take `rng: np.random.Generator`.

## PSD area

| Sampler | Role |
|---|---|
| `fixed(0.05 µm²)` | **default, a literature-based guess.** There are no owl or avian PSD measurements, and the DAB fill hid the PSDs. Mammalian spine synapses average about 0.04 µm² (AZ ≈ PSD area; Schikorski & Stevens 1997), with skewed distributions (Bartol et al. 2015) |
| `lognormal(median, sigma)` | for uncertainty quantification later; a suggested median of 0.04 µm² |
| `morph_scaled(feature, ...)` | hypothesis scenarios |

Why the area matters:
- **local input resistance**, through the spreading term
  $\approx r_L/(4a)$ with $a = \sqrt{A/\pi}$: about 4.4, 2.0 and 1.0 MΩ for 0.01, 0.05
  and 0.2 µm² at r_L = 100 Ω·cm. That is percent-level against spine-local
  resistances of tens of MΩ;
- **surface resolution:** triangles must be small compared with the patch;
- **packing:** how many synapses fit;
- **synaptic strength**, if $g_\mathrm{max} \propto$ area.

An `area_sweep` (proposal) would recompute site metrics over 0.01–0.3 µm² to
measure how much conclusions depend on the guess.

## Scenarios *(proposal)*

A `Scenario` is a reproducible recipe (samplers + parameters + seed) in
TOML. Its name and seed go into every output's metadata.

```toml
name = "clustered_axons"
seed = 7
[locations]
sampler = "clustered"
n_axons = 10
synapses_per_axon = { counts = [8, 4, 4, 2, 2, 1, 1, 1, 1, 1] }  # TS1
total = 25
cluster_scale_um = 0.5
[areas]
sampler = "fixed"
area_um2 = 0.05
```

Planned built-ins: `uniform`, `clustered_axons`, `distal_bias`,
`neck_bias`, `isolated_bias`, `landmarks`, `area_sweep`.

## Synapse placement analysis

Does morphology relate to synapse placement? This is always run on the
**FEM reference**, never on a reduced model fitted to the same sites.
- **Implemented for TS1** (notebook 02): loopiness and spine-local
  resistance of the real AZs against 100 uniform re-draws and geodesic
  jitter, reported as percentiles.
- **Proposal:** a null model matched to the AZs' geodesic spacing and axon
  clustering; axon-level questions (are same-axon synapses electrically
  closer than chance?); effect sizes with intervals. Power is modest with
  about 9 spines.

## Activation patterns, shared across models *(proposal)*

Activation patterns are tables of site, time and weight, plus a
`SynapseModel` (kinetics, reversal, $g_\mathrm{max}$). They are defined on
sites, so the same pattern drives the FEM, the cabaret reduced model and
jaxley.
- **Pattern samplers:** `single_site`, `axon_synchronous` (jitter, release
  probability), `poisson_axons`, `random_subset`, planted patterns.
- **Conductance synapses** sum nonlinearly (driving force, shunting).
- **FEM transients with synapses:** backward Euler, with low-rank (Woodbury)
  updates on the patch rows.
- **Metrics:** peak, time-to-peak, half-width, integral, and the summation
  index (pattern response divided by the sum of single responses).

What exists now: current injection transients (`simulate_current`). Synapse
biology is deferred (open question 13): synapses are expected to be
excitatory, the AMPA:NMDA ratio is unknown, and inhibition was previously
modelled as effective inputs to the soma.
