# Validation

Back to the [design index](README.md).

## Reference ladder

1. **Analytic:** isopotential sphere ($R_\mathrm{in}$, $\tau$); sealed
   cylinder ($\cosh$ attenuation, $R_\mathrm{in} = r_a \lambda \coth(L/\lambda)$);
   ball-and-stick; Rall equivalent cylinders. `camovox/validate.py` has the
   sphere and the sealed cylinder.
2. **FEM reference:** agrees with level 1 and converges under mesh
   refinement, globally and locally at necks. Done for the sphere and
   cylinder in plan 001 and notebook 01; the numbers are in
   [fem-reference.md](fem-reference.md).
3. **Reduced model:** checked against level 2. Now: port matrices at DC,
   100 Hz and 1 kHz, and EPSPs at chosen synapses (TS1: 2.8% and 0.2%;
   [reduction.md](reduction.md)). With the refinement loop: on the **locked
   test set**, plus fresh uniform samples and `landmarks`.
4. **jaxley:** the exported cell reproduces the camovox reduced model
   exactly (plan 006).

## Metrics

- input and transfer resistance;
- |Z(f)| and phase for 1–1000 Hz;
- the slowest passive time constants;
- single-synapse EPSP peak, time-to-peak and half-width;
- multi-synapse pattern responses and the summation index *(proposal)*.

Each is evaluated in at least two membrane states: resting
(2.5×10⁻⁵ S/cm²) and high-conductance (2.63×10⁻⁴ S/cm²).

**Loop diagnostics** *(proposal)*: error against iteration and against
compartment count, the probe/test gap, and where sites were added.

**Fidelity diagnostics:** surface area, volume, genus and local deviation
before and after repair and tetrahedralization; thinnest-neck preservation;
element quality (the share of negative couplings).

## Test shapes

Sphere, cylinder, ball-and-stick, a spine on a sphere (stubby, with a ring),
a torus, the 9 toric spines (`data/`, also in `mascaf/data/mesh/processed`),
and the reconstructed cell.

## Baselines

MASCAF cable models, evaluated by us. BF's published voxel-model numbers as a
consistency check only; their MASCAF comparisons are set aside for now.
