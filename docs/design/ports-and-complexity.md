# Ports, bounds and loopiness

Back to the [design index](README.md).

How a spine is characterized electrically before any reduction, and what
that says about the reduced model it needs. Code: `cabaret/ports.py`,
`cabaret/circuit.py` (`port_impedance`, `PortSolver`). Applied to TS1 in
notebook 02 (plans 003–004).

## Port impedance matrices *(implemented)*

A **port** is a site where current enters or voltage is read: each synapse
patch, plus the neck electrode ([fem-reference.md](fem-reference.md),
section 5). The **port impedance matrix** $Z_{ij}(f)$ is the voltage at port i
per unit current injected at port j:
- the diagonal is each synapse's input resistance;
- off-diagonals are transfer resistances (cross-talk);
- the neck column is what reaches the parent.

Two versions:
- **isolated spine** (neck floating, nothing attached). This is the complete
  multiport description of the spine, and **the target a reduced model is
  fitted to**. A passive spine joined to its parent only through its neck
  is exact under any parent once this multiport is matched;
- **neck clamped** (Schur complement on the neck). This is the spine's
  internal structure, and it doesn't depend on any parent.

## Effective complexity of a toric spine

**The question:** given a target behaviour and a tolerance ε, what is the
smallest reduced model that reproduces the FEM within ε? That means both its
**compartment count** and its **configuration**: topology, which loops are
kept, and where each synapse attaches.

**Lower bounds on the count, from the FEM alone** *(implemented)*:
- **Rank bound.** If every port of a reduced model sits on one of m
  compartments, its port matrix has rank at most m. The best rank-m
  approximation error of the FEM matrix (Eckart–Young) lower-bounds the
  error of any such model (`rank_bound`).
- **Diagonal + low-rank bound.** Each synapse also has a private series
  resistance (its patch's spreading resistance) that only full rank can
  capture in the rank bound. Allowing Z ≈ D + L Lᵀ, with D a non-negative
  diagonal and L of rank m, gives a fairer floor (`diag_lowrank_bound`;
  alternating principal-axis factoring, which estimates a local optimum).

**Achieved models, from reduction plus fitting:** see
[reduction.md](reduction.md). The gap between the achieved front and the
bound shows whether configuration is the limitation. The gap between
tree-only and looped models measures the electrical value of the loops.

| Target behaviour | What must match | Expected complexity |
|---|---|---|
| Somatic response to each synapse | $Z_{i,\mathrm{neck}}$ | low: neck-dominated |
| Local voltage at each synapse (NMDA, plasticity) | $Z_{ii}$ | higher |
| Interactions between synapses (shunting, nonlinear summation, cross-talk) | all $Z_{ij}$, plus conductance-synapse transients | highest |

**TS1 results** (notebooks 02–03; neck electrode; PSD placeholder 0.05 µm²):
- spine-local resistance 7.7–30.5 MΩ, a 4× spread;
- the isolated spine seen from the neck is about 5.3 GΩ (spine membrane
  only);
- 100 Hz and 1 kHz matrices are almost identical to DC (the spine is
  electrically small);
- rank bound: m ≥ 7 for 10% and m ≥ 13 for 5%;
- diagonal + low-rank bound: m = 8 gives 1.3%, and m = 12 gives 0.5%.

## Loopiness: the four-point condition *(implemented)*

**Resistance distance** $d_{ij}$ is the resistance between ports i and j with
the membrane removed (neck grounded, neck = port 0). For any four ports,
pair them up the three possible ways and sort the summed distances
a ≥ b ≥ c. The distances come from a tree with internal branch points
**if and only if a = b for every quadruple** (Buneman 1974).

Three normalizations (`four_point_violation`):
- **relative** (a − b)/a: the error a tree must accept on these distances
  (0 for a tree; 0.25 for four *points* evenly spaced on an ideal ring);
- **intrinsic** (a − b)/(a − c): unchanged by adding a series resistance to
  any port, so it measures topology (0 for a tree; 1 for an ideal ring).
  It is unstable when a ≈ c;
- **absolute** a − b, in Ω.

Patches dilute the relative measure, because each port's spreading
resistance acts like a twig on a tree. On the torus, four evenly spaced
patches give 0.159 relative but 0.996 intrinsic. `loopiness_index`
summarizes each measure over all quadruples (mean, 90th percentile, max).

**TS1:**
- relative violation: 90th percentile 0.011, max 0.098;
- intrinsic violation: 90th percentile 0.21, max 0.99;
- absolute gap: max 4.5 MΩ, against a median port distance of 28 MΩ.

So there is real loop structure, but it is electrically weak. Against 100
uniform re-draws and jitter, the real AZs are on the low side but not
significantly. An earlier "intrinsic maximum" lead turned out to be an
artifact of a placeholder neck patch.

## Coupling and lobes *(implemented)*

The **coupling coefficient** $Z_{ij}/\sqrt{Z_{ii}Z_{jj}}$ (neck clamped) lies in
[0, 1]. Average-linkage clustering on 1 − coupling (`electrical_groups`)
finds groups of synapses that strongly affect each other.

TS1 splits into **two lobes** (AZs 0–12 and 13–24):
- coupling within a lobe: median 0.40;
- coupling across lobes: median 0.010;
- **no axon has synapses in both lobes**.
