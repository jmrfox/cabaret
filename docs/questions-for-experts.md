# Questions for the anatomy/physiology experts (pinned, not yet sent)

Drafted 2026-10-23 after a literature check. **On hold:** the user decided to
continue with literature-based guesses (see `design.md`, "Area samplers") and
revisit these later, possibly as part of uncertainty quantification.

## Background from the literature check

- In Sanculi et al. 2020 only the postsynaptic SSN was DAB-filled (retrograde
  microruby from the optic tectum, streptavidin-HRP + DAB). The dark product
  hid the PSDs ("an unworkable requirement"). Presynaptic axons were
  unlabeled. AZs were scored from the presynaptic side: ≥5 vesicles,
  apposed membranes, seen in ≥2 sections (5 nm pixels, 70 nm z-steps). No
  synapse-size measures were reported.
- AZ and PSD areas are closely matched (one "synaptic apposition surface",
  PMC3701254). AZ area is therefore a usable proxy for PSD area.
- Mammalian reference: mouse CA1 spine synapses average about 0.04 µm² (AZ
  and PSD), with skewed distributions. Sizes correlate with docked vesicles
  and bouton vesicle content (Schikorski & Stevens 1997; Bartol et al. 2015).
  No avian or owl PSD measurements were found.

## Questions

**Measuring synapse size from the existing data**
1. With unlabeled presynaptic boutons, can the active-zone extent
   (apposition length per section × number of 70 nm sections) be measured
   per AZ?
2. Are the presynaptic boutons segmented, or could they be (volume, vesicle
   counts as size proxies)?
3. Is there unlabeled tissue in this volume, or other owl ICx EM, with
   visible PSDs, to give an owl-specific size distribution?

**AZ files**
4. *(Answered by the user)* Each AZ row's xyz is an AZ location. AZs are
   indexed implicitly by row number, and `*_syn_axon_dist.txt` lists those
   row numbers, one presynaptic axon per line.
5. Bifurcated synapses (one bouton, two or more AZs): one point per AZ?
6. AZ files for TS21, TS24 and other spines? Axon assignments beyond TS1?

**Biology**
7. Expected synapse size on toric spines compared with typical spine
   synapses? Perforated synapses?
8. *(Partly answered)* Spine synapses are expected to be excitatory. The
   AMPA:NMDA ratio is unknown. Inhibition was previously modelled as
   effective inputs to the sink/soma.
9. Do strong inputs (high input fraction) have larger synapses?

**Reconstruction**
10. *(Answered by the user)* Neck points may have relatively large errors.
    They were placed on the postsynaptic membrane near or on the side of a
    neck, with the parent location judged by eye.
11. Near necks, which tracing is more reliable: the TS meshes or the whole
    cell?

**Physiology**
12. *(Data exists)* ICx electrophysiology is available for later calibration
    of synaptic conductances.
