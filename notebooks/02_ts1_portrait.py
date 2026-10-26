# %% [markdown]
# # 02 · TS1 electrical portrait
#
# **TS1** is a toric spine on a space-specific neuron in the barn owl
# auditory midbrain (Sanculi et al. 2020, *eNeuro*). It is about 11.5 µm
# long, has about 7 holes, and receives **25 synapses (active zones, AZs)
# from 10 presynaptic axons**. Its shape is far from the tree-like
# morphologies standard cable models assume.
#
# This notebook builds TS1's **reference model** (a 3D finite-element model
# of its interior; see notebook 01) and measures the electrical properties a
# **reduced cable model** of TS1 would have to reproduce:
#
# 1. how **heterogeneous** its synapses are electrically (section 4);
# 2. whether it splits into **electrically separate parts** (section 5);
# 3. the **fewest compartments** any reduced model needs for a given
#    accuracy (section 7);
# 4. whether its **holes matter**, or whether a loop-free tree of resistors
#    could stand in for the spine (section 8);
# 5. whether the **real synapse positions** are special compared with random
#    ones (sections 9–10).
#
# TS1 is analysed **on its own**, with its neck as a port, and with no model of
# the soma or dendrite it attaches to. A passive spine joined to its parent
# only through the neck is fully described this way: whatever parent it is
# later connected to, its behaviour follows exactly. The neck port is where
# the separately traced TS1 mesh meets the whole-cell mesh (section 2).
#
# Everything is passive (leak + capacitance), in the high-conductance
# (in vivo) membrane state. Terms are defined in notebook 01's glossary.
# Runtime is about 1–2 minutes (meshing is cached after the first run).

# %%
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

import cabaret as cb
from cabaret import graft, ports, viz

pv.set_jupyter_backend("static")  # 3D views as static images in the notebook

# %% [markdown]
# ## Parameters and placeholders
#
# All choices in one place. **Placeholders** are values not yet backed by
# data. Results that depend on them say so.
#
# | Name | Value | Meaning and source |
# |---|---|---|
# | `SCALE` | 0.005 µm/px | All data files are in 5 nm SBEM pixel coordinates |
# | `MEMBRANE` | r_L 100 Ω·cm, c_m 1 µF/cm², g_pas 2.63×10⁻⁴ S/cm² | High-conductance (in vivo) passive membrane, as in the earlier voxel study of this cell ("BF") |
# | `PSD_AREA` | 0.05 µm² | **Placeholder** synapse patch area. PSD sizes were not measured in the EM data |
# | `CONTACT_DISTANCE` | 0.2 µm | How close TS1's surface must come to the whole-cell mesh to count as attached. It allows for mismatch between the two tracings (AZ points sit 0.03–0.1 µm off TS1, neck points 0.1–0.45 µm off the cell) |
# | `CONTACT_RADIUS` | 3 µm | Contacts are only searched for this close to the neck point, so places where the spine merely lies against other cell parts are ignored |
# | `EDGE`, `ENVELOPE` | 0.2 µm, 0.005 µm | Tetrahedron size, and how far the meshed surface may move from the reconstruction. The envelope is kept far below the thinnest neck radius (about 0.06 µm). At 0.2 µm, spine-local quantities read about 1% low against a 0.05 µm mesh (notebook 05). The finer mesh takes about 40 minutes to build, too slow for a demo |
# | `rng` | seed 0 | Random generator for the null models, so results are reproducible |
#
# There is no parent placeholder: the spine is characterized on its own.

# %%
SCALE = 0.005  # µm per pixel
MEMBRANE = cb.Passive(r_L=100.0, c_m=1.0, g_pas=2.63e-4)
PSD_AREA = 0.05  # µm², placeholder
CONTACT_DISTANCE, CONTACT_RADIUS = 0.2, 3.0  # µm
EDGE, ENVELOPE = 0.2, 0.005  # µm
I_INJ = 10e-12  # test current for the voltage map: 10 pA
rng = np.random.default_rng(0)

# %% [markdown]
# ## 1. Mesh and FEM reference model
#
# TS1's reconstructed surface (`TS1.obj`) is filled with tetrahedra. The FEM
# model is an electrical network on the mesh vertices: interior couplings
# come from the tetrahedra, and each surface vertex carries a share of the
# membrane.
#
# **What to check:**
# - **area and volume change** say how faithfully the tetrahedral mesh
#   follows the reconstruction; below 1% is good;
# - **negative couplings** come from badly shaped tetrahedra. The FEM
#   answer is still accurate (notebook 01), but later reduction steps need
#   to tolerate them.
#
# Meshing takes about 2 minutes the first time. The result is cached in
# `../data/cache/`; delete that file to re-mesh with other settings.

# %%
mesh = cb.load_mesh("../data/TS1.obj", scale=SCALE)
tets, fidelity = cb.tetrahedralize_cached(
    mesh,
    f"../data/cache/TS1_e{EDGE:g}_env{ENVELOPE:g}.npz",
    edge_length=EDGE,
    envelope=ENVELOPE,
)
model, quality = cb.fem_model(tets)  # FEM network (one node per mesh vertex)

print(f"{fidelity.n_tets} tetrahedra, {model.n_nodes} mesh vertices")
print(f"membrane area {fidelity.area_out:.1f} µm² ({100 * fidelity.area_error:+.2f}%)")
print(f"volume {fidelity.volume_out:.2f} µm³ ({100 * fidelity.volume_error:+.2f}%)")
print(f"negative couplings carry {100 * quality.negative_fraction:.1f}% of total |κ|")

# %% [markdown]
# ## 2. The neck port: where TS1 meets the cell
#
# TS1 was traced **separately** from the whole cell (`cell.obj`), so the two
# meshes don't share a cut face. The whole cell has been segmented into
# parts (by the neuromesh-cgal pipeline: `../data/neuromesh/parts/`):
# - the **cell's own version of TS1** = the parts mostly covered by the
#   TS1 mesh;
# - the **parent** = all other parts (soma, dendrites).
#
# Near the neck point, the TS1 tracing reaches **into** the parent: its base
# overlaps the soma or dendrite. Everything of TS1 that lies inside the parent,
# or within `CONTACT_DISTANCE` of it, is really parent cytoplasm. Those FEM
# nodes are tied together into one **electrode**, the neck port, and their
# membrane is removed. Electrically the parent is nearly isopotential at
# spine scale, so an equipotential electrode is a good stand-in.

# %%
parts = graft.load_parts("../data/neuromesh/parts", scale=SCALE)
own_parts = graft.region_parts(mesh, parts)  # the cell's own tracing of TS1
neck_points = cb.load_points("../data/TS1_neckpoint.txt", SCALE)
(contact,) = graft.attachment_contacts(
    tets,
    parts,
    own_parts,
    neck_points,
    distance=CONTACT_DISTANCE,
    radius=CONTACT_RADIUS,
)
spine, neck_node = cb.add_electrode(model, contact)  # TS1 with its neck port
neck_port = cb.point_port(spine, neck_node)  # port weights for the neck
removed = model.area[contact].sum()
print(f"cell parts that are TS1's own tracing: {own_parts.tolist()}")
print(
    f"neck port: {len(contact)} FEM nodes tied together; "
    f"{removed:.2f} µm² of membrane removed ({100 * removed / model.area.sum():.0f}% of TS1)"
)

# %% [markdown]
# ## 3. Sites: active zones
#
# - **Active zones** (`TS1_AZ.txt`): one point per synapse. AZs are
#   *presynaptic*, so the points sit just outside the spine membrane, across
#   the synaptic cleft. Each is **snapped** to the nearest membrane vertex
#   (the snap distance is printed) and grown into a membrane patch of
#   `PSD_AREA` along the surface.
# - **Axons** (`TS1_syn_axon_dist.txt`): which of the 10 presynaptic axons
#   made each synapse. Axons are numbered 0–9 in the order of that file.
#
# The view shows each AZ as a sphere coloured by axon, and the neck port
# (its tied region) as −1. Patches are placed only on real membrane, so
# never on the removed neck region.

# %%
membrane = cb.MembraneSurface.from_model(tets, spine)  # real membrane, for patches
az = cb.sites_from_points(
    membrane,
    cb.load_points("../data/TS1_AZ.txt", SCALE),
    "synapse",
    area=PSD_AREA,
    axon_groups=cb.load_axon_groups("../data/TS1_syn_axon_dist.txt"),
)
axon = az.table.axon_id.to_numpy(dtype=float)  # axon of each AZ, 0–9
print(
    f"{len(az)} AZs; snapped {az.table.snap_distance.min():.3f}–"
    f"{az.table.snap_distance.max():.3f} µm onto the membrane"
)
neck_region = np.zeros(spine.n_nodes)
neck_region[contact] = 1.0  # for display: the tied region's centre
pl = viz.plot_sites(
    tets, [*az.patches, neck_region], labels=[*axon, -1], name="axon (neck = -1)"
)
pl.show()

# %% [markdown]
# ## 4. Port impedance matrix
#
# A **port** is a site through which current can enter or voltage can be
# read: the 25 AZs plus the neck. The **impedance matrix** Z holds, in row i
# and column j, the voltage at port i per unit current injected at port j:
# - the **diagonal** $Z_{ii}$ is the *input resistance* at synapse i, which
#   sets the size of its local EPSP;
# - **off-diagonal** $Z_{ij}$ are *transfer resistances*: how much synapse j
#   depolarizes synapse i (cross-talk);
# - the **last column** is what each synapse delivers to the neck, and so to
#   the parent.
#
# Z is computed at DC and at 100 Hz and 1 kHz (roughly the frequency content
# of synaptic signals).
#
# Two versions of Z:
# - **Isolated spine** (neck port floating, nothing attached): the complete
#   description of TS1 as a multiport, and the **target a reduced spine model
#   is fitted to**. Its entries are large (GΩ), because an isolated spine can
#   only lose current through its own small membrane.
# - **Neck clamped** (neck held at rest, as if attached to a huge parent):
#   what is left is the spine's **own internal structure**, how synapses
#   interact inside the spine. It is mathematically derived from the
#   isolated matrix (a Schur complement), so it carries no extra assumptions.
#
# **How to read the heat maps:** ports are in the order of the AZ file, with
# the neck last on the left plot. Bright blocks are groups of synapses that
# strongly affect each other. A uniform matrix would mean the spine acts as
# one compartment.

# %%
freqs = [0.0, 100.0, 1000.0]  # Hz
z = cb.port_impedance(
    spine, MEMBRANE, [*az.patches, neck_port], freqs=freqs
)  # (3, 26, 26), Ω
z_dc = z[0].real
z_clamped = ports.clamp_port(z_dc, len(z_dc) - 1)  # neck held at rest -> (25, 25)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for ax, mat, title in [
    (axs[0], z_dc, "isolated spine, DC (MΩ); last = neck"),
    (axs[1], z_clamped, "Z with neck clamped (MΩ)"),
]:
    im = ax.imshow(mat / 1e6, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("port j (injected)")
    ax.set_ylabel("port i (recorded)")
    fig.colorbar(im, ax=ax)
plt.show()

r_local = np.diag(z_clamped) / 1e6  # spine-local input resistance at each AZ, MΩ
print(
    f"isolated spine, seen from the neck: {z_dc[-1, -1] / 1e6:.0f} MΩ (spine membrane only)"
)
print(
    f"spine-local resistance (neck clamped): {r_local.min():.1f}–{r_local.max():.1f} MΩ "
    f"({r_local.max() / r_local.min():.1f}× spread)"
)

# %% [markdown]
# **Reading:** the spine-local resistance varies about 4× across the 25
# synapses. Some AZs are electrically close to the neck, others sit behind
# thin loop arms. A single-compartment spine model would give all of them
# the same value. The 100 Hz and 1 kHz matrices are almost identical to DC,
# because the spine is electrically small at these frequencies. They are
# used in section 7.

# %% [markdown]
# ## 5. Two electrical lobes
#
# The clamped heat map shows two bright blocks. To quantify this, define the
# **coupling coefficient** between two synapses, $Z_{ij}/\sqrt{Z_{ii}Z_{jj}}$:
# 1 if they always see the same voltage, 0 if they don't influence each
# other. Clustering the synapses on this coefficient (into two groups)
# finds the blocks without hand-picking them.

# %%
coupling = ports.coupling_coefficients(z_clamped)
lobe = ports.electrical_groups(z_clamped, n_groups=2)  # lobe label (0 or 1) per AZ
off_diag = ~np.eye(len(lobe), dtype=bool)
same_lobe = (lobe[:, None] == lobe[None, :]) & off_diag
across = lobe[:, None] != lobe[None, :]
for k in (0, 1):
    members = np.flatnonzero(lobe == k)
    axons_k = sorted({int(a) for a in axon[members]})
    print(f"lobe {k}: AZs {members.tolist()} | axons {axons_k}")
print(
    f"coupling within a lobe: median {np.median(coupling[same_lobe]):.2f} "
    f"(min {coupling[same_lobe].min():.2f})"
)
print(
    f"coupling across lobes:  median {np.median(coupling[across]):.3f} "
    f"(max {coupling[across].max():.2f})"
)
pl = viz.plot_sites(
    tets, [*az.patches, neck_region], labels=[*lobe, -1], name="lobe (neck = -1)"
)
pl.show()

# %% [markdown]
# **Reading:** TS1 behaves as **two lobes** joined at the neck. Synapses in
# the same lobe are strongly coupled; synapses in different lobes barely
# interact once the neck is held fixed. Each lobe gets inputs from a
# different set of axons, and no axon has synapses in both. A reduced model
# can start from this structure: the neck plus one sub-tree per lobe.

# %% [markdown]
# ## 6. Voltage map for one synapse
#
# 10 pA into the AZ with the largest spine-local resistance (the most
# electrically isolated synapse), with the neck held at rest. Colour shows
# the voltage over the membrane. Where the colour changes fastest is where the resistance is:
# thin necks and arms. Regions of nearly uniform colour are close to
# isopotential and could each be one compartment.

# %%
iso = int(np.argmax(r_local))  # index of the most isolated AZ
print(
    f"most isolated AZ: #{iso} (lobe {lobe[iso]}, axon {int(axon[iso])}), "
    f"{r_local[iso]:.1f} MΩ spine-local"
)
clamped = cb.attach_sink(
    spine, neck_port, resistance=1.0
)  # neck held at rest (1 Ω to ground)
w = az.patches[iso]
v = cb.solve_dc(clamped, MEMBRANE, I_INJ * w / w.sum())[
    : len(tets.points)
]  # V, mesh vertices
pl = viz.plot_surface(tets, 1e3 * v, name="V (mV), 10 pA into the most isolated AZ")
pl.show()

# %% [markdown]
# ## 7. Rank bound: the fewest compartments for a given error
#
# Suppose a reduced model puts each synapse on one of m compartments. Its
# impedance matrix between synapses then has **rank at most m**. The best
# rank-m approximation of the FEM matrix (from its singular values) is
# therefore a **lower bound** on the error of *any* such model, however it
# is wired. The plot shows that bound (relative error, Frobenius norm)
# against m.
#
# **How to read it:**
# - where a curve crosses your tolerance (grey lines at 5% and 1%) gives a
#   floor on the number of compartments that carry synapses;
# - the **neck-clamped** curve is the one that describes the spine's
#   internal structure. The isolated matrix is dominated by one large shared
#   term (the spine's total membrane, seen through the neck), which makes it
#   look deceptively simple;
# - the 100 Hz and 1 kHz curves lie on the DC curve, so faster synaptic
#   signals don't need extra structure.
#
# **Caveat: this bound is pessimistic.** Each synapse patch has its own
# spreading resistance (current must fan out from a 0.05 µm² disk, about
# 2 MΩ). That adds a separate term for every synapse, which only full rank
# can capture. A reduced model represents it cheaply as one small resistor
# per synapse, not as extra compartments. A "diagonal + low-rank" bound that
# separates the two is the next analysis step.

# %%
fig, ax = plt.subplots(figsize=(5, 3.5))
for label, mat in [("isolated spine, DC", z_dc), ("neck clamped, DC", z_clamped)]:
    err = ports.rank_bound(mat)  # err[m] = lower bound for m compartments
    ax.semilogy(np.arange(len(err) - 1), err[:-1], "o-", ms=3, label=label)
for f, z_f in zip(freqs[1:], z[1:], strict=True):
    err = ports.rank_bound(ports.clamp_port(z_f, len(z_f) - 1))
    ax.semilogy(
        np.arange(len(err) - 1), err[:-1], ".--", ms=3, label=f"neck clamped, {f:g} Hz"
    )
for tol in (0.05, 0.01):
    ax.axhline(tol, color="gray", lw=0.8)
ax.set_xlabel("m (compartments carrying synapses)")
ax.set_ylabel("lower bound on relative error")
ax.legend(frameon=False, fontsize=8)
plt.show()
err_clamped = ports.rank_bound(z_clamped)
for tol in (0.1, 0.05, 0.01):
    m = int(np.argmax(err_clamped <= tol))
    print(f"error ≤ {tol:.0%} needs m ≥ {m} (neck clamped)")

# %% [markdown]
# ## 8. Loopiness: could a loop-free tree stand in for the spine?
#
# Simulators handle trees of compartments natively, and loops only with
# workarounds. So it matters whether TS1's holes change its electrical
# behaviour in a way no tree can mimic. The test (worked through on a ring
# in notebook 01, section 5) uses:
#
# - **resistance distance** $d_{ij}$: the resistance between ports i and j
#   with the membrane removed (neck = port 0);
# - the **four-point condition**: for four ports, pair them up in the three
#   possible ways, add the two distances in each pairing, and sort the sums
#   a ≥ b ≥ c. The distances come from a tree with internal branch points
#   **if and only if a = b for every quadruple** (Buneman 1974; the same
#   test is used to fit phylogenetic trees).
#
# ```
#   i         k        d_ij + d_kl  = smallest
#    \       /         d_ik + d_jl  = largest  } equal in a tree: both
#     o-----o          d_il + d_jk  = largest  } cross the middle once
#    /       \
#   j         l
# ```
#
# A loop gives current two routes, which shortens some distances but not
# others, so a and b come apart. Two measures per quadruple:
#
# - **relative violation** (a − b)/a: the error a tree must accept on these
#   distances (0 = tree, 0.25 = four points evenly spaced on an ideal ring).
#   This is the one that matters for **model accuracy**;
# - **intrinsic violation** (a − b)/(a − c): ignores each synapse's own
#   spreading resistance, which acts like a twig a tree can always add
#   (0 = tree, 1 = ideal ring). It asks whether there is loop structure at
#   all, however weak.
#
# The **loopiness index** summarizes each measure over all
# C(26, 4) = 14,950 quadruples of the 26 ports: mean, 90th percentile
# (`p90`) and maximum.

# %%
distances = ports.ResistanceMetric(spine, MEMBRANE, ground=neck_port)  # factorized once
d_real = distances.distances(az.patches)  # (26, 26) Ω; row/col 0 = neck
MEASURES = {"relative": "max", "intrinsic": "gap"}  # display name -> normalization
loopiness_real = {m: ports.loopiness_index(d_real, n) for m, n in MEASURES.items()}
for m, li in loopiness_real.items():
    print(f"{m:>9} violation:", {k: round(v_, 4) for k, v_ in li.items()})

gap = ports.four_point_violation(d_real, "none")  # a − b in Ω, per quadruple
pair_d = d_real[np.triu_indices(len(d_real), 1)]  # all port-to-port distances
print(
    f"absolute gap a − b: 90th percentile {np.quantile(gap, 0.9) / 1e6:.2f} MΩ, "
    f"max {gap.max() / 1e6:.2f} MΩ"
)
print(f"typical port-to-port distance: {np.median(pair_d) / 1e6:.1f} MΩ (median)")

# %% [markdown]
# Where would a tree be most wrong? The four ports with the largest absolute
# gap, shown as spheres (the label is just the order within the quadruple).
# Expect them to straddle a hole, where current has two comparable paths.
# Check this in the view rather than assuming it.

# %%
quad, _ = ports.worst_quadruples(d_real, k=1, normalize="none")
quad = quad[0]  # port indices; 0 = neck, i > 0 = AZ number i − 1
names = ["neck" if i == 0 else f"AZ {i - 1}" for i in quad]
print("most loop-like quadruple:", ", ".join(names))
pl = viz.plot_sites(
    tets,
    [neck_region if i == 0 else az.patches[i - 1] for i in quad],
    name="most loop-like quadruple",
)
pl.show()

# %% [markdown]
# ## 9. Is the real synapse placement special?
#
# A low violation could be a property of TS1's shape (any synapses on it
# would look tree-like) or of where the real AZs happen to sit. Two "null
# models" separate these:
#
# - **Uniform re-draws** (100 draws): 25 sites placed at random over the
#   membrane. Same patch area, and at least 0.2 µm apart so patches don't
#   overlap. The removed neck region has no membrane, so no random site lands
#   there. Question: is the real placement unusually tree-like
#   compared with random placement?
# - **Jitter** (30 draws per radius): each real AZ moved to a random
#   membrane point within a given distance along the surface (0.25, 0.5, 1
#   or 2 µm). Question: do small changes to the real positions make loops
#   matter more (real = a finely tuned optimum), or is the real value
#   typical of its neighbourhood?
#
# **The percentile** is the share of null draws with a violation at or below
# the real one. About 50% means typical, and below about 5% would mean the
# real placement is unusually tree-like.

# %%
min_spacing = 0.2  # µm, see above


def random_sites() -> cb.Sites:
    """One uniform re-draw of 25 synapse sites."""
    return cb.sample_uniform(
        membrane,
        len(az),
        rng,
        area=PSD_AREA,
        min_spacing=min_spacing,
    )


d_uniform = [distances.distances(random_sites().patches) for _ in range(100)]
jitter_radii = [0.25, 0.5, 1.0, 2.0]  # µm along the surface
d_jitter = {
    r: [
        distances.distances(cb.jitter_sites(membrane, az, r, rng).patches)
        for _ in range(30)
    ]
    for r in jitter_radii
}


def percentile(null: list[np.ndarray], measure: str, stat: str) -> float:
    """Share (%) of null draws whose violation is at or below the real one."""
    norm = MEASURES[measure]
    real = loopiness_real[measure][stat]
    return 100 * np.mean([ports.loopiness_index(d, norm)[stat] <= real for d in null])


print(
    "percentile of the real AZs among null draws (50 = typical, < 5 = unusually tree-like)"
)
for m in MEASURES:
    for stat in ("mean", "p90", "max"):
        jit = ", ".join(
            f"{r:g} µm: {percentile(d_jitter[r], m, stat):.0f}" for r in jitter_radii
        )
        print(
            f"{m:>9} {stat:>4}: real {loopiness_real[m][stat]:.4f}; "
            f"uniform {percentile(d_uniform, m, stat):.0f}; jitter {jit}"
        )

fig, axs = plt.subplots(2, 3, figsize=(12, 6))
for row, (m, norm) in zip(axs, MEASURES.items(), strict=True):
    for ax, stat in zip(row, ["mean", "p90", "max"], strict=True):
        ax.hist(
            [ports.loopiness_index(d, norm)[stat] for d in d_uniform],
            bins=20,
            color="lightgray",
            label="uniform re-draws",
        )
        ax.axvline(loopiness_real[m][stat], color="C3", lw=2, label="real AZs")
        ax.set_xlabel(f"{m} violation, {stat}")
    row[0].set_ylabel("number of draws")
axs[0, 0].legend(frameon=False, fontsize=8)
fig.tight_layout()
plt.show()

# %% [markdown]
# **How to read the histograms:** grey bars are the 100 random placements,
# the red line is TS1's real AZs. Top row: relative measure; bottom row:
# intrinsic measure.
#
# **Interpretation:**
# - **The holes create real loop structure, but it is electrically weak.**
#   The intrinsic measure is large for some quadruples, so topologically
#   there are loops between some synapses. But the absolute gaps (90th
#   percentile below 1 MΩ, worst a few MΩ) are small next to typical
#   port-to-port distances (about 28 MΩ). So the relative measure is small:
#   a tree with one branch per synapse should reproduce TS1's port distances
#   to about 1% typically and about 10% for the worst quadruple.
# - **The real AZs are on the low side, but not significantly.** For the
#   relative measure they sit around the 10th–25th percentile of uniform
#   draws, and for the intrinsic measure around the 30th–40th. The real
#   synapse positions are not detectably special with respect to loops.
# - **Jitter doesn't make things worse**, so the real placement is not a
#   finely tuned optimum.
# - **A lead that didn't survive.** With a placeholder neck patch (an earlier
#   version of this analysis), the real AZs seemed to avoid "ring-like"
#   quadruples (intrinsic maximum at the 4th percentile). With the neck port
#   defined from the cell mesh, that effect is gone: the intrinsic maximum is
#   typical. It was an artifact of how the neck was modelled. This is a
#   reminder that maxima of unstable ratios are fragile.
# - **Caveats:** the uniform null ignores that real AZs cluster by axon and
#   by lobe (section 5). Spacing is measured in a straight line, not along
#   the surface. Patch area is a placeholder, and smaller PSDs would dilute
#   the relative measure more. A stronger test would match the real AZs'
#   clustering.

# %% [markdown]
# ## 10. Do real synapses sit at unusually isolated places?
#
# The same comparison for a quantity that directly shapes synaptic
# responses: the spine-local input resistance at each synapse (neck
# clamped), for the real AZs and for 30 uniform re-draws (750 random sites).
# If real AZs sat at systematically more or less isolated places, the two
# histograms would separate.

# %%
solver = cb.PortSolver(clamped, MEMBRANE)  # neck at rest; factorized once, reused
r_local_uniform = [np.diag(solver.impedance(random_sites().patches)) for _ in range(30)]
r_local_uniform = np.concatenate(r_local_uniform) / 1e6  # MΩ

fig, ax = plt.subplots(figsize=(5, 3.2))
ax.hist(r_local_uniform, bins=30, density=True, alpha=0.5, label="random sites")
ax.hist(r_local, bins=15, density=True, alpha=0.5, label="real AZs")
ax.set_xlabel("spine-local input resistance, neck clamped (MΩ)")
ax.set_ylabel("density")
ax.legend(frameon=False)
plt.show()
print(
    f"median: real {np.median(r_local):.1f} MΩ, random {np.median(r_local_uniform):.1f} MΩ"
)

# %% [markdown]
# **Reading:** the distributions overlap. Real synapses are neither
# especially isolated nor especially close to the neck.
#
# ## Summary: what a reduced TS1 model needs
#
# - **Heterogeneous synapses:** a 4× spread in spine-local resistance
#   (about 8–31 MΩ). One
#   compartment for the whole spine is not enough when local voltages or
#   cross-talk matter.
# - **Two lobes:** the spine splits electrically into two lobes joined at the
#   neck (coupling across lobes about 0.01), each fed by its own set of
#   axons. Start the reduced model from the
#   neck plus one sub-tree per lobe.
# - **Holes matter little electrically:** there is real loop structure, but
#   a loop-free tree should reproduce the port distances to about 1%
#   typically and about 10% at worst. Loops act mostly through the
#   resistances they create.
# - **Compartment count:** the raw rank bound is high, but inflated by
#   per-synapse spreading resistance. A diagonal + low-rank bound and an
#   actual tree fit (next steps) will give the real number.
