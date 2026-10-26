# %% [markdown]
# # 04 · Fitting the reduced TS1 model and running it in jaxley
#
# Notebook 03 built a reduced model of toric spine TS1: a tree of resistors
# with membrane on its nodes, derived directly from the FEM reference model.
# This notebook takes the last two steps toward a usable simulation model:
#
# 1. **Fit** the tree's branch resistances and membrane areas to the FEM, by
#    gradient descent;
# 2. **Export** it to [jaxley](https://github.com/jaxleyverse/jaxley), the
#    simulator for later population models, and check that jaxley gives the
#    same voltages as the FEM.
#
# **What is fitted.** Each branch resistance and each node's share of the
# membrane. Total membrane is held exactly at the FEM's value, because
# channel densities will later be specified per membrane area.
#
# **Fitting targets.** The FEM port impedance matrices (25 synapses + neck)
# at DC, 100 Hz and 1 kHz, mainly with the neck clamped (the spine's internal
# structure; notebook 02, section 4). For a passive network these matrices
# determine its response to *any* input waveform, so matching them is a
# complete test. Section 4 checks that with time-domain voltage traces.

# %%
import matplotlib.pyplot as plt
import numpy as np

import cabaret as cb
from cabaret import fit, graft, jaxley_io, ports, reduce, toric

SCALE = 0.005  # µm per pixel
MEMBRANE = cb.Passive(r_L=100.0, c_m=1.0, g_pas=2.63e-4)  # high-conductance, as before
FREQS = [0.0, 100.0, 1000.0]  # Hz

# %% [markdown]
# ## 1. Reference model and starting tree (as in notebook 03)

# %%
parts = graft.load_parts("../data/neuromesh/parts", scale=SCALE)
ts1 = toric.load_toric_spine("TS1", "../data", parts, scale=SCALE)
neck_port = ts1.attachment_ports[0]
port_patches = [neck_port, *ts1.sites.patches]  # port 0 = neck, port k = AZ k−1
z_fem = cb.port_impedance(ts1.network, MEMBRANE, port_patches, FREQS)  # FEM targets

distances = ports.ResistanceMetric(ts1.network, MEMBRANE, ground=neck_port)
tree = reduce.fit_tree(distances.distances(ts1.sites.patches))
owner = reduce.assign_membrane(
    ts1.network, MEMBRANE, ts1.sites.patches, ground=neck_port
)
has_membrane = owner >= 0
port_area = np.bincount(
    owner[has_membrane],
    weights=ts1.network.area[has_membrane],
    minlength=len(port_patches),
)


def start_areas(t: reduce.PortTree) -> np.ndarray:
    """Membrane per tree node from the fingerprint assignment (notebook 03)."""
    area = np.zeros(t.n_nodes)
    np.add.at(area, t.port_node, port_area)
    return area


def clamped_error(t: reduce.PortTree, area: np.ndarray) -> float:
    """Neck-clamped port-matrix error against the FEM, averaged over FREQS."""
    g = reduce.tree_to_graph(t, MEMBRANE, area)
    z = cb.port_impedance(
        g, MEMBRANE, [cb.point_port(g, int(n)) for n in t.port_node], FREQS
    )
    return float(
        np.mean(
            [
                reduce.port_matrix_error(
                    ports.clamp_port(z[k], 0), ports.clamp_port(z_fem[k], 0)
                )
                for k in range(len(FREQS))
            ]
        )
    )


print(
    f"starting tree: {tree.n_nodes} compartments, clamped error "
    f"{100 * clamped_error(tree, start_areas(tree)):.2f}%"
)

# %% [markdown]
# ## 2. Fit the full tree
#
# The fit minimizes the relative error of the neck-clamped matrix, plus a
# small weight on the isolated matrix, using the Adam optimizer and exact
# gradients from JAX. Resistances and areas are adjusted in log space, so
# they stay positive. The loss curve shows the optimization settling.

# %%
full = fit.fit_tree_model(tree, start_areas(tree), z_fem, MEMBRANE, FREQS, steps=800)
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(100 * full.loss)
ax.set_xlabel("optimization step")
ax.set_ylabel("loss (≈ % error)")
plt.show()
print(
    f"full tree: {100 * clamped_error(tree, start_areas(tree)):.2f}% before fitting -> "
    f"{100 * clamped_error(full.tree, full.node_area):.2f}% after"
)
print(f"total membrane: {full.node_area.sum():.2f} µm² (FEM {port_area.sum():.2f} µm²)")

# %% [markdown]
# **Reading:** fitting barely changes the full tree (about 2.8%). Running
# longer or with larger steps doesn't help either, so this is a **floor set by
# the tree's shape**, not by the optimizer. A loop-free tree can't do better
# than about 3% for TS1's 26 ports. That is the electrical cost of leaving out
# the loops, and the number a model with real loops would have to beat.

# %% [markdown]
# ## 3. Smaller models: fitted accuracy against size
#
# Smaller trees come from merging short branches (notebook 03), and each is
# then fitted. The plot compares the error before and after fitting at each
# size. Fitting recovers much of what merging loses, roughly halving the
# error for medium-sized models.

# %%
positive = tree.resistance[tree.resistance > 0]
rows = []
for q in [None, 0.2, 0.35, 0.5, 0.6, 0.7, 0.8]:
    t = tree if q is None else reduce.contract_tree(tree, np.quantile(positive, q))
    res = fit.fit_tree_model(t, start_areas(t), z_fem, MEMBRANE, FREQS, steps=600)
    rows.append(
        (
            t.n_nodes,
            clamped_error(t, start_areas(t)),
            clamped_error(res.tree, res.node_area),
        )
    )
n_comp, err_before, err_after = map(np.array, zip(*rows, strict=True))

fig, ax = plt.subplots(figsize=(5.5, 3.6))
ax.plot(n_comp, 100 * err_before, "o--", color="gray", label="merged, not fitted")
ax.plot(n_comp, 100 * err_after, "o-", label="merged, then fitted")
ax.set_xlabel("compartments in the reduced model")
ax.set_ylabel("neck-clamped error vs FEM (%)")
ax.legend(frameon=False)
fig.tight_layout()
plt.show()
for n, b, a in rows:
    print(f"{n:3d} compartments: {100 * b:5.1f}% -> {100 * a:4.1f}% after fitting")

# %% [markdown]
# ## 4. Export to jaxley and compare voltage traces
#
# The fitted full tree becomes a jaxley cell: one compartment per tree node,
# rooted at the neck, with passive leak and capacitance. A synaptic-like
# current (an α-function, 50 pA peak, 0.5 ms time constant) is injected at
# three synapses in turn:
# - the synapse electrically closest to the neck;
# - the most isolated synapse;
# - a synapse from the other lobe.
#
# The neck is held near rest (tied to rest through 1 kΩ), as if attached to a
# large parent, so the traces show the spine's own local EPSPs.
#
# Each panel overlays three simulations of the same input:
# - **FEM** reference: the full 3D model, about 34,000 nodes;
# - **reduced (cabaret)**: the fitted 50-compartment tree in cabaret's solver;
# - **reduced (jaxley)**: the same tree simulated by jaxley.
#
# What to look for: the jaxley and cabaret curves should be identical (this
# tests the export). Both should lie close to the FEM (this tests the
# reduction).

# %%
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jaxley as jx  # noqa: E402

NECK_SHUNT = 1e3  # Ω: holds the neck near rest
DT_MS = 0.025
t_ms = np.arange(0, 15, DT_MS)
i_pa = 50 * (t_ms / 0.5) * np.exp(1 - t_ms / 0.5)  # α-function current, pA

reduced_graph = reduce.tree_to_graph(full.tree, MEMBRANE, full.node_area)
reduced_clamped = cb.attach_sink(
    reduced_graph,
    cb.point_port(reduced_graph, int(full.tree.port_node[0])),
    resistance=NECK_SHUNT,
)
fem_clamped = cb.attach_sink(ts1.network, neck_port, resistance=NECK_SHUNT)

zc = ports.clamp_port(z_fem[0].real, 0)
lobe = ports.electrical_groups(zc, 2)
closest, isolated = int(np.argmin(np.diag(zc))), int(np.argmax(np.diag(zc)))
other = int(np.flatnonzero(lobe != lobe[isolated])[0])
cases = {"closest to neck": closest, "most isolated": isolated, "other lobe": other}

fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, (label, az_i) in zip(axs, cases.items(), strict=True):
    port = az_i + 1
    w_fem = ts1.sites.patches[az_i]
    v_fem = cb.simulate_current(
        fem_clamped, MEMBRANE, w_fem, i_pa * 1e-12, DT_MS * 1e-3, w_fem[None]
    )[:, 0]
    w_red = cb.point_port(reduced_graph, int(full.tree.port_node[port]))
    v_red = cb.simulate_current(
        reduced_clamped, MEMBRANE, w_red, i_pa * 1e-12, DT_MS * 1e-3, w_red[None]
    )[:, 0]
    jm = jaxley_io.to_jaxley(full.tree, full.node_area, MEMBRANE, root_shunt=NECK_SHUNT)
    site = jm.cell.branch(int(jm.port_branch[port])).comp(0)
    site.stimulate(i_pa * 1e-3, verbose=False)  # nA
    site.record("v", verbose=False)
    v_jx = np.asarray(jx.integrate(jm.cell, delta_t=DT_MS))[0, 1:]  # drop t = 0
    ax.plot(t_ms, 1e3 * v_fem, lw=3, color="lightgray", label="FEM")
    ax.plot(t_ms, 1e3 * v_red, lw=1.5, label="reduced (cabaret)")
    ax.plot(t_ms, v_jx, "--", lw=1.5, label="reduced (jaxley)")
    ax.set_title(f"AZ {az_i}: {label}")
    ax.set_xlabel("time (ms)")
    peak_err = 100 * (1e3 * v_red.max() / (1e3 * v_fem.max()) - 1)
    print(
        f"AZ {az_i:2d} ({label}): EPSP peak FEM {1e3 * v_fem.max():.3f} mV, reduced "
        f"{1e3 * v_red.max():.3f} mV ({peak_err:+.1f}%), jaxley {v_jx.max():.3f} mV"
    )
axs[0].set_ylabel("voltage at the synapse (mV)")
axs[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
plt.show()

# %% [markdown]
# **Reading:** the jaxley and cabaret traces coincide, so the export is exact.
# Both follow the FEM closely in amplitude and time course, with EPSP peaks
# within about 0.2%, though the 3D model has about 700 times more nodes.
#
# Why 0.2% here but about 2.8% in section 2? The 2.8% is over the *whole*
# matrix, and most of it is in cross-talk between pairs of synapses (the
# off-diagonal terms), where the missing loops matter. Each synapse's own
# local response, which sets its EPSP, is reproduced much more accurately
# (notebook 03: median 0.3% for spine-local input resistance).
#
# **Technical note.** Each compartment's outgoing branch-point load in jaxley
# is set to a negligible 1e-6 (about 1 Ω). Setting it to essentially zero
# (1e-12) makes jaxley's solvers lose precision (`jaxley_io` module notes).
#
# ## Summary
#
# - The reduced TS1 model is **fitted and running in jaxley**. Its EPSPs
#   match the 3D FEM within about 0.2% at synapses across both lobes.
# - A loop-free tree has an accuracy floor of about **3%** for TS1: the
#   electrical cost of the holes. A model with real loops would be needed
#   to go below it.
# - All errors are against the FEM on a 0.2 µm mesh, which itself reads
#   spine-local input resistance about 1% low (notebook 05). That error is
#   in the reference, not in the reduction: notebook 03's tree reaches about
#   3% against both the 0.2 and the 0.05 µm FEM.
# - **Fitting matters for small models.** Merged-then-fitted trees reach
#   about 6–7% with 21–26 compartments, against 11–12% unfitted. Below about
#   20 compartments the error grows quickly with this simple
#   merge-then-fit approach.
# - Next: smarter small models (choosing which synapses share a
#   compartment, with per-synapse series resistances). The bound in notebook
#   03 suggests about 8 shared compartments could reach about 1%. Then the
#   other spines.
