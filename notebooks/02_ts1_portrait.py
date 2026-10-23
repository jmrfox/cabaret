# %% [markdown]
# # TS1 electrical portrait
# FEM reference model of toric spine TS1 with its 25 real active zones (AZs)
# and its neck. From the port impedance matrix we read off how many
# compartments a reduced model needs at least (rank bound), and how far the
# ports are from a tree (loopiness), for real and randomly placed synapses.
#
# PLACEHOLDERS: PSD area 0.05 µm² per AZ; the neck joins a parent of 80 MΩ and
# 100 pF (open question 4). Neck-clamped results don't depend on the parent.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

import camovox as cv
from camovox import ports, viz

pv.set_jupyter_backend("static")
SCALE = 0.005  # µm per pixel

# %% [markdown]
# ## Mesh and FEM reference model
# The tetrahedral mesh is cached in `../data/cache/` (meshing takes about
# 2 minutes the first time).

# %%
mesh = cv.load_mesh("../data/TS1.obj", scale=SCALE)
tets, rep = cv.tetrahedralize_cached(
    mesh, "../data/cache/TS1_e0.2_env0.005.npz", edge_length=0.2, envelope=0.005
)
ref, q = cv.fem_model(tets)
surf = cv.MembraneSurface.from_model(tets, ref)
print(
    f"{rep.n_tets} tets, {ref.n_nodes} nodes; area {rep.area_out:.1f} µm² "
    f"({100 * rep.area_error:+.2f}%), volume {rep.volume_out:.2f} µm³ "
    f"({100 * rep.volume_error:+.2f}%); negative couplings "
    f"{100 * q.negative_fraction:.1f}% of |κ|"
)

# %% [markdown]
# ## Sites: 25 AZs (coloured by axon) and the neck

# %%
az = cv.sites_from_points(
    surf,
    cv.load_points("../data/TS1_AZ.txt", SCALE),
    "synapse",
    axon_groups=cv.load_axon_groups("../data/TS1_syn_axon_dist.txt"),
)
neck = cv.sites_from_points(
    surf, cv.load_points("../data/TS1_neckpoint.txt", SCALE), "neck", area=0.2
)
print(
    f"AZ snap distance {az.table.snap_distance.min():.3f}–"
    f"{az.table.snap_distance.max():.3f} µm"
)
axon = az.table.axon_id.to_numpy(dtype=float)
pl = viz.plot_sites(
    tets, az.patches + neck.patches, labels=[*axon, -1], name="axon (neck = -1)"
)
pl.show()

# %% [markdown]
# ## Port impedance matrix
# Ports are the 25 AZ patches plus the neck (last). Membrane is in the
# high-conductance state (in vivo).

# %%
p = cv.Passive(r_L=100.0, c_m=1.0, g_pas=2.63e-4)
spine = cv.attach_sink(ref, neck.patches[0], resistance=80e6, capacitance=100e-12)
freqs = [0.0, 100.0, 1000.0]
z = cv.port_impedance(spine, p, az.patches + neck.patches, freqs=freqs)
zdc = z[0].real
z_clamped = ports.clamp_port(
    zdc, len(zdc) - 1
)  # neck clamped: the spine's own structure

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
im = axs[0].imshow(zdc / 1e6, cmap="viridis")
axs[0].set_title("Z (DC, MΩ), neck = last")
fig.colorbar(im, ax=axs[0])
im = axs[1].imshow(z_clamped / 1e6, cmap="viridis")
axs[1].set_title("Z with neck clamped (MΩ)")
fig.colorbar(im, ax=axs[1])
plt.show()
zii, zloc = np.diag(zdc)[:-1] / 1e6, np.diag(z_clamped) / 1e6
print(f"Z_ii {zii.min():.1f}–{zii.max():.1f} MΩ")
print(f"spine-local (neck clamped) {zloc.min():.1f}–{zloc.max():.1f} MΩ")

# %% [markdown]
# ## Voltage field for current into one AZ

# %%
k = int(np.argmax(np.diag(z_clamped)))  # the most isolated AZ
w = az.patches[k]
v = cv.solve_dc(spine, p, 10e-12 * w / w.sum())  # 10 pA
pl = viz.plot_surface(tets, 1e3 * v, name="V (mV), 10 pA into the most isolated AZ")
pl.show()

# %% [markdown]
# ## Rank bound: fewest compartments for a given error
# `err[m]` lower-bounds the relative error of any reduced model whose ports
# sit on m compartments. The clamped curve ignores the shared parent and is
# the honest one for the spine's internal structure.

# %%
fig, ax = plt.subplots(figsize=(5, 3.5))
for label_, zz in [("full Z (with parent)", zdc), ("neck clamped", z_clamped)]:
    err = ports.rank_bound(zz)
    ax.semilogy(np.arange(len(err) - 1), err[:-1], "o-", ms=3, label=label_)
for f, zf in zip(freqs[1:], z[1:], strict=True):
    err = ports.rank_bound(ports.clamp_port(zf, len(zf) - 1))
    ax.semilogy(
        np.arange(len(err) - 1), err[:-1], ".--", ms=3, label=f"clamped, {f:g} Hz"
    )
ax.axhline(0.05, color="gray", lw=0.8)
ax.axhline(0.01, color="gray", lw=0.8)
ax.set_xlabel("rank m (compartments carrying ports)")
ax.set_ylabel("relative error lower bound")
ax.legend(frameon=False, fontsize=8)
plt.show()
err_c = ports.rank_bound(z_clamped)
for tol in (0.1, 0.05, 0.01):
    print(
        f"error ≤ {tol:.0%} needs rank ≥ {int(np.argmax(err_c <= tol))} (neck clamped)"
    )

# %% [markdown]
# ## Loopiness: could a tree reproduce these ports?
# Four-point violation of the axial resistance distances (neck grounded).
# Zero means a tree with internal nodes reproduces the ports exactly.
# Real AZs are compared with uniform re-draws of 25 sites (same area, at
# least 0.2 µm apart, at least 1 µm from the neck).

# %%
d_real = ports.resistance_distances(ref, p, az.patches, neck.patches[0])
li_real = ports.loopiness_index(d_real)
rng = np.random.default_rng(0)
neck_xyz = neck.table[["x", "y", "z"]].to_numpy()
li_uniform = []
for _ in range(20):
    u = cv.sample_uniform(
        surf, len(az), rng, min_spacing=0.2, exclude=neck_xyz, exclude_radius=1.0
    )
    li_uniform.append(
        ports.loopiness_index(
            ports.resistance_distances(ref, p, u.patches, neck.patches[0])
        )
    )
print("real AZs:", {k_: round(v_, 4) for k_, v_ in li_real.items()})
for key in ("median", "p90", "max"):
    vals = np.array([u[key] for u in li_uniform])
    lo, hi = vals.min(), vals.max()
    print(f"uniform {key}: {np.median(vals):.4f} (range {lo:.4f}–{hi:.4f})")

fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
for ax, key in zip(axs, ["p90", "max"], strict=True):
    ax.hist(
        [u[key] for u in li_uniform],
        bins=10,
        color="lightgray",
        label="uniform re-draws",
    )
    ax.axvline(li_real[key], color="C3", label="real AZs")
    ax.set_xlabel(f"four-point violation, {key}")
    ax.legend(frameon=False, fontsize=8)
plt.show()

# %% [markdown]
# ## Placement sensitivity of spine-local resistance
# Spread of the neck-clamped input resistance at real AZs against uniform
# re-draws.

# %%
local_uniform = []
for _ in range(10):
    u = cv.sample_uniform(
        surf, len(az), rng, min_spacing=0.2, exclude=neck_xyz, exclude_radius=1.0
    )
    zu = cv.port_impedance(spine, p, u.patches + neck.patches)[0].real
    local_uniform.append(np.diag(ports.clamp_port(zu, len(zu) - 1)))
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.hist(
    np.concatenate(local_uniform) / 1e6,
    bins=30,
    density=True,
    alpha=0.5,
    label="uniform sites",
)
ax.hist(np.diag(z_clamped) / 1e6, bins=15, density=True, alpha=0.5, label="real AZs")
ax.set_xlabel("spine-local input resistance, neck clamped (MΩ)")
ax.legend(frameon=False)
plt.show()
