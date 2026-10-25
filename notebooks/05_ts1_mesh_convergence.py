# %% [markdown]
# # 05 · Mesh convergence of the TS1 reference
#
# Everything downstream (the tree, the fit, the jaxley model) is matched to
# the TS1 FEM reference built on a 0.2 µm mesh. This notebook checks how
# much that reference changes with mesh resolution (plan 008).
#
# **Why it might matter.** At a 0.2 µm target edge the thinnest neck
# (about 0.12 µm across) may be only a tetrahedron or two wide, and the
# 0.05 µm² PSD patches cover only a few surface vertices. Neck resistance
# and the spine-local input resistance at each synapse are exactly what the
# reduction is fitted to.
#
# **Meshes.**
# - fTetWild with target edge 0.2, 0.14 and 0.1 µm, and 0.2 µm with its
#   `coarsen` option. The envelope is 0.005 µm throughout.
# - **The reference:** the 0.2 µm mesh refined uniformly (every tetrahedron
#   split into eight, `TetMesh.refined`). Its boundary is the same surface,
#   so only the resolution changes.
#
# Meshing takes a few minutes per mesh the first time; the results are
# cached in `../data/cache/`.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cabaret as cb
from cabaret import graft, ports, reduce, toric

SCALE = 0.005  # µm per pixel
MEMBRANE = cb.Passive(r_L=100.0, c_m=1.0, g_pas=2.63e-4)  # as in notebooks 03–04
FREQS = [0.0, 100.0, 1000.0]  # Hz
MESHES = {  # fTetWild meshes, label: (edge length in µm, coarsen)
    "0.2 coarse": (0.2, True),
    "0.2": (0.2, False),
    "0.14": (0.14, False),
    "0.1": (0.1, False),
}
REF = "0.2 refined"
REFINED_CACHE = Path("../data/cache/TS1_e0.2_env0.005_refined.npz")

# %% [markdown]
# ## 1. The meshes
#
# For each mesh: its size, how faithfully it follows the reconstruction,
# the share of negative couplings, how many surface vertices each AZ patch
# covers, and how many FEM nodes are tied into the neck electrode.
#
# **What to check:** the vertices per AZ patch. Below about 5, the patch is
# barely resolved and its spreading resistance is set by the mesh.

# %%
parts = graft.load_parts("../data/neuromesh/parts", scale=SCALE)
spines = {
    label: toric.load_toric_spine(
        "TS1", "../data", parts, scale=SCALE, edge_length=e, coarsen=c
    )
    for label, (e, c) in MESHES.items()
}
if not REFINED_CACHE.exists():
    spines["0.2"].tets.refined().save(REFINED_CACHE)
spines[REF] = toric.load_toric_spine(
    "TS1", "../data", parts, scale=SCALE, tets=cb.TetMesh.load(REFINED_CACHE)
)

rows = []
for label, ts in spines.items():
    _, quality = cb.fem_model(ts.tets)
    verts = [np.count_nonzero(p) for p in ts.sites.patches]
    rows.append(
        {
            "mesh": label,
            "tets": ts.fidelity.n_tets,
            "vertices": ts.fidelity.n_points,
            "surface vertices": int((ts.fem.area > 0).sum()),
            "area change (%)": 100 * ts.fidelity.area_error,
            "volume change (%)": 100 * ts.fidelity.volume_error,
            "negative |κ| (%)": 100 * quality.negative_fraction,
            "vertices per AZ patch (median)": np.median(verts),
            "vertices per AZ patch (min)": np.min(verts),
            "neck electrode nodes": len(ts.attachment_contacts[0]),
        }
    )
pd.DataFrame(rows).set_index("mesh").round(2)

# %% [markdown]
# ## 2. Port quantities on each mesh
#
# Ports are ordered neck first, then AZ 0 to AZ 24, as in notebooks 03–04.
# For each mesh:
# - the port matrices at DC, 100 Hz and 1 kHz, isolated and with the neck
#   clamped;
# - the **spine-local input resistance** of each AZ: the diagonal of the
#   neck-clamped DC matrix;
# - the **isolated input resistance** of each AZ: the diagonal of the
#   isolated DC matrix (dominated by the total membrane);
# - the resistance from the neck to each AZ, with the membrane removed.

# %%
results = {}
for label, ts in spines.items():
    neck = ts.attachment_ports[0]
    z = cb.port_impedance(ts.network, MEMBRANE, [neck, *ts.sites.patches], FREQS)
    zc = np.stack([ports.clamp_port(zk, 0) for zk in z])
    d = ports.ResistanceMetric(ts.network, MEMBRANE, ground=neck).distances(
        ts.sites.patches
    )
    results[label] = {
        "z": z,
        "zc": zc,
        "r_local": np.diag(zc[0].real),
        "r_iso": np.diag(z[0].real)[1:],
        "r_neck": d[0, 1:],
    }

# %% [markdown]
# ## 3. Errors against the refined mesh
#
# Relative errors against the uniformly refined 0.2 µm mesh. Matrix errors are relative
# Frobenius norms (`port_matrix_error`), per frequency; per-AZ quantities
# give the median and the largest error over the 25 AZs.

# %%
ref = results[REF]


def rel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a / b - 1)


rows = []
for label, r in results.items():
    if label == REF:
        continue
    row = {"mesh": label}
    for key, name in [
        ("r_local", "spine-local R_in"),
        ("r_iso", "isolated R_in"),
        ("r_neck", "neck-AZ resistance"),
    ]:
        e = rel(r[key], ref[key])
        row[f"{name}: median (%)"] = 100 * np.median(e)
        row[f"{name}: max (%)"] = 100 * e.max()
    for k, f in enumerate(FREQS):
        row[f"clamped Z @ {f:g} Hz (%)"] = 100 * reduce.port_matrix_error(
            r["zc"][k], ref["zc"][k]
        )
        row[f"isolated Z @ {f:g} Hz (%)"] = 100 * reduce.port_matrix_error(
            r["z"][k], ref["z"][k]
        )
    rows.append(row)
pd.DataFrame(rows).set_index("mesh").T.round(2)

# %% [markdown]
# **Per AZ.** Spine-local input resistance error of each AZ against the
# refined mesh (left), and the median error against mesh size (right).
# Changing fTetWild's target edge length barely changes the vertex count,
# because the tight envelope already sets the surface resolution.

# %%
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for label, r in results.items():
    if label != REF:
        axs[0].plot(100 * (r["r_local"] / ref["r_local"] - 1), "o-", label=label)
axs[0].axhline(0, color="k", lw=0.5)
axs[0].set_xlabel("AZ")
axs[0].set_ylabel("spine-local R_in error vs refined (%)")
axs[0].legend()
labels = [m for m in results if m != REF]
n = [spines[m].fidelity.n_points for m in labels]
e = [100 * np.median(rel(results[m]["r_local"], ref["r_local"])) for m in labels]
axs[1].semilogx(n, e, "o")
for m, x, y in zip(labels, n, e, strict=True):
    axs[1].annotate(m, (x, y), textcoords="offset points", xytext=(5, 5))
axs[1].set_xlabel("mesh vertices")
axs[1].set_ylabel("spine-local R_in error, median (%)")
plt.tight_layout()

# %% [markdown]
# ## 4. Reading
#
# - **fTetWild's edge length is a weak knob.** Halving the target from 0.2
#   to 0.1 µm adds only 38% more vertices: the 0.005 µm envelope already
#   forces fine surface elements. Uniform refinement is the way to actually
#   resolve the mesh further.
# - **Isolated quantities have converged** (under 0.05% on every mesh):
#   they are set by the total membrane.
# - **The neck-clamped quantities, which the reduction is fitted to, have
#   not.** The 0.2 µm mesh used in notebooks 02–04 reads the spine-local
#   input resistance 1.2% low on average (1.3% median, 5.0% at AZ 8), and
#   the neck-clamped port matrix is 1.3% off. Coarse meshes read low, as P1
#   elements are too conductive when under-resolved.
# - **AZs 7 and 8 are the hard spot.** They are about 4–5% low on every
#   fTetWild mesh and only resolved by refinement, so a thin feature near
#   them is under-resolved at this surface resolution.
# - **`coarsen` costs too much accuracy** (4.4% median, 10% max).
# - **The refined mesh is affordable** (223k vertices; about 30 s for all 26
#   ports at DC, 12 minutes for this whole notebook), but whether it has
#   itself converged is not shown here: the next uniform level (about 1.7M
#   vertices) needs an iterative solver.
#
# For comparison, the tree's own floor is 2.8% on the neck-clamped matrix
# (notebook 04), so mesh error at 0.2 µm is not negligible next to it.
