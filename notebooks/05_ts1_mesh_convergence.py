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
# **Meshes.** fTetWild is rerun from the reconstruction with target edge
# 0.2, 0.1 and 0.05 µm (envelope 0.005 µm throughout). Each run makes a new
# mesh, so the surface moves slightly (within the envelope) as well as the
# resolution: the test is of the whole meshing step. The 0.05 µm mesh is the
# reference. For comparison, the 0.2 µm mesh refined uniformly (every
# tetrahedron split into eight, `TetMesh.refined`) changes resolution only.
#
# Meshing the 0.05 µm mesh takes about 40 minutes the first time; meshes
# are cached in `../data/cache/`. The solves on the two fine meshes take
# tens of minutes (about an hour with three frequencies, before this was cut
# to DC).

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cabaret as cb
from cabaret import graft, ports, reduce, toric

SCALE = 0.005  # µm per pixel
MEMBRANE = cb.Passive(r_L=100.0, c_m=1.0, g_pas=2.63e-4)  # as in notebooks 03–04
FREQS = [0.0]  # Hz; 100 Hz and 1 kHz gave the same clamped errors (plan 008)
EDGES = {"0.2": 0.2, "0.1": 0.1, "0.05": 0.05}  # fTetWild target edge, µm
REF = "0.05"
REFINED = "0.2 refined"
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
    label: toric.load_toric_spine("TS1", "../data", parts, scale=SCALE, edge_length=e)
    for label, e in EDGES.items()
}
if not REFINED_CACHE.exists():
    spines["0.2"].tets.refined().save(REFINED_CACHE)
spines[REFINED] = toric.load_toric_spine(
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
# - the DC port matrices, isolated and with the neck clamped (at 100 Hz and
#   1 kHz the neck-clamped errors were the same to 0.01%, so they are left
#   out to save time);
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
# ## 3. Errors against the 0.05 µm mesh
#
# Relative errors against the 0.05 µm mesh. Matrix errors are relative
# Frobenius norms (`port_matrix_error`); per-AZ quantities give the median,
# the largest and the mean signed error over the 25 AZs.

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
        row[f"{name}: mean signed (%)"] = 100 * np.mean(r[key] / ref[key] - 1)
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
# **Successive differences.** If the 0.2 → 0.1 change is clearly larger
# than 0.1 → 0.05, the meshing is converging and 0.05 µm is a fair
# reference.

# %%
steps = [("0.2", "0.1"), ("0.1", "0.05")]
pd.DataFrame(
    {
        f"{a} → {b}": {
            "spine-local R_in, median (%)": 100
            * np.median(rel(results[a]["r_local"], results[b]["r_local"])),
            "spine-local R_in, max (%)": 100
            * rel(results[a]["r_local"], results[b]["r_local"]).max(),
            "clamped Z @ DC (%)": 100
            * reduce.port_matrix_error(results[a]["zc"][0], results[b]["zc"][0]),
        }
        for a, b in steps
    }
).round(2)

# %% [markdown]
# **Per AZ.** Spine-local input resistance error of each AZ against the
# 0.05 µm mesh (left), and the median error against mesh size (right).

# %%
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for label, r in results.items():
    if label != REF:
        axs[0].plot(100 * (r["r_local"] / ref["r_local"] - 1), "o-", label=label)
axs[0].axhline(0, color="k", lw=0.5)
axs[0].set_xlabel("AZ")
axs[0].set_ylabel("spine-local R_in error vs 0.05 µm (%)")
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
# - **At 0.2 µm, the reference is about 1% low where it matters.** Against
#   the 0.05 µm mesh, spine-local input resistance is 1.1% low on average
#   (1.0% median, 4.1% max at AZ 12), and the neck-clamped port matrix is
#   1.1% off. At 0.1 µm: 0.7% low, 0.8% median, 2.8% max.
# - **The edge-length series alone does not show convergence.** The step
#   0.1 → 0.05 changes as much as 0.2 → 0.1 (0.8% median each). Part of it
#   is geometry: every rerun of fTetWild moves the surface a little (area
#   −0.80% at 0.2 µm, −1.04% at 0.05 µm), and the isolated input resistance,
#   which follows total membrane, shifts by 0.5%.
# - **The two fine meshes agree.** The 0.05 µm mesh and the uniformly
#   refined 0.2 µm mesh are built differently but differ by only 0.34%
#   (median) and 0.44% on the clamped matrix, with no systematic offset
#   (+0.14% mean). So the remaining spread between fine meshes is a few
#   tenths of a percent, mostly from surface differences, while the 0.2 µm
#   mesh's own error is about 1%.
# - **AZs 7–8** are 3–3.5% low at 0.2 µm and within 2% between the fine
#   meshes.
#
# For comparison, the tree's floor is 2.8% on the neck-clamped matrix
# (notebook 04): a 0.2 µm reference adds about 1% to every comparison.
