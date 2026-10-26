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
# **Saved results.** Meshing and solving take tens of minutes (about 40
# more the first time the 0.05 µm mesh is built), too long for a notebook.
# They run in a script, and this notebook only loads the saved results.
# To regenerate them, from the repository root:
#
# ```
# uv run python scripts/ts1_mesh_convergence.py data
# ```

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cabaret import reduce

RESULTS = "../data/cache/ts1_mesh_convergence.npz"  # from the script above
REF = "0.05"

saved = np.load(RESULTS)
labels = [str(s) for s in saved["labels"]]
FREQS = saved["freqs"]  # Hz; DC only, since 100 Hz and 1 kHz matched (plan 008)
QUANTITIES = ["z", "zc", "r_local", "r_iso", "r_neck"]
results = {m: {q: saved[f"{m}/{q}"] for q in QUANTITIES} for m in labels}

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
meshes = pd.DataFrame(
    saved["mesh_table"],
    index=pd.Index(labels, name="mesh"),
    columns=[str(c) for c in saved["mesh_columns"]],
)
meshes.round(2)

# %% [markdown]
# ## 2. Port quantities on each mesh
#
# Ports are ordered neck first, then AZ 0 to AZ 24, as in notebooks 03–04.
# The script computes, for each mesh:
# - the DC port matrices, isolated and with the neck clamped (at 100 Hz and
#   1 kHz the neck-clamped errors were the same to 0.01%, so they are left
#   out to save time);
# - the **spine-local input resistance** of each AZ: the diagonal of the
#   neck-clamped DC matrix;
# - the **isolated input resistance** of each AZ: the diagonal of the
#   isolated DC matrix (dominated by the total membrane);
# - the resistance from the neck to each AZ, with the membrane removed.

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
n = [meshes.loc[m, "vertices"] for m in labels]
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
