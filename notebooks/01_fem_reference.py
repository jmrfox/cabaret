# %% [markdown]
# # FEM reference model: checks and views
#
# camovox judges every reduced cable model against a **reference model**: a
# 3D finite-element (FEM) model of the cell interior on a tetrahedral mesh.
# This notebook checks that reference against exact answers on simple
# shapes, and introduces the views used in later notebooks:
#
# 1. **Sphere**: input resistance and time constant against the analytic
#    isopotential solution;
# 2. **Sealed cylinder**: voltage attenuation against cable theory;
# 3. **Torus** (the simplest loop): surface voltage, current streamlines and
#    slices;
# 4. equipotential surfaces, a preview of how the cell will be cut into
#    compartments;
# 5. the four-point violation, which measures how "loop-like" a set of
#    ports is;
# 6. site patches, mesh quality and export for ParaView.
#
# All voltages are relative to rest, for a passive membrane.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import trimesh

import camovox as cv
from camovox import ports, viz
from camovox import validate as va

pv.set_jupyter_backend("static")

# %% [markdown]
# ## 1. Sphere: isopotential check
#
# A small sphere is isopotential: axial resistance is negligible, so every
# point sits at the same voltage. Its input resistance is then just the
# membrane resistance, $1/(g_\mathrm{pas} A)$, from any injection site, and
# its membrane time constant is $c_m/g_\mathrm{pas}$ (40 ms at rest).
#
# What to look for: FEM within about 1% of analytic. The residual comes from
# the polyhedral surface having slightly less area than a true sphere.

# %%
p_rest = cv.Passive(r_L=100.0, c_m=1.0, g_pas=2.5e-5)
sphere = trimesh.creation.icosphere(subdivisions=3, radius=8.0)
s_tets, s_rep = cv.tetrahedralize(sphere, edge_length=2.0)
s_ref, s_q = cv.fem_model(s_tets)
w = cv.ball_patch(s_ref, [8.0, 0.0, 0.0], 1.0)
rin = cv.input_resistance(s_ref, p_rest, w)
rin_ref = va.sphere_input_resistance(8.0, p_rest)
tau = cv.time_constants(s_ref, p_rest, k=1)[0]
print(f"{s_rep.n_tets} tets, area change {100 * s_rep.area_error:+.2f}%")
print(f"R_in: FEM {rin / 1e6:.1f} MΩ, analytic {rin_ref / 1e6:.1f} MΩ")
print(f"τ: FEM {1e3 * tau:.2f} ms, analytic {1e3 * p_rest.tau_m:.2f} ms")

# %% [markdown]
# ## 2. Sealed cylinder: attenuation against cable theory
#
# Current is injected at one end of a 100 µm cylinder with sealed ends.
# Cable theory predicts $V(x)/V(0) = \cosh((L-x)/\lambda)/\cosh(L/\lambda)$,
# where λ is the length constant. A leaky (high-conductance) membrane makes
# the attenuation visible over this length.
#
# What to look for: the FEM dots (one per surface vertex) follow the cable
# curve. Small departures near the ends come from end caps (membrane the
# analytic sealed cable doesn't have) and from injection through a patch
# rather than a point. This is the regime where 1D cable models are
# accurate, which is why only the truly 3D parts of a cell need special
# treatment.

# %%
p_hi = cv.Passive(r_L=100.0, c_m=1.0, g_pas=1e-3)
radius, length = 2.0, 100.0
cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=24)
c_tets, _ = cv.tetrahedralize(cyl, edge_length=1.5, envelope=0.05)
c_ref, c_q = cv.fem_model(c_tets)
w0 = cv.ball_patch(c_ref, [0.0, 0.0, -length / 2], radius)
v_cyl = cv.solve_dc(c_ref, p_hi, 10e-12 * w0 / w0.sum())  # 10 pA, in V
print(f"λ = {va.cable_length_constant(radius, p_hi):.0f} µm")

x = viz.vertex_distance_along(c_tets, [0, 0, 1], [0, 0, -length / 2])
surface = c_ref.area > 0
v0 = np.average(v_cyl, weights=w0)
xs = np.linspace(0, length, 200)
viz.plot_profile(
    x[surface],
    {"FEM (surface vertices)": v_cyl[surface] / v0},
    xlabel="distance from injection (µm)",
    ylabel="V(x) / V(0)",
    reference={
        "cable theory": (xs, va.sealed_cylinder_attenuation(xs, radius, length, p_hi))
    },
)
plt.show()

# %%
pl = viz.plot_surface(c_tets, 1e3 * v_cyl, name="V (mV)")
pl.show()

# %% [markdown]
# ## 3. Torus: current around a hole
#
# A ring is the simplest loop, and toric spines are full of them. 10 pA is
# injected at one point on the outside of the ring.
#
# - **Surface voltage**: highest at the injection site, decaying symmetrically
#   both ways around the ring to a minimum on the opposite side.
# - **Streamlines** (tubes coloured by current density |J|, log scale): current
#   leaves the injection patch, splits, and flows both ways around the hole.
#   This is what a loop does electrically: it offers two parallel paths.
# - **Slice** (plane through the ring): the voltage is nearly uniform across
#   each cross-section, so the ring behaves like a 1D cable bent into a
#   circle. Only right at the injection patch is the field truly 3D.

# %%
torus = trimesh.creation.torus(major_radius=5.0, minor_radius=1.0)
t_tets, t_rep = cv.tetrahedralize(torus, edge_length=0.6, envelope=0.03)
t_ref, t_q = cv.fem_model(t_tets)
site = [6.0, 0.0, 0.0]
wt = cv.ball_patch(t_ref, site, 0.5)
v_t = cv.solve_dc(t_ref, p_hi, 10e-12 * wt / wt.sum())
j_t = cv.current_density(t_tets, v_t, p_hi)
print(
    f"{t_rep.n_tets} tets, negative couplings {100 * t_q.negative_fraction:.1f}% of |κ|"
)

# %%
pl = viz.plot_surface(t_tets, 1e3 * v_t, name="V (mV)")
pl.show()

# %%
pl = viz.plot_streamlines(t_tets, j_t, source_center=[5.0, 0.0, 0.0], source_radius=0.8)
pl.show()

# %%
pl = viz.plot_slice(t_tets, 1e3 * v_t, name="V (mV)", normal="z")
pl.show()

# %% [markdown]
# ## 4. Equipotentials as a partition preview
#
# With a very small leak, membrane current is nearly proportional to
# membrane area. That approximates the parameter-free "λ → ∞" field the
# design uses to cut a cell into compartments: each compartment is a slab
# between two equipotential surfaces.
#
# What to look for: on the ring the surfaces are cross-sections. Starting
# from the source they come in pairs, one on each arm, and the arms rejoin
# on the opposite side. In graph terms this is a loop, and it emerges from
# the field without any skeleton.

# %%
p_tiny = cv.Passive(r_L=100.0, c_m=1.0, g_pas=1e-8)
phi = cv.solve_dc(t_ref, p_tiny, 10e-12 * wt / wt.sum())
# The tiny leak puts φ at a large offset; only differences matter here.
pl = viz.plot_isosurfaces(t_tets, 1e3 * (phi - phi.min()), n=12, name="phi - min (mV)")
pl.show()

# %% [markdown]
# ## 5. Four-point violation on the ring
#
# Later notebooks ask whether a set of synapses on a looped spine could be
# reproduced by a loop-free **tree** of resistors. The test uses the
# **resistance distance** between ports (with membrane removed) and the
# **four-point condition**.
#
# For four ports, form the three pairings and their summed distances, sorted
# a ≥ b ≥ c. In a tree the two largest are equal (a = b). Two ways to
# measure the departure:
# - **relative** (a − b)/a: the error scale a tree must accept on these
#   distances. 0 for a tree, 0.25 for four *points* evenly spaced on an
#   ideal ring;
# - **intrinsic** (a − b)/(a − c): unchanged by adding resistance in series
#   with any port. 0 for a tree, 1 for an ideal 4-port ring.
#
# Why two: real ports are patches, and current must fan out from each patch
# (spreading resistance). That adds the same amount to every distance from
# that port, exactly like a short leaf branch in a tree. It dilutes the
# relative measure but leaves the intrinsic one unchanged.
#
# Here, on the torus:
# - four ports **evenly spaced** around the ring are as loop-like as it
#   gets: the intrinsic measure is near 1, and the relative one is below
#   0.25 because of the patches' spreading resistance;
# - four ports **bunched on one side** see the ring almost like a short
#   cable (the long way round is a weak parallel path), so both measures are
#   small.
#
# The first port is used as the electrical ground, which doesn't change
# the distances.

# %%
p_axial = cv.Passive(r_L=100.0)


def ring_ports(angles_deg: list[float]) -> list[np.ndarray]:
    """Ball patches on the outer rim of the torus at the given angles."""
    a = np.deg2rad(angles_deg)
    centers = np.column_stack([6.0 * np.cos(a), 6.0 * np.sin(a), np.zeros(len(a))])
    return [cv.ball_patch(t_ref, c, 0.4) for c in centers]


for name, angles in [
    ("evenly spaced", [0, 90, 180, 270]),
    ("bunched", [0, 20, 40, 60]),
]:
    pp = ring_ports(angles)
    d = ports.resistance_distances(t_ref, p_axial, pp[1:], pp[0])
    rel = ports.four_point_violation(d, "max")[0]
    intr = ports.four_point_violation(d, "gap")[0]
    print(f"{name:>13} {angles}: relative {rel:.3f}, intrinsic {intr:.3f}")

# %% [markdown]
# ## 6. Sites and mesh quality
#
# **Sites** are membrane patches where current enters or is read out. Here
# they are simple balls. On real spines they are geodesic patches grown to a
# target PSD area (notebook 02).
#
# **Mesh quality**: lines show FEM couplings with κ < 0. They come from
# obtuse tetrahedra and act like small negative conductances. A few percent
# is normal for fTetWild meshes and doesn't hurt FEM accuracy (sections 1
# and 2 are within about 1%). It matters for later reduction steps that
# assume positive couplings.

# %%
rng = np.random.default_rng(0)
angles = rng.uniform(0, 2 * np.pi, 8)
centers = np.column_stack([6.0 * np.cos(angles), 6.0 * np.sin(angles), np.zeros(8)])
patches = [cv.ball_patch(t_ref, c, 0.4) for c in centers]
pl = viz.plot_sites(t_tets, patches)
pl.show()

# %%
pl = viz.plot_mesh_quality(t_ref, t_tets)
pl.show()

# %% [markdown]
# ## 7. Export for ParaView
#
# Writes the torus solution (voltage, the λ → ∞ field and current density)
# to `../data/torus_dc.vtu`. In ParaView, open the file and colour by
# `V_mV`, or add a Stream Tracer on `J_A_per_um2`.

# %%
viz.write_vtu(
    "../data/torus_dc.vtu",
    t_tets,
    point_data={"V_mV": 1e3 * v_t, "phi_mV": 1e3 * phi},
    cell_data={"J_A_per_um2": j_t},
)
