# %% [markdown]
# # FEM reference model: checks and views
# Builds the FEM reference on three test shapes, checks it against analytic
# solutions, and shows the visualization helpers: surface voltage, slices,
# equipotentials, current streamlines, site patches and mesh quality.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import trimesh

import camovox as cv
from camovox import validate as va
from camovox import viz

pv.set_jupyter_backend("static")

# %% [markdown]
# ## Sphere: isopotential check
# A sphere is isopotential, so its input resistance is $1/(g_\mathrm{pas} A)$
# from any site.

# %%
p_rest = cv.Passive(r_L=100.0, c_m=1.0, g_pas=2.5e-5)
sphere = trimesh.creation.icosphere(subdivisions=3, radius=8.0)
s_tets, s_rep = cv.tetrahedralize(sphere, edge_length=2.0)
s_ref, s_q = cv.fem_model(s_tets)
w = cv.ball_patch(s_ref, [8.0, 0.0, 0.0], 1.0)
rin = cv.input_resistance(s_ref, p_rest, w)
print(f"{s_rep.n_tets} tets, area change {100 * s_rep.area_error:+.2f}%")
rin_ref = va.sphere_input_resistance(8.0, p_rest)
print(f"R_in FEM {rin / 1e6:.1f} MΩ vs analytic {rin_ref / 1e6:.1f} MΩ")

# %% [markdown]
# ## Sealed cylinder: attenuation against cable theory
# Inject at one end and compare $V(x)/V(0)$ with
# $\cosh((L-x)/\lambda)/\cosh(L/\lambda)$. The high-conductance membrane
# makes the attenuation visible over 100 µm.

# %%
p_hi = cv.Passive(r_L=100.0, c_m=1.0, g_pas=1e-3)
radius, length = 2.0, 100.0
cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=24)
c_tets, _ = cv.tetrahedralize(cyl, edge_length=1.5, envelope=0.05)
c_ref, c_q = cv.fem_model(c_tets)
w0 = cv.ball_patch(c_ref, [0.0, 0.0, -length / 2], radius)
v_cyl = cv.solve_dc(c_ref, p_hi, 10e-12 * w0 / w0.sum())  # 10 pA, in V

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
# ## Torus: current around a hole
# A ring is the simplest loop. Inject current at one point and watch it
# split both ways around the hole.

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
# ## Equipotentials as a partition preview
# With a very small leak, membrane current is nearly proportional to area,
# which approximates the parameter-free λ→∞ field. Its level sets are the
# planned compartment boundaries: on a ring they split into two arms that
# rejoin opposite the source, which is the loop in the Reeb graph.

# %%
p_tiny = cv.Passive(r_L=100.0, c_m=1.0, g_pas=1e-8)
phi = cv.solve_dc(t_ref, p_tiny, 10e-12 * wt / wt.sum())
pl = viz.plot_isosurfaces(t_tets, 1e3 * phi, n=12, name="φ (mV)")
pl.show()

# %% [markdown]
# ## Sites and mesh quality
# Patches are Euclidean balls for now (a placeholder for geodesic PSD
# patches). The last view shows FEM couplings with κ < 0, which come from
# obtuse tetrahedra.

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
# ## Export for ParaView
# Writes the torus solution to `../data/torus_dc.vtu`.

# %%
viz.write_vtu(
    "../data/torus_dc.vtu",
    t_tets,
    point_data={"V_mV": 1e3 * v_t, "phi_mV": 1e3 * phi},
    cell_data={"J_A_per_um2": j_t},
)
