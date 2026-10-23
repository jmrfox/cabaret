# %% [markdown]
# # 01 · The FEM reference model: checks and views
#
# **What camovox does.** It builds **reduced cable models** of neurons
# (compartmental models a simulator such as jaxley can run cheaply) that
# reproduce the electrical behaviour of a high-resolution 3D model of the
# same cell. The 3D model is the **reference model**: a finite-element (FEM)
# solution of current flow inside the cell's reconstructed surface. The full
# rationale is in `docs/design/` (start at its README; the physics is in
# `docs/design/fem-reference.md`).
#
# **This notebook** checks that the reference model gives the right answers
# on shapes with known solutions, and introduces the plots used in later
# notebooks:
#
# 1. **Sphere**: input resistance and time constant against the exact answer;
# 2. **Sealed cylinder**: voltage decay along a cable against cable theory;
# 3. **Torus** (the simplest loop): voltage, current paths and a cross-section;
# 4. **Equipotential surfaces**, a preview of how a cell will be cut into
#    compartments;
# 5. **Four-point violation**, which measures how "loop-like" a set of
#    recording/injection sites is;
# 6. **Sites and mesh quality**, and export for ParaView.
#
# **Glossary**
# - *Passive membrane*: leak conductance plus capacitance, no active
#   channels. All voltages are deviations from rest.
# - *r_L*: axial (cytoplasmic) resistivity, Ω·cm. *c_m*: membrane capacitance,
#   µF/cm². *g_pas*: leak conductance, S/cm².
# - *Tetrahedral mesh ("tets")*: the cell interior filled with tetrahedra.
#   The FEM solves for the voltage at every mesh vertex.
# - *Patch / site*: a small area of membrane where current is injected or
#   voltage is read out. It is a patch rather than a point, because a point
#   source in 3D has infinite resistance.
# - *κ (kappa)*: the geometric conductance of a coupling between two mesh
#   vertices, in µm (conductance = κ / r_L).
#
# **Running it**: from the `notebooks/` folder, with the project environment
# (`uv run jupyter lab`). It takes about 1 minute, mostly meshing.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import trimesh

import camovox as cv
from camovox import ports, viz
from camovox import validate as va  # analytic reference solutions

pv.set_jupyter_backend("static")  # 3D views as static images in the notebook

# Membrane parameters used below.
# - REST: a typical resting (slice) membrane.
# - LEAKY: deliberately leakier than physiological, so voltage decay along a
#   short test cylinder is large enough to see. It is a test setting only.
REST = cv.Passive(r_L=100.0, c_m=1.0, g_pas=2.5e-5)
LEAKY = cv.Passive(r_L=100.0, c_m=1.0, g_pas=1e-3)
I_INJ = 10e-12  # injected test current: 10 pA, in A

# %% [markdown]
# ## 1. Sphere: isopotential check
#
# A small sphere is **isopotential**: its interior resistance is negligible,
# so every point sits at the same voltage. Its input resistance is then just
# the membrane resistance, $1/(g_\mathrm{pas} A)$ for surface area A, from any
# injection site. Its membrane time constant is $c_m/g_\mathrm{pas}$, which is
# 40 ms with the resting parameters.
#
# Steps: build an 8 µm-radius sphere surface, fill it with tetrahedra of
# about 2 µm edge, build the FEM model, then inject through a 1 µm-radius
# patch on its surface.
#
# **What to look for:** FEM within about 1% of the exact value. The small
# excess comes from the faceted surface having slightly less area than a
# true sphere.

# %%
sphere = trimesh.creation.icosphere(subdivisions=3, radius=8.0)  # µm
sphere_tets, sphere_fidelity = cv.tetrahedralize(sphere, edge_length=2.0)
sphere_model, _ = cv.fem_model(sphere_tets)  # FEM network on the mesh vertices

patch = cv.ball_patch(sphere_model, center=[8.0, 0.0, 0.0], radius=1.0)
rin_fem = cv.input_resistance(sphere_model, REST, patch)  # Ω
rin_exact = va.sphere_input_resistance(8.0, REST)  # Ω
tau_fem = cv.time_constants(sphere_model, REST, k=1)[0]  # slowest time constant, s

print(
    f"mesh: {sphere_fidelity.n_tets} tetrahedra; surface area changed by "
    f"{100 * sphere_fidelity.area_error:+.2f}% from the input surface"
)
print(f"input resistance: FEM {rin_fem / 1e6:.1f} MΩ, exact {rin_exact / 1e6:.1f} MΩ")
print(f"time constant:    FEM {1e3 * tau_fem:.2f} ms, exact {1e3 * REST.tau_m:.2f} ms")

# %% [markdown]
# ## 2. Sealed cylinder: attenuation against cable theory
#
# 10 pA is injected at one end of a cylinder (radius 2 µm, length 100 µm)
# whose ends are sealed. Cable theory predicts
# $V(x)/V(0) = \cosh((L-x)/\lambda)/\cosh(L/\lambda)$, where L is the length
# and λ the length constant. The leaky test membrane gives λ ≈ 320 µm, so
# the voltage falls by about 5% over the cylinder.
#
# The plot shows one dot per surface vertex of the FEM model (voltage
# normalized by the voltage at the injection patch) against the cable-theory
# curve.
#
# **What to look for:** the dots follow the curve to within a fraction of a
# percent. The slight extra decay toward the far end comes from the end caps,
# which carry membrane that the idealized sealed cable doesn't have. Thin
# processes like this are where 1D cable models are accurate. Only the truly
# 3D parts of a cell (such as toric spines) need special treatment.

# %%
radius, length = 2.0, 100.0  # µm
cylinder = trimesh.creation.cylinder(radius=radius, height=length, sections=24)
# envelope: how far (µm) the meshed surface may deviate from the input surface
cyl_tets, _ = cv.tetrahedralize(cylinder, edge_length=1.5, envelope=0.05)
cyl_model, _ = cv.fem_model(cyl_tets)

end_patch = cv.ball_patch(cyl_model, center=[0.0, 0.0, -length / 2], radius=radius)
v_cyl = cv.solve_dc(cyl_model, LEAKY, I_INJ * end_patch / end_patch.sum())  # V
print(f"length constant λ = {va.cable_length_constant(radius, LEAKY):.0f} µm")

x = viz.vertex_distance_along(cyl_tets, axis=[0, 0, 1], origin=[0, 0, -length / 2])
on_surface = cyl_model.area > 0  # surface vertices carry membrane, interior ones don't
v_at_injection = np.average(v_cyl, weights=end_patch)
x_curve = np.linspace(0, length, 200)
viz.plot_profile(
    x[on_surface],
    {"FEM (surface vertices)": v_cyl[on_surface] / v_at_injection},
    xlabel="distance from injection (µm)",
    ylabel="V(x) / V(0)",
    reference={
        "cable theory": (
            x_curve,
            va.sealed_cylinder_attenuation(x_curve, radius, length, LEAKY),
        )
    },
)
plt.show()

# %% [markdown]
# The same solution on the cylinder's surface (in mV). Bright = high voltage
# at the injected end.

# %%
pl = viz.plot_surface(cyl_tets, 1e3 * v_cyl, name="V (mV)")
pl.show()

# %% [markdown]
# ## 3. Torus: current around a hole
#
# A ring (centreline radius 5 µm, tube radius 1 µm) is the simplest loop,
# and toric spines are full of loops. 10 pA is injected through a 0.5 µm
# patch on the outer rim. Three views of the same solution:
#
# - **Surface voltage**: highest at the injection site, falling symmetrically
#   both ways round to a minimum on the opposite side. The total range is
#   small (about 0.02 mV on top of about 5 mV), because the ring is
#   electrically compact.
# - **Current streamlines**: paths current follows inside the volume, as
#   tubes coloured by current density (log scale). Current leaves the
#   patch, splits, and flows both ways around the hole. This is what a loop
#   does electrically: it offers two parallel paths.
# - **Slice** through the ring's mid-plane: the voltage is nearly uniform
#   across each cross-section, so the ring behaves like a 1D cable bent into
#   a circle. Only right at the injection patch is the field truly 3D.

# %%
torus = trimesh.creation.torus(major_radius=5.0, minor_radius=1.0)  # µm
torus_tets, torus_fidelity = cv.tetrahedralize(torus, edge_length=0.6, envelope=0.03)
torus_model, torus_quality = cv.fem_model(torus_tets)

inject_patch = cv.ball_patch(
    torus_model, center=[6.0, 0.0, 0.0], radius=0.5
)  # outer rim
v_torus = cv.solve_dc(torus_model, LEAKY, I_INJ * inject_patch / inject_patch.sum())
j_torus = cv.current_density(torus_tets, v_torus, LEAKY)  # A/µm², one vector per tet
print(f"mesh: {torus_fidelity.n_tets} tetrahedra")

# %%
pl = viz.plot_surface(torus_tets, 1e3 * v_torus, name="V (mV)")
pl.show()

# %%
# Streamlines are seeded in a 0.8 µm sphere centred in the tube at the
# injection site, and traced forward and backward along the current.
pl = viz.plot_streamlines(
    torus_tets, j_torus, source_center=[5.0, 0.0, 0.0], source_radius=0.8
)
pl.show()

# %%
pl = viz.plot_slice(torus_tets, 1e3 * v_torus, name="V (mV)", normal="z")
pl.show()

# %% [markdown]
# ## 4. Equipotentials as a partition preview
#
# The planned reduction cuts a cell into compartments along **equipotential
# surfaces** of a particular field: current injected at a root and leaving
# through the membrane in proportion to area (the "λ → ∞" field; see
# `docs/design/reduction.md`). With an almost zero leak (g_pas = 1e-8 S/cm²) membrane
# current is nearly proportional to area, so the solution approximates that
# field. Because the leak is so small, the whole torus sits at a large,
# meaningless offset voltage. Only differences matter, so we plot
# φ − min(φ). VTK labels it "phi" because its font has no φ.
#
# **What to look for:** away from the source, the surfaces are clean
# cross-sections of the tube, one on each arm, and the two arms rejoin
# opposite the source. Cut into slabs between surfaces, this would give a
# ring of compartments, a loop that emerges from the field with no skeleton
# needed. Near the source the field is 3D and the surfaces bunch up.

# %%
NO_LEAK = cv.Passive(r_L=100.0, c_m=1.0, g_pas=1e-8)
phi = cv.solve_dc(torus_model, NO_LEAK, I_INJ * inject_patch / inject_patch.sum())
pl = viz.plot_isosurfaces(
    torus_tets, 1e3 * (phi - phi.min()), n=12, name="phi - min (mV)"
)
pl.show()

# %% [markdown]
# ## 5. Four-point violation on the ring
#
# Later notebooks ask whether a set of synapses on a looped spine could be
# reproduced by a **loop-free tree** of resistors, which is much easier to
# put into a simulator. The test uses two ingredients:
#
# - the **resistance distance** between two sites: the resistance you would
#   measure between them with the membrane removed;
# - the **four-point condition**. For any four sites there are three ways
#   to pair them up. Add the two distances in each pairing and sort the
#   three sums, a ≥ b ≥ c. Distances come from a tree **exactly when a = b**
#   for every four sites.
#
# Two ways to measure the departure from a tree:
# - **relative** violation (a − b)/a: the error scale a tree must accept on
#   these distances. 0 for a tree; 0.25 for four *points* evenly spaced on
#   an ideal ring;
# - **intrinsic** violation (a − b)/(a − c): unchanged by adding resistance
#   in series with any one site. 0 for a tree; 1 for an ideal 4-site ring.
#
# Why two: real sites are patches, and current must fan out from each patch
# (spreading resistance). That adds the same amount to every distance from
# that site, like a short twig on a tree. It dilutes the relative measure
# but leaves the intrinsic one unchanged.
#
# Below, four 0.4 µm patches sit on the outer rim of the torus:
# - **evenly spaced** (0°, 90°, 180°, 270°): as loop-like as it gets. Expect
#   intrinsic ≈ 1, and relative < 0.25 because of the patches' spreading
#   resistance;
# - **bunched** (0°–60°): the long way round is a weak parallel path, so
#   the ring looks almost like a short cable. Expect both measures small.
#
# Technical note: distances are computed with the first site grounded.
# That choice of reference doesn't change resistance distances.

# %%
AXIAL_ONLY = cv.Passive(r_L=100.0)  # membrane is ignored for resistance distances


def rim_patches(angles_deg: list[float]) -> list[np.ndarray]:
    """0.4 µm patches on the torus's outer rim (radius 6 µm) at given angles."""
    a = np.deg2rad(angles_deg)
    centers = np.column_stack([6.0 * np.cos(a), 6.0 * np.sin(a), np.zeros(len(a))])
    return [cv.ball_patch(torus_model, c, 0.4) for c in centers]


for label, angles in [
    ("evenly spaced", [0, 90, 180, 270]),
    ("bunched", [0, 20, 40, 60]),
]:
    sites = rim_patches(angles)
    d = ports.resistance_distances(torus_model, AXIAL_ONLY, sites[1:], ground=sites[0])
    rel = ports.four_point_violation(d, "max")[0]
    intr = ports.four_point_violation(d, "gap")[0]
    print(f"{label:>13} {angles}: relative {rel:.3f}, intrinsic {intr:.3f}")

# %% [markdown]
# ## 6. Sites and mesh quality
#
# **Sites** are drawn as coloured spheres at the centre of each patch,
# because real patches are too small to see at this scale. Here, 8 patches
# at random angles on the rim (fixed random seed, so the figure is
# reproducible). On real spines, sites are geodesic patches grown to a target
# synapse area (notebook 02).

# %%
rng = np.random.default_rng(0)  # fixed seed: same figure every run
random_sites = rim_patches(list(np.rad2deg(rng.uniform(0, 2 * np.pi, 8))))
pl = viz.plot_sites(torus_tets, random_sites)
pl.show()

# %% [markdown]
# **Mesh quality.** In the FEM network, most couplings between neighbouring
# vertices are positive conductances. Badly shaped (obtuse) tetrahedra give
# some **negative** couplings (κ < 0). The view shows the most negative 20%
# of them. Scattered lines like these are normal for this mesher and don't
# hurt accuracy (sections 1 and 2 match exact answers to about 1%). They
# matter for later reduction steps that assume all couplings are positive,
# so we track them.

# %%
print(
    f"negative couplings carry {100 * torus_quality.negative_fraction:.1f}% of total |κ|"
)
pl = viz.plot_mesh_quality(torus_model, torus_tets)
pl.show()

# %% [markdown]
# ## 7. Export for ParaView
#
# Writes the torus solution to `../data/torus_dc.vtu`. It contains the
# voltage (`V_mV`), the near-λ→∞ field (`phi_mV`) and the current density per
# tetrahedron (`J_A_per_um2`). In ParaView: open the file, colour by `V_mV`,
# or add a *Stream Tracer* filter on `J_A_per_um2`.

# %%
viz.write_vtu(
    "../data/torus_dc.vtu",
    torus_tets,
    point_data={"V_mV": 1e3 * v_torus, "phi_mV": 1e3 * phi},
    cell_data={"J_A_per_um2": j_torus},
)
