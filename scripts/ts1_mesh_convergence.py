"""TS1 mesh-convergence study (plan 008): meshes, solves, saved results.

Builds the TS1 reference model on fTetWild meshes at target edge 0.2, 0.1
and 0.05 µm and on the 0.2 µm mesh refined uniformly, then computes the DC
port quantities on each. Results go into one ``.npz`` that
``notebooks/05_ts1_mesh_convergence.py`` loads, so the notebook itself runs
in seconds.

This takes tens of minutes with the meshes cached, and about 40 minutes
more the first time the 0.05 µm mesh is built.

Usage (from the repository root)::

    uv run python scripts/ts1_mesh_convergence.py data

writes ``data/cache/ts1_mesh_convergence.npz`` (or the path given as a
second argument).
"""

import argparse
import time
from pathlib import Path

import numpy as np

import cabaret as cb
from cabaret import graft, ports, toric

SCALE = 0.005  # µm per pixel
MEMBRANE = cb.Passive(r_L=100.0, c_m=1.0, g_pas=2.63e-4)  # as in notebooks 03–04
FREQS = [0.0]  # Hz; 100 Hz and 1 kHz gave the same clamped errors (plan 008)
EDGES = {"0.2": 0.2, "0.1": 0.1, "0.05": 0.05}  # fTetWild target edge, µm
REFINED = "0.2 refined"
MESH_COLUMNS = [
    "tets",
    "vertices",
    "surface vertices",
    "area change (%)",
    "volume change (%)",
    "negative |κ| (%)",
    "vertices per AZ patch (median)",
    "vertices per AZ patch (min)",
    "neck electrode nodes",
]


def _mesh_row(ts: cb.ReferenceModel) -> list[float]:
    """Size and quality of one mesh, in the order of ``MESH_COLUMNS``."""
    _, quality = cb.fem_model(ts.tets)
    verts = [np.count_nonzero(p) for p in ts.sites.patches]
    return [
        ts.fidelity.n_tets,
        ts.fidelity.n_points,
        int((ts.fem.area > 0).sum()),
        100 * ts.fidelity.area_error,
        100 * ts.fidelity.volume_error,
        100 * quality.negative_fraction,
        np.median(verts),
        np.min(verts),
        len(ts.attachment_contacts[0]),
    ]


def _port_quantities(ts: cb.ReferenceModel) -> dict[str, np.ndarray]:
    """DC port matrices and per-AZ resistances of one mesh.

    Returns
    -------
    dict
        ``z``, ``zc``: (F, 26, 26) port matrices in Ω, isolated and with the
        neck (port 0) clamped; ``r_local``, ``r_iso``, ``r_neck``: (25,)
        spine-local and isolated input resistance and membrane-free
        neck-to-AZ resistance of each AZ, in Ω.
    """
    neck = ts.attachment_ports[0]
    z = cb.port_impedance(ts.network, MEMBRANE, [neck, *ts.sites.patches], FREQS)
    zc = np.stack([ports.clamp_port(zk, 0) for zk in z])
    d = ports.ResistanceMetric(ts.network, MEMBRANE, ground=neck).distances(
        ts.sites.patches
    )
    return {
        "z": z,
        "zc": zc,
        "r_local": np.diag(zc[0].real),
        "r_iso": np.diag(z[0].real)[1:],
        "r_neck": d[0, 1:],
    }


def main(data_dir: Path, out: Path) -> None:
    """Run the study and save the results to ``out``.

    Parameters
    ----------
    data_dir : pathlib.Path
        The repository's ``data`` folder (TS files, ``neuromesh/parts``,
        ``cache/``).
    out : pathlib.Path
        Output ``.npz``. Keys: ``labels`` (mesh names), ``mesh_columns`` and
        ``mesh_table`` (one row per mesh), ``freqs``, and
        ``"<label>/<quantity>"`` for each quantity of ``_port_quantities``.
    """
    t0 = time.perf_counter()

    def log(msg: str) -> None:
        print(f"[{time.perf_counter() - t0:7.0f} s] {msg}", flush=True)

    parts = graft.load_parts(data_dir / "neuromesh/parts", scale=SCALE)
    refined_cache = data_dir / "cache/TS1_e0.2_env0.005_refined.npz"
    labels = [*EDGES, REFINED]
    rows, saved = [], {}
    for label in labels:
        if label == REFINED:
            if not refined_cache.exists():
                coarse = toric.load_toric_spine("TS1", data_dir, parts, scale=SCALE)
                coarse.tets.refined().save(refined_cache)
            tets = cb.TetMesh.load(refined_cache)
            ts = toric.load_toric_spine("TS1", data_dir, parts, scale=SCALE, tets=tets)
        else:
            ts = toric.load_toric_spine(
                "TS1", data_dir, parts, scale=SCALE, edge_length=EDGES[label]
            )
        log(f"{label}: {ts.fidelity.n_points} vertices loaded")
        rows.append(_mesh_row(ts))
        for key, value in _port_quantities(ts).items():
            saved[f"{label}/{key}"] = value
        log(f"{label}: solved")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        labels=np.array(labels),
        mesh_columns=np.array(MESH_COLUMNS),
        mesh_table=np.array(rows, dtype=float),
        freqs=np.array(FREQS),
        **saved,
    )
    log(f"saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", type=Path, help="the repository's data folder")
    parser.add_argument(
        "out", type=Path, nargs="?", help="output .npz (default: in data_dir/cache)"
    )
    args = parser.parse_args()
    main(args.data_dir, args.out or args.data_dir / "cache/ts1_mesh_convergence.npz")
