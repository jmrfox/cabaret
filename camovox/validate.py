"""Analytic reference solutions for passive test shapes."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from camovox.params import UM2_TO_CM2, UM_TO_CM, Passive


def sphere_input_resistance(radius: float, params: Passive) -> float:
    """Input resistance of an isopotential sphere.

    Parameters
    ----------
    radius : float, in µm
    params : Passive

    Returns
    -------
    float, in Ω
    """
    return 1.0 / (params.g_pas * 4 * np.pi * radius**2 * UM2_TO_CM2)


def cable_length_constant(radius: float, params: Passive) -> float:
    """DC length constant λ of a cylinder.

    Parameters
    ----------
    radius : float, in µm
    params : Passive

    Returns
    -------
    float, in µm
    """
    a = radius * UM_TO_CM
    return np.sqrt(a / (2 * params.r_L * params.g_pas)) / UM_TO_CM


def sealed_cylinder_input_resistance(
    radius: float, length: float, params: Passive
) -> float:
    """Input resistance at one end of a sealed-end cylinder (cable theory).

    Parameters
    ----------
    radius, length : float, in µm
    params : Passive

    Returns
    -------
    float, in Ω
    """
    a = radius * UM_TO_CM
    r_a = params.r_L / (np.pi * a**2)  # Ω/cm
    lam = cable_length_constant(radius, params)
    return r_a * lam * UM_TO_CM / np.tanh(length / lam)


def sealed_cylinder_attenuation(
    x: ArrayLike, radius: float, length: float, params: Passive
) -> NDArray[np.float64]:
    """Steady-state ``V(x) / V(0)`` for current injected at ``x = 0``.

    Parameters
    ----------
    x : array_like of float, in µm
        Distance from the injection end.
    radius, length : float, in µm
    params : Passive

    Returns
    -------
    ndarray of float
    """
    lam = cable_length_constant(radius, params)
    return np.cosh((length - np.asarray(x, dtype=float)) / lam) / np.cosh(length / lam)
