"""Passive biophysical parameters and unit conversions.

Morphology is stored in µm (lengths), µm² (areas) and µm (geometric
conductance factors κ). Biophysical parameters use the conventional
neuroscience units below. The methods here convert to SI (S, F, Ω, s).
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

UM_TO_CM = 1e-4
UM2_TO_CM2 = 1e-8


@dataclass(frozen=True)
class Passive:
    """Uniform passive membrane and cytoplasm parameters.

    Parameters
    ----------
    r_L : float
        Axial (cytoplasmic) resistivity, in Ω·cm.
    c_m : float
        Specific membrane capacitance, in µF/cm².
    g_pas : float
        Specific passive membrane conductance, in S/cm².

    Examples
    --------
    >>> p = Passive(r_L=100.0, c_m=1.0, g_pas=2.5e-5)
    >>> round(p.tau_m * 1e3, 1)  # ms
    40.0
    """

    r_L: float = 100.0
    c_m: float = 1.0
    g_pas: float = 2.5e-5

    @property
    def tau_m(self) -> float:
        """Membrane time constant ``c_m / g_pas``, in s."""
        return self.c_m * 1e-6 / self.g_pas

    def axial_conductance(self, kappa: ArrayLike) -> NDArray[np.float64]:
        """Convert geometric conductance factors to conductances.

        Parameters
        ----------
        kappa : array_like of float, in µm
            Geometric factor (area / length) of each coupling.

        Returns
        -------
        ndarray of float, in S
        """
        return np.asarray(kappa, dtype=float) * UM_TO_CM / self.r_L

    def membrane_conductance(self, area: ArrayLike) -> NDArray[np.float64]:
        """Passive membrane conductance of membrane patches.

        Parameters
        ----------
        area : array_like of float, in µm²

        Returns
        -------
        ndarray of float, in S
        """
        return self.g_pas * np.asarray(area, dtype=float) * UM2_TO_CM2

    def capacitance(self, area: ArrayLike) -> NDArray[np.float64]:
        """Membrane capacitance of membrane patches.

        Parameters
        ----------
        area : array_like of float, in µm²

        Returns
        -------
        ndarray of float, in F
        """
        return self.c_m * 1e-6 * np.asarray(area, dtype=float) * UM2_TO_CM2
