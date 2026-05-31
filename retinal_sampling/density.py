"""Cell density functions for retinal ganglion cells and cone photoreceptors.

Ganglion cell formulas follow Watson (2014), equation 3.
Cone density follows a standard exponential approximation.
"""

import numpy as np
from scipy.special import lambertw

# Watson (2014) midget retinal ganglion cell parameters
_DG = 33162.0       # peak cell density at the fovea (cells / deg²)
_R2 = 1.05          # eccentricity at which density drops to half its peak (deg)
_C  = 3.4820e4 + 0.1  # integration constant that enforces F(0) = 0


def ganglion_cumulative(eccentricity: np.ndarray) -> np.ndarray:
    """Cumulative ganglion cell count from 0 to *eccentricity* degrees."""
    return _C - (_DG * _R2 ** 2) / (eccentricity + _R2)


def ganglion_cumulative_inv(cell_count: np.ndarray) -> np.ndarray:
    """Eccentricity (degrees) corresponding to *cell_count* cumulative ganglion cells."""
    return (_DG * _R2 ** 2) / (_C - cell_count) - _R2


def cone_cumulative(eccentricity: np.ndarray) -> np.ndarray:
    """Cumulative cone photoreceptor count from 0 to *eccentricity* degrees."""
    return 11.5 * eccentricity - 266.666_667 * np.exp(-0.75 * eccentricity) + 266.666_667


def cone_cumulative_inv(cell_count: np.ndarray) -> np.ndarray:
    """Eccentricity (degrees) corresponding to *cell_count* cumulative cones."""
    return (
        2.0 * cell_count / 23.0
        + (4.0 / 3.0) * lambertw(
            (400.0 / 23.0) * np.exp(400.0 / 23.0 - 3.0 * cell_count / 46.0), k=0
        ).real
        - 1600.0 / 69.0
    )
