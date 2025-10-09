import numpy as np
import pandas as pd
import pyshtools as pysh

class SHEmbedding:
    """
    Generate spherical harmonic (SH) positional embeddings for spatial coordinates.
    """

    def __init__(self, lmax: int = 10, normalization: str = '4pi'):
        """
        Parameters
        ----------
        lmax : int
            Maximum spherical harmonic degree.
        normalization : str
            Normalization for SH basis ('4pi', 'ortho', 'schmidt', etc.).
        """
        self.lmax = lmax
        self.normalization = normalization

    def compute_basis(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """
        Compute the real SH basis for each (lat, lon) coordinate.

        Returns
        -------
        np.ndarray
            Array of shape (N, (lmax+1)**2) with SH basis values.
        """
        n_points = len(lat)
        n_basis = (self.lmax + 1)**2
        Y = np.zeros((n_points, n_basis))

        # Convert to colatitude (0° at North Pole)
        theta = 90.0 - lat

        for i, (th, ph) in enumerate(zip(theta, lon)):
            # Returns ylm[i, l, m] with shape (2, lmax+1, lmax+1)
            ylm = pysh.expand.spharm(
                self.lmax, th, ph,
                normalization=self.normalization,
                kind='real'
            )

            # Flatten positive (m>=0) and negative (m<0) parts into one row
            # Positive m (index 0), negative m (index 1)
            y_list = []
            for l in range(self.lmax + 1):
                # negative orders first (m = -l..-1)
                for m in range(l, 0, -1):
                    y_list.append(ylm[1, l, m])   # these correspond to m = -m
                # then m = 0
                y_list.append(ylm[0, l, 0])
                # then positive orders (m = 1..l)
                for m in range(1, l + 1):
                    y_list.append(ylm[0, l, m]) 
            Y[i, :] = np.array(y_list)

        return Y

    def from_dataframe(self, df: pd.DataFrame, lon_col='lon', lat_col='lat') -> np.ndarray:
        lon = df[lon_col].values
        lat = df[lat_col].values
        return self.compute_basis(lon, lat)
