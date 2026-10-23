#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd

from scipy.spatial import cKDTree


def local_pca_dimension_cell_min(coords, radii, min_neighbors=5):
    """
    The per-cell minimum of the local effective dimension over several radii.

    coords is (n_cells, 2) and the rows of the returned frame line up with its
    rows, so it can be concatenated onto whatever table the coordinates came
    from.
    """

    coords = np.asarray(coords, dtype=float)

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (n_cells, 2)")

    radii = np.asarray(radii, dtype=float)

    if radii.ndim != 1 or len(radii) == 0:
        raise ValueError("radii must be a non-empty 1D array")
    if np.any(radii <= 0):
        raise ValueError("all radii must be positive")

    radii = np.sort(np.unique(radii))

    n_cells = len(coords)

    d_by_radius = np.full((n_cells, len(radii)), np.nan)

    if n_cells > 0:

        if not np.isfinite(coords).all():
            raise ValueError("coords contain NaN or infinite values")

        coords = coords - coords.mean(axis=0)

        x = coords[:, 0]
        y = coords[:, 1]

        tree = cKDTree(coords)

        for r_index, r in enumerate(radii):

            pairs = tree.query_pairs(r, output_type="ndarray")

            if len(pairs) > 0:
                centre = np.concatenate([pairs[:, 0], pairs[:, 1]])
                neighbour = np.concatenate([pairs[:, 1], pairs[:, 0]])
            else:
                centre = np.zeros(0, dtype=np.intp)
                neighbour = np.zeros(0, dtype=np.intp)

            count = np.bincount(centre, minlength=n_cells) + 1.0
            sum_x = np.bincount(centre, weights=x[neighbour], minlength=n_cells) + x
            sum_y = np.bincount(centre, weights=y[neighbour], minlength=n_cells) + y
            sum_xx = np.bincount(centre, weights=x[neighbour] ** 2, minlength=n_cells) + x ** 2
            sum_yy = np.bincount(centre, weights=y[neighbour] ** 2, minlength=n_cells) + y ** 2
            sum_xy = np.bincount(centre, weights=x[neighbour] * y[neighbour],
                                 minlength=n_cells) + x * y

            s_xx = sum_xx - sum_x ** 2 / count
            s_yy = sum_yy - sum_y ** 2 / count
            s_xy = sum_xy - sum_x * sum_y / count

            trace = s_xx + s_yy
            determinant = s_xx * s_yy - s_xy ** 2

            denominator = trace ** 2 - 2.0 * determinant

            usable = (count >= min_neighbors) & (denominator > 0)

            d_by_radius[usable, r_index] = np.clip(
                trace[usable] ** 2 / denominator[usable], 1.0, 2.0)

    n_usable_radii = np.isfinite(d_by_radius).sum(axis=1)
    usable_any = n_usable_radii > 0

    d_min_cell = np.full(n_cells, np.nan)

    if np.any(usable_any):
        d_min_cell[usable_any] = np.nanmin(d_by_radius[usable_any], axis=1)

    df_out = pd.DataFrame({"usable_any": usable_any,
                           "n_usable_radii": n_usable_radii,
                           "d_min_cell": d_min_cell})

    for r_index, r in enumerate(radii):
        df_out["d_r_" + format(r, "g")] = d_by_radius[:, r_index]

    return df_out
