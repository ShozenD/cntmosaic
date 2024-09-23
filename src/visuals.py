from typing import Sequence
import jax
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

def plot_surface_scatter(
    N_grid: int,
    X_grid: NDArray | None = None,
    y_grid: NDArray | None = None,
    X: NDArray | None = None,
    y: NDArray | None = None,
    test_ind: jax.Array | None = None,
    post_y: jax.Array | None = None,
    xz_lines: list[tuple[jax.Array, jax.Array, float]] | None = None,
    yz_lines: list[tuple[jax.Array, jax.Array, float]] | None = None,
    xy_annotate_lines: Sequence[
        tuple[tuple[float, float], tuple[float, float]] | None
    ] = None,
    fig_size: float = 8.0,
    label_size: float = 8.0,
    grid_alpha: float = 0.1,
    y_wireframe_alpha: float = 1.0,
    post_alpha: float = 0.1,
    point_size: float = 1.0,
    point_alpha: float = 0.5,
    ci_alpha: float = 0.1,
) -> None:
    # setup figure
    fig = plt.figure(figsize=(fig_size, fig_size))

    # plot the surface of the noiseless function and the data points
    x0_grid, x1_grid = (
        X_grid[:, 0].reshape((N_grid, N_grid)),
        X_grid[:, 1].reshape((N_grid, N_grid)),
    )
    ax = fig.add_subplot(projection="3d")

    # plot wireframes from draws from the posterior
    if post_y is not None:
        for i in range(post_y.shape[0]):
            post_y_grid = post_y[i, :].reshape((N_grid, N_grid))
            ax.plot_wireframe(
                x0_grid,
                x1_grid,
                post_y_grid,
                rstride=1,
                cstride=1,
                linewidth=1.0,
                alpha=post_alpha,
                color="tab:blue",
            )

    # plot the data points
    if X is not None and y is not None:
        color = (
            "tab:blue"
            if test_ind is None
            else np.where(test_ind, "tab:green", "tab:blue")
        )
        ax.scatter(
            xs=X[:, 0],
            ys=X[:, 1],
            zs=y,
            c=color,
            s=point_size,
            alpha=point_alpha,
        )

    # add confidence intervals at the boundaries
    if xz_lines:
        for line in xz_lines:
            x, z, y = line
            ax.plot(
                x, z, zs=y, zdir="y", color="tab:green", linestyle="--", alpha=ci_alpha
            )
    if yz_lines:
        for line in yz_lines:
            y, z, x = line
            ax.plot(
                y, z, zs=x, zdir="x", color="tab:green", linestyle="--", alpha=ci_alpha
            )

    # plot the surface of the noiseless function
    if y_grid is not None:
        ax.plot_wireframe(
            x0_grid,
            x1_grid,
            y_grid,
            rstride=1,
            cstride=1,
            linewidths=1.0,
            alpha=y_wireframe_alpha,
            color="tab:orange",
        )

    # add box in xy plane
    z_min = ax.get_zlim()[0]
    ax.set_zlim(ax.get_zlim())
    if xy_annotate_lines:
        for line in xy_annotate_lines:
            x_bounds, y_bounds = line
            z_bounds = (z_min, z_min)
            ax.plot(
                x_bounds,
                y_bounds,
                z_bounds,
                color="tab:gray",
                alpha=0.5,
                linestyle="--",
            )

    # remove background panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("w")
    ax.yaxis.pane.set_edgecolor("w")
    ax.zaxis.pane.set_edgecolor("w")

    # configure grid
    ax.xaxis._axinfo["grid"]["color"] = ("tab:gray", grid_alpha)
    ax.yaxis._axinfo["grid"]["color"] = ("tab:gray", grid_alpha)
    ax.zaxis._axinfo["grid"]["color"] = ("tab:gray", grid_alpha)

    # set labels and ticks
    ax.xaxis.set_tick_params(labelsize=label_size)
    ax.set_xlabel("x0", fontsize=label_size)
    ax.yaxis.set_tick_params(labelsize=label_size)
    ax.set_ylabel("x1", fontsize=label_size)
    ax.zaxis.set_tick_params(labelsize=label_size)
    ax.set_zlabel("y", fontsize=label_size)

    ax.set_box_aspect(aspect=None, zoom=0.9)
    return ax


def plot_line_scatter(
    X_grid: jax.Array,
    y_grid: jax.Array,
    X: jax.Array | None = None,
    y: jax.Array | None = None,
    test_ind: jax.Array | None = None,
    post_y: jax.Array | None = None,
    v_lines: Sequence[float] | None = None,
    ci: tuple[jax.Array, jax.Array] | None = None,
    fig_size: float = 5.0,
    label_size: float = 8.0,
    post_alpha: float = 0.25,
    point_size: float = 1.0,
    point_alpha: float = 0.25,
    ci_alpha: float = 0.1,
):
    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = fig.add_subplot()

    # plot draws of the function from the posterior
    if post_y is not None:
        for i in range(post_y.shape[0]):
            ax.plot(
                X_grid, post_y[i, :], linewidth=1.0, alpha=post_alpha, color="tab:blue"
            )

    # plot the data points
    if X is not None and y is not None:
        if test_ind is None:
            color = "tab:blue"
        else:
            test_ind = np.array(test_ind).squeeze()
            color = np.where(test_ind, "tab:green", "tab:blue")
        ax.scatter(X, y, c=color, s=point_size, alpha=point_alpha)

    # add confidence intervals
    if ci:
        ax.fill_between(
            X_grid.squeeze(), ci[0], ci[1], color="tab:blue", alpha=ci_alpha
        )

    # add the noiseless function
    ax.plot(X_grid, y_grid, linewidth=1.0, alpha=1.0, color="tab:orange")

    # add vertical lines denoting boundaries of the training data
    if v_lines:
        for v_line in v_lines:
            plt.axvline(v_line, color="tab:gray", linestyle="--", alpha=0.5)
            plt.axvline(v_line, color="tab:gray", linestyle="--", alpha=0.5)

    # set labels and ticks
    ax.set_xlabel("x", fontsize=label_size)
    ax.set_ylabel("y", fontsize=label_size)
    ax.xaxis.set_tick_params(labelsize=label_size)
    ax.yaxis.set_tick_params(labelsize=label_size)

    return ax