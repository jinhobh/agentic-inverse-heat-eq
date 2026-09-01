"""Numerical solver for the 2D diffusion (heat) equation.

    u_t = kappa * (u_xx + u_yy)   on   [0, Lx] x [0, Ly]

Uses the implicit backward-time centered-space (BTCS) scheme with a sparse
LU factorisation of the 5-point Laplacian.  Homogeneous Dirichlet boundary
conditions are enforced on all four edges.
"""

import numpy as np
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import splu


def gaussian_ic_2d(
    x,
    y,
    amplitude=1.0,
    center_x=0.5,
    center_y=0.5,
    width_x=0.05,
    width_y=0.05,
):
    """2D Gaussian pulse initial condition on the grid defined by *x* and *y*.

    The grid uses ``'ij'`` indexing so that ``u[i, j]`` corresponds to
    ``(x[i], y[j])``.
    """
    X, Y = np.meshgrid(x, y, indexing="ij")
    return amplitude * np.exp(
        -((X - center_x) ** 2) / (2.0 * width_x**2)
        - ((Y - center_y) ** 2) / (2.0 * width_y**2)
    )


def solve_diffusion_2d(kappa, u0, t, nx, ny, dt, lx=1.0, ly=1.0):
    """Solve ``u_t = kappa*(u_xx + u_yy)`` on ``[0, lx] x [0, ly]``.

    Parameters
    ----------
    kappa : float
        Diffusivity (>= 0).
    u0 : (nx, ny) array
        Initial field.
    t : float
        Final time (>= 0).
    nx, ny : int
        Grid points in x and y (each >= 3).
    dt : float
        Time step (> 0).
    lx, ly : float
        Domain lengths in x and y.

    Returns
    -------
    x, y : 1-D arrays
        Spatial coordinates.
    u : (nx, ny) array
        Field at time *t*.
    """
    if nx < 3 or ny < 3:
        raise ValueError("nx and ny must each be at least 3")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if t < 0:
        raise ValueError("t must be non-negative")
    if kappa < 0:
        raise ValueError("kappa must be non-negative")

    u = np.asarray(u0, dtype=float).copy()
    if u.shape != (nx, ny):
        raise ValueError(f"u0 must have shape ({nx}, {ny}), got {u.shape}")

    x = np.linspace(0.0, lx, nx)
    y = np.linspace(0.0, ly, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    rx = kappa * dt / dx**2
    ry = kappa * dt / dy**2
    nt = int(round(t / dt))

    # ---- build sparse system for interior points only ---------------------------
    nxi = nx - 2  # interior x count
    nyi = ny - 2  # interior y count
    ni = nxi * nyi  # total interior unknowns

    main = (1.0 + 2.0 * rx + 2.0 * ry) * np.ones(ni)

    # ±1  off-diagonal (y-neighbour): -ry
    off1 = -ry * np.ones(ni - 1)
    off1[nyi - 1 :: nyi] = 0.0  # break coupling across interior rows

    # ±nyi  off-diagonal (x-neighbour): -rx
    offn = -rx * np.ones(ni - nyi)

    a_mat = diags(
        [offn, off1, main, off1, offn],
        [-nyi, -1, 0, 1, nyi],
        format="csc",
    )
    lu = splu(a_mat)

    # ---- time stepping ---------------------------------------------------------
    for _ in range(nt):
        u_int = u[1:-1, 1:-1].ravel()
        u_int = lu.solve(u_int)
        u[1:-1, 1:-1] = u_int.reshape(nxi, nyi)
        u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0

    return x, y, u


def profile_width_2d(x, y, u):
    """Radial spread of the field, treating ``u`` as a mass distribution.

    Returns ``sqrt(var_x + var_y)``, which generalises the 1-D ``sigma`` to
    two dimensions.
    """
    X, Y = np.meshgrid(x, y, indexing="ij")

    mass = np.trapezoid(np.trapezoid(u, y, axis=1), x)
    if mass <= 0.0:
        raise ValueError("profile mass must be positive")

    mean_x = np.trapezoid(np.trapezoid(X * u, y, axis=1), x) / mass
    mean_y = np.trapezoid(np.trapezoid(Y * u, y, axis=1), x) / mass

    var_x = np.trapezoid(np.trapezoid((X - mean_x) ** 2 * u, y, axis=1), x) / mass
    var_y = np.trapezoid(np.trapezoid((Y - mean_y) ** 2 * u, y, axis=1), x) / mass

    return np.sqrt(var_x + var_y)


def diffusion_errors_2d(kappa, u0, T, Nx, Ny, dt, u_target, Lx=1.0, Ly=1.0):
    """Compare a guessed diffusivity against a fixed target observation.

    Returns
    -------
    u_error : float
        Relative L2 misfit of the field.
    x_error : float
        Signed radial-width misfit, ``sigma_guess - sigma_target``.
        Positive means the guessed kappa is too high; negative means it is
        too low.
    """
    x, y, u = solve_diffusion_2d(kappa, u0, T, Nx, Ny, dt, Lx, Ly)
    u_error = np.linalg.norm(u - u_target) / np.linalg.norm(u_target)
    x_error = profile_width_2d(x, y, u) - profile_width_2d(x, y, u_target)
    return u_error, x_error