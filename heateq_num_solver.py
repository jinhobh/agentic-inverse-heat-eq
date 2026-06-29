import numpy as np


def gaussian_ic(x, amplitude=1.0, center=0.5, width=0.05):
    """Localized Gaussian pulse initial condition."""
    return amplitude * np.exp(-(x - center) ** 2 / (2.0 * width ** 2))


def solve_heat(kappa, u0, t, nx, dt, l=1.0):
    """Solve u_t = kappa * u_xx on x in [0, l].

    Uses the implicit backward-time, centered-space scheme with zero
    Dirichlet boundary conditions.
    """
    if nx < 3:
        raise ValueError("nx must be at least 3")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if t < 0:
        raise ValueError("t must be non-negative")
    if kappa < 0:
        raise ValueError("kappa must be non-negative")

    u = np.asarray(u0, dtype=float).copy()
    if u.shape != (nx,):
        raise ValueError(f"u0 must have shape ({nx},), got {u.shape}")

    x = np.linspace(0.0, l, nx)
    dx = x[1] - x[0]
    r = kappa * dt / dx**2

    nt = int(round(t / dt))
    n_interior = nx - 2

    main = (1.0 + 2.0 * r) * np.ones(n_interior)
    off = -r * np.ones(n_interior - 1)
    a = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)

    u[0] = 0.0
    u[-1] = 0.0

    for _ in range(nt):
        u[1:-1] = np.linalg.solve(a, u[1:-1])
        u[0] = 0.0
        u[-1] = 0.0

    return x, u


def profile_width(x, u):
    """Spatial spread of the field, treating u as a mass distribution."""
    mass = np.trapezoid(u, x)
    if mass <= 0.0:
        raise ValueError("profile mass must be positive")

    mean = np.trapezoid(x * u, x) / mass
    var = np.trapezoid((x - mean) ** 2 * u, x) / mass
    return np.sqrt(var)


def heat_errors(kappa, u0, T, Nx, dt, u_target, L=1.0):
    """Compare a guessed diffusivity against a fixed target observation.

    Returns
    -------
    u_error : float
        Relative L2 misfit of the field.
    x_error : float
        Signed width misfit, sigma_guess - sigma_target. Positive means
        the guessed kappa is too high; negative means it is too low.
    """
    x, u = solve_heat(kappa, u0, T, Nx, dt, l=L)
    u_error = np.linalg.norm(u - u_target) / np.linalg.norm(u_target)
    x_error = profile_width(x, u) - profile_width(x, u_target)
    return u_error, x_error
