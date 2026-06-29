"""Compatibility wrapper for the importable heat equation solver module."""

from heateq_num_solver import gaussian_ic, heat_errors, profile_width, solve_heat

__all__ = ["gaussian_ic", "solve_heat", "profile_width", "heat_errors"]


