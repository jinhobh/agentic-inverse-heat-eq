"""Reconstruct the OpenRouter (real LLM) sweep figure from saved run data.

The sweep runs that used the actual OpenRouter API (model
``google/gemini-2.5-flash``) were captured only at the summary level -- the
per-iteration histories were not persisted, so the per-iteration trajectory
figure cannot be rebuilt. This script re-renders the *summary* performance view
(relative k error, estimated-vs-true, best field misfit, effort) from the real
recorded numbers so the genuine LLM behavior is preserved alongside the offline
bisection ("nonai") artifacts.

Data below is transcribed verbatim from the two real OpenRouter runs:
- RUN_12: the first complete sweep over 12 in-range k values.
- RUN_15: the second sweep (in-range + out-of-range) which reached 13 of 15
  values before OpenRouter returned HTTP 402 "Insufficient credits".
"""

import matplotlib.pyplot as plt
import numpy as np

from agentic_k_demo import MAX_ITERATIONS, TARGET_U_ERROR, K_MIN, K_MAX

MODEL = "google/gemini-2.5-flash"

# (k_true, k_est, best_u_error, n_iterations, converged)
RUN_12 = [
    (1e-4, 0.0002442, 9.88e-4, 13, True),
    (2e-4, 0.0003441, 9.86e-4, 12, True),
    (5e-4, 0.0005868, 5.91e-4, 11, True),
    (1e-3, 0.0001, 6.14e-3, 5, False),
    (5e-3, 0.004945, 3.52e-4, 10, True),
    (1e-2, 0.009974, 1.56e-4, 12, True),
    (5e-2, 0.04982, 7.07e-4, 10, True),
    (0.12, 0.1202, 4.84e-4, 10, True),
    (0.3, 0.3009, 1.01e-3, 8, False),
    (0.5, 0.5, 3.84e-5, 1, True),
    (0.8, 0.7501, 2.53e-2, 20, False),
    (1.0, 0.5005, 2.89e-1, 20, False),
]

RUN_15 = [
    (1e-5, 0.0001815, 1.18e-3, 20, False),
    (1e-4, 0.0002461, 1.00e-3, 12, False),
    (2e-4, 0.4982, 6.61e-1, 20, False),
    (5e-4, 0.4405, 6.40e-1, 20, False),
    (1e-3, 0.0009769, 1.57e-4, 10, True),
    (5e-3, 0.005125, 7.94e-4, 10, True),
    (1e-2, 0.08834, 2.85e-1, 20, False),
    (5e-2, 0.0499, 3.85e-4, 10, True),
    (0.12, 0.1202, 4.84e-4, 10, True),
    (0.3, 0.2976, 2.88e-3, 20, False),
    (0.5, 0.5, 3.84e-5, 1, True),
    (0.8, 0.7981, 9.50e-4, 15, True),
    (1.0, 0.97, 1.17e-2, 20, False),
    # k_true=2.0, 5.0 never ran: OpenRouter HTTP 402 (insufficient credits).
]


def to_results(rows):
    out = []
    for k_true, k_est, u_err, iters, converged in rows:
        out.append(
            {
                "k_true": k_true,
                "k_est": k_est,
                "k_abs_error": abs(k_est - k_true),
                "k_rel_error": abs(k_est - k_true) / k_true,
                "best_u_error": u_err,
                "n_iterations": iters,
                "converged": converged,
            }
        )
    return out


def plot_summary(results, model, save_path, note=None):
    """The original summary-style 4-panel performance figure."""
    k_true = np.array([r["k_true"] for r in results])
    k_est = np.array([r["k_est"] for r in results])
    k_rel = np.array([r["k_rel_error"] for r in results])
    u_err = np.array([r["best_u_error"] for r in results])
    iters = np.array([r["n_iterations"] for r in results])
    converged = np.array([r["converged"] for r in results])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Relative k error vs true k.
    axes[0, 0].loglog(k_true, k_rel, marker="o", color="C0")
    axes[0, 0].set_xlabel("k_true")
    axes[0, 0].set_ylabel("relative k error")
    axes[0, 0].set_title("Identification error vs true k")
    axes[0, 0].grid(True, which="both", alpha=0.3)

    # Estimated vs true k with the ideal y = x reference.
    lims = [min(k_true.min(), k_est.min()) * 0.7, max(k_true.max(), k_est.max()) * 1.3]
    axes[0, 1].plot(lims, lims, "k:", label="perfect (y=x)")
    axes[0, 1].scatter(
        k_true[converged], k_est[converged], color="C2", label="converged", zorder=3
    )
    axes[0, 1].scatter(
        k_true[~converged], k_est[~converged], color="C3", label="not converged",
        zorder=3,
    )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("k_true")
    axes[0, 1].set_ylabel("k_est")
    axes[0, 1].set_title("Estimated vs true k")
    axes[0, 1].legend()
    axes[0, 1].grid(True, which="both", alpha=0.3)

    # Best field error vs true k, against the stopping threshold.
    axes[1, 0].loglog(k_true, u_err, marker="s", color="C1")
    axes[1, 0].axhline(TARGET_U_ERROR, color="k", linestyle=":", label="stopping target")
    axes[1, 0].set_xlabel("k_true")
    axes[1, 0].set_ylabel("best field error")
    axes[1, 0].set_title("Best field misfit vs true k")
    axes[1, 0].legend()
    axes[1, 0].grid(True, which="both", alpha=0.3)

    # Iterations used vs true k, colored by convergence.
    colors = ["C2" if c else "C3" for c in converged]
    axes[1, 1].bar(range(len(k_true)), iters, color=colors)
    axes[1, 1].axhline(MAX_ITERATIONS, color="k", linestyle=":", label="iteration cap")
    axes[1, 1].set_xticks(range(len(k_true)))
    axes[1, 1].set_xticklabels([f"{k:.3g}" for k in k_true], rotation=45, ha="right")
    axes[1, 1].set_xlabel("k_true")
    axes[1, 1].set_ylabel("iterations used")
    axes[1, 1].set_title("Effort vs true k (green=converged)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, axis="y", alpha=0.3)

    suptitle = f"Real LLM agent performance (agent: openrouter:{model})"
    if note:
        suptitle += f"\n{note}"
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"plot saved: {save_path}")
    plt.close(fig)


def main():
    plot_summary(
        to_results(RUN_12),
        MODEL,
        "k_sweep_performance_openrouter.png",
        note="reconstructed from saved summary data; 12 in-range k values",
    )
    plot_summary(
        to_results(RUN_15),
        MODEL,
        "k_sweep_performance_openrouter_oor.png",
        note="reconstructed; 13/15 values (2.0, 5.0 aborted on HTTP 402)",
    )


if __name__ == "__main__":
    main()
