import argparse
import csv
import json
import math
import os
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, Field, ValidationError

from heateq_num_solver import gaussian_ic, heat_errors, solve_heat


K_MIN = 1e-4
K_MAX = 1.0
MAX_ITERATIONS = 20
TARGET_U_ERROR = 1e-3
HISTORY_CSV_PATH = "k_history.csv"
PLOT_PATH = "k_convergence.png"
SWEEP_CSV_PATH = "k_sweep_results.csv"
SWEEP_PLOT_PATH = "k_sweep_performance.png"
SWEEP_TRAJECTORY_PATH = "k_sweep_trajectories.png"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "google/gemini-2.5-flash"
GEMINI_MODEL = "gemini-3.5-flash"

# Default k_true values to sweep, from very small to very big, log-spaced. Small
# and large k are physically hard to identify because the field is insensitive to
# k there (little diffusion, or fully diffused). The first and last two values
# (1e-5, 2.0, 5.0) sit OUTSIDE the [K_MIN, K_MAX] range the agent may guess in, so
# the agent cannot reach the true k and can only rail against the nearest bound.
SWEEP_K_VALUES = [
    1e-5,
    1e-4,
    2e-4,
    5e-4,
    1e-3,
    5e-3,
    1e-2,
    5e-2,
    0.12,
    0.3,
    0.5,
    0.8,
    1.0,
    2.0,
    5.0,
]

BASE_CONFIG = {"L": 1.0, "Nx": 101, "T": 0.02, "dt": 1e-4}


class AgentDecision(BaseModel):
    action: Literal["evaluate", "stop"]
    k_guess: float | None = None
    rationale: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    stop_reason: str | None = None


def response_format_schema():
    return {
        "type": "text",
        "mime_type": "application/json",
        "schema": AgentDecision.model_json_schema(),
    }


def openai_response_format_schema():
    return {"type": "json_object"}


def extract_text(response):
    if hasattr(response, "text") and response.text:
        return response.text
    if hasattr(response, "choices") and response.choices:
        message = response.choices[0].message
        if hasattr(message, "content") and message.content:
            return message.content
    if isinstance(response, str):
        return response
    return str(response)


def parse_decision(raw_text):
    try:
        payload = json.loads(raw_text)
        return AgentDecision.model_validate(payload), None
    except (json.JSONDecodeError, ValidationError) as exc:
        return None, str(exc)


def valid_k_guess(decision):
    if decision.action != "evaluate":
        return False
    if decision.k_guess is None:
        return False
    return math.isfinite(decision.k_guess) and K_MIN <= decision.k_guess <= K_MAX


def build_prompt(history, best_so_far):
    compact_history = [
        {
            "iteration": item["iteration"],
            "k_guess": item["k_guess"],
            "u_error": item["u_error"],
            "width_error": item["width_error"],
        }
        for item in history[-8:]
    ]
    return (
        "You are estimating the heat diffusivity k for a 1D heat equation. "
        "Return only a JSON object with these fields: "
        "action ('evaluate' or 'stop'), k_guess (number or null), rationale "
        "(string), confidence (number from 0 to 1), and stop_reason "
        "(string or null).\n\n"
        f"Valid k bounds: [{K_MIN}, {K_MAX}]\n"
        f"Stopping target: u_error <= {TARGET_U_ERROR}\n"
        "Rule: positive width_error means k is too high; negative width_error "
        "means k is too low.\n"
        f"Best result so far: {best_so_far}\n"
        f"Recent history: {compact_history}\n\n"
        "Choose action='evaluate' with a finite in-bounds k_guess unless the "
        "best result is already good enough, in which case choose action='stop'."
    )


class AgentClient:
    def __init__(self, provider, client, model):
        self.provider = provider
        self.client = client
        self.model = model


def create_agent_client(mock=False):
    if mock:
        return AgentClient(provider="mock", client=None, model="log-bisection")

    if os.environ.get("OPENROUTER_API_KEY"):
        from openai import OpenAI

        return AgentClient(
            provider="openrouter",
            client=OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=os.environ["OPENROUTER_API_KEY"],
                default_headers={
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "Agentic Physics Heat Diffusivity Demo",
                },
            ),
            model=os.environ.get("OPENROUTER_MODEL", OPENROUTER_MODEL),
        )

    if os.environ.get("GEMINI_API_KEY"):
        from google import genai

        return AgentClient(
            provider="gemini",
            client=genai.Client(),
            model=os.environ.get("GEMINI_MODEL", GEMINI_MODEL),
        )

    raise RuntimeError(
        "Set OPENROUTER_API_KEY for OpenRouter or GEMINI_API_KEY for Google "
        "Gemini, or pass --mock for the offline solver-in-the-loop agent."
    )


def env_status():
    openrouter_set = bool(os.environ.get("OPENROUTER_API_KEY"))
    gemini_set = bool(os.environ.get("GEMINI_API_KEY"))
    return {
        "OPENROUTER_API_KEY": "set" if openrouter_set else "unset",
        "GEMINI_API_KEY": "set" if gemini_set else "unset",
    }


def call_agent(agent_client, prompt):
    if agent_client.provider == "openrouter":
        return agent_client.client.chat.completions.create(
            model=agent_client.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return only a valid JSON object. Do not include markdown, "
                        "code fences, or explanatory text outside the JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            response_format=openai_response_format_schema(),
        )

    return agent_client.client.interactions.create(
        model=agent_client.model,
        input=prompt,
        response_format=response_format_schema(),
    )


class MockProposer:
    """Deterministic offline agent: log-domain bisection on width-error sign.

    Stands in for the LLM so the loop and sweep can run without API access. It
    also cleanly exposes the *physics* limit: at extreme k the field barely
    responds to k, so even a perfect bracketing search leaves a large k error
    once u_error dips below the stopping target.
    """

    def __init__(self):
        self.lo = K_MIN
        self.hi = K_MAX

    def propose(self, history, best_so_far):
        if history:
            last = history[-1]
            k = last["k_guess"]
            width_error = last["width_error"]
            if width_error > 0:  # guess diffused too much -> k too high
                self.hi = min(self.hi, k)
            elif width_error < 0:  # k too low
                self.lo = max(self.lo, k)
        guess = math.sqrt(self.lo * self.hi)
        decision = AgentDecision(
            action="evaluate",
            k_guess=guess,
            rationale="mock log-bisection on width_error sign",
            confidence=0.5,
        )
        return decision, None


class LLMProposer:
    def __init__(self, agent_client):
        self.agent_client = agent_client

    def propose(self, history, best_so_far):
        prompt = build_prompt(history, best_so_far)
        response = call_agent(self.agent_client, prompt)
        return parse_decision(extract_text(response))


def make_proposer(agent_client):
    if agent_client.provider == "mock":
        return MockProposer()
    return LLMProposer(agent_client)


def evaluate_guess(k_guess, u0, target_u, config):
    u_error, width_error = heat_errors(
        k_guess,
        u0,
        config["T"],
        config["Nx"],
        config["dt"],
        target_u,
        L=config["L"],
    )
    return float(u_error), float(width_error)


def run_single(
    agent_client,
    k_true,
    config=None,
    max_iterations=MAX_ITERATIONS,
    verbose=True,
):
    """Run the agent loop against one true diffusivity and return a result dict."""
    cfg = dict(BASE_CONFIG)
    if config:
        cfg.update(config)
    cfg["k_true"] = k_true

    x = np.linspace(0.0, cfg["L"], cfg["Nx"])
    u0 = gaussian_ic(x)
    _, target_u = solve_heat(
        k_true, u0, cfg["T"], cfg["Nx"], cfg["dt"], l=cfg["L"]
    )

    proposer = make_proposer(agent_client)
    history = []
    best_so_far = None
    iters_to_converge = None

    if verbose:
        print(f"provider: {agent_client.provider}")
        print(f"model:    {agent_client.model}")
        print(f"k_true:   {k_true:.6g}\n")
        print("iter  k_guess       u_error      width_error   confidence  rationale")
        print("----  ------------  -----------  ------------  ----------  ---------")

    for iteration in range(1, max_iterations + 1):
        try:
            decision, parse_error = proposer.propose(history, best_so_far)
        except Exception as exc:
            # A transient API error should not discard the guesses already made;
            # if none were made yet, let the caller decide how to record it.
            if best_so_far is not None:
                if verbose:
                    print(f"{iteration:>4}  api error, stopping run: {exc}")
                break
            raise
        if decision is None:
            if verbose:
                print(f"{iteration:>4}  invalid response: {parse_error}")
            continue

        if decision.action == "stop":
            if verbose:
                print(f"{iteration:>4}  stop: {decision.stop_reason or decision.rationale}")
            break

        if not valid_k_guess(decision):
            if verbose:
                print(f"{iteration:>4}  invalid k_guess: {decision.k_guess!r}")
            continue

        u_error, width_error = evaluate_guess(decision.k_guess, u0, target_u, cfg)
        observation = {
            "iteration": iteration,
            "k_guess": decision.k_guess,
            "u_error": u_error,
            "width_error": width_error,
            "confidence": decision.confidence,
            "rationale": decision.rationale,
        }
        history.append(observation)
        if best_so_far is None or u_error < best_so_far["u_error"]:
            best_so_far = observation

        if verbose:
            print(
                f"{iteration:>4}  {decision.k_guess:>12.8f}  {u_error:>11.4e}  "
                f"{width_error:>12.4e}  {decision.confidence:>10.2f}  {decision.rationale}"
            )

        if u_error <= TARGET_U_ERROR:
            if iters_to_converge is None:
                iters_to_converge = len(history)
            break

    if best_so_far is None:
        raise RuntimeError(
            f"No valid in-bounds agent guesses were evaluated for k_true={k_true}."
        )

    k_abs_error = abs(best_so_far["k_guess"] - k_true)
    result = {
        "k_true": k_true,
        "k_est": best_so_far["k_guess"],
        "k_abs_error": k_abs_error,
        "k_rel_error": k_abs_error / k_true,
        "best_u_error": best_so_far["u_error"],
        "n_iterations": len(history),
        "converged": best_so_far["u_error"] <= TARGET_U_ERROR,
        "iters_to_converge": iters_to_converge,
        "k_true_in_bounds": K_MIN <= k_true <= K_MAX,
        "history": history,
        "config": cfg,
        "x": x,
        "target_u": target_u,
        "u0": u0,
    }
    return result


def run_single_report(agent_client, k_true, max_iterations=MAX_ITERATIONS):
    """Run one k_true and emit the detailed single-run table, CSV and figure."""
    result = run_single(agent_client, k_true, max_iterations=max_iterations)
    cfg = result["config"]

    _, best_u = solve_heat(
        result["k_est"], result["u0"], cfg["T"], cfg["Nx"], cfg["dt"], l=cfg["L"]
    )

    print_history_table(result["history"], k_true)
    csv_path = write_history_csv(result["history"], k_true, HISTORY_CSV_PATH)

    print("\nFinal report")
    print(f"estimated k: {result['k_est']:.8f}")
    print(f"true k:      {k_true:.8f}")
    print(f"abs error:   {result['k_abs_error']:.4e}")
    print(f"rel error:   {result['k_rel_error']:.4e}")
    print(f"best loss:   {result['best_u_error']:.4e}")
    print(f"iterations:  {result['n_iterations']} valid evaluations")
    print(f"converged:   {result['converged']}")
    print(f"history csv: {csv_path}")

    plot_results(
        result["x"], result["target_u"], best_u, result["history"], k_true, PLOT_PATH
    )
    return result


def run_sweep(agent_client, k_values, max_iterations=MAX_ITERATIONS, show=True):
    """Run the agent against many true k values and produce aggregate artifacts."""
    print(f"provider: {agent_client.provider}")
    print(f"model:    {agent_client.model}")
    print(f"sweep over {len(k_values)} k_true values\n")

    results = []
    failures = []
    for k_true in k_values:
        try:
            result = run_single(
                agent_client, k_true, max_iterations=max_iterations, verbose=False
            )
        except Exception as exc:  # keep partial sweeps: one bad k must not wipe all
            failures.append((k_true, exc))
            print(f"  k_true={k_true:>10.4g}  FAILED: {type(exc).__name__}: {exc}")
            continue
        results.append(result)
        conv = "yes" if result["converged"] else "NO "
        print(
            f"  k_true={k_true:>10.4g}  k_est={result['k_est']:>10.4g}  "
            f"rel_err={result['k_rel_error']:>9.2e}  "
            f"u_err={result['best_u_error']:>9.2e}  "
            f"iters={result['n_iterations']:>2}  converged={conv}"
        )

    if failures:
        print(f"\n{len(failures)} of {len(k_values)} runs failed and were skipped:")
        for k_true, exc in failures:
            print(f"  k_true={k_true:.4g}: {type(exc).__name__}: {exc}")
    if not results:
        raise RuntimeError("Every sweep run failed; no artifacts to write.")

    print_sweep_table(results)
    csv_path = write_sweep_csv(results, SWEEP_CSV_PATH)
    print(f"\nsweep csv: {csv_path}")

    plot_sweep_performance(results, SWEEP_PLOT_PATH, show=show)
    plot_sweep_trajectories(results, SWEEP_TRAJECTORY_PATH, show=show)

    hardest = max(results, key=lambda r: r["k_rel_error"])
    print(
        f"\nHardest k_true to identify: {hardest['k_true']:.4g} "
        f"(relative k error {hardest['k_rel_error']:.2e})"
    )
    return results


def print_history_table(history, k_true):
    """Print a per-iteration table of k guesses and errors."""
    print("\nPer-iteration guesses")
    header = (
        f"{'iter':>4}  {'k_guess':>12}  {'k_abs_error':>12}  "
        f"{'u_error':>11}  {'width_error':>12}  {'confidence':>10}"
    )
    print(header)
    print("-" * len(header))
    for item in history:
        print(
            f"{item['iteration']:>4}  {item['k_guess']:>12.8f}  "
            f"{abs(item['k_guess'] - k_true):>12.4e}  {item['u_error']:>11.4e}  "
            f"{item['width_error']:>12.4e}  {item['confidence']:>10.2f}"
        )


def write_history_csv(history, k_true, path):
    """Write the per-iteration history to a CSV file and return its path."""
    fields = [
        "iteration",
        "k_guess",
        "k_abs_error",
        "u_error",
        "width_error",
        "confidence",
        "rationale",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in history:
            writer.writerow(
                {
                    "iteration": item["iteration"],
                    "k_guess": item["k_guess"],
                    "k_abs_error": abs(item["k_guess"] - k_true),
                    "u_error": item["u_error"],
                    "width_error": item["width_error"],
                    "confidence": item["confidence"],
                    "rationale": item["rationale"],
                }
            )
    return path


def print_sweep_table(results):
    """Print the across-k summary table of agent performance."""
    print("\nSweep summary (agent performance vs true k)")
    header = (
        f"{'k_true':>12}  {'k_est':>12}  {'k_abs_error':>12}  "
        f"{'k_rel_error':>12}  {'best_u_error':>13}  {'iters':>6}  "
        f"{'to_conv':>7}  {'converged':>9}  {'in_bnds':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        to_conv = r["iters_to_converge"] if r["iters_to_converge"] is not None else "-"
        print(
            f"{r['k_true']:>12.6g}  {r['k_est']:>12.6g}  {r['k_abs_error']:>12.4e}  "
            f"{r['k_rel_error']:>12.4e}  {r['best_u_error']:>13.4e}  "
            f"{r['n_iterations']:>6}  {str(to_conv):>7}  {str(r['converged']):>9}  "
            f"{str(r['k_true_in_bounds']):>7}"
        )


def write_sweep_csv(results, path):
    """Write the across-k summary to CSV and return its path."""
    fields = [
        "k_true",
        "k_est",
        "k_abs_error",
        "k_rel_error",
        "best_u_error",
        "n_iterations",
        "iters_to_converge",
        "converged",
        "k_true_in_bounds",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({field: r[field] for field in fields})
    return path


def plot_results(x, target_u, best_u, history, k_true, save_path=None):
    iterations = [item["iteration"] for item in history]
    k_guesses = [item["k_guess"] for item in history]
    u_errors = [item["u_error"] for item in history]
    k_errors = [abs(item["k_guess"] - k_true) for item in history]
    width_errors = [item["width_error"] for item in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Target vs best estimated profile.
    axes[0, 0].plot(x, target_u, label="target")
    axes[0, 0].plot(x, best_u, "--", label="best estimate")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("u")
    axes[0, 0].set_title("Field profile")
    axes[0, 0].legend()

    # k guess per iteration with true-k reference line.
    axes[0, 1].plot(iterations, k_guesses, marker="o", label="k_guess")
    axes[0, 1].axhline(k_true, color="k", linestyle=":", label="k_true")
    axes[0, 1].set_xlabel("iteration")
    axes[0, 1].set_ylabel("k")
    axes[0, 1].set_title("k guess per iteration")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Field and k errors per iteration (log scale).
    axes[1, 0].semilogy(iterations, u_errors, marker="o", label="relative field error")
    axes[1, 0].semilogy(iterations, k_errors, marker="s", label="k abs error")
    axes[1, 0].set_xlabel("iteration")
    axes[1, 0].set_ylabel("error")
    axes[1, 0].set_title("Error per iteration")
    axes[1, 0].legend()
    axes[1, 0].grid(True, which="both", alpha=0.3)

    # Signed width error per iteration.
    axes[1, 1].plot(iterations, width_errors, marker="o", color="C3")
    axes[1, 1].axhline(0.0, color="k", linestyle=":")
    axes[1, 1].set_xlabel("iteration")
    axes[1, 1].set_ylabel("width_error (guess - target)")
    axes[1, 1].set_title("Signed width error per iteration")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"plot saved: {save_path}")
    if save_path is None:
        plt.show()
    else:
        plt.close(fig)


def plot_sweep_performance(results, save_path=None, show=True):
    """Per-run convergence trajectories (3 panels) plus the effort bar chart.

    The three trajectory panels each put the iteration number on the x axis and,
    on the y axis, the guessed k, the relative k error, and the absolute k error.
    Every run is one line, colored by its true k (dark = small, bright = large).
    """
    from matplotlib.colors import LogNorm

    k_true = np.array([r["k_true"] for r in results])
    iters_used = np.array([r["n_iterations"] for r in results])
    converged = np.array([r["converged"] for r in results])

    cmap = plt.get_cmap("viridis")
    norm = LogNorm(vmin=k_true.min(), vmax=k_true.max())

    def color_for(k):
        return cmap(norm(k))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Guessed k per iteration, with the allowed guess band drawn in.
    ax = axes[0, 0]
    for r in results:
        history = r["history"]
        if not history:
            continue
        xs = [item["iteration"] for item in history]
        ys = [item["k_guess"] for item in history]
        ax.plot(xs, ys, marker="o", ms=3, color=color_for(r["k_true"]))
    ax.axhline(K_MIN, color="k", linestyle="--", lw=1, alpha=0.6)
    ax.axhline(K_MAX, color="k", linestyle="--", lw=1, alpha=0.6, label="guess bounds")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("guessed k")
    ax.set_title("Guessed k per iteration")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # Relative k error per iteration.
    ax = axes[0, 1]
    for r in results:
        history = r["history"]
        if not history:
            continue
        xs = [item["iteration"] for item in history]
        ys = [abs(item["k_guess"] - r["k_true"]) / r["k_true"] for item in history]
        ax.semilogy(xs, ys, marker="o", ms=3, color=color_for(r["k_true"]))
    ax.set_xlabel("iteration")
    ax.set_ylabel("relative k error")
    ax.set_title("Relative k error per iteration")
    ax.grid(True, which="both", alpha=0.3)

    # Absolute k error per iteration.
    ax = axes[1, 0]
    for r in results:
        history = r["history"]
        if not history:
            continue
        xs = [item["iteration"] for item in history]
        ys = [abs(item["k_guess"] - r["k_true"]) for item in history]
        ax.semilogy(xs, ys, marker="o", ms=3, color=color_for(r["k_true"]))
    ax.set_xlabel("iteration")
    ax.set_ylabel("absolute k error")
    ax.set_title("Absolute k error per iteration")
    ax.grid(True, which="both", alpha=0.3)

    # Iterations used vs true k, colored by convergence (kept from before).
    colors = ["C2" if c else "C3" for c in converged]
    axes[1, 1].bar(range(len(k_true)), iters_used, color=colors)
    axes[1, 1].axhline(MAX_ITERATIONS, color="k", linestyle=":", label="iteration cap")
    axes[1, 1].set_xticks(range(len(k_true)))
    axes[1, 1].set_xticklabels([f"{k:.3g}" for k in k_true], rotation=45, ha="right")
    axes[1, 1].set_xlabel("k_true")
    axes[1, 1].set_ylabel("iterations used")
    axes[1, 1].set_title("Effort vs true k (green=converged)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, axis="y", alpha=0.3)

    # Shared colorbar for the three trajectory panels.
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(
        sm,
        ax=[axes[0, 0], axes[0, 1], axes[1, 0]],
        label="k_true",
        fraction=0.05,
        pad=0.02,
    )

    fig.suptitle("Agent convergence across the k sweep", fontsize=14)
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"plot saved: {save_path}")
    if show and save_path is None:
        plt.show()
    else:
        plt.close(fig)


def plot_sweep_trajectories(results, save_path=None, show=True):
    """Overlay each run's relative k-error trajectory, colored by true k."""
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("viridis")
    log_k = np.log10([r["k_true"] for r in results])
    lo, hi = log_k.min(), log_k.max()

    for r in results:
        history = r["history"]
        if not history:
            continue
        iters = [item["iteration"] for item in history]
        rel = [abs(item["k_guess"] - r["k_true"]) / r["k_true"] for item in history]
        frac = 0.0 if hi == lo else (math.log10(r["k_true"]) - lo) / (hi - lo)
        ax.semilogy(
            iters, rel, marker="o", color=cmap(frac), label=f"k={r['k_true']:.3g}"
        )

    ax.set_xlabel("iteration")
    ax.set_ylabel("relative k error")
    ax.set_title("Convergence trajectory per true k")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"plot saved: {save_path}")
    if show and save_path is None:
        plt.show()
    else:
        plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Agentic estimation of 1D heat-equation diffusivity k."
    )
    parser.add_argument(
        "--mode",
        choices=["sweep", "single"],
        default="sweep",
        help="Sweep many true k values (default) or run a single detailed estimate.",
    )
    parser.add_argument(
        "--k-true",
        type=float,
        default=0.12,
        help="True k for --mode single (default 0.12).",
    )
    parser.add_argument(
        "--values",
        type=float,
        nargs="+",
        default=None,
        help="Override the list of true k values to sweep.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=MAX_ITERATIONS,
        help=f"Max agent iterations per run (default {MAX_ITERATIONS}).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use the offline log-bisection agent instead of an LLM.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    status = env_status()
    print(
        "environment: "
        f"OPENROUTER_API_KEY={status['OPENROUTER_API_KEY']}, "
        f"GEMINI_API_KEY={status['GEMINI_API_KEY']}"
    )
    agent_client = create_agent_client(mock=args.mock)

    if args.mode == "single":
        run_single_report(agent_client, args.k_true, max_iterations=args.max_iters)
    else:
        k_values = args.values if args.values else SWEEP_K_VALUES
        run_sweep(agent_client, k_values, max_iterations=args.max_iters)


if __name__ == "__main__":
    main()
