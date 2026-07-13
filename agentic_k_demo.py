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
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "google/gemini-2.5-flash"
GEMINI_MODEL = "gemini-3.5-flash"


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


def create_agent_client():
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
        "Set OPENROUTER_API_KEY for OpenRouter or GEMINI_API_KEY for Google Gemini."
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


def run_agent_loop(agent_client):
    config = {"k_true": 0.12, "L": 1.0, "Nx": 101, "T": 0.02, "dt": 1e-4}
    x = np.linspace(0.0, config["L"], config["Nx"])
    u0 = gaussian_ic(x)
    _, target_u = solve_heat(
        config["k_true"],
        u0,
        config["T"],
        config["Nx"],
        config["dt"],
        l=config["L"],
    )

    history = []
    best_so_far = None

    print(f"provider: {agent_client.provider}")
    print(f"model:    {agent_client.model}")
    print()
    print("iter  k_guess       u_error      width_error   confidence  rationale")
    print("----  ------------  -----------  ------------  ----------  ---------")

    for iteration in range(1, MAX_ITERATIONS + 1):
        prompt = build_prompt(history, best_so_far)
        response = call_agent(agent_client, prompt)
        decision, parse_error = parse_decision(extract_text(response))
        if decision is None:
            print(f"{iteration:>4}  invalid response: {parse_error}")
            continue

        if decision.action == "stop":
            print(f"{iteration:>4}  stop: {decision.stop_reason or decision.rationale}")
            break

        if not valid_k_guess(decision):
            print(f"{iteration:>4}  invalid k_guess: {decision.k_guess!r}")
            continue

        u_error, width_error = evaluate_guess(decision.k_guess, u0, target_u, config)
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

        print(
            f"{iteration:>4}  {decision.k_guess:>12.8f}  {u_error:>11.4e}  "
            f"{width_error:>12.4e}  {decision.confidence:>10.2f}  {decision.rationale}"
        )

        if u_error <= TARGET_U_ERROR:
            break

    if best_so_far is None:
        raise RuntimeError("No valid in-bounds agent guesses were evaluated.")

    _, best_u = solve_heat(
        best_so_far["k_guess"],
        u0,
        config["T"],
        config["Nx"],
        config["dt"],
        l=config["L"],
    )

    k_true = config["k_true"]

    print_history_table(history, k_true)
    csv_path = write_history_csv(history, k_true, HISTORY_CSV_PATH)

    print("\nFinal report")
    print(f"estimated k: {best_so_far['k_guess']:.8f}")
    print(f"true k:      {k_true:.8f}")
    print(f"abs error:   {abs(best_so_far['k_guess'] - k_true):.4e}")
    print(f"best loss:   {best_so_far['u_error']:.4e}")
    print(f"iterations:  {len(history)} valid evaluations")
    print(f"history csv: {csv_path}")

    plot_results(x, target_u, best_u, history, k_true, PLOT_PATH)
    return best_so_far


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
    plt.show()


def main():
    status = env_status()
    print(
        "environment: "
        f"OPENROUTER_API_KEY={status['OPENROUTER_API_KEY']}, "
        f"GEMINI_API_KEY={status['GEMINI_API_KEY']}"
    )
    agent_client = create_agent_client()
    run_agent_loop(agent_client)


if __name__ == "__main__":
    main()
