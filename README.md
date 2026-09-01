# Agentic Physics

Plain Python demo for estimating an unknown 1D heat-equation diffusivity `k`
from synthetic target data. A language model proposes candidate values, and the
local numerical solver evaluates each proposal.

## What It Does

- Solves `u_t = k u_xx` on `[0, 1]` with zero Dirichlet boundaries.
- Uses an agent loop to propose `k` guesses within `[1e-4, 1.0]`.
- Evaluates each guess with relative field error and signed width error.
- **Sweeps many true `k` values** (very small `1e-4` to very big `1.0`) and
  measures where the agent struggles to recover the true `k`.
- Supports OpenRouter first, with Google Gemini as a fallback, plus an offline
  `--mock` bracketing agent so the pipeline runs without an API key.

## Why Some k Are Hard

Over the fixed observation time `T`, the field is only weakly sensitive to `k`
at the extremes: very small `k` barely diffuses, and very large `k` has already
diffused almost completely. In those regimes many different `k` values produce
nearly the same field, so the field misfit (`u_error`) can drop below the
stopping target while the recovered `k` is still far off. The sweep makes this
identifiability gap visible: relative `k` error is large at small `k` even for a
perfect bracketing search, and larger still for the LLM.

## Setup

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Set one API key in your shell. OpenRouter is preferred:

```bash
export OPENROUTER_API_KEY="your_openrouter_key"
```

Or use Gemini directly:

```bash
export GEMINI_API_KEY="your_gemini_key"
```

Optional model overrides:

```bash
export OPENROUTER_MODEL="google/gemini-2.5-flash"
export GEMINI_MODEL="gemini-3.5-flash"
```

## Run

Sweep across many true `k` values (default mode):

```bash
.venv/bin/python agentic_k_demo.py                 # LLM agent
.venv/bin/python agentic_k_demo.py --mock          # offline, no API key
.venv/bin/python agentic_k_demo.py --values 1e-4 1e-3 0.12 0.5 1.0
```

The default sweep also includes true `k` values **outside** the `[1e-4, 1.0]`
range the agent may guess in (`1e-5`, `2.0`, `5.0`), so the agent cannot reach
the true value and can only rail against the nearest bound.

The sweep prints a per-`k` summary table and writes artifacts **tagged with the
agent that produced them**, so LLM output is never confused with the offline
baseline. LLM runs write `*_openrouter.*` (or `*_gemini.*`); the `--mock` agent
writes `*_bisection_nonai.*` because it is a deterministic bisection search, not
an AI model. For the mock agent the files are:

- `k_sweep_results_bisection_nonai.csv` - one row per true `k` (estimate,
  absolute/relative `k` error, best field error, iterations, convergence flag,
  in-bounds flag).
- `k_sweep_performance_bisection_nonai.png` - 4 panels. Three plot per-run
  convergence with the iteration on the x axis: guessed `k`, relative `k` error,
  and absolute `k` error, one line per run colored by true `k` (dark = small,
  bright = large). The guessed-`k` panel draws the allowed guess band so
  out-of-range runs are visible as lines pinned to a bound. The fourth panel
  keeps iterations used per `k` (green = converged).
- `k_sweep_trajectories_bisection_nonai.png` - each run's relative `k`-error
  trajectory in a single larger panel.

Every figure's title also names the agent (e.g. `agent: bisection search (NOT an
AI model)` or `agent: openrouter:google/gemini-2.5-flash`).

### Real LLM figures

`reconstruct_openrouter_plot.py` re-renders the summary performance figure from
the recorded results of the real OpenRouter (`google/gemini-2.5-flash`) sweeps,
writing `k_sweep_performance_openrouter.png` (12 in-range values) and
`k_sweep_performance_openrouter_oor.png` (13/15 values; `2.0` and `5.0` were
aborted when OpenRouter ran out of credits). Only summary metrics were captured
for those runs, so the per-iteration trajectory figure cannot be rebuilt for the
LLM; re-run `agentic_k_demo.py` with a funded key to regenerate it live.

Single detailed run against one `k`:

```bash
.venv/bin/python agentic_k_demo.py --mode single --k-true 0.12
```

This prints a per-iteration table (`k` guess, `k` absolute error, field error,
signed width error, confidence) and a final estimate, writes the full history to
`k_history.csv`, and saves a 4-panel figure to `k_convergence.png` (target vs.
best profile, `k` guesses per iteration against `k_true`, field/`k` errors per
iteration on a log scale, and the signed width error per iteration).

Other flags: `--max-iters N` caps agent iterations per run.

## Test

```bash
.venv/bin/python test_agentic_k_demo.py
.venv/bin/python -m py_compile agentic_k_demo.py heateq_num_solver.py test_agentic_k_demo.py
```

The tests cover solver shape, boundary conditions, target recovery, signed width
error direction, invalid agent decisions, and OpenAI-compatible response parsing.

## Project Layout

- `agentic_k_demo.py` - agent loop, provider selection, reporting, and plotting.
- `heateq_num_solver.py` - importable heat-equation solver and error metrics.
- `heateq-num-solver.py` - compatibility wrapper for the original hyphenated name.
- `test_agentic_k_demo.py` - lightweight assertion-based test script.
- `requirements.txt` - runtime dependencies.

## Engineering Notes

- Keep API keys in environment variables, not source files or transcripts.
- Do not commit `.venv`, caches, or local command output.
- Python validates model responses with Pydantic before evaluating guesses.
- The solver remains deterministic; the model only controls candidate selection.
- OpenRouter requests cap `max_tokens` because the expected response is a small
  JSON object.
