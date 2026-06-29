# Agentic Physics

Plain Python demo for estimating an unknown 1D heat-equation diffusivity `k`
from synthetic target data. A language model proposes candidate values, and the
local numerical solver evaluates each proposal.

## What It Does

- Solves `u_t = k u_xx` on `[0, 1]` with zero Dirichlet boundaries.
- Generates target data with `k_true = 0.12`.
- Uses an agent loop to propose `k` guesses within `[1e-4, 1.0]`.
- Evaluates each guess with relative field error and signed width error.
- Supports OpenRouter first, with Google Gemini as a fallback.

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

```bash
.venv/bin/python agentic_k_demo.py
```

The script prints an iteration table and final estimate, then opens a Matplotlib
plot comparing the target profile with the best estimated profile.

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
