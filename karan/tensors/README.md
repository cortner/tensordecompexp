# `tensors/` — experiments and notebooks

Code for **Chebyshev-basis Tucker/CP** approximation of 2D/3D test functions and ACE-style energy models.

**Start at the repo handoff doc:** [`../HANDOFF.md`](../HANDOFF.md) (setup, reading order, what was excluded from git).

## Contents

| Path | Role |
|------|------|
| [`workflow/`](workflow/README.md) | Runnable experiment scripts (ALS, optimizers, baselines, plotting) |
| [`notebooks/`](notebooks/) | Jupyter exploration history (optional for reruns) |

## Typical workflow

1. Pick a script from [`workflow/README.md`](workflow/README.md) (e.g. `workflow/als/als_tucker.py`).
2. Run from that directory: `python als_tucker.py` → creates `*_results/<timestamp>/`.
3. After rank sweeps, point [`workflow/plotter/2d/plot_tucker.py`](workflow/plotter/2d/plot_tucker.py) at the generated JSON files.

Each `.py` file under `workflow/` includes a **one-line summary** at the top describing what it does.
