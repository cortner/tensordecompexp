# `workflow/` — experiment scripts

Scripts are grouped by **method**. All share the same test functions and Chebyshev sampling pattern unless noted in the file header.

**Full handoff context:** [`../../HANDOFF.md`](../../HANDOFF.md)

## Folders

| Folder | Purpose | Start here |
|--------|---------|------------|
| [`als/`](als/) | Main ALS / Tucker line; production 2D/3D experiments | `als_tucker.py` |
| [`als_discuss/`](als_discuss/) | Meeting prototypes: Zulip ALS, ACE-Tucker, operator form | `als_zulip.py`, `als_zulip_ace_3d.py` |
| [`baseline/`](baseline/) | Dense Chebyshev interpolation (no low rank) | `baseline_2d.py` |
| [`tucker/`](tucker/) | TensorLy Tucker rank sweeps on coefficient tensors | `tucker.py` |
| [`cp/`](cp/) | CP / SVD baselines | `cp3d.py`, `svd.py` |
| [`tt/`](tt/) | Tensor-train on coefficient tensors | `tt2d.py` |
| [`adam/`](adam/) | Adam on CP/Tucker factors (`autograd`) | `adam2d_tucker.py` |
| [`bfgs/`](bfgs/) | BFGS on CP/Tucker factors | `bfgs2d_tucker.py` |
| [`lbfgs/`](lbfgs/) | L-BFGS-B on CP factors (`scipy`) | `lbfgs2d.py` |
| [`plotter/2d/`](plotter/2d/) | Overlay metrics from JSON rank sweeps | `plot_tucker.py` |

## File index

Open any `.py` file — the **first line** is a module docstring with a short description. For navigation by task, see the **“Which script should I use?”** table in [`HANDOFF.md`](../../HANDOFF.md).

## Outputs

Running a script typically creates a directory such as:

- `als_tucker_results/<timestamp>/` — metrics JSON, plots, config snapshot  
- Or a single `*_rank_sweep_results.json` next to the script when driven from `if __name__ == "__main__"`

These paths are **gitignored** in the handoff; regenerate on your machine after clone.
