# Research handoff — tensor decomposition experiments (Karan)

**Audience:** Professor / successor taking over this RA work.  
**Goal:** Everything needed to understand, rerun, and extend the Chebyshev + Tucker/CP experiments without relying on Karan’s machine or memory.

**Original server project:** `/home/kanand03/Research/Tensor_research`  
**This repository folder:** `tensordecompexp/karan/` (mirror of the code + notes committed here)

---

## Executive summary

We studied **low-rank approximations of smooth 2D/3D test functions** (and later **ACE-style configuration energies**) by:

1. Sampling \(f\) on Chebyshev / uniform / random grids in \([-1,1]^d\).
2. Representing the sample tensor with **Chebyshev Vandermonde** factors \(T_x, T_y, (\,T_z\,)\).
3. Fitting **Tucker** \(F \approx (T_x A_x,\, T_y A_y,\, [T_z A_z],\, G)\) or **CP** via:
   - **ALS** (main line of work),
   - **Christoph-style ALS** (meeting variants in `als_discuss/` — more stable than naive normal equations),
   - **Gradient methods** (Adam, BFGS, L-BFGS-B),
   - **TensorLy baselines** (Tucker / PARAFAC / TT on coefficient tensors).

**Primary 2D entry point:** `tensors/workflow/als/als_tucker.py`  
**Stable ALS variant (recommended after reading math):** `tensors/workflow/als_discuss/als_zulip.py`  
**Dense reference (no low rank):** `tensors/workflow/baseline/baseline_2d.py`  
**ACE application (3D):** `tensors/workflow/als_discuss/als_zulip_ace_3d.py` + `latex_notes/ace_3d_configuration_energy_setup.pdf`

Every Python script under `tensors/workflow/` has a **one-line module summary** at the top of the file (added for this handoff).

---

## Quick start

### Environment

```bash
cd karan
python3 -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

| Package | Used for |
|---------|----------|
| `numpy`, `matplotlib` | All scripts |
| `tensorly` | Tucker / CP / TT baselines, some ALS variants |
| `autograd` | `workflow/adam/*` |
| `scipy` | `workflow/lbfgs/*`, `workflow/bfgs/*` |

On the **university server**, Python 3.10 was available via mamba: run `setup_mambapython3.sh` once (creates `~/bin/mambapython3` symlink). Adjust `TARGET` in that script if the mamba path changes.

### Run one experiment

```bash
cd tensors/workflow/als
python als_tucker.py
# Creates: als_tucker_results/<timestamp>/ with metrics JSON + plots
```

### Compare methods (after sweeps)

1. Run rank sweeps (scripts write JSON like `*_rank_sweep_results.json` next to the script or under `*_results/`).
2. Edit the `FILES` dict in `tensors/workflow/plotter/2d/plot_tucker.py` to point at those JSON paths.
3. `python plot_tucker.py`

---

## What is in this handoff vs what was left on disk

| Included in git | Why |
|-----------------|-----|
| All `workflow/**/*.py` | Reproducible experiments |
| `tensors/notebooks/*.ipynb` | Exploration timeline / teaching |
| `latex_notes/*.pdf` | Algorithm write-ups (see `latex_notes/README.md`) |
| `requirements.txt`, `setup_mambapython3.sh` | Environment |
| This `HANDOFF.md` | Navigation |

| Excluded (regenerate locally) | Why |
|--------------------------------|-----|
| `*_results*/`, `results/`, `experiments/` | ~2.5 GB timestamped outputs |
| `*.json`, `*.png`, `*.npz` from old runs | Metrics / plots / saved tensors |
| `.venv/`, duplicate `* copy.py` | Environment / scratch |

**To reproduce figures:** rerun the relevant script; outputs land in a new timestamped folder under the script’s `outdir` (default names like `als_tucker_results/`).

---

## Repository layout

```
karan/
├── HANDOFF.md              ← start here
├── requirements.txt
├── setup_mambapython3.sh   ← server only: mamba Python 3.10 symlink
├── latex_notes/                     ← PDF notes (index: latex_notes/README.md)
│   ├── tucker_als_chebyshev_3d_methods_comparison.pdf  ← main Tucker ALS reference
│   ├── ls_svd_functional_tucker_chebyshev_bases.pdf    ← Christoph LS–SVD method
│   └── ace_3d_configuration_energy_setup.pdf           ← ACE + Tucker (3D)
└── tensors/
    ├── README.md           ← short index into workflow + notebooks
    ├── notebooks/          ← chronological exploration (see table below)
    └── workflow/           ← production scripts by method
        ├── README.md       ← method folders + “which script when”
        ├── als/            ← main ALS / Tucker line
        ├── als_discuss/    ← meeting variants: Zulip ALS, ACE, operators
        ├── adam/, bfgs/, lbfgs/
        ├── baseline/, tucker/, cp/, tt/
        └── plotter/2d/
```

Related material elsewhere in the repo: `toymodel/notes/` (Quarto notes on Tucker/ALS — separate from this `karan/` tree).

---

## Shared problem setup

**2D test function** (most scripts):

\[
f(x,y) = \frac{1}{1 + \|C[x,y]^\top\|^2}, \quad C \in \mathbb{R}^{2\times 2}
\]

Typical sweep uses several \(C\) cases (stretch, rotation, identity scales) — see `if __name__ == "__main__"` blocks.

**3D test function:**

\[
f(x,y,z) = \frac{1}{1 + \|C[x,y,z]^\top\|^2}, \quad C \in \mathbb{R}^{3\times 3}
\]

**Pipeline (conceptual):**

```
sample f on grid  →  F (value tensor)  →  fit Tucker/CP factors in Chebyshev basis
                      ↓
              optional: dense Chebyshev coeffs C_ref for error metrics
```

**Metrics logged** (naming varies slightly by script): train/test RMSE on values, error on a dense Chebyshev coefficient reference, sometimes max error; saved as JSON + PNG.

---

## Which script should I use?

| Goal | Script | Notes |
|------|--------|-------|
| **Default 2D Tucker ALS** | `als/als_tucker.py` | Rank sweep; main implementation |
| **Equivalence check** (interp + Tucker vs ALS) | `als/als_tucker_prof.py` | Documents that the two views should match |
| **Stable 2D ALS** (avoid bad normal equations) | `als_discuss/als_zulip.py` | Christoph-style; Dec 2025 meeting |
| **3D Tucker ALS** | `als_discuss/als_zulip3d.py` or `als/als_3d_tensorly.py` | Zulip = iterative ALS; tensorly = HOOI + lift |
| **ACE energies + Tucker** | `als_discuss/als_zulip_ace_3d.py` | See ACE LaTeX note |
| **Dense interpolation baseline** | `baseline/baseline_2d.py` | No rank constraint |
| **TensorLy Tucker on coeffs** | `tucker/tucker.py` | Rank sweep on interpolated \(C\) |
| **Adam / BFGS Tucker** | `adam/adam2d_tucker.py`, `bfgs/bfgs2d_tucker.py` | Nonlinear optimization baselines |
| **CP instead of Tucker** | `als/als.py`, `adam/adam2d.py` | Rank-1 CP |
| **Plot sweeps** | `plotter/2d/plot_tucker.py` | Point `FILES` at your JSON |

### Lineage of similar-looking files (avoid confusion)

| Family | Variants | Difference |
|--------|----------|------------|
| **ALS CP 2D** | `als.py`, `als_test.py`, `als_test_eps.py` | `als_test_eps` uses SVD ridge subproblems; more stable |
| **Block ALS CP** | `als_whole.py`, `als_wholec.py`, `als_whole_early.py` | Updates full \(A,B\) per iter; `wholec` disables \(\lambda\); `whole_early` enables early stopping |
| **ALS Tucker init** | `als_tucker_init.py`, `adam2d_tucker_init.py`, `bfgs2d_tucker_init.py` | Optional TensorLy Tucker initialization |
| **3D cached reference** | `als_3d_cached.py` | Reuses precomputed dense Chebyshev coeff tensor |
| **als_discuss/als_tucker.py** | vs `als/als_tucker.py` | Discuss version: orthonormalize in Chebyshev metric; classical normal-equation baseline |

---

## Notebooks (`tensors/notebooks/`)

Rough chronological / pedagogical order:

| Notebook | Focus |
|----------|--------|
| `demo.ipynb`, `tensors.ipynb` | Imports, tensor basics |
| `2D_10.ipynb`, `2D_128.ipynb`, `td.ipynb` | CP on Chebyshev coefficients; error vs rank |
| `als.ipynb` | Early ALS + Chebyshev |
| `decomp.ipynb` | TensorLy PARAFAC |
| `directd.ipynb` | Hand-rolled 2D CP-ALS |
| `sgd.ipynb`, `subset.ipynb` | SGD / Adam on grids (exploratory) |
| `tt.ipynb` | Tensor train |
| `test.ipynb` | TensorLy smoke test |
| `3d.ipynb`, `3d/baseline.ipynb` | 3D experiments / PARAFAC baseline |

Notebooks are **not required** to run production sweeps; they document how ideas were developed.

---

## PDF notes (`latex_notes/`)

Full index: [`latex_notes/README.md`](latex_notes/README.md).

| PDF | Read when |
|-----|-----------|
| `tucker_als_chebyshev_3d_methods_comparison.pdf` | Tucker ALS in Chebyshev bases; method comparison (main reference) |
| `ls_svd_functional_tucker_chebyshev_bases.pdf` | Christoph LS–SVD functional Tucker (Zulip / stable variant) |
| `ace_3d_configuration_energy_setup.pdf` | ACE pooled features + configuration energies (3D) |
| `ace_tucker_2d_energy_model_and_verification.pdf` | 2D ACE–Tucker model and coefficient recovery |

---

## Suggested reading order for a new researcher

1. `latex_notes/tucker_als_chebyshev_3d_methods_comparison.pdf` — notation and ALS structure  
2. `workflow/baseline/baseline_2d.py` — error floor without low rank  
3. `workflow/als/als_tucker.py` — main code path + rank sweep  
4. `workflow/als/als_tucker_prof.py` — why ALS-on-\(F\) relates to interp-then-Tucker  
5. `workflow/als_discuss/als_zulip.py` — stable variant used in discussions  
6. `workflow/als_discuss/als_zulip_ace_3d.py` + `latex_notes/ace_3d_configuration_energy_setup.pdf` — ACE extension  
7. `workflow/plotter/2d/plot_tucker.py` — compare methods after rerunning sweeps  

---

## Copying from the server

```bash
# From laptop
rsync -av kanand03@<server>:/home/kanand03/Research/tensordecompexp/karan/ \
  ./karan/

# Or on server
tar -czvf karan_handoff.tar.gz -C /home/kanand03/Research/tensordecompexp karan
```

---

## Handoff checklist (for professor)

- [ ] Clone/pull `tensordecompexp` and read this file  
- [ ] `pip install -r karan/requirements.txt`  
- [ ] Run `tensors/workflow/als/als_tucker.py` once; confirm `als_tucker_results/` appears  
- [ ] Skim module docstrings in `tensors/workflow/` (one-line summary per file)  
- [ ] Read `latex_notes/tucker_als_chebyshev_3d_methods_comparison.pdf`  
- [ ] If ACE work continues: read `latex_notes/ace_3d_configuration_energy_setup.pdf` + run `als_discuss/als_zulip_ace_3d.py`  
- [ ] Re-run rank sweeps as needed; update `plotter/2d/plot_tucker.py` `FILES` dict  

---

## Open points / context for follow-up

- **Value space vs coefficient space:** Several scripts (`als_tensorly.py`, `als_weight_same.py`, comments in `als_3d_*`) explore weighted/orthogonal formulations so minimizing on \(F\) aligns with minimizing Chebyshev coefficients. See `als_tucker_prof.py` for the equivalence experiment.  
- **Christoph operator formulation:** `als_discuss/als2d_operator.py`, `als_3d_operator.py` — alternative derivation (Dec 2025); preferred stable implementation is the Zulip-style updates.  
- **HOEVD / energy optimization:** `als_zulip_ace_hosvd_opt.py`, `als_zulip_ace_hoevd_energy_opt.py` — ACE init + gradient refinement experiments.  

---

*Handoff packaged May 2026. Source-only snapshot; original tree ~2.6 GB with result artifacts.*
