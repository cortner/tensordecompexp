# LaTeX / PDF notes

Written-up algorithms and meeting notes (exported as PDF). Use this index to find the right document.

## Start here

| PDF | Contents |
|-----|----------|
| [`tucker_als_chebyshev_3d_methods_comparison.pdf`](tucker_als_chebyshev_3d_methods_comparison.pdf) | **Main reference:** 3D Tucker ALS in Chebyshev bases, mode updates, comparison of Direct ALS vs LS–SVD (11 pp.) |
| [`ls_svd_functional_tucker_chebyshev_bases.pdf`](ls_svd_functional_tucker_chebyshev_bases.pdf) | **Christoph / Zulip method:** LS–SVD functional Tucker (paper-style write-up with abstract) |
| [`tucker_als_chebyshev_3d_with_results.pdf`](tucker_als_chebyshev_3d_with_results.pdf) | Shorter 3D Tucker ALS note with result figures (subset of the comparison doc) |

## Tucker ALS (functional approximation)

| PDF | Contents |
|-----|----------|
| [`tucker_als_3d_anisotropic_chebyshev.pdf`](tucker_als_3d_anisotropic_chebyshev.pdf) | 3D anisotropic \(f\), Tucker ALS update structure (Christoph-style) |
| [`tucker_als_3d_factor_merge_and_svd_update.pdf`](tucker_als_3d_factor_merge_and_svd_update.pdf) | Professor’s notation: merge Tucker index, LS, then SVD for factor \(A\) |
| [`als_tucker_2d_chebyshev_basis_notes.pdf`](als_tucker_2d_chebyshev_basis_notes.pdf) | 2D ALS–Tucker in Chebyshev basis; metric matrices and closed-form \(G\) |
| [`chebyshev_tucker_low_rank_approximation_3d_intro.pdf`](chebyshev_tucker_low_rank_approximation_3d_intro.pdf) | Short intro: Chebyshev expansion + Tucker on coefficient tensor (3D) |

## ACE + Tucker (configuration energies)

| PDF | Contents |
|-----|----------|
| [`ace_3d_configuration_energy_setup.pdf`](ace_3d_configuration_energy_setup.pdf) | 3D ACE: pooled features, mean energy per configuration |
| [`ace_tucker_2d_energy_model_and_verification.pdf`](ace_tucker_2d_energy_model_and_verification.pdf) | 2D ACE–Tucker model, ALS+SVD updates, verification vs \(C^{\mathrm{ref}}\) |
| [`ace_tucker_2d_als_svd_update_formulas.pdf`](ace_tucker_2d_als_svd_update_formulas.pdf) | One-page: 2D ACE–Tucker ALS/SVD update equations only |
| [`ace_tucker_3d_mode1_als_update.pdf`](ace_tucker_3d_mode1_als_update.pdf) | 3D ACE–Tucker: mode-1 (update \(A\)) least-squares step |

## CP / optimization documentation

| PDF | Contents |
|-----|----------|
| [`als_2d_chebyshev_cp_isotropic_decomposition.pdf`](als_2d_chebyshev_cp_isotropic_decomposition.pdf) | 2D CP–ALS on Chebyshev grid, isotropic \(f=1/(1+c^2(x^2+y^2))\), results |
| [`bfgs_2d_chebyshev_cp_anisotropic_documentation.pdf`](bfgs_2d_chebyshev_cp_anisotropic_documentation.pdf) | Documents `bfgs2d.py`: anisotropic CP + BFGS |
| [`anisotropic_test_function_C_matrix_cases.pdf`](anisotropic_test_function_C_matrix_cases.pdf) | Definition of \(f(x,y;C)\) and example \(C\) matrices |

## Tensor methods background

| PDF | Contents |
|-----|----------|
| [`hosvd_hooi_explained_2x2x2_tensor.pdf`](hosvd_hooi_explained_2x2x2_tensor.pdf) | Toy HOSVD / HOOI on a \(2\times2\times2\) tensor |
| [`hoevd_higher_order_eigenvalue_decomposition.pdf`](hoevd_higher_order_eigenvalue_decomposition.pdf) | HOEVD definition and mode unfoldings |
| [`symmetric_tucker_decomposition_intro.pdf`](symmetric_tucker_decomposition_intro.pdf) | Symmetric Tucker: shared factor \(Q\) across modes |

## Map to code

| Topic | PDF | Related script |
|-------|-----|----------------|
| 3D Tucker ALS | `tucker_als_chebyshev_3d_methods_comparison.pdf` | `workflow/als/als_tucker.py`, `als_discuss/als_zulip3d.py` |
| LS–SVD Tucker | `ls_svd_functional_tucker_chebyshev_bases.pdf` | `workflow/als_discuss/als_zulip.py` |
| ACE 3D | `ace_3d_configuration_energy_setup.pdf` | `workflow/als_discuss/als_zulip_ace_3d.py` |
| BFGS CP | `bfgs_2d_chebyshev_cp_anisotropic_documentation.pdf` | `workflow/bfgs/bfgs2d.py` |
