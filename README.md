# Coupled anharmonic oscillators in four and five dimensions

**Faster eigenvalue solvers for matrix mechanics, and the 4D and 5D systems the thesis left unexplored.**

This is a follow-up to sections 4.1.4 and 6.1.2 of

> Eva Obeed, *Numerical Modelling of Some Quantum Systems*, master thesis, NTNU, Trondheim, May 2016.

## The open problems

The thesis solves anharmonic oscillators in one, two and three dimensions with numerical matrix mechanics and leaves two things open:

1. Its use of the iterative solver `eigsh` for the lowest eigenvalues is described as "a straightforward (but less than optimal) solution", which sometimes fails to converge.
2. Four dimensions (50⁴ = 6 250 000 basis states) and five dimensions (50⁵ = 312 500 000) are called within reach "with some time and patience", but "these last two possibilities have not been explored".

This project solves both for the non-separable Hamiltonian

```
H = Σ_j ( p_j² + q_j⁴ )  +  c Σ_{j<k} q_j² q_k² ,     j = 1..D
```

For c = 0 the problem splits into D independent one-dimensional oscillators. The coupling c is what makes higher dimensions hard.

## Three solvers

| Solver | Idea |
|---|---|
| **Lanczos** (`solve_lanczos`) | The thesis approach: `eigsh` on the product basis of harmonic oscillator states. |
| **Preconditioned LOBPCG** (`solve_lobpcg`) | Same basis, block solver, and a preconditioner that inverts the separable part of H exactly. |
| **Contracted basis** (`ContractedBasis`) | Solve the 1D anharmonic problem first and keep only its lowest n eigenstates per direction. |

**Why Lanczos is slow here.** The largest eigenvalue of the truncated Hamiltonian grows quickly with the basis size, while the levels of interest stay near the bottom. Lanczos needs more and more iterations as that spread grows.

**The preconditioner.** Write H = H_sep + cW. In the basis of 1D eigenvectors H_sep is diagonal, so (H_sep − σ)⁻¹ costs only a few small matrix transforms. Applied to the residual, it removes the spread of the spectrum, and the number of iterations stops depending on the basis size.

**The contracted basis.** In the basis of 1D anharmonic eigenstates the Hamiltonian is

```
H = diag(e_a + e_b + ...) + c Σ_{j<k} X_j X_k ,      X = Uᵀ q² U
```

Far fewer states are needed per direction (6 to 12 instead of 30 to 50), so the matrix shrinks from N^D to n^D.

## Results

All numbers are for c = 1 and the sector where every direction has even parity, unless stated.

### Validation

- In 2D all three solvers agree with dense diagonalization of the 900 × 900 matrix to 3 × 10⁻¹².
- For c = 0 the contracted basis reproduces the sums of 1D energies exactly in 3, 4 and 5 dimensions.

### Solver benchmark (3D, six lowest levels)

| N per direction | Basis size | Lanczos | LOBPCG + preconditioner | Contracted, n = 10 |
|---|---|---|---|---|
| 20 | 8 000 | 2.3 s, 2562 products | 0.27 s, 140 products | 0.18 s |
| 30 | 27 000 | 14.1 s, 5510 products | 0.79 s, 140 products | 0.20 s |
| 40 | 64 000 | 69.2 s, 9978 products | 2.03 s, 132 products | 0.18 s |

"Products" counts applications of H to a vector. Timings are from a two-core machine.

![Solver benchmark](figures/solver_benchmark.png)

- The preconditioned solver is 34 times faster than Lanczos at N = 40, and its work no longer grows with N.
- Lanczos also **missed a level**: the fifth level is doubly degenerate, and `eigsh` returned only one copy in every run (5 of the 6 lowest levels found). The block solver and the contracted basis found all six.

### Convergence of the contracted basis (4D ground state)

| n per direction | Matrix size | Error, c = 1 | Error, c = 10 |
|---|---|---|---|
| 3 | 81 | 1.7 × 10⁻⁷ | 1.0 × 10⁻² |
| 4 | 256 | 3.8 × 10⁻¹⁰ | 5.0 × 10⁻⁴ |
| 6 | 1 296 | below 10⁻¹⁵ | 1.2 × 10⁻⁶ |
| 8 | 4 096 | below 10⁻¹⁴ | 3.2 × 10⁻⁹ |
| 12 | 20 736 | reference level | 3.8 × 10⁻¹⁴ |

![Convergence of the contracted basis](figures/contracted_convergence.png)

Errors are measured against the n = 16 result. Six states per direction give machine precision at c = 1, where the product basis needs about 30.

### Four and five dimensions

| System | Method | Matrix size | Time | Six lowest levels |
|---|---|---|---|---|
| 4D | contracted, n = 12 | 20 736 | 0.5 s | 4.8957210759, 11.8932359599 (×3), 12.5027521365, 19.6021185249 |
| 4D | direct LOBPCG, N = 30 | 810 000 | 63 s | agrees to 3.8 × 10⁻¹² |
| 5D | contracted, n = 10 | 100 000 | 1.2 s | 6.3477665501, 13.5692984209 (×4), 14.2835806862 |
| 5D | direct LOBPCG, N = 16 | 1 048 576 | 50 s | agrees to 7.7 × 10⁻⁷ |

The two methods are independent, so their agreement is a real check. The remaining 5D difference comes from the direct calculation: N = 16 is too small a product basis for more digits.

### Lowest levels for D = 1 to 5

| D | E₀ (all even) | E₁ (one direction odd) | Gap |
|---|---|---|---|
| 1 | 1.0603620905 | 3.7996730298 | 2.7393109393 |
| 2 | 2.2409100202 | 5.1516802904 | 2.9107702702 |
| 3 | 3.5236030868 | 6.5870834155 | 3.0634803287 |
| 4 | 4.8957210759 | 8.0975992989 | 3.2018782230 |
| 5 | 6.3477665501 | 9.6766953763 | 3.3289288262 |

### Ground state energy against coupling

![Ground state energy against coupling](figures/ground_state_vs_coupling.png)

| c | D = 2 | D = 3 | D = 4 | D = 5 |
|---|---|---|---|---|
| 0 | 2.120724 | 3.181086 | 4.241448 | 5.301810 |
| 1 | 2.240910 | 3.523603 | 4.895721 | 6.347767 |
| 4 | 2.521816 | 4.244422 | 6.167545 | 8.256487 |
| 10 | 2.923071 | 5.186198 | 7.739711 | 10.528141 |

## Conclusion

The thesis expected 4D and 5D to need "time and patience" and possibly a high-end laptop. With a contracted basis the 5D ground state takes about one second on two cores, to ten digits. Solving directly in the original product basis is also feasible with a preconditioned block solver (a million basis states in under a minute), and serves as an independent check.

## Usage

```bash
pip install -r requirements.txt
python run_experiments.py           # all experiments, about 4 minutes
python run_experiments.py --quick   # skips the largest direct runs
```

In your own code:

```python
from coupled_oscillators import ContractedBasis, solve_lobpcg

# 5D ground state and next levels, coupling c = 1
print(ContractedBasis(dim=5, n=10, c=1.0).solve(k=6)["E"])

# Sector with odd parity in the first direction
print(ContractedBasis(dim=4, n=12, parities=(1, 0, 0, 0)).solve(k=3)["E"])

# Independent check in the harmonic oscillator product basis
print(solve_lobpcg(dim=4, N=20, k=6)["E"])
```

## Files

| File | Content |
|---|---|
| `coupled_oscillators.py` | the three solvers |
| `run_experiments.py` | validation, benchmark, convergence, 4D/5D runs, figures |
| `matrix_mechanics.py` | the `MatrixQM` class from the thesis implementation (product basis Hamiltonian) |
| `figures/` | the three figures shown above |

## Limitations

- The coupling is the same constant c for every pair of directions, and the single-direction potential is fixed to q⁴. The `MatrixQM` class accepts general coefficients; the contracted basis would need a small extension.
- The contracted basis converges more slowly as c grows (see the c = 10 column), because the 1D eigenstates become a poorer starting point.
- Only one parity sector is solved per call. The full spectrum is the union of all 2^D sectors.
- Timings depend on the machine and on the BLAS library behind NumPy.
