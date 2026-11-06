#!/usr/bin/env python3
"""
Coupled anharmonic oscillators in 3, 4 and 5 dimensions
=======================================================

Follow-up to sections 4.1.4 and 6.1.2 of

    Eva Obeed, "Numerical Modelling of Some Quantum Systems",
    Master thesis, NTNU, Trondheim, May 2016.

The thesis solves anharmonic oscillators up to three dimensions and notes
that (a) its use of the iterative solver eigsh is "less than optimal" and
(b) four and five dimensions "have not been explored". This module does
both, for the non-separable Hamiltonian

    H = sum_j ( p_j^2 + q_j^4 )  +  c * sum_{j<k} q_j^2 q_k^2 ,   j = 1..D.

(For c = 0 the problem separates into D copies of the one-dimensional
oscillator; the coupling c is what makes higher dimensions hard.)

Three solvers are implemented:

  1. `solve_lanczos`     - the thesis approach: eigsh on the product basis
                           of harmonic oscillator states.
  2. `solve_lobpcg`      - the same basis, but with the block solver LOBPCG
                           and a preconditioner built from the separable
                           part of H, which can be inverted exactly.
  3. `ContractedBasis`   - a change of basis: the one-dimensional anharmonic
                           problem is solved first, and only its lowest n
                           eigenstates are kept in each direction. The
                           matrix shrinks from N^D to n^D with n << N.

Units as in the thesis: hbar = 1, H0 = p^2 + q^2 has levels 2n + 1.
"""

import time
import warnings

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, eigsh, lobpcg

from matrix_mechanics import MatrixQM


class CountingOperator:
    """Wraps a matvec function and counts how often it is called."""

    def __init__(self, matvec, size):
        self.matvec, self.size, self.count = matvec, size, 0

    def __call__(self, x):
        self.count += 1
        return self.matvec(x)

    def linear_operator(self):
        def matmat(X):
            return np.column_stack([self(X[:, i]) for i in range(X.shape[1])])
        return LinearOperator((self.size, self.size), matvec=self,
                              matmat=matmat, dtype=float)


def product_basis_problem(dim, N, c=1.0, parities=None):
    """MatrixQM object for H in the harmonic oscillator product basis."""
    return MatrixQM(N=N, dim=dim, b2=0.0, b4=1.0, b22=c, parities=parities)


def one_dimensional_parts(N, parities):
    """For each direction: all eigenvalues e and eigenvectors U of
    h = p^2 + q^4 in the parity sector, and the matrix of q^2."""
    parts = {}
    for p in set(parities):
        qm = MatrixQM(N=N, dim=1, b2=0.0, b4=1.0, parities=(p,))
        e, U = eigh(qm.matrix())
        Q20, Q22 = qm.q2[0]
        Q2 = np.diag(Q20) + np.diag(Q22, 1) + np.diag(Q22, -1)
        parts[p] = (e, U, Q2)
    return [parts[p] for p in parities]


def apply_along(M, phi, axis):
    """Apply the matrix M along one axis of the array phi."""
    return np.moveaxis(np.tensordot(M, phi, axes=(1, axis)), 0, axis)


# ---------------------------------------------------------------------------
# Solver 1: Lanczos (eigsh), as in the thesis
# ---------------------------------------------------------------------------

def solve_lanczos(dim, N, k=6, c=1.0, parities=None, tol=1e-10):
    qm = product_basis_problem(dim, N, c, parities)
    op = CountingOperator(qm.H, qm.size)
    t0 = time.time()
    E = eigsh(op.linear_operator(), k=k, which="SA", tol=tol,
              return_eigenvectors=False)
    return dict(E=np.sort(E), time=time.time() - t0, matvecs=op.count,
                size=qm.size)


# ---------------------------------------------------------------------------
# Solver 2: LOBPCG with a separable preconditioner
# ---------------------------------------------------------------------------

def solve_lobpcg(dim, N, k=6, c=1.0, parities=None, tol=1e-6, maxiter=200):
    """Lowest k eigenvalues with LOBPCG.

    Preconditioner. Write H = H_sep + c * W, where H_sep = sum_j h_j is the
    separable part. In the product basis of the 1D eigenvectors of h_j,
    H_sep is diagonal with entries e_a + e_b + ..., so (H_sep - sigma)^-1
    costs only 2 D small matrix transforms. Applying it to the residual
    removes the huge spread of the spectrum (the largest eigenvalue grows
    like N^2), which is what slows Lanczos down.

    Starting vectors: the k lowest eigenvectors of H_sep.

    tol is the residual norm |H x - E x|; the error of an eigenvalue is of
    the order of the residual squared.
    """
    parities = (0,) * dim if parities is None else tuple(parities)
    qm = product_basis_problem(dim, N, c, parities)
    parts = one_dimensional_parts(N, parities)
    shape = (N,) * dim

    # energies of H_sep on the product grid
    Esep = np.zeros(shape)
    for j, (e, _, _) in enumerate(parts):
        Esep = Esep + e.reshape([-1 if i == j else 1 for i in range(dim)])
    sigma = Esep.min() - 1.0                    # H_sep - sigma is positive

    def to_sep(phi):                            # HO basis -> 1D eigenbasis
        for j, (_, U, _) in enumerate(parts):
            phi = apply_along(U.T, phi, j)
        return phi

    def from_sep(phi):
        for j, (_, U, _) in enumerate(parts):
            phi = apply_along(U, phi, j)
        return phi

    def precond(R):
        R = R.reshape(shape + (-1,))
        out = np.empty_like(R)
        for i in range(R.shape[-1]):
            out[..., i] = from_sep(to_sep(R[..., i]) / (Esep - sigma))
        return out.reshape(qm.size, -1)

    M = LinearOperator((qm.size, qm.size), matvec=precond, matmat=precond,
                       dtype=float)

    # starting block: lowest eigenvectors of the separable part
    lowest = np.argsort(Esep, axis=None)[:k]
    X = np.zeros((qm.size, k))
    for i, idx in enumerate(lowest):
        e = np.zeros(shape)
        e[np.unravel_index(idx, shape)] = 1.0
        X[:, i] = from_sep(e).ravel()

    op = CountingOperator(qm.H, qm.size)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")         # tolerance notices
        E, _ = lobpcg(op.linear_operator(), X, M=M, largest=False, tol=tol,
                      maxiter=maxiter)
    return dict(E=np.sort(E), time=time.time() - t0, matvecs=op.count,
                size=qm.size)


# ---------------------------------------------------------------------------
# Solver 3: contracted basis
# ---------------------------------------------------------------------------

class ContractedBasis:
    """H in the product basis of 1D anharmonic eigenstates.

    In this basis the separable part is diagonal and only the coupling
    c * q_j^2 q_k^2 is a full (but small, n x n per direction) matrix:

        H = diag(e_a + e_b + ...) + c * sum_{j<k} X_j X_k ,

    with X = U^T q^2 U the matrix of q^2 between 1D eigenstates. N is the
    size of the underlying harmonic oscillator basis used for the 1D
    problem (N = 50 gives the 1D states to machine precision).
    """

    def __init__(self, dim, n, c=1.0, parities=None, N=50):
        self.dim, self.n, self.c = dim, n, c
        self.parities = (0,) * dim if parities is None else tuple(parities)
        self.shape = (n,) * dim
        self.size = n ** dim
        parts = one_dimensional_parts(N, self.parities)
        self.e = [e[:n] for e, _, _ in parts]
        self.X = [U[:, :n].T @ Q2 @ U[:, :n] for _, U, Q2 in parts]
        self.Esep = np.zeros(self.shape)
        for j, e in enumerate(self.e):
            self.Esep = self.Esep + e.reshape(
                [-1 if i == j else 1 for i in range(dim)])

    def H(self, psi):
        phi = np.asarray(psi, dtype=float).reshape(self.shape)
        out = self.Esep * phi
        # sum_{j<k} X_j X_k phi = sum_k X_k ( sum_{j<k} X_j phi )
        partial = np.zeros_like(phi)
        for k in range(self.dim):
            if k > 0:
                out += self.c * apply_along(self.X[k], partial, k)
            if k < self.dim - 1:
                partial = partial + apply_along(self.X[k], phi, k)
        return out.ravel()

    def matrix(self):
        M = np.empty((self.size, self.size))
        identity = np.eye(self.size)
        for i in range(self.size):
            M[:, i] = self.H(identity[:, i])
        return M

    def solve(self, k=6, dense_limit=2000, tol=1e-6, maxiter=200):
        """Lowest k eigenvalues: dense for small sizes, otherwise LOBPCG
        with the diagonal preconditioner 1 / (E_sep - sigma)."""
        t0 = time.time()
        if self.size <= dense_limit:
            E = np.linalg.eigvalsh(self.matrix())[:k]
            return dict(E=E, time=time.time() - t0, matvecs=self.size,
                        size=self.size)
        Esep = self.Esep.ravel()
        sigma = Esep.min() - 1.0
        M = LinearOperator((self.size, self.size), dtype=float,
                           matvec=lambda r: r / (Esep - sigma),
                           matmat=lambda R: R / (Esep - sigma)[:, None])
        X = np.zeros((self.size, k))
        X[np.argsort(Esep)[:k], np.arange(k)] = 1.0
        op = CountingOperator(self.H, self.size)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            E, _ = lobpcg(op.linear_operator(), X, M=M, largest=False,
                          tol=tol, maxiter=maxiter)
        return dict(E=np.sort(E), time=time.time() - t0, matvecs=op.count,
                    size=self.size)


def separable_levels(dim, k=6, parities=None, N=50):
    """Exact lowest levels for c = 0: sums of 1D energies."""
    parities = (0,) * dim if parities is None else tuple(parities)
    parts = one_dimensional_parts(N, parities)
    total = np.zeros(1)
    for e, _, _ in parts:
        total = np.sort((total[:, None] + e[None, :12]).ravel())[:200]
    return total[:k]
