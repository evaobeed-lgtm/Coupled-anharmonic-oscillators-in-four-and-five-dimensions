#!/usr/bin/env python3
"""
Numerical matrix mechanics
==========================

Python 3 implementation of the first problem in

    Eva Obeed, "Numerical Modelling of Some Quantum Systems",
    Master thesis, NTNU, Trondheim, May 2016.

Covers
  * Section 4.1   - truncated q and ip matrices, commutator, eigh / eigsh checks
  * Section 2.4.1 - "discrete Schroedinger" representation  H = p^2 + V(q)
  * Section 6.1   - (an)harmonic oscillators in D dimensions (class MatrixQM)

        H = sum_j [ a2_j p_j^2 + b2_j q_j^2 + b4_j q_j^4 ]
            + sum_{j<k} b22_jk q_j^2 q_k^2                       (eq. 6.6)

Units: hbar = 1 and m*omega = 1, and H0 = p^2 + q^2, so that the harmonic
oscillator levels are E_n = 2n + 1 (the thesis convention, no factor 1/2).
"""

import time

import numpy as np
from scipy.linalg import eigh, eigvalsh
from scipy.sparse.linalg import LinearOperator, eigsh


# ---------------------------------------------------------------------------
# Section 4.1: q and ip as linear operators on N-component vectors
# ---------------------------------------------------------------------------

def make_q_ip(N):
    """Return the linear operators q and ip (= i*p, a real matrix) truncated
    to the lowest N harmonic oscillator states (eq. 2.31)."""
    Q11 = np.sqrt(np.arange(1, N) / 2)        # <n|q|n+1> = sqrt((n+1)/2)

    def q(psi):
        out = np.zeros_like(psi)
        out[:-1] += Q11 * psi[1:]
        out[1:] += Q11 * psi[:-1]
        return out

    def ip(psi):
        out = np.zeros_like(psi)
        out[:-1] += Q11 * psi[1:]
        out[1:] -= Q11 * psi[:-1]
        return out

    return q, ip


def matrix(operator, N):
    """Dense N x N matrix of a linear operator given as a function."""
    M = np.zeros((N, N))
    identity = np.eye(N)
    for n in range(N):
        M[:, n] = operator(identity[:, n])     # n'th column
    return M


def commutator_check(N=5):
    """[q, ip] = -1, except for the last diagonal element (the trace of a
    commutator of finite matrices must vanish)."""
    q, ip = make_q_ip(N)
    Mq, Mip = matrix(q, N), matrix(ip, N)
    return Mq @ Mip - Mip @ Mq


def dense_diagonalization_check(N=50):
    """Diagonalize H = p^2 + q^4 with scipy.linalg.eigh and measure the
    numerical quality of the result (sections 4.1.2 - 4.1.3)."""
    t0 = time.time()
    q, ip = make_q_ip(N)
    Mq, Mip = matrix(q, N), matrix(ip, N)
    Mp2, Mq2 = -Mip @ Mip, Mq @ Mq
    H = Mp2 + Mq2 @ Mq2
    E, S = eigh(H)
    inf = np.inf
    return {
        "N": N,
        "E": E,
        "eigerr": max(np.linalg.norm(H @ S[:, n] - E[n] * S[:, n], ord=inf)
                      for n in range(N)),
        "ortherr": np.linalg.norm(S.T @ S - np.eye(N), ord=inf),
        "comperr": np.linalg.norm(S @ S.T - np.eye(N), ord=inf),
        "decomperr": np.linalg.norm(S.T @ H @ S - np.diag(E), ord=inf),
        "time": time.time() - t0,
    }


def sparse_diagonalization_check(N=200, k=20):
    """Lowest k eigenvalues of H = p^2 + q^4 with the iterative routine
    scipy.sparse.linalg.eigsh, using only the action of H (section 4.1.4)."""
    t0 = time.time()
    q, ip = make_q_ip(N)

    def H(psi):
        psi = np.asarray(psi, dtype=float).ravel()
        p2psi = -ip(ip(psi))
        q2psi = q(q(psi))
        return p2psi + q(q(q2psi))

    A = LinearOperator((N, N), matvec=H, dtype=float)
    E, S = eigsh(A, k=k, which="SA")           # smallest algebraic
    err = max(np.linalg.norm(H(S[:, n]) - E[n] * S[:, n], ord=np.inf)
              for n in range(k))
    return {"N": N, "k": k, "E": np.sort(E), "eigerr": err,
            "time": time.time() - t0}


# ---------------------------------------------------------------------------
# Section 2.4.1: discrete Schroedinger representation, H = a2 p^2 + V(q)
# ---------------------------------------------------------------------------

def discrete_schrodinger(V, N=100, a2=1.0):
    """Eigenvalues/eigenvectors of H = a2*p^2 + V(q) for an arbitrary
    potential V (a vectorized function of q).

    q_N is diagonalized, q_N = S diag(qbar) S^T, so V(q_N) becomes the diagonal
    matrix V(qbar) and the kinetic energy becomes the full matrix S^T p^2 S.
    The eigenvalues qbar act as a (non-uniform) position grid.

    Returns (E, psi, qbar); psi[:, n] is eigenvector n on the grid qbar.
    """
    q, ip = make_q_ip(N)
    qbar, S = eigh(matrix(q, N))
    P2 = S.T @ (-matrix(ip, N) @ matrix(ip, N)) @ S
    H = a2 * P2 + np.diag(V(qbar))
    E, psi = eigh(H)
    return E, psi, qbar


# ---------------------------------------------------------------------------
# Section 6.1: oscillators in D dimensions
# ---------------------------------------------------------------------------

class MatrixQM:
    """H of eq. (6.6) in a product basis of harmonic oscillator states.

    Each direction uses N states of one parity (even: |0>,|2>,..., odd:
    |1>,|3>,...), since H only connects states of equal parity in every
    direction. The matrix elements are those of the infinite matrices
    (eq. 6.4), restricted afterwards to the lowest states.

    Parameters
    ----------
    N    : number of basis states per direction and parity
    dim  : number of dimensions D
    a2, b2, b4 : scalars or length-D sequences (defaults 1, 1, 0)
    b22  : couplings of q_j^2 q_k^2, j < k. Either a scalar (same value for
           all pairs), a dict {(j, k): value} or a (D, D) array (upper
           triangle is used). Default 0.
    parities : length-D sequence of 0 (even) / 1 (odd); default all even.
    """

    def __init__(self, N=50, dim=1, a2=1.0, b2=1.0, b4=0.0, b22=0.0,
                 parities=None):
        self.N, self.dim = N, dim
        self.shape = (N,) * dim
        self.size = N ** dim

        n = np.arange(2 * N, dtype=float)
        # Nonzero elements of q^2 and q^4 in the full basis |n>:
        self.Q20 = (2 * n + 1) / 2                               # <n|q^2|n>
        self.Q22 = (np.sqrt((n + 1) * (n + 2)) / 2)[:-2]         # <n|q^2|n+2>
        self.Q40 = (6 * n * n + 6 * n + 3) / 4                   # <n|q^4|n>
        self.Q42 = ((4 * n + 6) * np.sqrt((n + 1) * (n + 2)) / 4)[:-2]
        self.Q44 = (np.sqrt((n + 1) * (n + 2) * (n + 3) * (n + 4)) / 4)[:-4]
        # p^2 has the same diagonal as q^2 and opposite off-diagonal.

        self.a2 = self._per_dim(a2)
        self.b2 = self._per_dim(b2)
        self.b4 = self._per_dim(b4)
        self.b22 = self._pairs(b22)
        self.set_parities(parities)

    # -- parameter handling -------------------------------------------------
    def _per_dim(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        if x.size == 1:
            x = np.full(self.dim, x[0])
        if x.size != self.dim:
            raise ValueError("expected a scalar or %d values" % self.dim)
        return x

    def _pairs(self, b22):
        pairs = [(j, k) for j in range(self.dim) for k in range(j + 1, self.dim)]
        if isinstance(b22, dict):
            out = {(min(j, k), max(j, k)): float(v) for (j, k), v in b22.items()}
        elif np.ndim(b22) == 0:
            out = {jk: float(b22) for jk in pairs}
        else:
            b22 = np.asarray(b22, dtype=float)
            out = {(j, k): b22[j, k] for (j, k) in pairs}
        return {jk: v for jk, v in out.items() if v != 0.0}

    def set_parities(self, parities=None):
        """Select the parity sector and build the banded 1D operators."""
        if parities is None:
            parities = (0,) * self.dim
        parities = tuple(np.atleast_1d(parities))
        if len(parities) != self.dim:
            raise ValueError("need one parity per dimension")
        self.parities = parities
        self.h1 = []    # per direction: bands (d0, d1, d2) of a2 p^2+b2 q^2+b4 q^4
        self.q2 = []    # per direction: bands (d0, d1) of q^2
        for j, p in enumerate(parities):
            Q20, Q22 = self.Q20[p::2], self.Q22[p::2]
            Q40, Q42, Q44 = self.Q40[p::2], self.Q42[p::2], self.Q44[p::2]
            a2, b2, b4 = self.a2[j], self.b2[j], self.b4[j]
            self.h1.append(((b2 + a2) * Q20 + b4 * Q40,
                            (b2 - a2) * Q22 + b4 * Q42,
                            b4 * Q44))
            self.q2.append((Q20, Q22))

    # -- action of operators ------------------------------------------------
    @staticmethod
    def _apply(bands, phi, axis):
        """Apply a symmetric banded matrix along one axis of the array phi.
        bands[k] holds the k'th super(=sub)diagonal."""
        x = np.moveaxis(phi, axis, -1)
        out = np.zeros_like(phi)
        y = np.moveaxis(out, axis, -1)          # a view into out
        y += bands[0] * x
        for k in range(1, len(bands)):
            y[..., :-k] += bands[k] * x[..., k:]
            y[..., k:] += bands[k] * x[..., :-k]
        return out

    def H(self, psi):
        """Action of the Hamiltonian on a state vector of length N**dim."""
        phi = np.asarray(psi, dtype=float).reshape(self.shape)
        out = np.zeros_like(phi)
        for j in range(self.dim):
            out += self._apply(self.h1[j], phi, j)
        q2phi = {}
        for (j, k), c in self.b22.items():
            if j not in q2phi:
                q2phi[j] = self._apply(self.q2[j], phi, j)
            out += c * self._apply(self.q2[k], q2phi[j], k)
        return out.ravel()

    def linear_operator(self):
        return LinearOperator((self.size, self.size), matvec=self.H,
                              dtype=float)

    def matrix(self):
        """Dense matrix of H (only sensible for moderate N**dim)."""
        M = np.zeros((self.size, self.size))
        identity = np.eye(self.size)
        for n in range(self.size):
            M[:, n] = self.H(identity[:, n])
        return M

    # -- eigenvalues ----------------------------------------------------------
    def eigenvalues(self, k=20, dense_limit=4000):
        """Lowest k eigenvalues in the current parity sector. Dense
        diagonalization for small systems, Lanczos (eigsh) otherwise."""
        if self.size <= dense_limit:
            return eigvalsh(self.matrix())[:k]
        E = eigsh(self.linear_operator(), k=k, which="SA",
                  return_eigenvectors=False)
        return np.sort(E)

    def spectrum(self, k=20, dense_limit=4000):
        """Lowest k eigenvalues over all 2**dim parity sectors.
        Returns a list of (energy, parities) sorted by energy."""
        keep = self.parities
        levels = []
        for parities in np.ndindex((2,) * self.dim):
            self.set_parities(parities)
            kk = min(k, self.size - 1)
            levels += [(E, parities) for E in self.eigenvalues(kk, dense_limit)]
        self.set_parities(keep)
        return sorted(levels)[:k]


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

def main():
    np.set_printoptions(precision=10, suppress=True, linewidth=100)

    print("== Section 4.1.1: commutator [q, ip] for N = 5 ==")
    print(commutator_check(5))

    print("\n== Sections 4.1.2-4.1.3: eigh on H = p^2 + q^4 ==")
    for N in [5, 20, 50, 200, 500, 2000]:
        r = dense_diagonalization_check(N)
        print("N=%4d: eigerr=%.3e ortherr=%.3e comperr=%.3e "
              "decomperr=%.3e time=%.3f s" % (N, r["eigerr"], r["ortherr"],
              r["comperr"], r["decomperr"], r["time"]))

    print("\n== Section 4.1.4: eigsh on H = p^2 + q^4 ==")
    for N, k in [(50, 20), (200, 20), (500, 40)]:
        r = sparse_diagonalization_check(N, k)
        print("N=%4d (k=%3d): maxerror=%.3e time=%.3f s"
              % (N, k, r["eigerr"], r["time"]))

    print("\n== Section 6.1: harmonic oscillators, E = sum_j (2 n_j + 1) ==")
    for dim in (1, 2, 3):
        qm = MatrixQM(N=8, dim=dim)
        print("%d-dim:" % dim, np.array([E for E, _ in qm.spectrum(k=10)]))

    print("\n== Section 6.1.2: anharmonic oscillator H = p^2 + q^4 ==")
    for N in (10, 20, 30, 40, 50):
        E0 = MatrixQM(N=N, b2=0, b4=1).eigenvalues(k=1)[0]
        print("N=%2d: E0 = %.14f" % (N, E0))
    qm = MatrixQM(N=50, b2=0, b4=1)
    print("lowest levels (both parities):")
    print(np.array([E for E, _ in qm.spectrum(k=10)]))

    print("\n== Section 2.4.1: discrete Schroedinger, V(q) = q^4 ==")
    E, _, _ = discrete_schrodinger(lambda q: q ** 4, N=100)
    print(E[:6])

    print("\n== 2D: H = p_x^2 + p_y^2 + x^4 + y^4 + x^2 y^2 ==")
    t0 = time.time()
    qm = MatrixQM(N=30, dim=2, b2=0, b4=1, b22=1)
    for E, par in qm.spectrum(k=8):
        print("  E = %.10f   parities %s" % (E, par))
    print("  (%.1f s)" % (time.time() - t0))

    print("\n== 3D: H = sum_j (p_j^2 + q_j^4), even-even-even sector ==")
    t0 = time.time()
    qm = MatrixQM(N=20, dim=3, b2=0, b4=1)
    E = qm.eigenvalues(k=6)
    print("  ", E)
    print("   3 * E0(1D) = %.10f  (separable check), %.1f s"
          % (3 * MatrixQM(N=50, b2=0, b4=1).eigenvalues(k=1)[0],
             time.time() - t0))


if __name__ == "__main__":
    main()
