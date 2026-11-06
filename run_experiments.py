#!/usr/bin/env python3
"""
Experiments for coupled anharmonic oscillators in up to five dimensions.

    H = sum_j ( p_j^2 + q_j^4 ) + c * sum_{j<k} q_j^2 q_k^2

Run:  python run_experiments.py           (about 5 minutes on a laptop)
      python run_experiments.py --quick   (skips the largest direct runs)
"""

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coupled_oscillators import (ContractedBasis, product_basis_problem,
                                 separable_levels, solve_lanczos,
                                 solve_lobpcg)

OUTDIR = "figures"


def validation():
    print("== 1. Validation ==")
    qm = product_basis_problem(2, 30)
    ref = np.linalg.eigvalsh(qm.matrix())[:6]
    print("2D, c = 1, dense reference (900 x 900):")
    print("   ", ref)
    for name, E in [("Lanczos", solve_lanczos(2, 30)["E"]),
                    ("LOBPCG", solve_lobpcg(2, 30)["E"]),
                    ("contracted n=10", ContractedBasis(2, 10, N=30).solve()["E"])]:
        print("  %-16s max deviation %.1e" % (name, np.abs(E - ref).max()))
    print("c = 0 (separable), against sums of 1D energies:")
    for dim in (3, 4, 5):
        E = ContractedBasis(dim, 8, c=0.0).solve()["E"]
        print("  %dD contracted: max deviation %.1e"
              % (dim, np.abs(E - separable_levels(dim)).max()))
    E = solve_lobpcg(3, 30, c=0.0)["E"]
    print("  3D LOBPCG (N=30): max deviation %.1e"
          % np.abs(E - separable_levels(3)).max())


def benchmark(sizes=(20, 30, 40)):
    print("\n== 2. Solver benchmark, 3D, c = 1, six lowest levels ==")
    ref = ContractedBasis(3, 16).solve(dense_limit=5000)["E"]
    print("reference:", ref)
    print("%4s %8s | %-28s | %-28s | %s"
          % ("N", "size", "Lanczos (thesis)", "LOBPCG + preconditioner",
             "contracted n=10"))
    rows = []
    for N in sizes:
        a = solve_lanczos(3, N)
        b = solve_lobpcg(3, N)
        c = ContractedBasis(3, 10).solve()
        found = [int(np.sum(np.abs(r["E"] - ref) < 1e-6)) for r in (a, b, c)]
        rows.append((N, a, b, c, found))
        print("%4d %8d | %6.1f s %5d matvecs %d/6 | %6.2f s %5d matvecs %d/6"
              " | %5.2f s %d/6"
              % (N, a["size"], a["time"], a["matvecs"], found[0],
                 b["time"], b["matvecs"], found[1], c["time"], found[2]))
    print("(k/6 = how many of the six lowest levels were found; the fifth "
          "level is doubly degenerate)")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    size = [r[1]["size"] for r in rows]
    ax.loglog(size, [r[1]["time"] for r in rows], "o-", label="Lanczos (thesis)")
    ax.loglog(size, [r[2]["time"] for r in rows], "s-",
              label="LOBPCG + preconditioner")
    ax.loglog(size, [r[3]["time"] for r in rows], "^-",
              label="contracted basis, n = 10")
    ax.set(xlabel="size of product basis $N^3$", ylabel="time (s)",
           title="3D coupled oscillators, six lowest levels")
    ax.legend()
    fig.savefig(os.path.join(OUTDIR, "solver_benchmark.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    return rows


def contracted_convergence():
    print("\n== 3. Convergence of the contracted basis with n (4D) ==")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ns = (3, 4, 5, 6, 8, 10, 12)
    for c, marker in ((1.0, "o-"), (10.0, "s-")):
        E = [ContractedBasis(4, n, c=c).solve(k=1)["E"][0] for n in ns]
        best = ContractedBasis(4, 16, c=c).solve(k=1)["E"][0]
        err = np.abs(np.array(E) - best)
        print("c = %4.1f: E0 = %.10f" % (c, best))
        for n, e in zip(ns, err):
            print("   n = %2d (size %6d): error %.1e" % (n, n ** 4, e))
        ax.semilogy(ns, np.maximum(err, 1e-16), marker, label="c = %g" % c)
    ax.set(xlabel="states kept per direction, n", ylabel="error of $E_0$",
           title="4D ground state: convergence of the contracted basis")
    ax.legend()
    fig.savefig(os.path.join(OUTDIR, "contracted_convergence.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def four_and_five_dimensions(quick=False):
    print("\n== 4. Four and five dimensions, c = 1 ==")
    for dim, n, direct in ((4, 12, (20, 30)), (5, 10, (12, 16))):
        r = ContractedBasis(dim, n).solve()
        print("%dD contracted n=%d (size %d): %.2f s, %d matvecs"
              % (dim, n, r["size"], r["time"], r["matvecs"]))
        print("    ", r["E"])
        for N in direct[:1] if quick else direct:
            d = solve_lobpcg(dim, N)
            print("%dD direct LOBPCG N=%d (size %d): %.1f s, %d matvecs, "
                  "max deviation from contracted %.1e"
                  % (dim, N, d["size"], d["time"], d["matvecs"],
                     np.abs(d["E"] - r["E"]).max()))


def spectrum_table():
    print("\n== 5. Lowest levels for D = 1..5, c = 1 ==")
    print("%2s %16s %16s %12s" % ("D", "E0 (all even)", "E1 (one odd)", "gap"))
    table = []
    for dim in range(1, 6):
        n = 12 if dim <= 4 else 10
        E0 = ContractedBasis(dim, n).solve(k=1)["E"][0]
        E1 = ContractedBasis(dim, n, parities=(1,) + (0,) * (dim - 1)
                             ).solve(k=1)["E"][0]
        table.append((dim, E0, E1))
        print("%2d %16.10f %16.10f %12.10f" % (dim, E0, E1, E1 - E0))
    return table


def coupling_scan():
    print("\n== 6. Ground state energy against coupling strength ==")
    cs = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    print("%5s" % "c" + "".join("%12s" % ("D=%d" % d) for d in range(2, 6)))
    data = {}
    for dim in range(2, 6):
        n = 10 if dim <= 4 else 8
        data[dim] = [ContractedBasis(dim, n, c=c).solve(k=1)["E"][0]
                     for c in cs]
        ax.plot(cs, data[dim], "o-", label="D = %d" % dim)
    for i, c in enumerate(cs):
        print("%5.1f" % c + "".join("%12.6f" % data[d][i] for d in range(2, 6)))
    ax.set(xlabel="coupling $c$", ylabel="ground state energy $E_0$",
           title=r"$H=\sum_j(p_j^2+q_j^4)+c\sum_{j<k}q_j^2q_k^2$")
    ax.legend()
    fig.savefig(os.path.join(OUTDIR, "ground_state_vs_coupling.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def main():
    quick = "--quick" in sys.argv
    os.makedirs(OUTDIR, exist_ok=True)
    np.set_printoptions(precision=10, linewidth=120)
    validation()
    benchmark(sizes=(20, 30) if quick else (20, 30, 40))
    contracted_convergence()
    four_and_five_dimensions(quick)
    spectrum_table()
    coupling_scan()
    print("\nFigures written to", os.path.abspath(OUTDIR))


if __name__ == "__main__":
    main()
