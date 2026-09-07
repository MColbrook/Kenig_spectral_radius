# Computational certificate

This directory contains the computational certificate for *A counterexample
to Kenig's conjecture for the Laplace double-layer operator* by Matthew J.
Colbrook and Siavash Sadeghi. It contains the exact polynomial witness, all
six computational data files, their generators, and a consolidated verifier. All mathematical
data needed for reproduction are in this directory. The analytic kernel
identities, continuous operator estimates and perturbation arguments in
the paper are part of the proof.

The directory is supplied as `certificate/` inside the
[GitHub repository](https://github.com/MColbrook/Kenig_spectral_radius).
All commands below run from the repository root, which contains this
subdirectory. The paper is distributed separately; its TeX source and PDF
are not inputs to the programs.

The coarse inverse bound uses rigorous Arb ball arithmetic within FLINT.
The remaining acceptance calculations use arbitrary-precision integers and
rationals. This is a computer-assisted proof; no proof-assistant verification
is claimed.

## Installation

The package was tested with Python 3.12.14, NumPy 2.3.5 and python-flint 0.9.0.
The tested python-flint wheel includes FLINT 3.6.0. Use an isolated Python
environment, then install the pinned dependencies from the repository root:

```text
python -m pip install -r certificate/requirements.txt
```

The default stored-data check needs only the Python standard library. NumPy
is needed for the polynomial generators and principal-value replay;
python-flint is needed for the coarse matrix calculation. NumPy polynomial
proposals are accepted only after exact residual checks.

Run without `-O`, `-OO` or `PYTHONOPTIMIZE`. The consolidated verifier and
exact-polynomial generators reject these modes so that their mathematical
assertions remain active. The Arb calculation uses explicit checks that
remain active under optimisation as well.
The `-B` option prevents bytecode files in the certificate directory.

## Verification

From the repository root:

```text
python -B certificate/verify.py
python -B certificate/verify.py --replay --coarse-replay
```

Both commands are read-only. A failed mathematical check gives a nonzero
exit status.

The first command:

- Checks the exact Arb endpoints, all block ranges, their sums, the series
  and pi bounds, and the rational inequalities giving the continuous inverse.
- Reconstructs continuity, means, input norms and matrix moments directly
  from the exact polynomial witness.
- Integrates every saved residual polynomial again, assembles its analytic
  errors and checks the strict continuous residual and positive-matrix bounds.

The second command additionally:

- Rebuilds all 784 principal-value, coefficient and final assembly rows from
  the saved smooth-companion data, without a preparation cache, and compares
  their exact coefficients. A separate integer implementation integrates
  their squared norms.
- Recomputes every entry of the 4096-by-4096 coarse matrix and its square
  using Arb, checks the strict norm inequalities and compares the resulting
  enclosures with the supplied ones.

The options `--replay` and `--coarse-replay` may also be used separately.
Neither command regenerates the finite-image two-variable kernel enclosures
or all periodic-tail action polynomials. Use the generators below for those
calculations. Checking saved coarse subtotals alone does not establish that
they enclose the true matrix quantities; the full Arb calculation supplies
that part of the verification.

To save a verification report after the requested checks pass, add
`--write-report`. This creates or replaces `certificate/verification.json`.
The report is optional output, rather than a supplied certificate input.
Its scope fields state which computations were actually repeated.

## Coarse inverse calculation

The coarse matrix is `B_ij = K_P(x_i,x_j)/4096`, where
`x_i = (2i+1)/8192`. The exact rational profile and its first two derivatives
are converted outwards to Arb balls. Machin's identity encloses pi using
exact alternating sums. The entire-function polynomials retain terms
through `k=24`; exact geometric bounds enclose their complex remainders.
Every off-diagonal evaluation checks the series domain and excludes zero
from the divisors. The removable diagonal is evaluated separately.

Rigorous Arb matrix multiplication encloses `B^2`, in blocks of 128 rows.
The 32 block subtotals have outward dyadic endpoints. Their final sums and
acceptance comparisons use exact rational arithmetic. The supplied result
uses 96-bit working precision and implies:

| Quantity | Lower endpoint | Upper endpoint |
| --- | --- | --- |
| Squared Hilbert--Schmidt norm of B | 0.36879506825017661274 | 0.36879506825017661275 |
| Squared Hilbert--Schmidt norm of B^2 | 0.00878723246832160633 | 0.00878723246832160634 |

These decimal endpoints are exact outward bounds, checked against the
sharper dyadic endpoints in the JSON file. They give `||B||_HS < 61/100`
and `||B^2||_HS < 95/1000`. The manuscript's derivative bounds 500 and 900
give `||P-B||_HS < 93/1000`. Its continuous Neumann-series argument then yields
`||(P* + i/2)^(-1)|| < 1203000/32891 < 40`.

To regenerate the coarse data file:

```text
python -B certificate/periodic_squared_resolvent_certificate.py
```

The default output is `certificate/periodic_squared_resolvent_certificate.json`;
it is replaced only after the full calculation passes. The defaults are
96-bit precision, four FLINT threads and 128 rows per product block. The
matrix has 16,777,216 entries; allow several gigabytes of available memory.
Actual runtime and software versions appear in the result file. Change
`--threads`, `--precision` or `--block-rows` when appropriate for the machine;
precision must be at least 96 bits.

Quick checks that do not replace the full result are:

```text
python -B certificate/periodic_squared_resolvent_certificate.py --self-test
python -B certificate/periodic_squared_resolvent_certificate.py --smoke
```

The self-tests check profile boundary values, compare the polynomial kernel
with an independent Arb cotangent evaluation, and test multiplication on a
nonsymmetric matrix. The smoke calculation uses 64 grid points and is
explicitly labelled as a smoke test. It proves no continuous inverse bound.

## Regenerate all computations

The following sequence replaces all six computed data files and produces
a new verification report. The exact input witness is retained.

```text
python -B certificate/periodic_squared_resolvent_certificate.py
python -B certificate/pair_exact_moments.py
python -B certificate/pair_exact_periodic_tail.py
python -B certificate/pair_pstar_residual_certificate.py
python -B certificate/pair_companion_smooth_certificate.py
python -B certificate/pair_companion_pv_certificate.py
python -B certificate/verify.py --replay --write-report
```

Set `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to `1`
before starting Python for the NumPy-based stages. The initial step already
performs the full Arb calculation; adding `--coarse-replay` to the last step
would repeat it.

The two finite-image generators are the expensive stages. Each accepts
`--shard k --shards 4` for `k = 0, 1, 2, 3`, followed by
`--combine --shards 4` on that same generator. Execute every shard before
combining. This is an optional alternative to the respective single command
above. Temporary shards and progress logs go to `dump/certificate/` relative
to the repository root; `CERTIFICATE_SCRATCH` can select another scratch
directory. All target rows are retained in the completed data files, so no
scratch directory is an input to verification or reproduction.

## File roles

| Data file | Purpose |
| --- | --- |
| `pair_rational_approximants.json` | Exact input coefficients and rational means; never generated by verification |
| `periodic_squared_resolvent_certificate.json` | Arb enclosures for the coarse matrix and its square, with exact scalar bounds |
| `pair_exact_moments.json` | Exact witness Gram matrix and positive maps |
| `pair_exact_periodic_tail.json` | Infinite periodic-tail action polynomials |
| `pair_pstar_residual_certificate.json` | All 128 adjoint-residual target rows |
| `pair_companion_smooth_certificate.json` | All 128 smooth-companion target rows |
| `pair_companion_pv_certificate.json` | All 784 companion-residual rows |

Each computed mathematical data file has a generator with the same basename.
`coarse_inverse_model.py` contains the exact coarse inputs and report checks.
`pair_finite_kernel_boxes.py` implements the rational kernel enclosures.
`exact_arithmetic.py` supplies polynomial operations, rational logarithm and
scalar-tail intervals, the optimisation-mode guard and scratch paths.
`verify.py` consolidates the checks; `requirements.txt` pins dependencies.

## Arithmetic and reproduction

Arb's real and complex balls enclose conversion, truncation and arithmetic
errors. The matrix calculation uses rigorous `arb_mat` multiplication and
retains all radii. The Python implementation, FLINT/Arb and the execution environment
are trusted software components.

For the remaining calculations, the complete integer polynomial residual
`N-D S` determines acceptance of a proposed quotient. Floating-point proposal
quality is not a mathematical premise. All omitted tails, singular limits,
seam terms and kernel-to-action estimates are controlled by the bounds in
the manuscript and the corresponding exact arithmetic checks. Residual
thresholds are checked with a positive remaining error margin before squaring.

## Software references

- F. Johansson, *Arb: efficient arbitrary-precision midpoint-radius interval
  arithmetic*, IEEE Transactions on Computers 66 (2017), no. 8, 1281--1292.
  [Published paper](https://doi.org/10.1109/TC.2017.2690633).
- The FLINT team, *FLINT: Fast Library for Number Theory*, version 3.6.0 (2026).
  [FLINT](https://flintlib.org).
- [Arb ball semantics](https://flintlib.org/doc/arb.html) and
  [rigorous matrix arithmetic](https://flintlib.org/doc/arb_mat.html).
- [Python-FLINT documentation](https://python-flint.readthedocs.io/en/stable/).
