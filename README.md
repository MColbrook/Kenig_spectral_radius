# A counterexample to Kenig's conjecture for the Laplace double-layer operator

This repository contains the computational certificate accompanying the
paper *A counterexample to Kenig's conjecture for the Laplace double-layer
operator* by
**Matthew J. Colbrook and Siavash Sadeghi**.

The paper concerns the Laplace double-layer operator in the layer-potential
solution of boundary value problems on Lipschitz domains. It disproves
Kenig's 1994 spectral-radius conjecture and shows that the associated
classical Neumann series need not converge in operator norm.

The certificate establishes the local operator bounds and positive matrix
inequality used in the paper's construction. It includes the exact
polynomial inputs, computed data, verification programs and generators.
The analytic arguments that connect these calculations to the spectral
counterexample are given in the paper, which is distributed separately.

## Repository contents

```text
README.md
certificate/
    README.md
    requirements.txt
    verify.py
    ... certificate data and generating scripts
```

The [certificate documentation](certificate/README.md) explains every file,
the mathematical checks, dependencies and complete reproduction procedure.
All mathematical data required by the programs are in `certificate/`.

## Check the supplied data

Run the following command from the repository root, using Python 3.12:

```text
python -B certificate/verify.py
```

This check uses only the Python standard library. It checks the stored
interval bounds, reconstructs the exact input moments and residual norms,
and verifies the final inequalities. A successful run reports `"status":
"PASS"`; a failed check exits with a nonzero status. It does not modify the
certificate files.

## Recompute the principal-value assembly and coarse inverse bound

In an isolated Python environment, install the
[pinned dependencies](certificate/requirements.txt), then run:

```text
python -m pip install -r certificate/requirements.txt
python -B certificate/verify.py --replay --coarse-replay
```

In addition to the stored-data checks, this reconstructs all 784
principal-value assembly rows and recomputes the full 4096-by-4096 coarse
matrix and its square with rigorous FLINT/Arb interval enclosures. Allow
several gigabytes of available memory for the coarse calculation.

These options do not regenerate the two-dimensional finite-image kernel
enclosures or the periodic-tail polynomials. The
[complete regeneration instructions](certificate/README.md#regenerate-all-computations)
give the commands for those calculations as well.

Run the programs without `-O`, `-OO` or `PYTHONOPTIMIZE`, so that all
mathematical assertions remain active. The `-B` option prevents creation of
Python bytecode files in the certificate directory.

## Validation

The coarse inverse calculation uses rigorous ball arithmetic in FLINT/Arb.
The remaining acceptance calculations use exact integers and rationals;
floating-point polynomial proposals are accepted only after exact residual
checks. The paper supplies the continuous kernel, tail and perturbation
estimates needed to interpret these finite calculations. This is a
computer-assisted proof; no proof-assistant verification is claimed.

If using this certificate, please cite the accompanying paper by Colbrook
and Sadeghi.
