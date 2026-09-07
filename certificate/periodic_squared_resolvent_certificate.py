"""Rigorous Arb enclosure of the sampled operator and its square.

Use the dependencies in certificate/requirements.txt (Python 3.12 tested).
A full run saves periodic_squared_resolvent_certificate.json beside this file.
Self-tests and smoke tests do not replace that certificate. --no-output makes
a full run read-only; --output selects an explicit JSON destination.
"""

import sys
sys.dont_write_bytecode = True

import argparse
from datetime import datetime, timezone
from fractions import Fraction as Q
import json
from pathlib import Path
import platform
from time import perf_counter
from tempfile import NamedTemporaryFile

try:
    import flint
    from flint import arb, acb, acb_poly, arb_mat, fmpq, ctx
except ImportError as exc:
    raise SystemExit("Install the dependencies in certificate/requirements.txt") from exc

from coarse_inverse_model import (SIZE, PRECISION, SERIES_RADIUS, NORM_BOUND, SQUARE_NORM_BOUND,
                   MIDPOINT_BOUND, profile, pi_bounds, series_data, operator_bounds,
                   require, read_rational, check_report, SCHEMA, PASS_STATUS,
                   PI_IDENTITY, ANALYTIC_SOURCE, ANALYTIC_NOTE)

HERE = Path(__file__).resolve().parent
CANONICAL_OUTPUT = HERE / "periodic_squared_resolvent_certificate.json"


def ball(q):
    """Enclose an exact rational without a binary64 conversion."""
    q = Q(q)
    return arb(fmpq(q.numerator, q.denominator))


def rational(q):
    return {"numerator": str(q.numerator), "denominator": str(q.denominator)}


def endpoints(value):
    """Outward dyadic endpoints; decimal printing never controls acceptance."""
    require(value.is_finite(), "Nonfinite interval")
    lower = value.lower().fmpq()
    upper = value.upper().fmpq()
    lo = Q(int(lower.numerator), int(lower.denominator))
    hi = Q(int(upper.numerator), int(upper.denominator))
    require(lo <= hi, "Reversed interval")
    return lo, hi


def enclosure(value):
    lo, hi = endpoints(value)
    return {"lower": rational(lo), "upper": rational(hi)}


class Kernel:
    def __init__(self, n):
        require(n >= 2 and n % 2 == 0, "An even grid size is required")
        self.n = n
        lo, hi = pi_bounds()
        self.pi = ball(lo).union(ball(hi))
        self.period = acb(1, ball(Q(1, 2)))
        s, t, _, _ = series_data()
        self.s = acb_poly([fmpq(q.numerator, q.denominator) for q in s])
        self.t = acb_poly([fmpq(q.numerator, q.denominator) for q in t])
        # Each component of a complex remainder of modulus < 2^-85 lies here.
        self.tail = acb(arb(0, (1, -85)), arb(0, (1, -85)))
        data = [[ball(v) for v in profile(Q(2*i+1, 2*n))] for i in range(n)]
        self.f, self.slope, self.second = map(list, zip(*data))
        self.offsets = [arb(k)/n for k in range(-n//2, n//2)]
        self.tangent = [acb(1, slope) for slope in self.slope]
        self.two_pi = 2*self.pi

    def coordinates(self, i, j):
        wrap = (i-j+self.n//2)//self.n
        r = self.offsets[i-j-wrap*self.n+self.n//2]
        delta = self.f[i]-self.f[j]-ball(Q(wrap, 2))
        return r, delta

    def entry(self, i, j):
        """Enclose K_P(x_i,x_j)/n, including the exact removable diagonal."""
        if i == j:
            value = self.second[i]/(4*self.pi*(1+self.slope[i]**2))
        else:
            r, delta = self.coordinates(i, j)
            z = self.pi*acb(r, delta)/self.period
            # Check the series domain at every sample, with outward arithmetic.
            require(abs(z) < 6, f"Series domain failed at ({i}, {j})")
            v = z*z
            s = self.s(v)+self.tail
            require(not s.contains(0), f"Sinc denominator includes zero at ({i}, {j})")
            local_denominator = self.two_pi*(r*r+delta*delta)
            require(local_denominator > 0, f"Local denominator failed at ({i}, {j})")
            regular = z*(self.t(v)+self.tail)/(self.period*s)
            value = ((delta-r*self.slope[j])/local_denominator
                     - (regular*self.tangent[j]).imag/2)
        result = value/self.n
        require(result.is_finite(), f"Nonfinite kernel entry at ({i}, {j})")
        return result

    def cot_entry(self, i, j):
        """Independent transcendental formula, used only in self-tests."""
        require(i != j, "Cotangent is singular on the diagonal")
        r, delta = self.coordinates(i, j)
        z = self.pi*acb(r, delta)/self.period
        return -(z.cot()*self.tangent[j]/self.period).imag/(2*self.n)


def self_test():
    """Boundary values, independent cotangent evaluation, and matrix orientation."""
    expected = {
        Q(0): (Q(0), Q(3, 2), Q(-70)),
        Q(1, 16): (Q(3, 1024), Q(-1, 2), Q(0)),
        Q(11, 16): (Q(-317, 1024), Q(-1, 2), Q(0)),
        Q(3, 4): (Q(-157, 512), Q(3, 2), Q(70)),
        Q(13, 16): (Q(-125, 1024), Q(7, 2), Q(0)),
        Q(15, 16): (Q(323, 1024), Q(7, 2), Q(0)),
    }
    for x, value in expected.items():
        require(profile(x) == value, f"Profile boundary mismatch at {x}")
    # The 4096-point grid includes close pairs, all five pieces and the seam.
    k = Kernel(SIZE)
    indices = [0, 1, 255, 256, 1024, 2815, 2816, 3071, 3072,
               3327, 3328, 3839, 3840, 4094, 4095]
    for i in indices:
        for j in indices:
            a = k.entry(i, j)
            if i != j:
                require(a.overlaps(k.cot_entry(i, j)), f"Cotangent comparison failed at ({i}, {j})")
            require(a.rad() < ball(Q(1, 10**18)), "Unexpectedly wide sample enclosure")
    # A nonsymmetric integer matrix distinguishes a square from A A^T.
    a = arb_mat([[1, 2, -1], [0, 3, 4], [-2, 0, 5]])
    square = a*a
    require(square.contains(arb_mat([[3, 8, 2], [-8, 9, 32], [-12, -4, 27]])),
            "Matrix multiplication self-test failed")
    require(operator_bounds()["inverse_norm_upper"] == Q(1203000, 32891),
            "Final rational estimate changed")


def build_matrix(n, block_rows):
    kernel = Kernel(n)
    matrix = arb_mat(n, n)
    blocks = []
    start_time = perf_counter()
    for start in range(0, n, block_rows):
        stop = min(start+block_rows, n)
        squared = arb(0)
        for i in range(start, stop):
            for j in range(n):
                value = kernel.entry(i, j)
                matrix[i, j] = value
                squared += value**2
        blocks.append({"rows": [start, stop], "B_squared_frobenius_sum": enclosure(squared)})
        print(f"Kernel rows {start}:{stop} enclosed ({perf_counter()-start_time:.1f}s)", flush=True)
    return matrix, blocks


def square_blocks(matrix, blocks):
    """Every block is rows(B) times B, using rigorous arb_mat multiplication.

    Reuse B as a common enclosing matrix: dependency between repeated entries
    can widen the enclosure, but cannot invalidate containment of B^2.
    Only one rectangular product is retained at a time.
    """
    n = matrix.nrows()
    start_time = perf_counter()
    for block in blocks:
        start, stop = block["rows"]
        rows = arb_mat(stop-start, n)
        for i in range(start, stop):
            for j in range(n):
                rows[i-start, j] = matrix[i, j]
        product = rows*matrix
        del rows
        squared = arb(0)
        for i in range(stop-start):
            for j in range(n):
                squared += product[i, j]**2
        block["B_square_squared_frobenius_sum"] = enclosure(squared)
        del product
        print(f"Square rows {start}:{stop} enclosed ({perf_counter()-start_time:.1f}s)", flush=True)


def aggregate(blocks, key):
    """Add outward endpoints as exact rationals, including all rows once."""
    lo = sum((read_rational(b[key]["lower"]) for b in blocks), Q(0))
    hi = sum((read_rational(b[key]["upper"]) for b in blocks), Q(0))
    require(0 <= lo <= hi, "Invalid squared-norm enclosure")
    return lo, hi


def run(n, block_rows, smoke):
    require(type(n) is int and n == (64 if smoke else SIZE), "Invalid grid size")
    require(type(block_rows) is int and 0 < block_rows <= SIZE, "Invalid block size")
    require(ctx.prec >= PRECISION and ctx.threads > 0, "Invalid Arb context")
    require(flint.__version__ == "0.9.0", "Use the pinned python-flint 0.9.0 dependency")
    started = perf_counter()
    self_test()
    print("Self-tests passed", flush=True)
    matrix, blocks = build_matrix(n, block_rows)
    square_blocks(matrix, blocks)
    del matrix
    b_lo, b_hi = aggregate(blocks, "B_squared_frobenius_sum")
    c_lo, c_hi = aggregate(blocks, "B_square_squared_frobenius_sum")
    if not smoke:
        require(n == SIZE, "The continuous estimate requires n=4096")
        require(b_hi < NORM_BOUND**2, "Sampled operator norm bound failed")
        require(c_hi < SQUARE_NORM_BOUND**2, "Sampled square norm bound failed")
    pi_lo, pi_hi = pi_bounds()
    _, _, s_tail, t_tail = series_data()
    result = {
        "schema": SCHEMA,
        "status": "SMOKE TEST ONLY" if smoke else PASS_STATUS,
        "grid_size": n,
        "precision_bits": ctx.prec,
        "block_rows": block_rows,
        "environment": {"python": sys.version, "python_flint": flint.__version__,
                        "flint": flint.__FLINT_VERSION__, "platform": platform.platform(),
                        "flint_threads": ctx.threads},
        "pi": {"lower": rational(pi_lo), "upper": rational(pi_hi),
               "identity": PI_IDENTITY, "terms": [32, 10]},
        "series": {"last_k": 24, "zeta_modulus_upper": 6,
                   "S_tail_bound": rational(s_tail), "T_tail_bound": rational(t_tail),
                   "component_radius_used": rational(SERIES_RADIUS)},
        "B_squared_frobenius_sum": {"lower": rational(b_lo), "upper": rational(b_hi)},
        "B_square_squared_frobenius_sum": {"lower": rational(c_lo), "upper": rational(c_hi)},
        "blocks": blocks,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": perf_counter()-started,
    }
    if not smoke:
        result["strict_norm_upper_bounds"] = {
            "B": rational(NORM_BOUND), "B_square": rational(SQUARE_NORM_BOUND),
            "P_minus_B": rational(MIDPOINT_BOUND),
        }
        result["analytic_inputs"] = {
            "target_derivative_bound": 500, "source_derivative_bound": 900,
            "source": ANALYTIC_SOURCE,
            "note": ANALYTIC_NOTE,
        }
        result["continuous_bounds"] = {k: rational(v) for k, v in operator_bounds().items()}
        check_report(result)
    return result


def output_path(requested, smoke, no_output):
    """Only a complete run may replace the designated coarse certificate."""
    require(not (requested and no_output), "--output and --no-output cannot be combined")
    if no_output or (smoke and requested is None):
        return None
    destination = (requested if requested is not None else CANONICAL_OUTPUT).resolve()
    require(destination.suffix == ".json", "Output must be a JSON file")
    if destination.is_relative_to(HERE):
        require(destination == CANONICAL_OUTPUT and not smoke,
                "Only a full run may replace the canonical coarse certificate; other certificate inputs are protected")
    return destination


def save_report(destination, result):
    """Replace the output only after the computation and all checks succeed."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                prefix=destination.stem+".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="JSON destination; full runs default to the canonical certificate")
    parser.add_argument("--no-output", action="store_true", help="Do not save a report (including for a full run)")
    parser.add_argument("--block-rows", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--precision", type=int, default=PRECISION)
    parser.add_argument("--smoke", action="store_true", help="Use n=64; do not claim a continuous inverse bound")
    parser.add_argument("--self-test", action="store_true", help="Only the independent consistency checks")
    args = parser.parse_args()
    require(0 < args.block_rows <= SIZE and args.threads > 0 and args.precision >= PRECISION, "Invalid run parameters")
    require(flint.__version__ == "0.9.0", "Use the pinned python-flint 0.9.0 dependency")
    require(not (args.self_test and args.smoke), "--self-test and --smoke are separate modes")
    require(not (args.self_test and args.output), "Self-tests do not produce a certificate")
    destination = output_path(args.output, args.smoke, args.no_output or args.self_test)
    ctx.prec = args.precision
    ctx.threads = args.threads
    if args.self_test:
        self_test()
        print("Self-tests passed")
        return
    result = run(64 if args.smoke else SIZE, args.block_rows, args.smoke)
    if destination is not None:
        save_report(destination, result)
        print("Saved", destination)
    print(result["status"])
    for key in ("B_squared_frobenius_sum", "B_square_squared_frobenius_sum"):
        print(key, result[key])
    if "continuous_bounds" in result:
        print("inverse_norm_upper", result["continuous_bounds"]["inverse_norm_upper"])


if __name__ == "__main__":
    main()
