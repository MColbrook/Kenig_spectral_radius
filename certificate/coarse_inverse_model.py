"""Exact inputs and rational inequalities for the coarse inverse estimate.

This module uses only the standard library. The analytic interpretation of
these finite calculations is described in the manuscript and README.md.
"""

from fractions import Fraction as Q
from datetime import datetime
from math import factorial, isfinite
import re

SIZE = 4096
PRECISION = 96
SERIES_RADIUS = Q(1, 2**85)
NORM_BOUND = Q(61, 100)
SQUARE_NORM_BOUND = Q(95, 1000)
MIDPOINT_BOUND = Q(93, 1000)
PAPER_SQUARED_NORM_BRACKETS = {
    "B_squared_frobenius_sum": (Q("0.36879506825017661274"), Q("0.36879506825017661275")),
    "B_square_squared_frobenius_sum": (Q("0.00878723246832160633"), Q("0.00878723246832160634")),
}
SCHEMA = "coarse-inverse-arb-v1"
PASS_STATUS = "PASS: rigorous Arb enclosure"
PI_IDENTITY = "16 atan(1/5) - 4 atan(1/239)"
ANALYTIC_SOURCE = "manuscript.tex, Appendix A: cert:target-derivative and cert:source-derivative"
ANALYTIC_NOTE = "These analytic bounds and the kernel identity are proved in the manuscript."


def require(condition, message):
    """Checks remain active under python -O."""
    if not condition:
        raise ArithmeticError(message)


def arctan_bounds(denominator, terms):
    """Alternating-series enclosure of arctan(1/denominator)."""
    require(denominator > 1 and terms > 0, "Invalid arctangent parameters")
    partial = sum((Q((-1)**k, (2*k + 1)*denominator**(2*k + 1))
                   for k in range(terms)), Q(0))
    adjacent = partial + Q((-1)**terms,
                           (2*terms + 1)*denominator**(2*terms + 1))
    return min(partial, adjacent), max(partial, adjacent)


def pi_bounds():
    """Machin's identity, with exact rational alternating-series bounds."""
    a, b = arctan_bounds(5, 32)
    c, d = arctan_bounds(239, 10)
    lower, upper = 16*a - 4*d, 16*b - 4*c
    require(Q(3) < lower < upper < Q(22, 7), "Invalid pi enclosure")
    return lower, upper


def profile(x):
    """Return the exact rational values F(x), F'(x), F''(x), 0 <= x < 1.

    F(x) = 7x/2 - 4 integral_0^x w(t) dt, with the piecewise-polynomial
    rounded indicator w in the manuscript. All breakpoints are rational.
    """
    x = Q(x)
    require(0 <= x < 1, "Profile argument outside [0, 1)")
    if x < Q(1, 16):
        u, sign = 16*x, 1
        branch = "left"
    elif x < Q(11, 16):
        return Q(7, 2)*x - 4*(Q(221, 4096) + x - Q(1, 16)), Q(-1, 2), Q(0)
    elif x < Q(13, 16):
        u, sign = 16*(x - Q(3, 4)), -1
        branch = "falling"
    elif x < Q(15, 16):
        return Q(7, 2)*x - 4*(Q(3, 4) - Q(35, 4096)), Q(7, 2), Q(0)
    else:
        u, sign = 16*(x - 1), 1
        branch = "right"
    v = u*u
    theta = u*(35 + v*(-35 + v*(21 - 5*v)))/16
    integral = v*(140 + v*(-70 + v*(28 - 5*v)))/128
    w = (1 + sign*theta)/2
    if branch == "left":
        primitive = (u + integral)/32
    elif branch == "falling":
        primitive = Q(221, 4096) + Q(10, 16) + ((u - integral)/2 + Q(221, 256))/16
    else:
        primitive = Q(3, 4) - Q(35, 4096) + ((u + integral)/2 + Q(35, 256))/16
    return Q(7, 2)*x - 4*primitive, Q(7, 2) - 4*w, -70*sign*(1-v)**3


def series_data():
    """Coefficients and geometric tail bounds on |zeta| <= 6.

    S(v) truncates S_0(v) at k=24; T(v) truncates T_0(v) at k=24.
    The first omitted terms have k=25. Consecutive absolute terms have
    ratios at most the two rational numbers below for every k >= 25.
    """
    s = [Q((-1)**k, factorial(2*k + 1)) for k in range(25)]
    t = [Q((-1)**k*2*k, factorial(2*k + 1)) for k in range(1, 25)]
    s_ratio = Q(36, 52*53)
    t_ratio = Q(26, 25)*s_ratio
    s_tail = Q(6**50, factorial(51))/(1 - s_ratio)
    t_tail = Q(50*6**48, factorial(51))/(1 - t_ratio)
    require(0 < s_tail < SERIES_RADIUS, "S series tail exceeds allowance")
    require(0 < t_tail < SERIES_RADIUS, "T series tail exceeds allowance")
    return s, t, s_tail, t_tail


def operator_bounds():
    """Exact final inequalities, conditional on the two sampled norm bounds.

    The derivative bounds 500 and 900 are analytic inputs from the paper.
    """
    interpolation_squared = (Q(500**2 + 900**2, 12) + Q(500*900, 8))/SIZE**2
    require(interpolation_squared < MIDPOINT_BOUND**2, "Midpoint bound failed")
    p = NORM_BOUND + MIDPOINT_BOUND
    p_squared = SQUARE_NORM_BOUND + MIDPOINT_BOUND*(2*NORM_BOUND + MIDPOINT_BOUND)
    require(p_squared < Q(1, 4), "Neumann-series denominator is not positive")
    inverse = (p + Q(1, 2))/(Q(1, 4) - p_squared)
    require(inverse < 40, "Coarse inverse estimate failed")
    return {
        "midpoint_error_squared_upper": interpolation_squared,
        "P_norm_upper": p,
        "P_squared_norm_upper": p_squared,
        "inverse_norm_upper": inverse,
    }


def exact_keys(value, keys, label):
    require(type(value) is dict and set(value) == set(keys), "Unexpected fields in "+label)


def read_rational(value):
    """Read a reduced exact rational; JSON decimal numbers are never accepted."""
    exact_keys(value, ("numerator", "denominator"), "rational")
    numerator, denominator = value["numerator"], value["denominator"]
    require(type(numerator) is str and re.fullmatch(r"-?(0|[1-9][0-9]*)", numerator),
            "Invalid integer numerator")
    require(type(denominator) is str and re.fullmatch(r"[1-9][0-9]*", denominator),
            "Invalid positive integer denominator")
    result = Q(int(numerator), int(denominator))
    require(str(result.numerator) == numerator and str(result.denominator) == denominator,
            "Rational is not in canonical reduced form")
    return result


def read_enclosure(value, dyadic=False):
    exact_keys(value, ("lower", "upper"), "enclosure")
    lo, hi = read_rational(value["lower"]), read_rational(value["upper"])
    require(0 <= lo <= hi, "Invalid squared-norm enclosure")
    if dyadic:
        require(all(q.denominator & (q.denominator-1) == 0 for q in (lo, hi)),
                "Arb endpoints must be dyadic")
    return lo, hi


def check_report(saved):
    """Audit the saved report and its exact arithmetic, without rerunning Arb.

    Block sums are computational output. Checking their consistency does not
    reconstruct kernel entries or certify their numerical origin; --coarse-replay
    in verify.py regenerates the matrix and its square using Arb.
    """
    keys = ("schema", "status", "grid_size", "precision_bits", "block_rows",
            "environment", "pi", "series", "B_squared_frobenius_sum",
            "B_square_squared_frobenius_sum", "blocks", "created_utc", "elapsed_seconds",
            "strict_norm_upper_bounds", "analytic_inputs", "continuous_bounds")
    exact_keys(saved, keys, "coarse certificate")
    require(saved["schema"] == SCHEMA and saved["status"] == PASS_STATUS,
            "Expected a completed rigorous Arb certificate")
    require(type(saved["grid_size"]) is int and saved["grid_size"] == SIZE, "Wrong matrix size")
    require(type(saved["precision_bits"]) is int and saved["precision_bits"] >= PRECISION,
            "Insufficient precision")
    width = saved["block_rows"]
    require(type(width) is int and 0 < width <= SIZE, "Invalid row-block size")
    env = saved["environment"]
    exact_keys(env, ("python", "python_flint", "flint", "platform", "flint_threads"), "environment")
    require(env["python_flint"] == "0.9.0", "Unexpected python-flint version")
    require(all(type(env[k]) is str and env[k].strip() for k in ("python", "flint", "platform")),
            "Missing execution metadata")
    require(type(env["flint_threads"]) is int and env["flint_threads"] > 0, "Invalid thread count")
    require(type(saved["created_utc"]) is str, "Missing creation time")
    created = datetime.fromisoformat(saved["created_utc"])
    require(created.utcoffset() is not None and created.utcoffset().total_seconds() == 0,
            "Creation time must be UTC")
    elapsed = saved["elapsed_seconds"]
    require(type(elapsed) in (int, float) and isfinite(elapsed) and elapsed >= 0,
            "Invalid elapsed time")  # Timing metadata is not a proof input.
    pi = saved["pi"]
    exact_keys(pi, ("lower", "upper", "identity", "terms"), "pi enclosure")
    require(pi["identity"] == PI_IDENTITY and pi["terms"] == [32, 10]
            and all(type(v) is int for v in pi["terms"]), "Wrong Machin-series metadata")
    require((read_rational(pi["lower"]), read_rational(pi["upper"])) == pi_bounds(),
            "Pi enclosure disagrees with the exact alternating series")
    series = saved["series"]
    exact_keys(series, ("last_k", "zeta_modulus_upper", "S_tail_bound", "T_tail_bound",
                        "component_radius_used"), "series metadata")
    require(type(series["last_k"]) is int and series["last_k"] == 24 and
            type(series["zeta_modulus_upper"]) is int and series["zeta_modulus_upper"] == 6,
            "Wrong series parameters")
    _, _, s_tail, t_tail = series_data()
    require(read_rational(series["S_tail_bound"]) == s_tail and
            read_rational(series["T_tail_bound"]) == t_tail and
            read_rational(series["component_radius_used"]) == SERIES_RADIUS,
            "Wrong series tail enclosures")
    norm_bounds = saved["strict_norm_upper_bounds"]
    expected_norms = {"B": NORM_BOUND, "B_square": SQUARE_NORM_BOUND, "P_minus_B": MIDPOINT_BOUND}
    exact_keys(norm_bounds, expected_norms, "norm thresholds")
    require({k: read_rational(v) for k, v in norm_bounds.items()} == expected_norms,
            "Wrong strict norm thresholds")
    analytic = saved["analytic_inputs"]
    exact_keys(analytic, ("target_derivative_bound", "source_derivative_bound", "source", "note"),
               "analytic inputs")
    require(type(analytic["target_derivative_bound"]) is int and analytic["target_derivative_bound"] == 500
            and type(analytic["source_derivative_bound"]) is int and analytic["source_derivative_bound"] == 900,
            "Wrong derivative bounds")
    require(analytic["source"] == ANALYTIC_SOURCE and analytic["note"] == ANALYTIC_NOTE,
            "Stale analytic-input metadata")
    continuous = operator_bounds()
    exact_keys(saved["continuous_bounds"], continuous, "continuous bounds")
    require({k: read_rational(v) for k, v in saved["continuous_bounds"].items()} == continuous,
            "Continuous operator bounds disagree with exact arithmetic")
    blocks = saved["blocks"]
    require(type(blocks) is list and len(blocks) == (SIZE+width-1)//width, "Wrong block count")
    norm_keys = ("B_squared_frobenius_sum", "B_square_squared_frobenius_sum")
    sums = {key: [Q(0), Q(0)] for key in norm_keys}
    for start, block in zip(range(0, SIZE, width), blocks):
        exact_keys(block, ("rows", *norm_keys), "matrix block")
        require(type(block["rows"]) is list and block["rows"] == [start, min(start+width, SIZE)]
                and all(type(v) is int for v in block["rows"]), "Missing, repeated or unordered matrix rows")
        for key in norm_keys:
            lo, hi = read_enclosure(block[key], dyadic=True)
            sums[key][0] += lo
            sums[key][1] += hi
    for key, bound in zip(norm_keys, (NORM_BOUND, SQUARE_NORM_BOUND)):
        require(tuple(sums[key]) == read_enclosure(saved[key], dyadic=True),
                "Stored aggregate disagrees with exact block sums")
        require(sums[key][1] < bound**2, "Sampled norm threshold failed")
        printed_lo, printed_hi = PAPER_SQUARED_NORM_BRACKETS[key]
        require(printed_lo < sums[key][0] <= sums[key][1] < printed_hi < bound**2,
                "Arb enclosure does not establish the exact decimal bracket printed in the manuscript: "+key)
    return {"stored_arb_block_enclosures_checked": True, "matrix_regenerated": False,
            "printed_squared_norm_brackets_checked": True,
            "rows_checked": SIZE, "blocks_checked": len(blocks),
            **{key: {"lower": str(sums[key][0]), "upper": str(sums[key][1])} for key in norm_keys},
            **{key: str(value) for key, value in continuous.items()}}
