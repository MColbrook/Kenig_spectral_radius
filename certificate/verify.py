"""Read-only exact checks of the supplied certificate.

--replay adds uncached reconstruction of every companion PV row.
--coarse-replay independently regenerates the coarse matrix and its square
using Arb. Finite-image kernels and periodic-tail actions are regenerated
by their separate generators. Only --write-report changes verification.json.
"""
import argparse
import json
import math
import sys
from fractions import Fraction as F
from pathlib import Path

if not __debug__:
    raise RuntimeError("Certificate checks require assertions; do not use Python -O or -OO")
sys.dont_write_bytecode = True
from exact_arithmetic import power_tail_interval
from coarse_inverse_model import (check_report, read_enclosure, pi_bounds, series_data,
                                  operator_bounds, PRECISION, SIZE, require)

HERE = Path(__file__).resolve().parent
DATA_FILES = [
    "pair_rational_approximants.json",
    "periodic_squared_resolvent_certificate.json",
    "pair_exact_moments.json",
    "pair_exact_periodic_tail.json",
    "pair_pstar_residual_certificate.json",
    "pair_companion_smooth_certificate.json",
    "pair_companion_pv_certificate.json",
]


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: "+key)
        result[key] = value
    return result


def reject_nonfinite(value):
    raise ArithmeticError("Nonfinite JSON number: "+value)


def read_json(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"),
                      object_pairs_hook=unique_object, parse_constant=reject_nonfinite)


def check_scalar_envelopes():
    # Exact analytic inputs accompanying the Arb matrix calculation.
    lo, hi = pi_bounds()
    assert 3 < lo < hi and hi**2 < 10
    _, _, s_tail, t_tail = series_data()
    assert max(s_tail, t_tail) < F(1, 2**85)
    bounds = operator_bounds()
    assert bounds["midpoint_error_squared_upper"] < F(93, 1000)**2
    assert bounds["inverse_norm_upper"] == F(1203000, 32891) < 40
    # Both kernel derivatives, using only 3 < pi and pi^2 < 10.
    m2, m3, slope, oscillation = F(70), F(6720), F(4), F(3, 2)
    remote_remainder = 40*oscillation*(1+slope)+F(112, 3)*oscillation**2
    target_derivative = m3/36+m2**2/24+((slope**2+2*slope)*10+remote_remainder)/6
    source_derivative = m3/18+m2**2/24+m2/3+((m2*oscillation+2*slope+slope**2)*10+remote_remainder)/6
    assert target_derivative == F(2969, 6) < 500
    assert source_derivative == F(5279, 6) < 900
    return {"status": "Exact series-tail and continuous-operator inequalities pass",
            "S0_tail_bound": str(s_tail), "T0_tail_bound": str(t_tail),
            "target_derivative_bound": str(target_derivative),
            "source_derivative_bound": str(source_derivative),
            **{key: str(value) for key, value in bounds.items()}}


def independent_square(row, left, right):
    """Integrate a complex power polynomial using a triangular integer Gram sum."""
    coefficients = row["coefficients"]
    denominator = row["coefficient_denominator"]
    assert type(denominator) is int and denominator > 0 and coefficients
    assert all(len(pair) == 2 and all(type(v) is int for v in pair)
               for pair in coefficients)
    common = math.lcm(*range(1, 2*len(coefficients)))
    total = 0
    for i, (a, b) in enumerate(coefficients):
        for j in range(i, len(coefficients)):
            if (i+j) % 2:
                continue
            c, d = coefficients[j]
            total += (1 if i == j else 2)*(a*c+b*d)*(common//(i+j+1))
    squared = (right-left)*F(total, denominator*denominator*common)
    assert squared >= 0
    return squared


def positive(v):
    return v[0] > 0 and v[0]*v[1]-v[2]**2-v[3]**2 > 0


def matvec(matrix, vector):
    return [sum((x*y for x, y in zip(row, vector)), F(0)) for row in matrix]


def check_witness_and_moments(witness, saved, companion):
    """Reconstruct moments directly in the saved Chebyshev representation."""
    assert witness["panels"] == 64 and witness["order"] == 20 and witness["scale_bits"] == 64
    raw = witness["integers"]
    assert len(raw) == 64 and all(len(panel) == 20 for panel in raw)
    assert all(len(row) == 4 and all(len(pair) == 2 and all(type(v) is int for v in pair)
               for pair in row) for panel in raw for row in panel)
    means = [[F(*v) for v in pair] for pair in witness["means"]]
    scale = 1 << 64
    denominator = math.lcm(scale, *(v.denominator for pair in means for v in pair))
    coefficients = [[[[v*(denominator//scale) for v in pair] for pair in row]
                     for row in panel] for panel in raw]
    for panel in coefficients:
        for channel in range(4):
            for part in range(2):
                panel[0][channel][part] -= int(means[channel][part]*denominator)
        panel[0][0][0] += denominator
    for panel in range(64):
        for channel in range(4):
            for part in range(2):
                right = sum(coefficients[panel][k][channel][part] for k in range(20))
                left = sum((-1)**k*coefficients[(panel+1)%64][k][channel][part]
                           for k in range(20))
                assert left == right
    for channel in range(2):
        for part in range(2):
            mean = sum((F(coefficients[p][k][channel][part], denominator*64*(1-k*k))
                        for p in range(64) for k in range(0, 20, 2)), F(0))
            assert mean == (1 if channel == 0 and part == 0 else 0)
    moment_lcm = math.lcm(*(abs(1-k*k) for k in range(0, 39, 2)))
    factors = [[0 if (i+j)%2 else moment_lcm//(1-(i+j)**2)+moment_lcm//(1-(i-j)**2)
                for j in range(20)] for i in range(20)]

    def gram(a, b, panels):
        re = im = 0
        for p in panels:
            for i in range(20):
                ar, ai = coefficients[p][i][a]
                for j in range(20):
                    br, bi = coefficients[p][j][b]
                    factor = factors[i][j]
                    re += factor*(ar*br+ai*bi)
                    im += factor*(ar*bi-ai*br)
        den = 128*moment_lcm*denominator*denominator
        return F(re, den), F(im, den)

    source_squared = [gram(c, c, range(64))[0] for c in range(2)]
    assert all(0 <= value < 16 for value in source_squared)
    g = [[gram(a, b, range(5, 43)) for b in range(4)] for a in range(4)]
    columns = [
        [g[0][0][0], g[1][1][0], g[0][1][0], g[0][1][1]],
        [g[2][2][0], g[3][3][0], g[2][3][0], g[2][3][1]],
        [2*g[0][2][0], 2*g[1][3][0], g[0][3][0]+g[2][1][0], g[0][3][1]+g[2][1][1]],
        [-2*g[0][2][1], -2*g[1][3][1], -g[0][3][1]+g[2][1][1], g[0][3][0]-g[2][1][0]],
    ]
    k = [list(row) for row in zip(*columns)]
    assert k == [[F(*v) for v in row] for row in saved["K"]]
    signs = [1, 1, 1, -1]
    a = [[sum((k[i][r]*signs[r]*k[r][j]*signs[j] for r in range(4)), F(0))
          for j in range(4)] for i in range(4)]
    assert a == [[F(*v) for v in row] for row in saved["A"]]
    ident = [F(1), F(1), F(0), F(0)]
    x = [F(1, 3), F(37, 40), F(0), F(37, 200)]
    assert positive([v-u/4 for v, u in zip(x, ident)])
    assert positive([u-v for v, u in zip(x, ident)])
    ki = matvec(k, ident)
    assert positive([F(7, 2)*u-v for u, v in zip(ident, ki)])
    assert F(*saved["Frobenius_squared_integral"]) == ki[0]+ki[1] < 4
    ax = matvec(a, x)
    assert positive([v-F(21, 20)*w-u/50 for v, w, u in zip(ax, x, ident)])
    # Recheck every density-seam jump and the true three-panel Hilbert bound.
    chebyshev = [[1], [0, 1]]
    for k in range(2, 20):
        polynomial = [0]*(k+1)
        for j, value in enumerate(chebyshev[-1]):
            polynomial[j+1] += 2*value
        for j, value in enumerate(chebyshev[-2]):
            polynomial[j] -= value
        chebyshev.append(polynomial)
    powers = [[[[sum((F(coefficients[p][k][ch][part]*chebyshev[k][j], denominator)
                              for k in range(j, 20)), F(0)) for j in range(20)]
                for part in range(2)] for ch in range(2)] for p in range(64)]
    seam_omega = [F(0), F(0)]
    eta = F(1, 4096)
    for seam in range(64):
        for channel in range(2):
            omega = F(0)
            for part in range(2):
                left = powers[(seam-1)%64][channel][part]
                right = powers[seam][channel][part]
                for degree in range(20):
                    jump = 128**degree*sum(((left[k]-(-1)**(k-degree)*right[k])*math.comb(k, degree)
                                           for k in range(degree, 20)), F(0))
                    if degree == 0:
                        assert jump == 0
                    omega += abs(jump)*eta**degree
            seam_omega[channel] = max(seam_omega[channel], omega)
    seam_squared = [128*eta*(12*w/6)**2 for w in seam_omega]
    assert all(v < F(1, 10**16) for v in seam_squared)
    assert all(12*w < F(1, 10**6) for w in seam_omega)
    assert seam_squared == [F(*v) for v in companion["seam_discard_error_squared"]]
    true_h = []
    for channel in range(2):
        sup = max(sum((abs(v) for part in p[channel] for v in part), F(0)) for p in powers)
        derivative = max(sum((128*k*abs(v) for part in p[channel] for k, v in enumerate(part)), F(0))
                         for p in powers)
        true_h.append(F(3, 64)*derivative+sup)
    assert max(true_h) < 19
    assert true_h == [F(*v) for v in companion["true_three_panel_H_uniform_bounds"]]
    return {"all_four_functions_continuous_periodic": True,
            "first_two_density_means_zero": True,
            "source_L2_squared": [[v.numerator, v.denominator] for v in source_squared],
            "source_L2_squared_decimal": [float(v) for v in source_squared],
            "K_and_A_reconstructed_from_witness": True,
            "all_density_seam_and_Hilbert_bounds_reconstructed": True,
            "positive_matrix_inequalities": True}


def check_coarse(saved):
    return check_report(saved)


def replay_coarse(saved, threads, precision, block_rows):
    """Regenerate interval enclosures without replacing any supplied data.

    Different supported platforms or settings may produce different outward
    endpoints. Both runs must pass their own exact bounds, and their aggregate
    enclosures must overlap; matching partitions are also compared blockwise.
    """
    import periodic_squared_resolvent_certificate as coarse
    coarse.ctx.prec = precision if precision is not None else saved["precision_bits"]
    coarse.ctx.threads = threads
    width = block_rows if block_rows is not None else saved["block_rows"]
    rebuilt = coarse.run(SIZE, width, False)
    result = check_report(rebuilt)
    keys = ("B_squared_frobenius_sum", "B_square_squared_frobenius_sum")

    def compare(left, right):
        for key in keys:
            old_lo, old_hi = read_enclosure(left[key], dyadic=True)
            new_lo, new_hi = read_enclosure(right[key], dyadic=True)
            require(max(old_lo, new_lo) <= min(old_hi, new_hi),
                    "Recomputed and supplied Arb enclosures are disjoint: "+key)

    compare(saved, rebuilt)
    blockwise = rebuilt["block_rows"] == saved["block_rows"]
    if blockwise:
        for old, new in zip(saved["blocks"], rebuilt["blocks"]):
            compare(old, new)
    result.update(matrix_regenerated=True, environment=rebuilt["environment"],
                  precision_bits=rebuilt["precision_bits"], block_rows=rebuilt["block_rows"],
                  elapsed_seconds=rebuilt["elapsed_seconds"],
                  aggregate_enclosures_overlap=True, block_enclosures_compared=blockwise)
    return result


def check_tail(saved):
    assert saved["panels"] == 64 and saved["scale_bits"] == 80
    assert saved["retained_line_images"] == [-8, 8]
    assert saved["retained_tail_odd_powers"] == 17
    outputs = saved["outputs"]
    assert len(outputs) == 64
    maximum = F(0)
    degree = 0
    for panel in outputs:
        assert len(panel) == 4
        assert {row["channel"] for row in panel} == {"Pstar_0", "Pstar_1", "R_0", "R_1"}
        for row in panel:
            error = F(*row["polynomial_error_bound"])
            assert 0 <= error < F(1, 10**12)
            maximum = max(maximum, error)
            assert row["coefficients"] and all(len(pair) == 2 and all(type(v) is int for v in pair)
                                              for pair in row["coefficients"])
            degree = max(degree, len(row["coefficients"])-1)
    assert saved["maximum_degree"] == degree
    coefficient_error = F(0)
    for k in range(9):
        lo, hi = power_tail_interval(2*k+2)
        midpoint = F(round((lo+hi)/2*(1 << 100)), 1 << 100)
        coefficient_error += max(abs(midpoint-lo), abs(midpoint-hi))*F(5, 2)**(2*k+1)*F(4, 3)
    geometric_error = F(4, 3)*F(5, 16)**19/(19*(1-F(25, 324)))
    assert geometric_error+coefficient_error < F(1, 10**10)
    error = 4*(geometric_error+coefficient_error)+maximum
    assert error == F(*saved["uniform_tail_action_error"]) < F(1, 10**9)
    return error


def partition(records, count):
    assert len(records) == count
    cursor = F(0)
    for row in records:
        left, right = map(F, row["target"])
        assert left == cursor and left < right <= 1
        cursor = right
    assert cursor == 1


def finite_records(saved):
    rows = saved["records"]
    partition(rows, 128)
    assert saved["target_intervals"] == len(rows)
    for row in rows:
        assert row["needs_subdivision"] is False
        assert 0 <= row["maximum_degree"] <= 48
        assert 0 <= F(*row["max_kernel_error"]) <= F(1, 10**9)
        assert 0 <= F(*row["pi_action_error"]) < F(1, 10**15)
        assert len(row["channels"]) == 2
        assert row["source_subdivisions"] == 0
        assert row["accepted_image_boxes"] == 128*17
    boxes = sum(row["accepted_image_boxes"] for row in rows)
    assert saved["accepted_image_boxes"] == boxes == 128*128*17
    return rows


def squared_norms(rows):
    total = [F(0), F(0)]
    for row in rows:
        left, right = map(F, row["target"])
        assert len(row["channels"]) == 2
        for channel in range(2):
            squared = independent_square(row["channels"][channel], left, right)
            assert squared == F(*row["channels"][channel]["L2_squared"])
            total[channel] += squared
    return total


def below_threshold(squared, error, threshold):
    # Both signs are essential: a negative margin cannot be squared safely.
    if not (0 <= error < threshold and 0 <= squared < (threshold-error)**2):
        raise ArithmeticError("Continuous residual threshold failed")


def check_residuals(pstar, smooth, companion, tail_error):
    p_rows = finite_records(pstar)
    s_rows = finite_records(smooth)
    eta = F(1, 2**14)
    m5 = F(55050240)
    p_error = 68*F(1, 10**9)+4*eta*(m5*eta**3/72+m5*eta**4/4)+tail_error+F(1, 10**15)
    assert p_error == F(*pstar["analytic_action_error_bound"])
    p_squared = squared_norms(p_rows)
    assert p_squared == [F(*v) for v in pstar["polynomial_residual_norms_squared"]]
    for squared in p_squared:
        below_threshold(squared, p_error, F(7, 10**8))
    smooth_error = 68*F(1, 10**9)+4*m5*eta**4/72+tail_error+F(1, 10**15)
    assert smooth_error == F(*smooth["action_L2_error_bound"]) < F(7, 10**8)
    for row in s_rows:
        left, right = map(F, row["target"])
        panel = min(int((left+right)*32), 63)
        assert row["original_target_panel"] == panel
        assert F(panel, 64) <= left < right <= F(panel+1, 64)
        for channel in row["channels"]:
            independent_square(channel, left, right)  # Also check the full coefficient representation.
    rows = companion["records"]
    partition(rows, 784)
    assert companion["target_intervals"] == len(rows)
    parent = 0
    for row in rows:
        left, right = map(F, row["target"])
        while F(s_rows[parent]["target"][1]) <= left:
            parent += 1
        oldleft, oldright = map(F, s_rows[parent]["target"])
        assert oldleft <= left < right <= oldright
        assert row["smooth_parent_target"] == s_rows[parent]["target"]
        assert row["original_target_panel"] == min(int((left+right)*32), 63)
        assert 0 <= F(*row["A_uniform_error"]) < F(1, 10**10)
        assert all(0 <= F(*e) <= F(1, 10**9) for e in row["H_uniform_errors"])
        assert all(0 <= F(*e) for e in row["final_coefficient_rounding"])
    r_squared = squared_norms(rows)
    assert r_squared == [F(*v) for v in companion["polynomial_residual_norms_squared"]]
    h_error = [max(F(*row["H_uniform_errors"][ch]) for row in rows) for ch in range(2)]
    a_error = max(F(*row["A_uniform_error"]) for row in rows)
    rounding = [max(F(*row["final_coefficient_rounding"][ch]) for row in rows) for ch in range(2)]
    errors = [smooth_error+F(1, 10**8)+h_error[ch]/6+20*a_error+rounding[ch] for ch in range(2)]
    assert errors == [F(*v) for v in companion["analytic_error_bounds"]]
    assert h_error == [F(*v) for v in companion["max_unweighted_H_approximation_errors"]]
    assert a_error == F(*companion["max_A_approximation_error"])
    for squared, error in zip(r_squared, errors):
        below_threshold(squared, error, F(8, 10**8))
    return {"Pstar_target_intervals": len(p_rows), "companion_target_intervals": len(rows),
            "finite_image_boxes_per_operator": 128*128*17,
            "all_saved_polynomial_squared_norms_reconstructed": True,
            "Pstar_residual_threshold": "7/100000000",
            "companion_residual_threshold": "8/100000000",
            "Pstar_polynomial_squared_norms": [[v.numerator, v.denominator] for v in p_squared],
            "companion_polynomial_squared_norms": [[v.numerator, v.denominator] for v in r_squared],
            "Pstar_analytic_error": str(p_error),
            "companion_analytic_errors": [str(v) for v in errors]}


def replay_companion(smooth, companion):
    from pair_companion_pv_certificate import PVCalculator
    calculator = PVCalculator()  # Never load an optional preparation cache.
    parent = 0
    rows = companion["records"]
    for count, row in enumerate(rows, 1):
        target = tuple(map(F, row["target"]))
        while F(smooth["records"][parent]["target"][1]) <= target[0]:
            parent += 1
        rebuilt = calculator.assemble(smooth["records"][parent], target)
        assert json.loads(json.dumps(rebuilt)) == row, ("uncached companion row", target)
        if count % 64 == 0:
            print("Uncached companion rows checked", count, "of", len(rows), flush=True)
    assert calculator.seam_squared == [F(*v) for v in companion["seam_discard_error_squared"]]
    assert calculator.h_true_bound == [F(*v) for v in companion["true_three_panel_H_uniform_bounds"]]
    assert all(value < F(1, 10**16) for value in calculator.seam_squared)
    assert max(calculator.h_true_bound) < 19
    assert all(12*w < F(1, 10**6) for w in calculator.seam_omega)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true", help="reconstruct all 784 companion PV rows without a cache")
    parser.add_argument("--coarse-replay", action="store_true", help="regenerate the full 4096-point Arb matrix and square; requires python-flint")
    parser.add_argument("--coarse-threads", type=int, default=4, help="FLINT threads for --coarse-replay (default: 4)")
    parser.add_argument("--coarse-precision", type=int, help="Arb bits for --coarse-replay (default: saved precision; minimum 96)")
    parser.add_argument("--coarse-block-rows", type=int, help="rows per Arb product block (default: saved block size)")
    parser.add_argument("--write-report", action="store_true", help="save the completed checks to verification.json; no input certificate is changed")
    args = parser.parse_args()
    if args.coarse_threads <= 0 or (args.coarse_precision is not None and args.coarse_precision < PRECISION):
        parser.error("The coarse replay requires a positive thread count and at least 96 bits")
    if args.coarse_block_rows is not None and not 0 < args.coarse_block_rows <= SIZE:
        parser.error("The coarse replay block size must be between 1 and 4096")
    if not args.coarse_replay and (args.coarse_precision is not None or args.coarse_block_rows is not None
                                  or args.coarse_threads != 4):
        parser.error("Coarse replay settings require --coarse-replay")
    data = {name: read_json(name) for name in DATA_FILES}
    scalar = check_scalar_envelopes()
    witness = check_witness_and_moments(data[DATA_FILES[0]], data[DATA_FILES[2]], data[DATA_FILES[6]])
    coarse = check_coarse(data[DATA_FILES[1]])
    tail_error = check_tail(data[DATA_FILES[3]])
    residuals = check_residuals(data[DATA_FILES[4]], data[DATA_FILES[5]], data[DATA_FILES[6]], tail_error)
    replay_count = replay_companion(data[DATA_FILES[5]], data[DATA_FILES[6]]) if args.replay else 0
    coarse_replay = (replay_coarse(data[DATA_FILES[1]], args.coarse_threads, args.coarse_precision,
                                 args.coarse_block_rows) if args.coarse_replay else None)
    assert 2*(F(20, 10**6)**2+F(205, 10**6)**2) < F(1, 3000)**2
    assert (4+F(1, 3000))/3000 < F(1, 700)
    assert (7+F(1, 700))/700 < F(1, 99)
    margin = F(1, 50)-F(1, 99)
    assert margin == F(49, 4950) > 0
    report = {"status": "PASS",
              "scope": "Stored exact arithmetic and analytic error assembly; optional reconstructions are identified below",
              "finite_image_kernels_regenerated": False, "periodic_tail_actions_regenerated": False,
              "witness_and_moments": witness, "scalar_envelopes": scalar,
              "coarse_stored_data": coarse, "coarse_replay": coarse_replay,
              "tail_action_error": str(tail_error), "continuous_residuals": residuals,
              "uncached_companion_rows_replayed": replay_count,
              "actual_A_X_minus_21_over_20_X_strict_margin": str(margin)}
    if args.write_report:
        (HERE/"verification.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "coarse_arb_matrix_replayed": coarse_replay is not None,
                      "uncached_companion_rows_replayed": replay_count,
                      "strict_matrix_margin": str(margin), "report_written": args.write_report}, indent=2))


if __name__ == "__main__":
    main()
