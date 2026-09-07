"""Exact rational checks for the explicit approximate functions only.

This certificate does not bound their continuous-operator residuals.
"""
import exact_arithmetic  # Enforce the execution contract before any checks.
import json
import math
from fractions import Fraction as F
from pathlib import Path

HERE=Path(__file__).parent


def dump_fraction(x):
    return [x.numerator,x.denominator]


def positive_test(x):
    # Coordinates (11,22,Re12,Im12); Sylvester's exact criterion.
    a,b,c,d=x
    determinant=a*b-c*c-d*d
    return {'strictly_positive':bool(a>0 and determinant>0),
            'first_diagonal':dump_fraction(a),
            'determinant':dump_fraction(determinant)}


def main():
    p=json.loads((HERE/'pair_rational_approximants.json').read_text())
    panels,order,scale=p['panels'],p['order'],1<<p['scale_bits']
    assert panels==64 and order==20 and p["scale_bits"]==64
    means=[[F(*v) for v in pair] for pair in p['means']]
    common=math.lcm(scale,*(v.denominator for pair in means for v in pair))
    coeff=[[[[int(a)*(common//scale) for a in pair] for pair in row] for row in panel]
           for panel in p['integers']]
    for panel in coeff:
        for c in range(4):
            for ri in range(2):
                panel[0][c][ri]-=int(means[c][ri]*common)
        # First matrix entry is 1+alpha.
        panel[0][0][0]+=common
    # Check continuity and exact means directly from the saved functions.
    for j in range(panels):
        for c in range(4):
            for ri in range(2):
                right=sum(coeff[j][k][c][ri] for k in range(order))
                left=sum((-1)**k*coeff[(j+1)%panels][k][c][ri] for k in range(order))
                assert left==right
    for c in range(2):
        for ri in range(2):
            mean=sum((F(coeff[j][k][c][ri],common*panels*(1-k*k))
                      for j in range(panels) for k in range(0,order,2)),F(0))
            assert mean == (1 if c==0 and ri==0 else 0)
    # Integral T_j(t)T_k(t) dx on a full panel of length1/64 is
    # [1/(1-(j+k)^2)+1/(1-(j-k)^2)]/128 when j+k is even, else0.
    moment_lcm=math.lcm(*(abs(1-k*k) for k in range(0,2*order-1,2)))
    factors=[[0 if (j+k)%2 else moment_lcm//(1-(j+k)**2)+moment_lcm//(1-(j-k)**2)
              for k in range(order)] for j in range(order)]
    gram=[[(F(0),F(0)) for b in range(4)] for a in range(4)]
    denominator=128*moment_lcm*common*common
    for a in range(4):
        for b in range(a,4):
            re=im=0
            # J=(5/64,43/64); endpoints have no integral mass.
            for panel in coeff[5:43]:
                for j in range(order):
                    ar,ai=panel[j][a]
                    for k in range(order):
                        fac=factors[j][k]
                        if not fac:
                            continue
                        br,bi=panel[k][b]
                        re+=fac*(ar*br+ai*bi)
                        im+=fac*(ar*bi-ai*br)
            gram[a][b]=(F(re,denominator),F(im,denominator))
            gram[b][a]=(gram[a][b][0],-gram[a][b][1])
    g=gram
    columns=[
        [g[0][0][0],g[1][1][0],g[0][1][0],g[0][1][1]],
        [g[2][2][0],g[3][3][0],g[2][3][0],g[2][3][1]],
        [2*g[0][2][0],2*g[1][3][0],g[0][3][0]+g[2][1][0],g[0][3][1]+g[2][1][1]],
        [-2*g[0][2][1],-2*g[1][3][1],-g[0][3][1]+g[2][1][1],g[0][3][0]-g[2][1][0]]]
    k=[[columns[j][i] for j in range(4)] for i in range(4)]
    signs=[1,1,1,-1]
    km=[[signs[i]*k[i][j]*signs[j] for j in range(4)] for i in range(4)]
    a=[[sum((k[i][v]*km[v][j] for v in range(4)),F(0)) for j in range(4)] for i in range(4)]
    witness=[F(1,3),F(37,40),F(0),F(37,200)]
    ki=[k[i][0]+k[i][1] for i in range(4)]
    defect=[sum((a[i][j]*witness[j] for j in range(4)),F(0))-F(21,20)*witness[i]
            for i in range(4)]
    lower=[defect[0]-F(1,50),defect[1]-F(1,50),defect[2],defect[3]]
    normtest=positive_test([F(7,2)-ki[0],F(7,2)-ki[1],-ki[2],-ki[3]])
    witness_test=positive_test(lower)
    trace=ki[0]+ki[1]
    assert trace<4 and normtest['strictly_positive'] and witness_test['strictly_positive']
    assert positive_test(witness)['strictly_positive']
    assert positive_test([1-witness[0],1-witness[1],-witness[2],-witness[3]])['strictly_positive']
    result={'status':'PASS: exact polynomial moments (continuous residuals are checked separately)',
            'all_functions_exactly_continuous_periodic':True,'first_two_exactly_zero_mean':True,
            'K':[[dump_fraction(x) for x in row] for row in k],
            'A':[[dump_fraction(x) for x in row] for row in a],
            'Frobenius_squared_integral':dump_fraction(trace),'Frobenius_squared_integral_less_than_4':True,
            'K_I_norm_less_than_7_over_2':normtest,
            'A_X_minus_21_over_20_X_minus_I_over_50_positive':witness_test,
            'decimal_diagnostics':{'Frobenius_squared_integral':float(trace),'K_I':[float(x) for x in ki],
                                   'defect':[float(x) for x in defect]}}
    (HERE/'pair_exact_moments.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'status':result['status'],**result['decimal_diagnostics']},indent=2))


if __name__=='__main__':
    main()
