"""Exact arithmetic shared by the certificate generators and verifier."""
import math
import os
import sys
from fractions import Fraction as F
from pathlib import Path

if not __debug__:
    raise RuntimeError("Certificate checks require assertions; do not use Python -O or -OO")
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def scratch_path(name):
    """Return a scratch filename; create its directory only when requested."""
    if Path(name).name != name:
        raise ValueError("Scratch output must be a filename")
    directory = Path(os.environ.get("CERTIFICATE_SCRATCH", HERE.parent / "dump" / "certificate"))
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name



def add(p, q):
    out = dict(p)
    for key, value in q.items():
        out[key] = out.get(key, F(0))+value
        if not out[key]:
            del out[key]
    return out


def scale(p, s):
    return {key: value*s for key, value in p.items() if value*s}


def mul(p, q, degree=None):
    out = {}
    for (i,j), a in p.items():
        for (k,l), b in q.items():
            if degree is None or i+j+k+l <= degree:
                key = (i+k,j+l)
                out[key] = out.get(key,F(0))+a*b
    return {key: value for key,value in out.items() if value}


def bernoulli_numbers(n):
    result = [F(1)]
    for k in range(1,n+1):
        result.append(-sum((F(math.comb(k+1,j))*result[j]
                            for j in range(k)),F(0))/F(k+1))
    return result


def rising(s,n):
    return math.prod(range(s,s+n))


def power_tail_interval(s, first_omitted=8, em_start=64, em_order=16):
    """Enclose sum_{j>first_omitted} j^-s using Euler--Maclaurin.

    The periodic Bernoulli remainder uses its exact polynomial coefficient
    l1 bound on [0,1], rather than any floating-point pi or zeta constant.
    """
    b = bernoulli_numbers(2*em_order)
    center = sum((F(1,j**s) for j in range(first_omitted+1,em_start)),F(0))
    center += F(1,(s-1)*em_start**(s-1))+F(1,2*em_start**s)
    for r in range(1,em_order+1):
        center += b[2*r]*F(rising(s,2*r-1),math.factorial(2*r)*em_start**(s+2*r-1))
    bernoulli_sup = sum((math.comb(2*em_order,k)*abs(b[k])
                         for k in range(2*em_order+1)),F(0))
    remainder = bernoulli_sup*F(rising(s,2*em_order-1),
                                math.factorial(2*em_order)*em_start**(s+2*em_order-1))
    return center-remainder, center+remainder


def log_interval(value, count=48):
    """Exact rational log enclosure, via scaling and the positive atanh series."""
    assert value>0
    exponent=0
    while value<1:
        value*=2
        exponent-=1
    while value>=2:
        value/=2
        exponent+=1
    def core(u):
        lo=2*sum((u**(2*k+1)/F(2*k+1) for k in range(count)),F(0))
        return lo,lo+2*u**(2*count+1)/(F(2*count+1)*(1-u*u))
    l2=core(F(1,3))
    lv=core((value-1)/(value+1))
    if exponent>=0:
        return lv[0]+exponent*l2[0],lv[1]+exponent*l2[1]
    return lv[0]+exponent*l2[1],lv[1]+exponent*l2[0]


def arctan_bracket(q,count):
    partial=sum((F((-1)**k,(2*k+1)*q**(2*k+1)) for k in range(count)),F(0))
    nxt=F((-1)**count,(2*count+1)*q**(2*count+1))
    return min(partial,partial+nxt),max(partial,partial+nxt)
