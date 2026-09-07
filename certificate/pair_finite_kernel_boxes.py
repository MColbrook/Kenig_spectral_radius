"""Exact certificates for rational finite-image P* kernel polynomials.

Floating Taylor coefficients only propose dyadic polynomials; every
accepted error is checked by the exact integer residual N-D*S.
"""
import exact_arithmetic  # Enforce the execution contract before any checks.
import math
from fractions import Fraction as F
from pathlib import Path
import numpy as np

HERE=Path(__file__).parent
from exact_arithmetic import add,mul,scale
from pair_exact_periodic_tail import geometry_coefficients

BITS=80
QMAX=48
IMAGE_ERROR=F(1,10**9)


def restrict_polynomial(co,a,b):
    """Exact coefficients of p(a+b*t)."""
    out=[F(0)]*len(co)
    for j,c in enumerate(co):
        for k in range(j+1):
            out[k]+=c*math.comb(j,k)*a**(j-k)*b**k
    while len(out)>1 and not out[-1]: out.pop()
    return out


def target_grid():
    breaks={F(j,64) for j in range(65)}
    for c in (F(1,16),F(11,16),F(13,16),F(15,16)):
        for k in range(7,15):
            breaks.add(c-F(1,2**k)); breaks.add(c+F(1,2**k))
    br=sorted(breaks)
    return list(zip(br[:-1],br[1:]))


class Geometry:
    def __init__(self):
        q,dq=geometry_coefficients()
        self.f=[[F(a+2*b,2*dq) for a,b in panel] for panel in q]
        self.cache={}

    def interval(self,left,right):
        key=(left,right)
        if key not in self.cache:
            center=(left+right)/2
            half=(right-left)/2
            original=min(int(center*64),63)
            co=restrict_polynomial(self.f[original],128*center-2*original-1,128*half)
            slope=[(k+1)*co[k+1]/half for k in range(len(co)-1)]
            self.cache[key]=(co,slope,center,half)
        return self.cache[key]


def embed(co,axis):
    return {(k,0) if axis==0 else (0,k):v for k,v in enumerate(co) if v}


def divide_linear_y(numerator,r):
    """Exact polynomial division by r00+r10*u+r01*v; None on remainder."""
    work=dict(numerator); quotient={}
    constant=r.get((0,0),F(0)); target=r.get((1,0),F(0)); source=r[(0,1)]
    highest=max((j for i,j in work),default=0)
    for j in range(highest,0,-1):
        row=[(i,val) for (i,jj),val in work.items() if jj==j]
        for i,val in row:
            coefficient=val/source
            quotient[(i,j-1)]=quotient.get((i,j-1),F(0))+coefficient
            del work[(i,j)]
            for key,v in [((i,j-1),-constant*coefficient),((i+1,j-1),-target*coefficient)]:
                work[key]=work.get(key,F(0))+v
                if not work[key]: del work[key]
    return None if work else {k:v for k,v in quotient.items() if v}


def patch_corner(target,source,image):
    if image: return False
    eta=F(1,2**14)
    a,b=target;c,d=source
    for corner in (F(1,16),F(11,16),F(13,16),F(15,16)):
        if ((corner-eta<=a<b<=corner and corner<=c<d<=corner+eta)
            or (corner-eta<=c<d<=corner and corner<=a<b<=corner+eta)):
            return True
    return False


def image_polynomials(geometry,target,source,image):
    if patch_corner(target,source,image):
        return {},{(0,0):F(1)},F(1),'patched_affine_corner'
    fx,sx,x0,ax=geometry.interval(*target)
    fy,_,y0,ay=geometry.interval(*source)
    r={(0,0):y0-x0-image,(1,0):-ax,(0,1):ay}
    delta=add(add(embed(fy,1),scale(embed(fx,0),-1)),{(0,0):-F(image,2)})
    q=divide_linear_y(delta,r) if abs(image)<=1 else None
    if q is not None:
        b=divide_linear_y(add(q,scale(embed(sx,0),-1)),r)
        assert b is not None
        return b,add({(0,0):F(1)},mul(q,q)),F(1),'factored'
    numerator=add(delta,scale(mul(embed(sx,0),r),-1))
    denominator=add(mul(r,r),mul(delta,delta))
    rlow=source[0]-target[1]-image
    rhigh=source[1]-target[0]-image
    distance=max(rlow,-rhigh,F(0))
    if not distance:
        raise ValueError(('Unremoved diagonal',target,source,image))
    return numerator,denominator,distance*distance,'separated'


def certify_polynomial(numerator,denominator,real_lower,error=IMAGE_ERROR):
    if not numerator:
        return {'coefficients':{},'degree':0,'error':F(0),'residual_numerator':0,'lower':F(1)}
    common=math.lcm(*(v.denominator for v in list(numerator.values())+list(denominator.values())))
    ni={key:int(val*common) for key,val in numerator.items()}
    di={key:int(val*common) for key,val in denominator.items()}
    d0=di[(0,0)]
    assert d0>0
    dl1=d0-sum(abs(v) for key,v in di.items() if key!=(0,0))
    lower=max(F(dl1),real_lower*common)
    dn=[(i,j,float(F(v,d0))) for (i,j),v in di.items() if (i,j)!=(0,0)]
    nn={key:float(F(v,d0)) for key,v in ni.items()}
    jet=np.zeros((QMAX+1,QMAX+1))
    candidate={}
    powerscale=1<<BITS
    for total in range(QMAX+1):
        for i in range(total+1):
            j=total-i
            value=nn.get((i,j),0.)
            for a,b,c in dn:
                if a<=i and b<=j: value-=c*jet[i-a,j-b]
            if not math.isfinite(value): return None
            jet[i,j]=value
            integer=round(value*powerscale)
            if integer: candidate[(i,j)]=integer
        if total not in (2,4,6,8,12,16,24,32,40,48): continue
        residual={key:val*powerscale for key,val in ni.items()}
        for (i,j),d in di.items():
            for (k,l),s in candidate.items():
                key=(i+k,j+l)
                residual[key]=residual.get(key,0)-d*s
        total_residual=sum(abs(v) for v in residual.values())
        bound=F(total_residual,6*powerscale)/lower
        if bound<=error:
            return {'coefficients':dict(candidate),'degree':total,'error':bound,
                    'residual_numerator':total_residual,'lower':lower}
    return None


def certify_image(geometry,target,source,image,error=IMAGE_ERROR):
    n,d,lower,kind=image_polynomials(geometry,target,source,image)
    result=certify_polynomial(n,d,lower,error)
    if result is not None: result['kind']=kind
    return result


def companion_image_polynomials(geometry,target,source,image):
    fx,sx,x0,ax=geometry.interval(*target)
    fy,_,y0,ay=geometry.interval(*source)
    target_panel=min(int(x0*64),63)
    source_panel=min(int(y0*64),63)-64*image
    near=source_panel in (target_panel-1,target_panel,target_panel+1)
    if near and patch_corner(target,source,image):
        return {},{(0,0):F(1)},F(1),'companion_corner_patch'
    r={(0,0):y0-x0-image,(1,0):-ax,(0,1):ay}
    delta=add(add(embed(fy,1),scale(embed(fx,0),-1)),{(0,0):-F(image,2)})
    denominator=add(mul(r,r),mul(delta,delta))
    rlow=source[0]-target[1]-image
    rhigh=source[1]-target[0]-image
    distance=max(rlow,-rhigh,F(0))
    if not near:
        assert distance>=F(1,64)
        return r,denominator,distance*distance,'companion_remote'
    s=embed(sx,0)
    target_den=add({(0,0):F(1)},mul(s,s))
    q=divide_linear_y(delta,r)
    if q is not None:
        b=divide_linear_y(add(q,scale(s,-1)),r)
        assert b is not None
        numerator=scale(mul(b,add(q,s)),-1)
        den=mul(add({(0,0):F(1)},mul(q,q)),target_den)
        return numerator,den,F(1),'companion_factored_remainder'
    if not distance: raise ValueError(('Unremoved companion corner',target,source,image))
    sr=mul(s,r)
    numerator=scale(mul(add(delta,scale(sr,-1)),add(delta,sr)),-1)
    den=mul(mul(r,denominator),target_den)
    sign=1 if r[(0,0)]>0 else -1
    return scale(numerator,sign),scale(den,sign),distance**3,'companion_separated_remainder'


def certify_companion_image(geometry,target,source,image,error=IMAGE_ERROR):
    n,d,lower,kind=companion_image_polynomials(geometry,target,source,image)
    result=certify_polynomial(n,d,lower,error)
    if result is not None: result['kind']=kind
    return result
