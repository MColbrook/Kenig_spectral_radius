"""Exact finite-rank periodic-tail actions on the rational pair inputs.

All polynomial algebra and enclosures are rational/integer. This certifies
the tail component only, not the seventeen retained line-image integrals.
"""
import exact_arithmetic  # Enforce the execution contract before any checks.
import json
import math
import time
from fractions import Fraction as F
from pathlib import Path

HERE=Path(__file__).parent
from exact_arithmetic import power_tail_interval
from exact_arithmetic import arctan_bracket


def cmul(a,b):
    return (a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0])


def cadd(a,b):
    return (a[0]+b[0],a[1]+b[1])


def pmul(a,b):
    out=[(0,0)]*(len(a)+len(b)-1)
    for i,ai in enumerate(a):
        if ai==(0,0):
            continue
        for j,bj in enumerate(b):
            if bj!=(0,0):
                out[i+j]=cadd(out[i+j],cmul(ai,bj))
    return out


def geometry_coefficients():
    all_f=[]
    all_x=[]
    for panel in range(64):
        x0=F(2*panel+1,128)
        width=F(1,128)
        xco=[x0,width]+[F(0)]*7
        if panel<4:
            center=F(0); height=F(0); sign=-1
        elif panel<44:
            all_f.append([F(35,1024)-x0/2,-width/2]+[F(0)]*7)
            all_x.append(xco)
            continue
        elif panel<52:
            center=F(3,4); height=F(-157,512); sign=1
        elif panel<60:
            all_f.append([F(7,2)*x0-F(3037,1024),F(7,2)*width]+[F(0)]*7)
            all_x.append(xco)
            continue
        else:
            center=F(1); height=F(1,2); sign=-1
        fco=[F(0)]*9
        terms={0:height,1:F(3,2),2:sign*35,4:-sign*4480,6:sign*458752,8:-sign*20971520}
        for power,value in terms.items():
            for k in range(power+1):
                fco[k]+=value*math.comb(power,k)*(x0-center)**(power-k)*width**k
        all_f.append(fco)
        all_x.append(xco)
    d=math.lcm(*(v.denominator for row in all_f+all_x for v in row))
    q=[]
    for f,x in zip(all_f,all_x):
        q.append([(int((4*xx+2*ff)*d),int((4*ff-2*xx)*d)) for xx,ff in zip(x,f)])
    return q,5*d


def exact_input_power_coefficients():
    p=json.loads((HERE/'pair_rational_approximants.json').read_text())
    scale=1<<p['scale_bits']
    means=[[F(*v) for v in pair] for pair in p['means']]
    den=math.lcm(scale,*(v.denominator for pair in means for v in pair))
    order=p['order']
    cheb=[[1],[0,1]]
    for k in range(2,order):
        row=[0]*(k+1)
        for j,v in enumerate(cheb[-1]): row[j+1]+=2*v
        for j,v in enumerate(cheb[-2]): row[j]-=v
        cheb.append(row)
    functions=[]
    for panel in p['integers']:
        channels=[]
        for channel in range(2):
            co=[(0,0)]*order
            for k in range(order):
                z=tuple(a*(den//scale) for a in panel[k][channel])
                if k==0:
                    z=(z[0]-int(means[channel][0]*den)+(den if channel==0 else 0),
                       z[1]-int(means[channel][1]*den))
                for j,v in enumerate(cheb[k]):
                    co[j]=cadd(co[j],(z[0]*v,z[1]*v))
            channels.append(co)
        functions.append(channels)
    return functions,den


def main():
    started=time.perf_counter()
    max_power=17
    q,dq=geometry_coefficients()
    inputs,df=exact_input_power_coefficients()
    integral_lcm=math.lcm(*range(1,8*max_power+20+1))
    # mu_l numerator denominator=64*df*integral_lcm*dq^l.
    moments=[[[[0,0] for l in range(max_power+1)] for c in range(2)] for sign in range(2)]
    powers=[]
    for panel in range(64):
        pp=[[(1,0)]]
        for l in range(1,max_power+1): pp.append(pmul(pp[-1],q[panel]))
        powers.append(pp)
        for l,poly in enumerate(pp):
            for sign in range(2):
                qq=poly if sign==0 else [(a,-b) for a,b in poly]
                for channel in range(2):
                    prod=pmul(qq,inputs[panel][channel])
                    for k in range(0,len(prod),2):
                        factor=integral_lcm//(k+1)
                        moments[sign][channel][l][0]+=factor*prod[k][0]
                        moments[sign][channel][l][1]+=factor*prod[k][1]
    tail_bits=100
    tail_int=[]
    tail_coefficient_error=F(0)
    for k in range(9):
        lo,hi=power_tail_interval(2*k+2)
        val=round((lo+hi)/2*(1<<tail_bits))
        mid=F(val,1<<tail_bits)
        tail_int.append(val)
        tail_coefficient_error+=max(abs(mid-lo),abs(mid-hi))*F(5,2)**(2*k+1)*F(4,3)
    geometric_error=F(4,3)*F(5,16)**19/(19*(1-F(25,324)))
    assert geometric_error+tail_coefficient_error<F(1,10**10)
    # A rational enclosure for1/(2pi), then a dyadic representative.
    lo5,hi5=arctan_bracket(5,64)
    lo239,hi239=arctan_bracket(239,24)
    pi_lo,pi_hi=16*lo5-4*hi239,16*hi5-4*lo239
    invlo,invhi=1/(2*pi_hi),1/(2*pi_lo)
    pi_bits=120
    pi_int=round((invlo+invhi)/2*(1<<pi_bits))
    pi_mid=F(pi_int,1<<pi_bits)
    pi_error=max(abs(pi_mid-invlo),abs(pi_mid-invhi))
    wden=(1<<tail_bits)*64*df*integral_lcm*dq**max_power
    outputs=[]
    max_rounding=F(0)
    max_degree=0
    out_bits=80
    for panel in range(64):
        panel_outputs=[]
        for channel in range(2):
            ww=[]
            for sign in range(2):
                out=[(0,0)]*(8*max_power+1)
                for k,s_int in enumerate(tail_int):
                    power=2*k+1
                    for l in range(power+1):
                        coefficient=s_int*math.comb(power,l)*(-1)**(power-l)*dq**(max_power-power)
                        mu=tuple(moments[sign][channel][l])
                        for j,term in enumerate(powers[panel][power-l]):
                            if sign: term=(term[0],-term[1])
                            value=cmul(term,mu)
                            out[j]=cadd(out[j],(coefficient*value[0],coefficient*value[1]))
                ww.append(out)
            qp=[(128*(j+1)*a,128*(j+1)*b) for j,(a,b) in enumerate(q[panel][1:])]
            plus=pmul(qp,ww[0])
            minus=pmul([(a,-b) for a,b in qp],ww[1])
            pnum=[(a[1]-b[1],-a[0]+b[0]) for a,b in zip(plus,minus)]
            rnum=[]
            for a,b in zip(ww[0],ww[1]):
                aa=cmul((4,-2),a); bb=cmul((4,2),b)
                rnum.append((-aa[0]-bb[0],-aa[1]-bb[1]))
            for kind,poly,den in [('Pstar',pnum,dq*wden),('R',rnum,5*wden)]:
                # Truncate only after bounding all discarded coefficients.
                degree=len(poly)-1
                remainder=F(0)
                while degree>0:
                    term=F(abs(poly[degree][0])+abs(poly[degree][1]),6*den)
                    if remainder+term>F(1,10**13): break
                    remainder+=term
                    degree-=1
                coeff=[]
                for term in poly[:degree+1]:
                    coeff.append([round(F(a*pi_int*(1<<out_bits),den*(1<<pi_bits))) for a in term])
                l1=F(sum(abs(a)+abs(b) for a,b in poly),den)
                error=remainder+pi_error*l1+F(degree+1,1<<out_bits)
                assert error<F(1,10**12)
                max_rounding=max(max_rounding,error)
                max_degree=max(max_degree,degree)
                panel_outputs.append({'channel':f'{kind}_{channel}', 'coefficients':coeff,
                                      'polynomial_error_bound':[error.numerator,error.denominator]})
        outputs.append(panel_outputs)
    # Both exact source norms are<4, so L1 norms are<4.
    total_error=4*(geometric_error+tail_coefficient_error)+max_rounding
    assert total_error<F(1,10**9)
    result={'status':'PASS: exact periodic tail enclosure (finite images are checked separately)',
            'profile':'F as defined in the manuscript; unwrapped coordinates x,y in [0,1]',
            'retained_line_images':[-8,8],'retained_tail_odd_powers':max_power,
            'panels':64,'scale_bits':out_bits,
            'representation':'power coefficients in t=128*x-2*panel-1; real/imaginary integer pairs divided by2^80',
            'uniform_tail_action_error':[total_error.numerator,total_error.denominator],
            'uniform_tail_action_error_decimal':float(total_error),'maximum_degree':max_degree,
            'outputs':outputs,'elapsed_seconds':time.perf_counter()-started}
    (HERE/'pair_exact_periodic_tail.json').write_text(json.dumps(result,separators=(',',':')),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='outputs'},indent=2))


if __name__=='__main__':
    main()
