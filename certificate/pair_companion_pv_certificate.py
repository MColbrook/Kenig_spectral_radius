"""Exact one-dimensional analytic PV and continuous companion residuals.

The smooth seventeen-image remainder plus tail is an INPUT certificate.
No full R claim is made unless that completed file exists and every exact
residual comparison passes. Floating numbers only occur in display text
or the independently verified N-D*S coefficient proposer.
"""
import exact_arithmetic  # Enforce the execution contract before any checks.
import hashlib
import json
import math
import time
from fractions import Fraction as F
from pathlib import Path

HERE=Path(__file__).parent
COORD=HERE
from pair_finite_kernel_boxes import Geometry,restrict_polynomial,certify_polynomial,target_grid
from pair_exact_periodic_tail import exact_input_power_coefficients
from pair_pstar_residual_certificate import Calculator,norm_squared
from exact_arithmetic import log_interval,scratch_path

ETA=F(1,4096)
HBUDGET=F(1,10**9)
ABUDGET=F(1,10**10)
ROUND_BITS=100


def trim(p):
    while len(p)>1 and not p[-1]: p.pop()
    return p


def add(p,q):
    r=list(p)+[F(0)]*max(0,len(q)-len(p))
    for k,c in enumerate(q): r[k]+=c
    return trim(r)


def mul(p,q):
    r=[F(0)]*(len(p)+len(q)-1)
    for i,a in enumerate(p):
        if a:
            for j,b in enumerate(q):
                if b: r[i+j]+=a*b
    return trim(r)


def value(p,t):
    out=F(0)
    for c in reversed(p): out=out*t+c
    return out


def quantize(p,bits=ROUND_BITS):
    den=1<<bits
    out=[F(round(c*den),den) for c in p]
    error=sum((abs(a-b) for a,b in zip(p,out)),F(0))
    return trim(out),error


def pair_restrict(p,a,b):
    return [restrict_polynomial(component,a,b) for component in p]


def pair_l1(p):
    return sum((abs(v) for component in p for v in component),F(0))


def rational_pair(value): return [value.numerator,value.denominator]


def all_input_polynomials(data):
    order=data['order'];scale=1<<data['scale_bits']
    cheb=[[1],[0,1]]
    for k in range(2,order):
        row=[0]*(k+1)
        for j,v in enumerate(cheb[-1]): row[j+1]+=2*v
        for j,v in enumerate(cheb[-2]): row[j]-=v
        cheb.append(row)
    output=[]
    for panel in data['integers']:
        channels=[]
        for c in range(4):
            pair=[]
            for ri in (0,1):
                co=[F(0)]*order
                for k in range(order):
                    v=F(panel[k][c][ri],scale)
                    if k==0: v-=F(*data['means'][c][ri])
                    for j,b in enumerate(cheb[k]): co[j]+=v*b
                pair.append(trim(co))
            channels.append(pair)
        output.append(channels)
    return output


class PVCalculator:
    def __init__(self):
        self.data=json.loads((COORD/'pair_rational_approximants.json').read_text())
        assert self.data['panels']==64 and self.data['order']==20
        self.all=all_input_polynomials(self.data)
        self.inputs=[]
        for p in self.all:
            a=[list(component) for component in p[0]];a[0][0]+=1
            self.inputs.append([a,p[1]])
        # Cross-check the raw density/mean normalization with the tail code.
        tc,td=exact_input_power_coefficients()
        for panel in range(64):
            for ch in range(2):
                for ri in range(2):
                    assert self.inputs[panel][ch][ri]==trim([F(v[ri],td) for v in tc[panel][ch]])
        self.geometry=Geometry()
        base=Calculator()
        self.pi=F(base.pi_int,1<<base.pi_bits);self.pi_error=base.pi_error
        self.cache={};self.log_cache={};self.local_cache={}
        self.seam_omega=self.check_seams()
        self.seam_squared=[128*ETA*(12*w/6)**2 for w in self.seam_omega]
        assert all(v<F(1,10**16) for v in self.seam_squared)
        # Direct coefficient bounds suffice; no sampled derivative maximum
        # or assumption of matching derivative jets enters this estimate.
        self.h_true_bound=[]
        for ch in range(2):
            sup=max(pair_l1(p[ch]) for p in self.inputs)
            derivative=max(sum((128*k*abs(v) for part in p[ch]
                                for k,v in enumerate(part)),F(0)) for p in self.inputs)
            self.h_true_bound.append(F(3,64)*derivative+sup) # log2<1
        assert max(self.h_true_bound)<19

    def check_seams(self):
        maxima=[F(0),F(0)]
        for seam in range(64):
            for ch in range(2):
                omega=F(0)
                for ri in range(2):
                    left=restrict_polynomial(self.inputs[(seam-1)%64][ch][ri],F(1),F(128))
                    right=restrict_polynomial(self.inputs[seam][ch][ri],F(-1),F(128))
                    diff=add(left,[-c for c in right]);assert diff[0]==0
                    omega+=sum((abs(v)*ETA**k for k,v in enumerate(diff)),F(0))
                maxima[ch]=max(maxima[ch],omega)
        return maxima

    def original_near(self,panel):
        if panel in self.cache: return self.cache[panel]
        channels=[]
        for ch in range(2):
            poly=[[F(0)],[F(0)]];source=[]
            for offset in (-1,0,1):
                pk=self.inputs[(panel+offset)%64][ch]
                source.append(pair_restrict(pk,F(-2*offset),F(1)))
                for ri in range(2):
                    # int_-1^1 (p(t)-p(z))/(t-z)dt, a polynomial in z.
                    q=[F(0)]*max(1,len(pk[ri])-1)
                    for n,pn in enumerate(pk[ri]):
                        for k in range(n):
                            exponent=n-1-k
                            if exponent%2==0: q[k]+=2*pn/F(exponent+1)
                    poly[ri]=add(poly[ri],restrict_polynomial(q,F(-2*offset),F(1)))
            logs=[]
            signs=[[-v for v in p] for p in source[0]]
            logs.append((F(panel-1,64),signs,False))
            for index in (0,1):
                diff=[add(source[index][ri],[-v for v in source[index+1][ri]]) for ri in range(2)]
                seam=F(panel+index,64)
                assert all(value(c,F(2*index-1))==0 for c in diff)
                logs.append((seam,diff,True))
            logs.append((F(panel+2,64),source[2],False))
            # The total log coefficient vanishes: physical/log-normalized
            # distance conventions agree before any selected omission.
            for ri in range(2):
                total=[F(0)]
                for _,co,_ in logs: total=add(total,co[ri])
                assert total==[0]
            channels.append((poly,logs))
        self.cache[panel]=channels
        return channels

    def log_polynomial(self,left,right,endpoint,coefficient_bound):
        center=(left+right)/2;half=(right-left)/2
        d=abs(center-endpoint);orientation=1 if center>endpoint else -1
        assert d>half
        ratio=orientation*half/d;rho=abs(ratio)
        assert rho<=F(1,2)
        if d not in self.log_cache: self.log_cache[d]=log_interval(d)
        lo,hi=self.log_cache[d]
        mid=F(round((lo+hi)/2*(1<<120)),1<<120)
        ce=max(abs(mid-lo),abs(mid-hi))
        degree=0
        while True:
            rem=rho**(degree+1)/F(degree+1)/(1-rho)
            if coefficient_bound*(ce+rem)<=HBUDGET/16: break
            degree+=1
            if degree>80: raise ArithmeticError('log degree limit')
        poly=[mid]+[(-1)**(k+1)*ratio**k/F(k) for k in range(1,degree+1)]
        return poly,ce+rem,degree

    def hilbert_on(self,target):
        left,right=target;center=(left+right)/2;half=(right-left)/2
        panel=min(int(center*64),63)
        assert F(panel,64)<=left<right<=F(panel+1,64)
        a=128*center-2*panel-1;b=128*half
        out=[];errors=[];omitted=[];degrees=[]
        for ch,(base,logs) in enumerate(self.original_near(panel)):
            hp=pair_restrict(base,a,b);err=F(0);omit=[];degree=0
            for endpoint,co,is_internal in logs:
                rc=pair_restrict(co,a,b)
                if is_internal and max(abs(left-endpoint),abs(right-endpoint))<=ETA:
                    omit.append(str(endpoint));continue
                if is_internal:
                    assert min(abs(left-endpoint),abs(right-endpoint))>=ETA
                bound=pair_l1(rc)
                if not bound: continue
                lp,le,deg=self.log_polynomial(left,right,endpoint,bound)
                for ri in range(2): hp[ri]=add(hp[ri],mul(rc[ri],lp))
                err+=bound*le;degree=max(degree,deg)
            for ri in range(2):
                hp[ri],rounding=quantize(hp[ri]);err+=rounding
            assert err<=HBUDGET
            assert len(omit)<=1
            out.append(hp);errors.append(err);omitted.append(omit);degrees.append(degree)
        return {'panel':panel,'polynomials':out,'errors':errors,'omitted':omitted,'log_degree':max(degrees)}

    def a_on(self,target):
        _,s,_,_=self.geometry.interval(*target)
        den=add([F(1)],mul(s,s))
        certificate=certify_polynomial({(0,0):F(1)},
                     {(k,0):v for k,v in enumerate(den) if v},F(1),ABUDGET/4)
        if certificate is None: raise ArithmeticError('A target refinement needed')
        candidate=certificate['coefficients']
        q=[F(candidate.get((k,0),0),1<<80) for k in range(certificate['degree']+1)]
        assert all(j==0 for i,j in candidate)
        a=[v*self.pi for v in q]
        error=certificate['error']+sum(abs(v) for v in q)*self.pi_error
        a,rounding=quantize(a,120);error+=rounding
        assert error<ABUDGET
        return a,error,certificate['degree']

    def local_on(self,target):
        if target not in self.local_cache:
            hp=self.hilbert_on(target)
            ap,ae,adeg=self.a_on(target)
            self.local_cache[target]=(hp,ap,ae,adeg)
        return self.local_cache[target]

    def assemble(self,record,target):
        left,right=target;center=(left+right)/2;half=(right-left)/2
        oldleft,oldright=map(F,record['target'])
        assert oldleft<=left<right<=oldright
        oldcenter=(oldleft+oldright)/2;oldhalf=(oldright-oldleft)/2
        hilbert,ap,ae,adeg=self.local_on(target)
        panel=hilbert['panel'];local_a=128*center-2*panel-1;local_b=128*half
        outputs=[];rounderrs=[]
        for ch in range(2):
            row=record['channels'][ch];den=row['coefficient_denominator']
            smooth=[restrict_polynomial([F(v[ri],den) for v in row['coefficients']],
                         (center-oldcenter)/oldhalf,half/oldhalf) for ri in range(2)]
            desired=pair_restrict(self.all[panel][ch+2],local_a,local_b)
            if ch==1: desired[0][0]-=1
            residual=[];re=F(0)
            for ri in range(2):
                action=add(smooth[ri],mul(ap,hilbert['polynomials'][ch][ri]))
                exact=add(desired[ri],[-v for v in action])
                qr,err=quantize(exact);residual.append(qr);re+=err
            count=max(map(len,residual))
            co=[(residual[0][k] if k<len(residual[0]) else F(0),
                 residual[1][k] if k<len(residual[1]) else F(0)) for k in range(count)]
            squared,integers,common=norm_squared(co,left,right)
            outputs.append({'coefficient_denominator':common,'coefficients':integers,
                            'L2_squared':rational_pair(squared)})
            rounderrs.append(re)
        return {'target':[str(left),str(right)],'original_target_panel':panel,
                'smooth_parent_target':record['target'],
                'omitted_seams':hilbert['omitted'],'log_degree':hilbert['log_degree'],
                'A_degree':adeg,'A_uniform_error':rational_pair(ae),
                'H_uniform_errors':[rational_pair(e) for e in hilbert['errors']],
                'final_coefficient_rounding':[rational_pair(e) for e in rounderrs],
                'channels':outputs}


def refined_grid(smooth_records):
    cuts={F(v) for row in smooth_records for v in row['target']}
    for panel in range(64):
        a,b=F(panel,64),F(panel+1,64)
        cuts.update((a,b))
        for k in range(7,13): cuts.update((a+F(1,2**k),b-F(1,2**k)))
    cuts=sorted(cuts)
    return list(zip(cuts[:-1],cuts[1:]))



def main():
    start=time.perf_counter();calculator=PVCalculator()
    path=COORD/'pair_companion_smooth_certificate.json'
    if not path.exists(): raise FileNotFoundError('Completed smooth companion certificate is required; no full R claim')
    smooth=json.loads(path.read_text());source=smooth['records']
    assert F(source[0]['target'][0])==0 and F(source[-1]['target'][1])==1
    assert all(F(a['target'][1])==F(b['target'][0]) for a,b in zip(source[:-1],source[1:]))
    tail=json.loads((COORD/'pair_exact_periodic_tail.json').read_text())
    expected=4*17*F(1,10**9)+4*F(55050240)*F(1,2**14)**4/72+F(*tail['uniform_tail_action_error'])+F(1,10**15)
    assert F(*smooth['action_L2_error_bound'])==expected
    assert expected<F(7,10**8)
    for row in source:
        assert not row['needs_subdivision']
        assert F(*row['max_kernel_error'])<=F(1,10**9)
        assert F(*row['pi_action_error'])<F(1,10**15)
    pending=refined_grid(source);completed=[];index=0
    progress=scratch_path('pair_companion_pv_progress.jsonl')
    with progress.open('w',encoding='utf-8') as stream:
        while pending:
            target=pending.pop(0)
            while F(source[index]['target'][1])<=target[0]: index+=1
            try: result=calculator.assemble(source[index],target)
            except ArithmeticError:
                a,b=target
                if b-a<F(1,2**20): raise
                mid=(a+b)/2;pending.insert(0,(mid,b));pending.insert(0,(a,mid));continue
            completed.append(result);stream.write(json.dumps(result,separators=(',',':'))+'\n');stream.flush()
            if len(completed)%16==0: print('PV rows',len(completed),'pending',len(pending),'seconds',round(time.perf_counter()-start,1),flush=True)
    assert F(completed[0]['target'][0])==0 and F(completed[-1]['target'][1])==1
    assert all(a['target'][1]==b['target'][0] for a,b in zip(completed[:-1],completed[1:]))
    squared=[sum((F(*r['channels'][ch]['L2_squared']) for r in completed),F(0)) for ch in range(2)]
    he=[max(F(*r['H_uniform_errors'][ch]) for r in completed) for ch in range(2)]
    ae=max(F(*r['A_uniform_error']) for r in completed)
    ce=[max(F(*r['final_coefficient_rounding'][ch]) for r in completed) for ch in range(2)]
    # True H<=19; omitted unweighted seam term<=12*omega<1e-6,
    # and the polynomial approximation error<1e-9, hence |H_poly|<20.
    assert all(12*w<F(1,10**6) for w in calculator.seam_omega)
    assert max(he)<=HBUDGET and ae<ABUDGET
    error=[expected+F(1,10**8)+he[ch]/6+20*ae+ce[ch] for ch in range(2)]
    threshold=F(8,10**8)
    passed=all(0<=e<threshold and 0<=ss<(threshold-e)**2 for ss,e in zip(squared,error))
    out={'status':'CONTINUOUS COMPANION RESIDUAL CERTIFICATE PASS' if passed else 'Companion residual thresholds NOT met',
         'both_continuous_residuals_below_8e_minus8':passed,
         'both_continuous_residuals_below_5e_minus6':passed,
         'polynomial_residual_norms_squared':[rational_pair(v) for v in squared],
         'polynomial_residual_norms_squared_decimal':[float(v) for v in squared],
         'analytic_error_bounds':[rational_pair(v) for v in error],
         'analytic_error_bounds_decimal':[float(v) for v in error],
         'seam_discard_error_squared':[rational_pair(v) for v in calculator.seam_squared],
         'true_three_panel_H_uniform_bounds':[rational_pair(v) for v in calculator.h_true_bound],
         'max_unweighted_H_approximation_errors':[rational_pair(v) for v in he],
         'max_A_approximation_error':rational_pair(ae),
         'target_intervals':len(completed),'records':completed,
         'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                   [path,COORD/'pair_rational_approximants.json',COORD/'pair_exact_periodic_tail.json',Path(__file__)]},
         'elapsed_seconds':time.perf_counter()-start}
    if not passed: raise ArithmeticError('Continuous companion residual threshold failed')
    HERE.joinpath('pair_companion_pv_certificate.json').write_text(json.dumps(out,separators=(',',':')),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k!='records'},indent=2),flush=True)
    if not passed: raise ArithmeticError('Continuous companion residual threshold failed')


if __name__=='__main__': main()
