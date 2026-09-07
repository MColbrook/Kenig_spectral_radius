"""Execute the finite-image continuous P* residual enclosure.

Every accepted kernel polynomial passes an exact integer residual test;
all source integration and final residual norms are exact rational.
"""
import exact_arithmetic  # Enforce the execution contract before any checks.
import json
import math
import argparse
import time
from fractions import Fraction as F
from pathlib import Path
from exact_arithmetic import scratch_path

from pair_finite_kernel_boxes import (Geometry,target_grid,restrict_polynomial,
                                     certify_image,BITS,QMAX,IMAGE_ERROR)
from pair_exact_periodic_tail import exact_input_power_coefficients
from exact_arithmetic import arctan_bracket

HERE=Path(__file__).parent


def norm_squared(coeff,left,right):
    common=math.lcm(*(v.denominator for pair in coeff for v in pair))
    integers=[(int(a*common),int(b*common)) for a,b in coeff]
    lcm=math.lcm(*range(1,2*len(coeff)))
    total=0
    for i,(ar,ai) in enumerate(integers):
        for j,(br,bi) in enumerate(integers):
            if (i+j)%2==0: total+=(ar*br+ai*bi)*(lcm//(i+j+1))
    squared=(right-left)*F(total,common*common*lcm)
    assert squared>=0
    return squared,integers,common


class Calculator:
    def __init__(self):
        self.geometry=Geometry()
        self.input_co,self.df=exact_input_power_coefficients()
        self.integration_lcm=math.lcm(*range(1,QMAX+20+1))
        self.moment_base=self.df*self.integration_lcm
        self.moment_cache={}
        self.tail=json.loads((HERE/'pair_exact_periodic_tail.json').read_text())
        lo5,hi5=arctan_bracket(5,64);lo239,hi239=arctan_bracket(239,24)
        pi_lo,pi_hi=16*lo5-4*hi239,16*hi5-4*lo239
        a,b=1/(2*pi_hi),1/(2*pi_lo)
        self.pi_bits=120
        self.pi_int=round((a+b)/2*(1<<self.pi_bits))
        self.pi_error=max(abs(F(self.pi_int,1<<self.pi_bits)-a),abs(F(self.pi_int,1<<self.pi_bits)-b))

    def input_on(self,interval):
        left,right=interval;center=(left+right)/2;half=(right-left)/2
        panel=min(int(center*64),63)
        a=128*center-2*panel-1;b=128*half
        result=[]
        for channel in range(2):
            re=restrict_polynomial([F(v[0],self.df) for v in self.input_co[panel][channel]],a,b)
            im=restrict_polynomial([F(v[1],self.df) for v in self.input_co[panel][channel]],a,b)
            count=max(len(re),len(im))
            re += [F(0)]*(count-len(re));im += [F(0)]*(count-len(im))
            result.append(list(zip(re,im)))
        return result

    def moments(self,source):
        if source not in self.moment_cache:
            functions=self.input_on(source)
            den=math.lcm(self.df,*(v.denominator for co in functions for pair in co for v in pair))
            width=source[1]-source[0]
            md=F(den*self.integration_lcm)/width
            assert md.denominator==1
            quotient=md.numerator//self.moment_base
            assert quotient*self.moment_base==md.numerator and quotient&(quotient-1)==0
            exponent=quotient.bit_length()-1
            moments=[]
            for co in functions:
                integers=[(int(a*den),int(b*den)) for a,b in co]
                row=[]
                for j in range(QMAX+1):
                    re=im=0
                    for k,(a,b) in enumerate(integers):
                        if (j+k)%2==0:
                            factor=self.integration_lcm//(j+k+1)
                            re+=a*factor;im+=b*factor
                    row.append((re,im))
                moments.append(row)
            self.moment_cache[source]=(moments,exponent)
        return self.moment_cache[source]

    def compute(self,target,source_depth_limit=4):
        start=time.perf_counter()
        acc=[[[0,0] for i in range(QMAX+1)] for c in range(2)]
        acc_exp=0;count=0;degree=0;max_error=F(0);subdivisions=0
        def visit(source,image,depth):
            nonlocal acc_exp,count,degree,max_error,subdivisions
            certificate=certify_image(self.geometry,target,source,image)
            if certificate is None:
                if depth>=source_depth_limit: raise ArithmeticError('target subdivision needed')
                subdivisions+=1
                center=(source[0]+source[1])/2
                visit((source[0],center),image,depth+1)
                visit((center,source[1]),image,depth+1)
                return
            count+=1;degree=max(degree,certificate['degree']);max_error=max(max_error,certificate['error'])
            if not certificate['coefficients']: return
            moments,exponent=self.moments(source)
            if exponent>acc_exp:
                shift=exponent-acc_exp
                for channel in acc:
                    for pair in channel: pair[0]<<=shift;pair[1]<<=shift
                acc_exp=exponent
            shift=acc_exp-exponent
            for (i,j),value in certificate['coefficients'].items():
                for channel in range(2):
                    re,im=moments[channel][j]
                    acc[channel][i][0]+=(value*re)<<shift
                    acc[channel][i][1]+=(value*im)<<shift
        try:
            for source in target_grid():
                for image in range(-8,9): visit(source,image,0)
        except ArithmeticError:
            return {'needs_subdivision':True,'target':[str(v) for v in target],
                    'elapsed_seconds':time.perf_counter()-start}
        before_pi_den=(1<<BITS)*self.moment_base*(1<<acc_exp)
        final_den=before_pi_den*(1<<self.pi_bits)
        pi_action_error=max(self.pi_error*F(sum(abs(a)+abs(b) for a,b in row),before_pi_den) for row in acc)
        assert pi_action_error<F(1,10**15)
        # Assemble the exact finite-image, stored-tail and local residual
        # polynomials BEFORE taking any coefficient or L2 bound.
        left,right=target;center=(left+right)/2;half=(right-left)/2
        panel=min(int(center*64),63)
        a=128*center-2*panel-1;b=128*half
        local_inputs=self.input_on(target)
        _,slope,_,_=self.geometry.interval(*target)
        result_channels=[]
        for channel in range(2):
            co=[(F(re*self.pi_int,final_den),F(im*self.pi_int,final_den)) for re,im in acc[channel]]
            tail_record=next(r for r in self.tail['outputs'][panel] if r['channel']==f'Pstar_{channel}')
            tr=restrict_polynomial([F(v[0],1<<80) for v in tail_record['coefficients']],a,b)
            ti=restrict_polynomial([F(v[1],1<<80) for v in tail_record['coefficients']],a,b)
            for i,val in enumerate(tr): co[i]=(co[i][0]+val,co[i][1])
            for i,val in enumerate(ti): co[i]=(co[i][0],co[i][1]+val)
            for i,(re,im) in enumerate(local_inputs[channel]):
                if channel==0 and i==0: re-=1
                co[i]=(co[i][0]-im/2,co[i][1]+re/2)
            if channel==1:
                for i,val in enumerate(slope): co[i]=(co[i][0]-val,co[i][1])
                co[0]=(co[0][0]+F(1,2),co[0][1])
            squared,ints,common=norm_squared(co,left,right)
            result_channels.append({'L2_squared':[squared.numerator,squared.denominator],
                                    'coefficient_denominator':common,'coefficients':ints})
        return {'needs_subdivision':False,'target':[str(v) for v in target],
                'accepted_image_boxes':count,'source_subdivisions':subdivisions,'maximum_degree':degree,
                'max_kernel_error':[max_error.numerator,max_error.denominator],
                'pi_action_error':[pi_action_error.numerator,pi_action_error.denominator],
                'channels':result_channels,'elapsed_seconds':time.perf_counter()-start}


CALCULATOR=None


def process_target(target):
    global CALCULATOR
    if CALCULATOR is None: CALCULATOR=Calculator()
    return CALCULATOR.compute(tuple(F(v) for v in target))


def combine(completed,start):
    completed.sort(key=lambda r:F(r['target'][0]))
    assert F(completed[0]['target'][0])==0 and F(completed[-1]['target'][1])==1
    assert all(F(r['target'][0])<F(r['target'][1]) and not r['needs_subdivision'] for r in completed)
    assert all(F(a['target'][1])==F(b['target'][0]) for a,b in zip(completed[:-1],completed[1:]))
    squared=[sum((F(*r['channels'][c]['L2_squared']) for r in completed),F(0)) for c in range(2)]
    tail=json.loads((HERE/'pair_exact_periodic_tail.json').read_text())
    eta=F(1,2**14);m5=F(55050240)
    corner=eta*(m5*eta**3/72+m5*eta**4/4)
    error=4*17*IMAGE_ERROR+4*corner+F(*tail['uniform_tail_action_error'])+F(1,10**15)
    threshold=F(7,10**8)
    passed=(0<=error<threshold and
            all(0<=s<(threshold-error)**2 for s in squared))
    if not passed:
        raise ArithmeticError("Continuous Pstar residual threshold failed")
    output={'status':'CONTINUOUS P* RESIDUAL CERTIFICATE PASS' if passed else 'CONTINUOUS P* residual certificate thresholds NOT met',
            'polynomial_residual_norms_squared':[[s.numerator,s.denominator] for s in squared],
            'polynomial_residual_norms_squared_decimal':[float(s) for s in squared],
            'analytic_action_error_bound':[error.numerator,error.denominator],
            'both_continuous_residuals_below_7e_minus8':passed,
            'both_continuous_residuals_below_5e_minus_7':passed,
            'target_intervals':len(completed),'accepted_image_boxes':sum(r['accepted_image_boxes'] for r in completed),
            'records':completed,'elapsed_seconds':time.perf_counter()-start}
    (HERE/'pair_pstar_residual_certificate.json').write_text(json.dumps(output,separators=(',',':')),encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k not in ('records','polynomial_residual_norms_squared','analytic_action_error_bound')},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--combine',action='store_true')
    args=parser.parse_args()
    start=time.perf_counter()
    if args.combine:
        records=[]
        for k in range(args.shards):
            records.extend(json.loads(scratch_path(f'pair_pstar_residual_shard_{k}.json').read_text())['records'])
        combine(records,start)
        return
    if not 0<=args.shard<args.shards: parser.error('require 0 <= shard < shards')
    progress=scratch_path(f'pair_pstar_residual_progress_{args.shard}.jsonl')
    completed=[]
    pending=[target for i,target in enumerate(target_grid()) if i%args.shards==args.shard]
    with progress.open('w',encoding='utf-8') as stream:
        while pending:
            target=pending.pop(0)
            result=process_target(tuple(str(v) for v in target))
            if result['needs_subdivision']:
                left,right=(F(v) for v in result['target']);mid=(left+right)/2
                if right-left<F(1,2**20): raise ArithmeticError('target refinement limit reached')
                pending.insert(0,(mid,right));pending.insert(0,(left,mid))
                print('subdivide target',result['target'],flush=True)
                continue
            completed.append(result)
            stream.write(json.dumps(result,separators=(',',':'))+'\n');stream.flush()
            print('shard',args.shard,'completed',len(completed),'pending',len(pending),'target',result['target'],
                  'row seconds',round(result['elapsed_seconds'],1),flush=True)
    output={'status':'Exact partial target partition completed; combine all shards for full P* residual certificate',
            'shard':args.shard,'shards':args.shards,'records':completed,'elapsed_seconds':time.perf_counter()-start}
    scratch_path(f'pair_pstar_residual_shard_{args.shard}.json').write_text(json.dumps(output,separators=(',',':')),encoding='utf-8')
    if args.shards==1: combine(completed,start)


if __name__=='__main__': main()
