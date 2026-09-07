"""Certify the finite smooth R remainders; analytic Hilbert part follows.

Uses the original three adjacent source panels for each original target
panel, without altering the fixed unwrapped seventeen-image split.
"""
import exact_arithmetic  # Enforce the execution contract before any checks.
import argparse
import json
import math
import time
from fractions import Fraction as F
from pathlib import Path
from exact_arithmetic import scratch_path

from pair_finite_kernel_boxes import target_grid,certify_companion_image,BITS,QMAX,IMAGE_ERROR,restrict_polynomial
from pair_pstar_residual_certificate import Calculator

HERE=Path(__file__).parent


class CompanionCalculator(Calculator):
    def compute(self,target,source_depth_limit=4):
        start=time.perf_counter()
        acc=[[[0,0] for i in range(QMAX+1)] for c in range(2)]
        acc_exp=0;count=0;degree=0;max_error=F(0);subdivisions=0
        def visit(source,image,depth):
            nonlocal acc_exp,count,degree,max_error,subdivisions
            certificate=certify_companion_image(self.geometry,target,source,image)
            if certificate is None:
                if depth>=source_depth_limit: raise ArithmeticError('target subdivision needed')
                subdivisions+=1;center=(source[0]+source[1])/2
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
        left,right=target;center=(left+right)/2;half=(right-left)/2
        panel=min(int(center*64),63);a=128*center-2*panel-1;b=128*half
        outputs=[]
        for channel in range(2):
            co=[(F(re*self.pi_int,final_den),F(im*self.pi_int,final_den)) for re,im in acc[channel]]
            tail_record=next(r for r in self.tail['outputs'][panel] if r['channel']==f'R_{channel}')
            tr=restrict_polynomial([F(v[0],1<<80) for v in tail_record['coefficients']],a,b)
            ti=restrict_polynomial([F(v[1],1<<80) for v in tail_record['coefficients']],a,b)
            for i,val in enumerate(tr): co[i]=(co[i][0]+val,co[i][1])
            for i,val in enumerate(ti): co[i]=(co[i][0],co[i][1]+val)
            common=math.lcm(*(v.denominator for pair in co for v in pair))
            outputs.append({'coefficient_denominator':common,
                            'coefficients':[[int(re*common),int(im*common)] for re,im in co]})
        return {'needs_subdivision':False,'target':[str(v) for v in target],
                'original_target_panel':panel,'accepted_image_boxes':count,
                'source_subdivisions':subdivisions,'maximum_degree':degree,
                'max_kernel_error':[max_error.numerator,max_error.denominator],
                'pi_action_error':[pi_action_error.numerator,pi_action_error.denominator],
                'channels':outputs,'elapsed_seconds':time.perf_counter()-start}


def combine(records):
    records.sort(key=lambda r:F(r['target'][0]))
    assert F(records[0]['target'][0])==0 and F(records[-1]['target'][1])==1
    assert all(F(r['target'][0])<F(r['target'][1]) and not r['needs_subdivision'] for r in records)
    assert all(F(a['target'][1])==F(b['target'][0]) for a,b in zip(records[:-1],records[1:]))
    tail=json.loads((HERE/'pair_exact_periodic_tail.json').read_text())
    eta=F(1,2**14);m5=F(55050240)
    error=4*17*IMAGE_ERROR+4*m5*eta**4/72+F(*tail['uniform_tail_action_error'])+F(1,10**15)
    assert error<F(7,10**8)
    output={'status':'PASS: smooth companion remainder and tail enclosure (principal value is assembled separately)',
            'action_L2_error_bound':[error.numerator,error.denominator],
            'target_intervals':len(records),'accepted_image_boxes':sum(r['accepted_image_boxes'] for r in records),
            'records':records}
    (HERE/'pair_companion_smooth_certificate.json').write_text(json.dumps(output,separators=(',',':')),encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k not in ('records','action_L2_error_bound')},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--combine',action='store_true')
    args=parser.parse_args();start=time.perf_counter()
    if args.combine:
        records=[]
        for k in range(args.shards):
            records.extend(json.loads(scratch_path(f'pair_companion_smooth_shard_{k}.json').read_text())['records'])
        combine(records);return
    if not 0<=args.shard<args.shards: parser.error('require 0 <= shard < shards')
    calculator=CompanionCalculator()
    pending=[t for i,t in enumerate(target_grid()) if i%args.shards==args.shard]
    completed=[]
    with scratch_path(f'pair_companion_smooth_progress_{args.shard}.jsonl').open('w',encoding='utf-8') as stream:
        while pending:
            target=pending.pop(0);result=calculator.compute(target)
            if result['needs_subdivision']:
                a,b=target;c=(a+b)/2
                if b-a<F(1,2**20): raise ArithmeticError('target refinement limit')
                pending.insert(0,(c,b));pending.insert(0,(a,c))
                print('subdivide target',result['target'],flush=True);continue
            completed.append(result);stream.write(json.dumps(result,separators=(',',':'))+'\n');stream.flush()
            print('R shard',args.shard,'completed',len(completed),'pending',len(pending),
                  'target',result['target'],'seconds',round(result['elapsed_seconds'],1),flush=True)
    out={'status':'Exact partial companion smooth target partition; principal-value term not included',
         'shard':args.shard,'shards':args.shards,'records':completed,'elapsed_seconds':time.perf_counter()-start}
    scratch_path(f'pair_companion_smooth_shard_{args.shard}.json').write_text(json.dumps(out,separators=(',',':')),encoding='utf-8')
    if args.shards==1: combine(completed)


if __name__=='__main__': main()
