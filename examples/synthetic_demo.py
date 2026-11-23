"""No data, model, network or weights: exercise the complete rescue API."""
from pathlib import Path
import argparse
import json
import numpy as np
from loveda_rescue.rescue import rescue, effects
from loveda_rescue.evaluation import confusion_matrix, summarize
from loveda_rescue.visualization import save_effects

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,default=Path('outputs/demo')); args=parser.parse_args()
    prob=np.full((7,2,4),.01,np.float32); prob[0]=.5; prob[1]=.45
    prob /= prob.sum(0,keepdims=True)
    gt=np.array([[1,0,4,255],[1,0,4,255]])
    before=prob.argmax(0).astype(np.uint8)
    after=rescue(prob,1,.22,'probability')
    result=dict(effects=effects(gt,before,after),metrics=summarize(confusion_matrix(gt,after)))
    assert result['effects']==dict(B=2,H=2,W=2,M=6,net=0)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    save_effects(gt,before,after,args.output/'effects.png')
    print(json.dumps(result['effects']))

if __name__=='__main__': main()
