"""Portable commands for the documented historical protocol.

Training and full-data inference are user-invoked reproduction steps, not run
as part of publishing this repository.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import numpy as np
from .calibration import calibrate
from .evaluation import confusion_matrix, summarize
from .rescue import rescue, effects

def load(path: Path):
    """Config files are JSON-compatible YAML to avoid ambiguous scalar parsing."""
    return json.loads(path.read_text(encoding='utf-8'))

def dump(path: Path, value):
    """Keep outputs separate and never silently replace a completed run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f: json.dump(value, f, indent=2, allow_nan=False)

def records(path: Path, branch: dict, require_source=False):
    """Verify direction and source calibration role before fitting thresholds."""
    meta = load(path/'manifest.json')
    if meta['direction'] != branch['direction']: raise ValueError('Wrong direction')
    if require_source and meta['role'] != 'calibration': raise ValueError('Source calibration cache required')
    return [path/(Path(n).stem+'.npz') for n in meta['names']]

def evaluate_files(paths, branch, threshold):
    """Accumulate pooled pixel metrics and image-level label-change counts."""
    before = np.zeros((7,7),np.int64); after=before.copy(); counts={k:0 for k in ['B','H','W','M','net']}
    per_image=[]
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            prob, gt = z['prob'], z['gt']
            b0=prob.argmax(0).astype(np.uint8)
            new=rescue(prob,branch['target'],threshold,branch['scoring'],branch['arithmetic'])
            before += confusion_matrix(gt,b0); after += confusion_matrix(gt,new)
            row=effects(gt,b0,new)
            for k in counts: counts[k]+=row[k]
            per_image.append(dict(image=path.stem,**row))
    return dict(B0=summarize(before),rescue=summarize(after),effects=counts,per_image=per_image)

def main(argv=None):
    """Run one explicitly selected reproduction command."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','train','infer','calibrate','evaluate','source-cv'])
    parser.add_argument('--config',type=Path,default=Path('configs/urban_to_rural.yaml'))
    parser.add_argument('--thresholds',type=Path,default=Path('configs/rescue_thresholds.yaml'))
    parser.add_argument('--branch',default='water_ratio')
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--weights',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--cache',type=Path)
    parser.add_argument('--role',choices=['calibration','selection','target'],default='target')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    root=Path(__file__).resolve().parents[2]
    if args.command in ['calibrate','evaluate','source-cv']:
        branch=load(args.thresholds)['branches'][args.branch]
        paths=records(args.cache,branch,args.command!='evaluate')
        if args.command=='calibrate':
            tau=calibrate(paths,branch['target'],branch['scoring'],branch['arithmetic'])
            dump(args.output,dict(branch=args.branch,threshold=tau,frozen_reference=branch['threshold'],negative_pool='all valid source GT != target',quantile=.999))
        elif args.command=='evaluate':
            if branch['role'].startswith('negative') and load(args.cache/'manifest.json')['role']=='target':
                raise ValueError('Historical negative control has no target evaluation')
            dump(args.output,evaluate_files(paths,branch,branch['threshold']))
        else:
            indices=np.random.default_rng(20261122).permutation(len(paths)); rows=[]
            if len(paths)<5: raise ValueError('Five folds require at least five images')
            for fold,test in enumerate(np.array_split(indices,5)):
                train=[p for i,p in enumerate(paths) if i not in set(test)]
                tau=calibrate(train,branch['target'],branch['scoring'],branch['arithmetic'])
                result=evaluate_files([paths[i] for i in test],branch,tau)
                rows.append(dict(fold=fold,threshold=tau,held_out=[paths[i].stem for i in test],**result))
            dump(args.output,dict(seed=20261122,folds=rows,scope='source image OOF, no weight retraining'))
        return
    # Heavy model dependencies are optional for the synthetic quick start.
    import torch
    from .training_data import LoveDAStore
    cfg=load(args.config); direction=cfg['direction']
    split=load(root/'configs/splits.json')[direction]
    if args.data_root is None: parser.error('--data-root is required')
    store=LoveDAStore(str(args.data_root),str(args.data_root),'Train' if args.command=='train' else 'Val',split['source'] if args.command in ['train','prepare'] or args.role!='target' else split['target'])
    if args.command=='prepare':
        if not set(split['model_selection']+split['calibration'])==set(store.names): raise ValueError('Source names differ from frozen manifest')
        dump(args.output,dict(source=store.describe(),split_counts={k:len(split[k]) for k in ['model_selection','calibration']})); return
    from .model import ResNet18UNet
    if args.weights is None: parser.error('--weights is required (official encoder weights; no implicit download)')
    torch.set_num_threads(cfg['cpu_threads'])
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    model=ResNet18UNet(args.weights,cfg)
    if args.command=='train':
        from .training import train_epoch, validate
        val=LoveDAStore(str(args.data_root),str(args.data_root),'Val',split['source'])
        if set(store.names) & set(val.names): raise ValueError('Train/Val filename overlap')
        args.output.mkdir(parents=True,exist_ok=False)
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=2,threshold=0,threshold_mode='abs',min_lr=1e-6)
        best=-float('inf'); bad=0; history=[]
        for epoch in range(1,cfg['effective_epochs']+1):
            tr=train_epoch(model,store,cfg,epoch,opt)
            result,_,_=validate(model,val,split['model_selection'],cfg,epoch)
            score=result['mIoU']; sched.step(score)
            if score>best:
                best=score; bad=0
                torch.save(dict(model_state_dict=model.state_dict(),epoch=epoch,config=cfg),args.output/'best.pt')
            else: bad+=1
            history.append(dict(epoch=epoch,train=tr,selection=result))
            if bad>=cfg['early_stopping_patience']: break
        dump(args.output/'history.json',history); return
    from .inference import sliding_logits
    if args.checkpoint is None: parser.error('--checkpoint required for inference')
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'],strict=True); model.eval()
    names=sorted(split['calibration' if args.role=='calibration' else 'model_selection']) if args.role!='target' else store.names
    if direction=='urban_to_rural' and args.role=='target': names=load(root/'configs/water_confirmation.json')
    if not set(names)<=set(store.names): raise ValueError('Required image names absent')
    args.output.mkdir(parents=True,exist_ok=False)
    from .data import map_labels
    for name in names:
        rgb,raw=store.read(name)
        _,logits,_=sliding_logits(model,rgb,cfg['normalization_mean'],cfg['normalization_std'],256,128,2)
        np.savez(args.output/(Path(name).stem+'.npz'),prob=logits.softmax(0).numpy(),gt=map_labels(raw))
    dump(args.output/'manifest.json',dict(direction=direction,role=args.role,names=names,checkpoint_epoch=ckpt.get('epoch')))

if __name__=='__main__': main()
