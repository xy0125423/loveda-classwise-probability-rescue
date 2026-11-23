"""Redraw five statistical figures from packaged aggregate records only."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyBboxPatch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DATA=json.loads((ROOT/'results/figure_statistics.json').read_text(encoding='utf-8'))
C={'status_colors':{'B':'#0072B2','H':'#D55E00','W':'#CC79A7'}}
NAMES={'fig01_design':'overview','fig02_failure':'failure_analysis','fig04_distributions':'probability_analysis','fig05_ablation':'ablation','fig06_target':'results'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False})

def save(fig,name,*args):
    """Keep all scientific values, changing only the portable font family."""
    fig.savefig(DEST/(NAMES[name]+'.png'),dpi=220,facecolor='white',bbox_inches='tight',pad_inches=.12); plt.close(fig)

def frame(rows: int, cols: int, height: float, **kwargs):
    return plt.subplots(rows,cols,figsize=(180/25.4,height/25.4),squeeze=False,**kwargs)

def title(ax, text: str) -> None:
    ax.set_title(text,loc='left',fontweight='bold',pad=9)

def box(ax,x,y,w,h,text,color='#F0F4F6',fs=9):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.012',linewidth=.7,edgecolor='#84929B',facecolor=color))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs)

def arrow(ax,x1,y1,x2,y2):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',color='#596973',lw=1))

def design():
    fig,axs=frame(1,1,141);ax=axs[0,0];ax.set_axis_off();ax.set_xlim(0,1);ax.set_ylim(0,1)
    title(ax,'(a) Two frozen models; class-specific branches')
    box(ax,.01,.78,.28,.15,'Urban-trained model\nSource calibration: 339\nWater: ratio / probability')
    box(ax,.37,.78,.28,.15,'Rural-trained model\nSource calibration: 496\nBuilding: probability / ratio')
    box(ax,.73,.78,.25,.15,'Separate weights\nNo joint multiclass\nrescue experiment',color='#FCF0DE')
    ax.text(.01,.69,'(b) One-step rescue with frozen source thresholds',weight='bold')
    for x,text in zip([.01,.265,.52,.775],['Background winner\nHighest non-BG = c','Score: Pc or\nlog[(Pc+eps)/(P0+eps)]','Q0.999 from all\nvalid source GT != c','Score >= threshold\nChange once to c']):box(ax,x,.50,.215,.13,text,fs=8.3)
    for x in [.225,.48,.735]:arrow(ax,x,.565,x+.04,.565)
    ax.text(.01,.42,'(c) Development chronology (historical contact is retained)',weight='bold')
    labels=['Target diagnosis\nand branch design','Water 981 fixed\n(excludes 11 pilot)','Urban 677 explored\nthen Building fixed','Source five-fold\nAFTER target results']
    for x,text in zip([.01,.265,.52,.775],labels):box(ax,x,.23,.215,.13,text,color='#F4F0E7',fs=8.1)
    for x in [.225,.48,.735]:arrow(ax,x,.295,x+.04,.295)
    ax.text(.01,.13,'Evaluation: B wrong-to-right | H right-to-wrong | W wrong-to-wrong',fontsize=9)
    ax.text(.01,.06,'Net = B - H; M = B + H + W.  No independent geographic benchmark.',fontsize=8.3)
    fig.subplots_adjust(left=.035,right=.98,top=.93,bottom=.03)
    save(fig,'fig01_design','Separate model branches and true chronology','S03_SPLIT;S09_CONFIG;S13_THRESH;S14_AUDIT')

def failure():
    classes=['Background','Building','Road','Water','Barren','Forest','Agriculture'];mats=[]
    for field in ['IoU','AUROC','Top2']:
        arr=np.zeros((7,2))
        for j,(cs,ps) in enumerate([('S06_CLASS','S08_PROB'),('T01_CLASS','S11_PROB')]):
            for i,c in enumerate(classes):
                cr=next(x for x in DATA[cs] if x['method']=='B0' and x['class_name'].lower()==c.lower())
                pr=next(x for x in DATA[ps] if x['class_name']==c and x['scope']=='GT_class' and x['domain'].startswith('Target'))
                arr[i,j]=float(cr['IoU'])*100 if field=='IoU' else float(pr['exact_unique_top2_rate'])*100 if field=='Top2' else float(pr.get('binned4096_one_vs_rest_AUROC',pr.get('binned4096_OVR_AUROC')))
        mats.append(arr)
    fig,axs=frame(1,3,100);fig.subplots_adjust(left=.16,right=.97,top=.83,bottom=.16,wspace=.26)
    for k,(a,arr,label) in enumerate(zip(axs[0],mats,['(a) Baseline IoU (%)','(b) OVR AUROC','(c) GT in Top-2 (%)'])):
        a.imshow(arr,aspect='auto',cmap='Blues',vmin=0,vmax=1 if k==1 else 100)
        title(a,label);a.set_xticks([0,1],['U → R\n992 images','R → U\n677 images']);a.set_yticks(range(7),classes if k==0 else ['']*7)
        a.tick_params(length=0)
        for i in range(7):
            for j in range(2):
                v=arr[i,j];display=f'{v:.3f}' if k==1 else ('<0.1' if 0<v<.1 else f'{v:.1f}');a.text(j,i,display,ha='center',va='center',color='white' if v>(.6 if k==1 else 60) else '#152F43',weight='bold' if v==0 else 'normal',fontsize=9)
    save(fig,'fig02_failure','Zero argmax output can coexist with residual probability ranking','S06_CLASS;T01_CLASS;S08_PROB;S11_PROB',list(axs[0]))

def distributions():
    z={k:np.asarray(v) for k,v in json.loads((ROOT/'results/distribution_bins.json').read_text()).items()};fig,axs=frame(2,2,166);fig.subplots_adjust(left=.10,right=.86,top=.93,bottom=.18,wspace=.35,hspace=.55)
    centers=(np.arange(10000)+.5)/10000
    for col,(key,xlabel) in enumerate([('pb_hists','Building probability'),('pbg_hists','Background probability')]):
        a=axs[0,col]
        for idx,label,color in [(0,'All valid non-Building','#687B8B'),(3,'All valid Building','#007D8A')]:
            hist=z[key][idx];bins=hist.reshape(1000,10).sum(1);dens=bins/hist.sum()/.001
            xc=centers.reshape(1000,10).mean(1)
            lim=(.12,.25) if col==0 else (0,.85)
            shown=(xc>=lim[0])&(xc<=lim[1])
            a.plot(xc[shown],dens[shown],label=label,color=color)
        if col==0:a.axvline(.22051838040351868,color='#007D8A',ls='--',lw=1);a.set_xlim(.12,.25)
        else:a.set_xlim(0,.85)
        title(a,f'({chr(97+col)}) '+xlabel);a.set_xlabel(xlabel);a.set_ylabel('Density');a.set_ylim(bottom=0)
    for col,(idx,label) in enumerate([(0,'True Building'),(1,'True Background')]):
        a=axs[1,col];counts=z['joint_2d_cand'][idx]
        im=a.pcolormesh(z['pbg_edges'],z['pb_edges'],np.ma.masked_equal(counts,0),cmap='Blues',norm=LogNorm(1,max(1,z['joint_2d_cand'][:2].max())),rasterized=True,shading='flat')
        slope=np.exp(-.1564700133800514)
        x=np.linspace(max(z['pbg_edges'][0],z['pb_edges'][0]/slope),min(z['pbg_edges'][-1],z['pb_edges'][-1]/slope),100);a.plot(x,slope*x,color='#6558A6',ls='--',label='Ratio threshold')
        a.axhline(.22051838040351868,color='#007D8A',label='Probability threshold')
        a.set_xlim(z['pbg_edges'][0],z['pbg_edges'][-1]);a.set_ylim(z['pb_edges'][0],z['pb_edges'][-1]);title(a,f'({chr(99+col)}) '+label+' candidates')
        a.set_xlabel('Background probability');a.set_ylabel('Building probability')
    cax=fig.add_axes([.89,.18,.018,.275]);cb=fig.colorbar(im,cax=cax);cb.set_label('Candidate pixels / saved bin',fontsize=7.5)
    cb.set_ticks([1,100,10000,1000000]);cb.set_ticklabels(['1','1e2','1e4','1e6']);cb.ax.tick_params(labelsize=8)
    h1,l1=axs[0,0].get_legend_handles_labels();h2,l2=axs[1,0].get_legend_handles_labels()
    fig.legend(h1+h2,l1+l2,loc='lower center',bbox_to_anchor=(.49,.035),ncols=2,fontsize=7.8,frameon=False)
    fig.text(.5,.012,'Source 496 images; fixed saved bins; empty bins unfilled. No target threshold fitting.',ha='center',fontsize=7.5)
    save(fig,'fig04_distributions','Score boundaries interact with class-conditional background probability','G1-13 saved NPZ;S13_THRESH;S12_THRESH',list(axs.flat))

def ablation():
    fig,axs=frame(1,3,105);fig.subplots_adjust(left=.12,right=.97,top=.85,bottom=.27,wspace=.55)
    a=axs[0,0];title(a,'(a) Triggered sets')
    vals=[484597,1925433,0];a.barh([2,1,0],np.array(vals)/1e6,color=['#6558A6','#007D8A','#999999'])
    a.set_yticks([2,1,0],['Ratio only','Prob. only','Both']);a.set_xlabel('Modified pixels (million)');a.set_xlim(0,2.5)
    for y,v in zip([2,1,0],vals):a.text(v/1e6+.05,y,f'{v/1e6:.3f}',va='center',fontsize=8)
    a=axs[0,1];title(a,'(b) Source change types')
    for j,method in enumerate(['G1_12_Ratio_PBR','G1_13_Prob_PBR']):
        row=next(x for x in DATA['S13_GLOBAL'] if x['method']==method);bottom=0
        for key in ['B','H','W']:
            v=int(row[key])/1e6;a.bar(j,v,bottom=bottom,color=C['status_colors'][key],label=key if j==0 else None);bottom+=v
    a.set_xticks([0,1],['Ratio','Probability']);a.set_ylabel('Pixels (million)');a.legend(ncols=3,loc='upper left',bbox_to_anchor=(0,1.01),fontsize=8);a.set_ylim(0,2.5)
    a=axs[0,2];title(a,'(c) Later source OOF')
    vals=[x['B_minus_H']/1e6 for x in DATA['S14_AUDIT']['OOF_pooled_counts']];a.barh([3,2,1,0],vals,color=['#6558A6','#007D8A']*2)
    a.set_yticks([3,2,1,0],['Water ratio','Water prob.','Building ratio','Building prob.']);a.axvline(0,color='#555',lw=.6);a.set_xlabel('Net pixels (million)');a.set_xlim(-1,14)
    for y,v in zip([3,2,1,0],vals):a.text(v+.3 if v>=0 else .6,y,f'{v:.2f}',ha='left',va='center',fontsize=8)
    fig.text(.5,.08,'Different coverage; not a matched-coverage causal comparison. OOF followed target evaluation.',ha='center',fontsize=8)
    save(fig,'fig05_ablation','Disjoint sets and later OOF support conditional score effects','S13_QUAD;S13_GLOBAL;S14_AUDIT',list(axs[0]))

def target():
    fig,axs=frame(2,3,158);fig.subplots_adjust(left=.10,right=.97,top=.92,bottom=.13,wspace=.56,hspace=.67)
    for r,(sid,method,cl,n) in enumerate([('S10_GLOBAL','PBR_Water_only','Water',981),('T01_GLOBAL','G1_13_Building_Prob_PBR','Building',677)]):
        row=next(x for x in DATA[sid] if x['method']==method and (sid!='S10_GLOBAL' or x['scope']=='Target_Confirm_981'))
        a=axs[r,0];title(a,f'({chr(97+r*3)}) {cl}: {n} fixed images')
        vals=[float(row[cl+'_'+k])*100 for k in ['IoU','Precision','Recall']];a.barh([2,1,0],vals,color=['#007D8A','#899EA3','#007D8A']);a.set_yticks([2,1,0],['IoU','Precision','Recall']);a.set_xlim(0,110);a.set_xlabel('Percent')
        for y,v in zip([2,1,0],vals):a.text(v+2,y,f'{v:.2f}',va='center',fontsize=8)
        a=axs[r,1];title(a,f'({chr(98+r*3)}) Paired gain vs B0')
        for j,metric in enumerate(['OA','mIoU']):
            if r==0:
                ci=next(x for x in DATA['S10_CI'] if x['method']==method and x['metric']=='Δ'+metric+' vs B0');v=float(ci['point_estimate']);lo=float(ci['ci95_low']);hi=float(ci['ci95_high'])
            else:
                ci=next(x for x in DATA['T01_CI'] if x['comparison']=='G1_13_minus_B0' and x['metric']==metric);v=float(ci['estimate']);lo=float(ci['CI95_low']);hi=float(ci['CI95_high'])
            a.errorbar(v,1-j,xerr=[[v-lo],[hi-v]],fmt='o',color='#007D8A',capsize=3,ms=4)
        a.set_yticks([1,0],['OA','mIoU']);a.set_ylim(-.5,1.5);a.set_xlim(0,.65);a.axvline(0,lw=.6,color='gray');a.set_xlabel('Gain (pp), 95% CI')
        a=axs[r,2];title(a,f'({chr(99+r*3)}) Label-change balance')
        fields=['B_wrong_to_right','H_right_to_wrong','W_wrong_to_wrong','net_benefit_B_minus_H'] if r==0 else ['B','H','W','net_B_minus_H']
        vals=[int(row[k])/1e6 for k in fields];a.barh([3,2,1,0],vals,color=[C['status_colors'][k] for k in ['B','H','W']]+['#394C59']);a.set_yticks([3,2,1,0],['B','H','W','B - H']);a.set_xlabel('Pixels (million)');a.set_xlim(0,3.5)
        for y,v in zip([3,2,1,0],vals):a.text(v+.06,y,f'{v:.3f}',va='center',fontsize=8)
    fig.text(.5,.025,'Historical target contact in both sets. CI: paired images, 10,000 (Water) / 2,000 (Building) resamples.',ha='center',fontsize=7.8)
    save(fig,'fig06_target','Positive fixed-set gain coexists with low class recall and harmful changes','S10_GLOBAL;S10_CI;T01_GLOBAL;T01_CI',list(axs.flat))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,default=Path('outputs/figures')); args=parser.parse_args()
    DEST=args.output; DEST.mkdir(parents=True,exist_ok=True)
    for fn in [design,failure,distributions,ablation,target]: fn()
    print('Redrew five figures from saved statistics; no model reproduction performed.')
