import csv, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
R='/home/claude/faultlines/results/'
r3={r['instance_id']:r['resolved']=='1' for r in csv.DictReader(open(R+'e3_rows.csv'))}
r5={r['instance_id']:r['resolved']=='1' for r in csv.DictReader(open(R+'e5_rows.csv'))}
r6={r[0]:r[1]=='True' for r in csv.reader(open(R+'e6_rows_basic.csv'))}
ids=list(r3); n=lambda t:r3[t]+r5[t]+r6[t]
groups=[('Always resolved',[t for t in ids if n(t)==3]),('Sometimes resolved',[t for t in ids if 0<n(t)<3]),('Never resolved',[t for t in ids if n(t)==0])]
for g in groups: g[1].sort(key=lambda t:(-n(t), t))
SURF='#fcfcfb'; BLUE='#2a78d6'; GRAY='#e3e2dd'; T1='#0b0b0b'; T2='#52514e'
fig,ax=plt.subplots(figsize=(11,2.6),dpi=220); fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
c,g,ggap=1.0,0.3,2.6
rows=[('E3',r3),('E5',r5),('E6 (E5 repeated)',r6)]
x=0
for name,ts in groups:
    x0=x
    for t in ts:
        for ri,(rn,rd) in enumerate(rows):
            ax.add_patch(FancyBboxPatch((x,(2-ri)*(c+g)),c,c,boxstyle='round,pad=0,rounding_size=0.2',fc=BLUE if rd[t] else GRAY,ec='none'))
        x+=c+g
    ax.text(x0, 3*(c+g)+0.35, f"{name}: {len(ts)}", ha='left', va='bottom', fontsize=8.5, color=T1, fontweight='bold')
    x+=ggap
for ri,(rn,rd) in enumerate(rows):
    ax.text(-0.8,(2-ri)*(c+g)+c/2,f"{rn}   {sum(rd.values())}/70",ha='right',va='center',fontsize=8,color=T2)
ly=-1.7
for i,(lab,col) in enumerate([('resolved',BLUE),('not resolved',GRAY)]):
    lx=i*9
    ax.add_patch(FancyBboxPatch((lx,ly),c,c,boxstyle='round,pad=0,rounding_size=0.2',fc=col,ec='none'))
    ax.text(lx+1.5,ly+c/2,lab,va='center',fontsize=8,color=T2)
ax.set_xlim(-14,x-ggap+0.3); ax.set_ylim(-2.0,5.6); ax.set_aspect('equal'); ax.axis('off')
plt.savefig('fig_variance.png',bbox_inches='tight',pad_inches=0.12,facecolor=SURF)
