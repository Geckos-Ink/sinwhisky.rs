#!/usr/bin/env python3
"""Hierarchical circles and bounded adaptive frequencies; private outputs outside repo.

Compare arbitrary-frequency geometry queries with direct sampled sums and matched
Hann FFT controls. Only aggregate measurements and rendered plots belong in docs.
"""
import argparse
from collections import Counter
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from aft_compare import ROOT, decode, periodic_hann
from geometric_compare import predict, save


def direct_queries(dense, zoom, omegas):
    dense=np.asarray(dense,dtype=float)
    y=dense*periodic_hann(len(dense))/(len(dense)/2)
    t=np.arange(len(dense))/zoom
    return np.asarray([np.dot(y,np.exp(-1j*omega*t)) for omega in omegas])


def run_frames(binary, frames, error=16, span=128, zoom=1, depth=12, budget=8192,
               repeats=3, coarse_error=None, prune=True):
    frames=np.asarray(frames,dtype=float)
    if frames.ndim!=2 or len(frames)==0:raise ValueError('complete frames required')
    coarse_error=error*8 if coarse_error is None else coarse_error
    text='\n'.join(' '.join(format(v,'.17g') for v in x) for x in frames)
    proc=subprocess.run([str(binary),str(frames.shape[1]),str(error),str(span),str(zoom),str(depth),
                         str(budget),str(repeats),str(coarse_error),str(int(prune))],input=text,
                        text=True,capture_output=True,check=True)
    rows=[json.loads(s) for s in proc.stdout.splitlines()]
    if len(rows)!=len(frames):raise ValueError('frame count mismatch')
    for x,row in zip(frames,rows):verify(x,row,error,zoom,budget)
    return rows


def verify(x,row,error,zoom,budget):
    dense=np.asarray(row['dense'])
    if dense.shape!=(len(x)*zoom,) or not np.all(np.isfinite(dense)):raise ValueError('dense shape/values')
    independent=predict(row,np.arange(len(dense))/zoom)
    if np.max(abs(dense-independent))>1e-7*max(1,float(np.max(abs(x)))):raise ValueError('circle prediction mismatch')
    peak=np.max(abs(dense[::zoom]-x))
    if peak>error or not math.isclose(peak,row['max_error'],rel_tol=1e-8,abs_tol=1e-8):raise ValueError('source error mismatch')
    q=row['queries']
    if q['coarse']+q['fine']>budget:raise ValueError('query budget exceeded')
    points=row['points']
    if not all(points[i][0]<points[i+1][0] for i in range(len(points)-1)):raise ValueError('frequencies unordered')
    # Every reported peak, five off-bin controls, and spread points (including
    # skipped fine probes) are checked independently, without Rust finite sums.
    samples=[points[i] for i in np.linspace(0,len(points)-1,min(33,len(points)),dtype=int)]
    omegas=[p['omega'] for p in row['peaks']]+[p[0] for p in row['check']]+[p[0] for p in samples]
    reference=direct_queries(dense,zoom,omegas)
    expected=[complex(*p['value']) for p in row['peaks']]+[complex(p[1],p[2]) for p in row['check']]+[complex(p[1],p[2]) for p in samples]
    uncertainties=[0.]*(len(row['peaks'])+len(row['check']))+[p[3] for p in samples]
    eps=3e-8*max(1,float(np.max(abs(reference))))
    errors=np.abs(reference-expected)
    if np.any(errors>np.asarray(uncertainties)+eps):raise ValueError('arbitrary frequency discrepancy')
    exact=errors[np.asarray(uncertainties)==0]
    row['_verification']={'checked_frequencies':len(omegas),'max_exact_absolute_difference':float(np.max(exact,initial=0)),
        'max_exact_relative_difference':float(np.max(exact,initial=0)/max(1,float(np.max(abs(reference)))))}


def fft_peaks(dense, zoom, rate, padding=64, relative_floor=.02):
    n=len(dense);y=np.asarray(dense)*periodic_hann(n)
    z=np.fft.rfft(y,n=n*padding)/(n/2)
    frequency=np.arange(len(z))*rate/(n*padding/zoom)
    mag=abs(z)
    mask=(mag[1:-1]>mag[:-2]) & (mag[1:-1]>=mag[2:]) & (mag[1:-1]>=relative_floor*np.max(mag))
    indices=np.flatnonzero(mask)+1
    if mag[0]>mag[1] and mag[0]>=relative_floor*np.max(mag):indices=np.r_[0,indices]
    return frequency,z,indices


def matched_peak_diagnostics(row,zoom,rate,n):
    f,z,idx=fft_peaks(row['dense'],zoom,rate)
    found=np.asarray([p['omega']*rate/(2*np.pi) for p in row['peaks']])
    reference=f[idx]
    distance=np.min(abs(reference[:,None]-found[None,:]),axis=1) if len(found) and len(reference) else np.full(len(reference),np.inf)
    missed=int(np.sum(distance>rate/n*.75))
    return {'dense_reference_peaks':len(reference),'adaptive_peaks':len(found),
            'unmatched_reference_peaks':missed,
            'matched_median_distance_hz':float(np.median(distance[distance<=rate/n*.75])) if np.any(distance<=rate/n*.75) else None}


def aggregate(rows,rate,n,zoom=1):
    hist=Counter(str(m['end']-m['start']) for r in rows for m in r['models'])
    timings={k+'_us':float(np.median([r[k+'_ns'] for r in rows])/1000) for k in ['fit','prepare','adaptive','fine_only','dp']}
    diagnostics=[matched_peak_diagnostics(r,zoom,rate,n) for r in rows]
    return {'frames':len(rows),'mean_descriptors':float(np.mean([r['circles']+r['lines'] for r in rows])),
        'mean_circles':float(np.mean([r['circles'] for r in rows])),
        'mean_unmerged_descriptors':float(np.mean([r['unmerged_count'] for r in rows])),
        'mean_dp_descriptors':float(np.mean([r['dp_count'] for r in rows])),
        'mean_coarse_descriptors':float(np.mean([r['coarse_count'] for r in rows])),
        'mean_merges':float(np.mean([r['merges'] for r in rows])),
        'mean_coarse_queries':float(np.mean([r['queries']['coarse'] for r in rows])),
        'mean_fine_queries':float(np.mean([r['queries']['fine'] for r in rows])),
        'mean_fine_only_queries':float(np.mean([r['queries']['fine_only'] for r in rows])),
        'mean_skipped_fine':float(np.mean([r['queries']['skipped_fine'] for r in rows])),
        'mean_predictor_disagreement_bound':float(np.mean([r['queries']['disagreement_bound'] for r in rows])),
        'budget_exhausted_frames':sum(r['queries']['budget_exhausted'] for r in rows),
        'mean_adaptive_peaks':float(np.mean([len(r['peaks']) for r in rows])),
        'source_peak_error':max(r['max_error'] for r in rows),'mean_source_rmse':float(np.mean([r['rmse'] for r in rows])),
        'stop_counts':{k:sum(r['stops'][k] for r in rows) for k in rows[0]['stops']},
        'span_histogram':dict(sorted(hist.items(),key=lambda kv:int(kv[0]))),
        'verification':{'checked_frequencies':sum(r['_verification']['checked_frequencies'] for r in rows),
            'max_exact_relative_difference':max(r['_verification']['max_exact_relative_difference'] for r in rows)},
        'peak_diagnostics':diagnostics,**timings}


def timed_fft(x,padding,repeats=21):
    y=x*periodic_hann(len(x));np.fft.rfft(y,n=len(x)*padding)
    durations=[]
    for _ in range(repeats):
        at=time.perf_counter_ns();np.fft.rfft(y,n=len(x)*padding);durations.append(time.perf_counter_ns()-at)
    return {'padding':padding,'output_bins':len(x)*padding//2+1,'median_us':float(np.median(durations)/1000),'ns':durations}


def render(report,selected,synthetic,depth_rows,output):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.right':False,'axes.spines.top':False})
    x,row=selected;n=len(x);rate=report['rate']
    f,z,idx=fft_peaks(row['dense'],1,rate)
    points=np.asarray(row['points'],dtype=float);freq=points[:,0]*rate/(2*np.pi)
    mag=np.hypot(points[:,1],points[:,2]);peaks=row['peaks']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    mask=f<5000;pmask=freq<5000
    axes[0].plot(f[mask],20*np.log10(np.maximum(abs(z[mask])/32768,1e-7)),color='#888888',lw=1,label='Same predictor, FFT padded 64×')
    axes[0].scatter(freq[pmask],20*np.log10(np.maximum(mag[pmask]/32768,1e-7)),s=7,color='#c06024',label='Nonuniform circle queries')
    for p in peaks:
        hz=p['omega']*rate/(2*np.pi)
        if hz<5000:axes[0].axvline(hz,color='#176650',lw=.6,alpha=.6)
    axes[0].set(xlabel='Frequency (Hz)',ylabel='Coefficient magnitude (dBFS)',title='Measured Beethoven window · adaptive peak locations',ylim=(-100,-10));axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    sx,srow=synthetic
    sf,sz,_=fft_peaks(srow['dense'],1,8000);f0=report['synthetic'][0]['given_frequency_hz']
    axes[1].plot(sf,abs(sz),color='#555555',label='Same predictor, FFT padded 64×')
    fx=np.fft.rfftfreq(len(sx),1/8000);zx=abs(np.fft.rfft(sx*periodic_hann(len(sx))))/(len(sx)/2)
    axes[1].stem(fx,zx,linefmt='#8c9ab3',markerfmt='o',basefmt=' ',label='Source FFT bins')
    strongest=max(srow['peaks'],key=lambda p:abs(complex(*p['value'])))
    axes[1].axvline(strongest['omega']*8000/(2*np.pi),color='#c06024',label='Adaptive circle peak')
    axes[1].axvline(f0,color='#176650',ls='--',label='Given synthetic frequency')
    axes[1].set(xlim=(f0-55,f0+55),xlabel='Frequency (Hz)',ylabel='Normalized coefficient magnitude',title='Off-bin peak · output precision differs from resolution');axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    save(fig,output/'adaptive-frequencies')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    starts=[e['start'] for e in report['excerpts']];xx=np.arange(len(starts));width=.24
    for i,(key,label) in enumerate([('mean_unmerged_descriptors','Before child merges'),('mean_descriptors','After checked refits'),('mean_dp_descriptors','Dyadic DP control')]):
        axes[0].bar(xx+(i-1)*width,[e[key] for e in report['excerpts']],width,label=label)
    axes[0].set(xticks=xx,xticklabels=[f'{s} s' for s in starts],ylabel='Mean circle + line count',title='Variable supports and merged children');axes[0].legend(fontsize=8)
    for key,label in [('adaptive_us','Coarse + fine search'),('fine_only_us','Fine-only search'),('fit_us','Hierarchy fitting'),('dp_us','Dyadic DP fitting')]:
        axes[1].plot(starts,[e[key]/1000 for e in report['excerpts']],'o-',label=label)
    axes[1].set(xlabel='Excerpt start (s)',ylabel='Median call time (ms)',title='Measured costs · fitting and queries separate',yscale='log');axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    save(fig,output/'adaptive-merging')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for d in report['depth']:
        axes[0].scatter(d['max_depth'],d['mean_adaptive_peaks'],s=40,color='#176650')
    axes[0].set(xlabel='Maximum refinement depth',ylabel='Retained relevant peaks',title='Depth limits determine when refinement stops');axes[0].grid(alpha=.2)
    zoom=report['zoom'];axes[1].plot([z['zoom'] for z in zoom],[z['strongest_hz'] for z in zoom],'o-',color='#c06024')
    axes[1].set(xlabel='Sampling zoom of same geometry',ylabel='Strongest estimated peak (Hz)',title='Zoom compares model assumptions, not new observations');axes[1].grid(alpha=.2)
    save(fig,output/'adaptive-depth-zoom')
    frames=[]
    for depth,drow in depth_rows:
        fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
        f,z,_=fft_peaks(drow['dense'],1,rate);mask=f<3500
        ax.plot(f[mask],abs(z[mask]),color='#888888',lw=1,label='Fixed fine predictor · FFT reference')
        pts=np.asarray(drow['points'],dtype=float);pf=pts[:,0]*rate/(2*np.pi);mask=pf<3500
        ax.scatter(pf[mask],np.hypot(pts[mask,1],pts[mask,2]),s=10,color='#c06024',label='Measured adaptive queries')
        for p in drow['peaks']:
            hz=p['omega']*rate/(2*np.pi)
            if hz<3500:ax.axvline(hz,color='#176650',lw=.8)
        ax.set(xlabel='Frequency (Hz)',ylabel='Normalized coefficient magnitude (PCM16)',
               title=f'Max depth {depth} · {len(drow["peaks"])} retained peaks · {drow["queries"]["fine"]} fine queries')
        ax.legend(fontsize=8);ax.grid(alpha=.2)
        data=io.BytesIO();fig.savefig(data,format='png',dpi=100,facecolor='white');plt.close(fig)
        data.seek(0);frames.append(Image.open(data).convert('RGB'))
    frames[0].save(output/'adaptive-depth.gif',save_all=True,append_images=frames[1:],duration=900,loop=0)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--ffmpeg',default=os.environ.get('FFMPEG','ffmpeg'));p.add_argument('--previous-report',type=Path);args=p.parse_args()
    source=args.input.resolve();output=args.output.resolve()
    if not source.is_file():p.error('input missing')
    if output.exists() or output==ROOT or ROOT in output.parents:p.error('output must be new and outside repository')
    ffmpeg=shutil.which(args.ffmpeg)
    if not ffmpeg:p.error('FFmpeg missing')
    subprocess.run(['cargo','build','--offline','--release','--example','adaptive_bench'],cwd=ROOT,check=True)
    meta=subprocess.run(['cargo','metadata','--offline','--no-deps','--format-version','1'],cwd=ROOT,capture_output=True,text=True,check=True)
    binary=Path(json.loads(meta.stdout)['target_directory'])/'release/examples/adaptive_bench';output.mkdir(parents=True)
    report={'experiment':'hierarchical-adaptive-FAFT-v1','date':time.strftime('%Y-%m-%d'),'rate':44100,'n':1024,
        'selection':'7 evenly spaced raw mono PCM16 windows per two-second excerpt, 21 total',
        'window':'Periodic Hann applied by prepared query AFTER circle fitting; no double windowing',
        'search':'Initial N/4 intervals, relative relevance floor 2%, midpoint refinement then bounded ternary peak refinement; merge distance half a source FFT bin',
        'limits':'Default interval depth 12, peak iterations 12, 8192 combined oracle calls; low-gain pruning is a heuristic, not peak completeness',
        'model':'Source max error 16 PCM16, max support 128, significand bits >=12; coarse error 128, bits >=8. Greedy source-informed split and up to four adjacent merge passes.',
        'timing':'Warmed in-process medians, output allocation included; fit excludes preparation, preparation excludes search. Coarse+fine search includes predictor L1 disagreement scan but excludes both model fits and spectrum preparation. FFT timed separately in NumPy.',
        'environment':{'platform':platform.platform(),'numpy':np.__version__,'python':platform.python_version(),
                       'rust':subprocess.run(['rustc','--version'],capture_output=True,text=True,check=True).stdout.strip()},
        'excerpts':[],'zoom':[],'depth':[],'precision_trials':[],'synthetic':[]}
    if args.previous_report is not None:
        previous=json.loads(args.previous_report.read_text())
        report['first_attempt']={'kernel':'Repeated closed-sum trigonometry per circle/query',
            'excerpts':previous['excerpts']}
    selected=None
    for start in [60,180,300]:
        print(f'{start} s: child merges and adaptive queries',flush=True)
        pcm=decode(ffmpeg,source,start,2,44100);frames=np.lib.stride_tricks.sliding_window_view(pcm,1024)[::512]
        indices=np.linspace(0,len(frames)-1,7,dtype=int);frames=frames[indices]
        rows=run_frames(binary,frames)
        summary=aggregate(rows,44100,1024);summary.update(start=start,frame_indices=indices.tolist())
        report['excerpts'].append(summary)
        if selected is None:selected=(frames[3],rows[3]);(output/'private-selected-model.json').write_text(json.dumps(rows[3]))
        print(f'  {summary["mean_descriptors"]:.1f} descriptors, {summary["mean_merges"]:.1f} merges, search {summary["adaptive_us"]/1000:.2f} ms',flush=True)
    x,row=selected
    for zoom in [1,2,4,8]:
        r=run_frames(binary,x[None,:],zoom=zoom)[0]
        strongest=max(r['peaks'],key=lambda p:abs(complex(*p['value'])))
        report['zoom'].append({'zoom':zoom,'strongest_hz':strongest['omega']*44100/(2*np.pi),**aggregate([r],44100,1024,zoom)})
    for coarse_error in [16,32,64,128]:
        r=run_frames(binary,x[None,:],coarse_error=coarse_error)[0]
        report['precision_trials'].append({'coarse_max_error':coarse_error,**aggregate([r],44100,1024)})
    depth_rows=[]
    for depth in [0,2,4,8,12]:
        r=run_frames(binary,x[None,:],depth=depth)[0];depth_rows.append((depth,r))
        report['depth'].append({'max_depth':depth,**aggregate([r],44100,1024)})
    # Bound-only control discloses whether the budget prevents completeness.
    r=run_frames(binary,x[None,:],prune=False)[0]
    report['bound_only_control']=aggregate([r],44100,1024)
    report['fft_controls']=[timed_fft(x,q) for q in [1,8,64]]
    n=512;t=np.arange(n);single=None
    for name,frequencies in [('off-bin-single',[401.337]),('separated-pair',[401.337,553.721]),('near-pair',[401.337,403.337])]:
        sx=sum((1 if i==0 else .65)*np.sin(2*np.pi*f*t/8000+.31*i) for i,f in enumerate(frequencies))
        r=run_frames(binary,sx[None,:],error=1e-5,coarse_error=1e-3)[0]
        dominant=max(r['peaks'],key=lambda p:abs(complex(*p['value'])))['omega']*8000/(2*np.pi)
        f,z,idx=fft_peaks(sx,1,8000,padding=1);fft_dominant=float(f[np.argmax(abs(z))])
        pf,pz,_=fft_peaks(sx,1,8000,padding=64);padded=float(pf[np.argmax(abs(pz))])
        report['synthetic'].append({'name':name,'given_frequency_hz':frequencies[0],'given_frequencies_hz':frequencies,
            'adaptive_dominant_hz':dominant,'source_fft_dominant_hz':fft_dominant,'padded64_fft_dominant_hz':padded,
            'above_10percent_peak_frequencies_hz':[p['omega']*8000/(2*np.pi) for p in r['peaks'] if abs(complex(*p['value']))>=.1*max(abs(complex(*q['value'])) for q in r['peaks'])],
            **aggregate([r],8000,n)})
        if single is None:single=(sx,r)
    render(report,selected,single,depth_rows,output)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(output/'report.json')

if __name__=='__main__':main()
