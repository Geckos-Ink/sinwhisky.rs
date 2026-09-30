#!/usr/bin/env python3
"""Geometry-first FAFT experiments; original FFT and integral AFT are references.

Output must be new/outside the repository. Only aggregate JSON and figures may
be copied to docs; exported circle models and dense predictions are private audio.
"""
import argparse
from collections import Counter
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
from matplotlib.animation import FuncAnimation, PillowWriter
from aft_compare import ROOT, decode, window_frames, periodic_hann, timed_fft, relative_l2, differences, rust_frames as aft_frames

TRIALS = [
    dict(name='dense-anchors', error=1, span=2, phases=False, bits=53),
    dict(name='span8-midpoint', error=16, span=8, phases=False, bits=53),
    dict(name='span32-midpoint', error=16, span=32, phases=False, bits=53),
    dict(name='span128-midpoint', error=16, span=128, phases=False, bits=53),
    dict(name='span128-phase-search', error=16, span=128, phases=True, bits=53),
    dict(name='adaptive-precision', error=16, span=128, phases=True, bits=8),
    dict(name='tight-error', error=1, span=128, phases=True, bits=8),
    dict(name='wide-error', error=128, span=128, phases=True, bits=8),
    dict(name='prevalidated-precision', error=16, span=128, phases=True, bits=8, prevalidate=True),
]


def as_complex(values):
    z=np.asarray(values,dtype=float)
    return z[...,0]+1j*z[...,1]


def predict(row, positions):
    """Independent direct circle evaluation from exported centers/radii/angles."""
    positions=np.asarray(positions,dtype=float)
    y=np.empty_like(positions);coverage=np.zeros_like(positions,dtype=int)
    for model in row['models']:
        mask=(positions>=model['start']) & (positions<model['end'])
        t=positions[mask]-model['start'];length=model['end']-model['start']
        if model['kind']=='line':
            value=model['first']+(model['last']-model['first'])*t/length
        else:
            m=model['middle'];a=model['angles']
            angle=np.where(t<m,a[0]+(a[1]-a[0])*t/m,
                           a[1]+(a[2]-a[1])*(t-m)/(length-m))
            value=model['cy']+model['radius']*np.sin(angle)
        y[mask]=value*row['scale'];coverage[mask]+=1
    if not np.all(coverage==1):raise ValueError('invalid coverage')
    return y


def verify(x,row,trial,zoom):
    n=len(x);pcm=np.asarray(row['decoded']);dense=np.asarray(row['dense'])
    if pcm.shape!=(n,) or dense.shape!=(n*zoom,) or not np.all(np.isfinite(dense)):
        raise ValueError('invalid output shapes or values')
    peak=float(np.max(np.abs(pcm-x)));rms=float(np.sqrt(np.mean((pcm-x)**2)))
    if peak>trial['error'] or not math.isclose(peak,row['max_error'],rel_tol=1e-8,abs_tol=1e-8):
        raise ValueError('source error bound / metric mismatch')
    if not math.isclose(rms,row['rmse'],rel_tol=1e-8,abs_tol=1e-8):raise ValueError('RMS mismatch')
    independent=predict(row,np.arange(n*zoom)/zoom)
    if np.max(np.abs(independent-dense))>1e-7*max(1.,float(np.max(abs(x)))):
        raise ValueError('geometry prediction mismatch')
    reference=np.fft.rfft(dense)[:n//2+1]/zoom
    for key in ['closed','rotations']:
        z=as_complex(row[key])
        if z.shape!=reference.shape or not np.all(np.isfinite(z)):raise ValueError('invalid spectrum')
        difference=np.linalg.norm(z-reference)
        # Numerical check concerns the SAME predicted waveform, not original audio.
        if difference>2e-8*max(1.,float(np.linalg.norm(reference))):
            raise ValueError(f'{key} spectral discrepancy: {difference/max(1,np.linalg.norm(reference))}')
    for model in row['models']:
        if model['kind']!='circle':continue
        for local,angle in zip([0,model['middle'],model['end']-model['start']],model['angles']):
            original=x[(model['start']+local)%n]/row['scale']
            vertical=(model['cy']+model['radius']*math.sin(angle)-original)*row['scale']
            horizontal=(model['cx']+model['radius']*math.cos(angle)-local*model['x_scale'])*row['scale']
            if max(abs(vertical),abs(horizontal))>trial['error']+1e-7:
                raise ValueError('quantized circle fitting-anchor discrepancy')


def run_frames(binary,frames,trial,zoom=1,repeats=1):
    text='\n'.join(' '.join(format(v,'.17g') for v in x) for x in frames)
    proc=subprocess.run([str(binary),str(frames.shape[1]),str(trial['error']),str(trial['span']),
                         str(int(trial['phases'])),str(trial['bits']),str(zoom),str(repeats),str(int(trial.get('prevalidate',False)))],
                        input=text,text=True,capture_output=True,check=True)
    rows=[json.loads(line) for line in proc.stdout.splitlines()]
    if len(rows)!=len(frames):raise ValueError('frame count mismatch')
    for x,row in zip(frames,rows):verify(x,row,trial,zoom)
    return rows


def summarize(rows,frames):
    fft=np.fft.rfft(frames,axis=1)
    spectrum=np.asarray([as_complex(r['rotations']) for r in rows])
    predicted=np.asarray([r['decoded'] for r in rows])
    moments=Counter();widths=Counter();phases=Counter();gains=Counter()
    for r in rows:
        for m in r['models']:
            widths[m['bits']]+=1;moments[m['end']-m['start']]+=1
            if m['kind']=='circle':
                phases[format(m['middle']/(m['end']-m['start']),'.3g')]+=1;gains[m['gain']]+=1
    return {'frames':len(rows),'mean_circles':float(np.mean([r['circles'] for r in rows])),
            'mean_lines':float(np.mean([r['lines'] for r in rows])),
            'mean_pieces':float(np.mean([r['pieces'] for r in rows])),
            'mean_estimated_model_bits':float(np.mean([r['estimated_model_bits'] for r in rows])),
            'peak_error':max(r['max_error'] for r in rows),
            'rms_error':float(np.sqrt(np.mean((predicted-frames)**2))),
            'spectrum_vs_source_fft_relative_l2':relative_l2(spectrum,fft),
            'rotations_vs_prediction_fft_relative_l2':relative_l2(spectrum,np.fft.rfft(predicted,axis=1)),
            'closed_vs_prediction_fft_relative_l2':relative_l2(np.asarray([as_complex(r['closed']) for r in rows]),np.fft.rfft(predicted,axis=1)),
            'precision_histogram':dict(sorted(widths.items())),'span_histogram':dict(sorted(moments.items())),
            'middle_fraction_histogram':dict(sorted(phases.items())),'aperture_gain_histogram':dict(sorted(gains.items())),
            **{key+'_us':float(np.median([r[key+'_ns'] for r in rows])/1000) for key in ['fit','closed','rotation','nine_query']}}


def save(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=160,facecolor='white')
    fig.savefig(path.with_suffix('.svg'),facecolor='white')
    svg=path.with_suffix('.svg');svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines())+'\n')
    plt.close(fig)


def render(report,selected,output):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.right':False,'axes.spines.top':False})
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    names=[t['name'] for t in TRIALS[1:6]]
    for excerpt in report['excerpts']:
        rows=excerpt['trials'][1:6]
        axes[0].plot(range(len(rows)),[r['mean_circles']+r['mean_lines'] for r in rows],'o-',label=f'{excerpt["start"]} s')
    axes[0].set(xticks=range(len(names)),xticklabels=['span 8','span 32','span 128','+phase search','+precision'],
                ylabel='Mean circle + line descriptors',title='Retrying geometry definitions · max error 16 PCM16')
    axes[0].tick_params(axis='x',labelrotation=22);axes[0].legend();axes[0].grid(alpha=.2)
    sweep=report['scaling'];n=[r['n'] for r in sweep]
    for key,label in [('fft_us','NumPy rFFT (source)'),('aft_us','Integral AFT reference'),('closed_us','Geometry finite sums'),
                      ('rotation_us','Geometry batched rotations'),('nine_query_us','Geometry 9 selected bins')]:
        axes[1].loglog(n,[r[key] for r in sweep],'o-',label=label)
    axes[1].set(xlabel='Source frame size N',ylabel='Median query time (µs)',title='Prepared geometry queries · fitting cost in report')
    axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    fig.suptitle('Geometry-first FAFT · actual circumcircles → spectrum',fontsize=14)
    save(fig,output/'geometric-search')

    source,row=selected;n=len(source);t=np.arange(n)/report['rate']*1000
    dense=predict(row,np.arange(n*4)/4);decoded=np.asarray(row['decoded'])
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    axes[0,0].plot(t,source/32768,color='#444444',label='Windowed source')
    axes[0,0].plot(np.arange(n*4)/4/report['rate']*1000,dense/32768,color='#b85827',lw=.9,label='Circle predictor, zoom 4')
    for m in row['models']:
        for local in ([0,m['middle']] if m['kind']=='circle' else [0]):
            i=m['start']+local;axes[0,0].plot(t[i],source[i]/32768,'.',color='#176650',ms=3)
    axes[0,0].set(title=f'{row["circles"]} waveform circles + {row["lines"]} lines · fitting points marked',xlabel='Local time (ms)',ylabel='Full-scale amplitude')
    axes[0,0].legend(fontsize=8)
    axes[1,0].plot(t,decoded-source,color='#b85827');axes[1,0].set(title=f'Peak {row["max_error"]:.3f}; RMS {row["rmse"]:.3f} PCM16',xlabel='Local time (ms)',ylabel='Sample prediction error')
    fft=np.fft.rfft(source);f=np.fft.rfftfreq(n,1/report['rate']);coef=2/(periodic_hann(n).sum()*32768)
    for spec,label in [(fft,'Source FFT'),(as_complex(row['rotations']),'Direct circle spectrum')]:
        axes[0,1].semilogx(f[1:-1],20*np.log10(np.maximum(abs(spec[1:-1])*coef,1e-8)),lw=1,label=label)
    axes[0,1].set(title='Phase-aware sum from centers/radii/angles',xlabel='Frequency (Hz)',ylabel='Magnitude (dBFS)',ylim=(-130,0));axes[0,1].legend(fontsize=8)
    circ=[m for m in row['models'] if m['kind']=='circle']
    axes[1,1].scatter([m['radius']*row['scale'] for m in circ],[m['bits'] for m in circ],s=18)
    axes[1,1].set(xscale='log',xlabel='Radius in scaled waveform coordinates',ylabel='Significand bits per field',title='Different circle ranges and parameter precision')
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle('Beethoven at 60 s excerpt midpoint · geometry fitted before spectral analysis',fontsize=14)
    save(fig,output/'geometric-model')

    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for excerpt in report['excerpts']:
        for r in excerpt['prediction_trials']:
            axes[0].scatter(r['stride'],r['heldout_rms'],s=50,label=f'{excerpt["start"]} s, step {r["stride"]}')
    axes[0].set(xticks=[2,4],xlabel='Measured source-point stride',ylabel='Held-out RMS error (PCM16)',title='Predicting omitted points needs new evidence');axes[0].legend(fontsize=7);axes[0].grid(alpha=.2)
    z=report['zoom'];axes[1].plot([r['zoom'] for r in z],[r['source_fft_difference']*100 for r in z],'o-',label='Difference from source FFT')
    axes[1].set(xlabel='Model zoom density',ylabel='Relative complex difference (%)',title='Zoom changes the sampled circle function',ylim=(0,max(r['source_fft_difference']*100 for r in z)*1.2));axes[1].grid(alpha=.2)
    save(fig,output/'geometric-prediction')

    # Show a real fitted circumcircle and its two angular half rates, alongside the
    # complete measured predictor. x on the left is scaled geometric x, not time.
    circle=max(circ,key=lambda m:m['end']-m['start']);length=circle['end']-circle['start']
    fig,(ax,wave)=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    angles=np.linspace(min(circle['angles']),max(circle['angles']),400)
    ax.plot(circle['cx']+circle['radius']*np.cos(angles),circle['cy']+circle['radius']*np.sin(angles),color='#b85827',lw=1.2)
    for j in [0,circle['middle'],length]:
        ax.plot(j*circle['x_scale'],source[(circle['start']+j)%n]/row['scale'],'o',color='#176650',ms=4)
    marker,=ax.plot([],[],'o',color='#27589c',ms=6)
    ax.set(title='Circle through three waveform points',xlabel='Local scaled x',ylabel='Normalized amplitude',aspect='equal');ax.grid(alpha=.2)
    # Match geometric axes to the arc extent to avoid a huge empty full circle.
    wave.plot(t,source/32768,color='#555555',label='Source');wave.plot(np.arange(n*4)/4/report['rate']*1000,dense/32768,color='#b85827',lw=.8,label='Geometry predictor')
    cursor=wave.axvline(0,color='#27589c',lw=.8);mark,=wave.plot([],[],'o',color='#27589c',ms=4)
    wave.axvspan(t[circle['start']],(circle['end'])/report['rate']*1000,color='#c4dded',alpha=.4)
    wave.set(title='Angular progress projects to waveform amplitude',xlabel='Local time (ms)',ylabel='Full-scale amplitude');wave.legend(fontsize=8);wave.grid(alpha=.2)
    def update(index):
        j=index*length/47
        a=circle['angles'];m=circle['middle']
        theta=a[0]+(a[1]-a[0])*j/m if j<m else a[1]+(a[2]-a[1])*(j-m)/(length-m)
        px=circle['cx']+circle['radius']*math.cos(theta);py=circle['cy']+circle['radius']*math.sin(theta)
        marker.set_data([px],[py]);time_ms=(circle['start']+j)/report['rate']*1000
        cursor.set_xdata([time_ms]*2);mark.set_data([time_ms],[py*row['scale']/32768])
        fig.suptitle(f'Measured circle · span {length} · bits {circle["bits"]} · slowed angular projection')
        return marker,cursor,mark
    FuncAnimation(fig,update,frames=48,interval=100).save(output/'geometric-circle.gif',writer=PillowWriter(fps=10),dpi=100)
    plt.close(fig)



def supplemental_checks(report, pcm, binaries, ffmpeg, output):
    """Phase and actual encoded-bitrate controls, distinct from point/field precision."""
    frames=window_frames(pcm,report['n'],report['hop'])
    indices=np.linspace(0,len(frames)-1,21,dtype=int)
    reference=frames[indices]
    trial=TRIALS[8]
    rows=run_frames(binaries/'geometric_bench',reference,trial)
    report['first_excerpt_phase_comparison']=differences(
        np.fft.rfft(reference,axis=1),np.asarray([as_complex(r['rotations']) for r in rows]),report['rate'],report['n'])
    report['encoded_bitrate_controls']=[]
    for bitrate in [80,160]:
        encoded=output/f'private-mono-{bitrate}kbps.mp3'
        subprocess.run([ffmpeg,'-v','error','-nostdin','-f','s16le','-ar',str(report['rate']),'-ac','1','-i','pipe:0',
                        '-c:a','libmp3lame','-b:a',f'{bitrate}k',str(encoded)],
                       input=pcm.astype('<i2').tobytes(),capture_output=True,check=True)
        decoded=decode(ffmpeg,encoded,0,report['seconds'] if 'seconds' in report else 2,report['rate'])
        comparison=window_frames(decoded,report['n'],report['hop'])
        if comparison.shape[0]!=frames.shape[0]:raise ValueError('encoded control alignment/length mismatch')
        comparison=comparison[indices]
        rows=run_frames(binaries/'geometric_bench',comparison,trial)
        report['encoded_bitrate_controls'].append({
            'requested_mono_kbps':bitrate,'file_bytes':encoded.stat().st_size,'decoded_samples':len(decoded),
            'method':'Re-encode first 2-second mono decoded excerpt; FFmpeg honors gapless delay metadata; identical sample rate and window indices.',
            'decoded_fft_vs_original_relative_l2':relative_l2(np.fft.rfft(comparison,axis=1),np.fft.rfft(reference,axis=1)),
            'geometry':summarize(rows,comparison),
        })


def build():
    subprocess.run(['cargo','build','--offline','--release','--example','geometric_bench','--example','aft_bench'],cwd=ROOT,check=True)
    r=subprocess.run(['cargo','metadata','--offline','--no-deps','--format-version','1'],cwd=ROOT,capture_output=True,text=True,check=True)
    return Path(json.loads(r.stdout)['target_directory'])/'release/examples'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--ffmpeg',default=os.environ.get('FFMPEG','ffmpeg'));p.add_argument('--previous-report',type=Path);args=p.parse_args()
    source=args.input.resolve();output=args.output.resolve()
    if not source.is_file():p.error('input missing')
    if output.exists() or output==ROOT or ROOT in output.parents:p.error('output must be new, outside repository')
    ffmpeg=shutil.which(args.ffmpeg)
    if not ffmpeg:p.error('FFmpeg missing')
    binaries=build();output.mkdir(parents=True)
    report={'experiment':'geometry-first-FAFT-v1','date':time.strftime('%Y-%m-%d'),'n':1024,'rate':44100,'hop':512,
            'window':'periodic Hann applied to mono PCM16 before geometry fitting and source FFT',
            'selection':'21 evenly spaced windows from each two-second excerpt; all source samples checked; not every possible window',
            'objective':'minimum circle+line count, then hypothetical model bits, within a dyadic candidate bank',
            'timing':'warmed calls in-process; includes output allocation, excludes decoding/windowing/text/JSON/graphics; prepared geometry queries exclude fitting',
            'environment':{'platform':platform.platform(),'python':platform.python_version(),'numpy':np.__version__,
                           'rust':subprocess.run(['rustc','--version'],capture_output=True,text=True,check=True).stdout.strip()},
            'excerpts':[],'zoom':[],'scaling':[]}
    selected=None;first_pcm=None
    for start in [60,180,300]:
        pcm=decode(ffmpeg,source,start,2,44100)
        if first_pcm is None:first_pcm=pcm
        allframes=window_frames(pcm,1024,512);indices=np.linspace(0,len(allframes)-1,21,dtype=int);frames=allframes[indices]
        excerpt={'start':start,'frame_indices':indices.tolist(),'trials':[],'prediction_trials':[]}
        for trial in TRIALS:
            print(f'{start} s · {trial["name"]}',flush=True)
            rows=run_frames(binaries/'geometric_bench',frames,trial)
            data={**trial,**summarize(rows,frames)};excerpt['trials'].append(data)
            print(f'  {data["mean_circles"]:.1f} circles + {data["mean_lines"]:.1f} lines, fit {data["fit_us"]:.1f} µs, rotations {data["rotation_us"]:.1f} µs',flush=True)
            if start==60 and trial['name']=='prevalidated-precision':
                selected=(frames[10],rows[10]);(output/'private-selected-model.json').write_text(json.dumps(rows[10]))
        for stride in [2,4]:
            print(f'{start} s · predict held-out points, stride {stride}',flush=True)
            trial=TRIALS[8];rows=run_frames(binaries/'geometric_bench',frames[:,::stride],trial,zoom=stride)
            predictions=np.asarray([r['dense'] for r in rows]);mask=np.arange(1024)%stride!=0
            excerpt['prediction_trials'].append({'stride':stride,'fitted_point_max_error':max(r['max_error'] for r in rows),
                'heldout_rms':float(np.sqrt(np.mean((predictions[:,mask]-frames[:,mask])**2))),
                'heldout_max':float(np.max(abs(predictions[:,mask]-frames[:,mask]))),
                'spectrum_vs_full_source_fft_relative_l2':relative_l2(np.asarray([as_complex(r['rotations']) for r in rows])*stride,np.fft.rfft(frames,axis=1)[:,:1024//stride//2+1])})
        report['excerpts'].append(excerpt)
    first_report = args.previous_report
    if first_report is not None and first_report.is_file():
        first = json.loads(first_report.read_text())
        report['first_attempt'] = {
            'implementation': 'All quantization widths tried on nominal failures; repeated sine/cosine preparation in candidate checks; derivative-based line moments for all bins.',
            'scaling': first['scaling'],
            'adaptive_trials': [{'start':e['start'], **e['trials'][5]} for e in first['excerpts']],
        }
    x,row=selected
    for q in [1,2,4,8,16]:
        r=run_frames(binaries/'geometric_bench',x[None,:],TRIALS[8],zoom=q)[0]
        report['zoom'].append({'zoom':q,'source_fft_difference':relative_l2(as_complex(r['rotations']),np.fft.rfft(x)),
            'same_curve_fft_difference':relative_l2(as_complex(r['rotations']),np.fft.rfft(r['dense'])[:513]/q),
            'rotation_us':float(np.median(r['rotation_ns'])/1000)})
    for n in [256,512,1024,2048,4096]:
        print(f'Timing geometry at N={n}',flush=True)
        begin=len(first_pcm)//2-n//2;x=first_pcm[begin:begin+n]*periodic_hann(n)
        r=run_frames(binaries/'geometric_bench',x[None,:],TRIALS[8],repeats=5)[0]
        aft=aft_frames(binaries/'aft_bench',x[None,:],repeats=5)[0];fft_ns=timed_fft(x[None,:],101)[0].tolist()
        report['scaling'].append({'n':n,'circles':r['circles'],'lines':r['lines'],'pieces':r['pieces'],
             **{key+'_ns':r[key+'_ns'] for key in ['fit','closed','rotation','nine_query']},'aft_ns':aft['aft_ns'],'fft_ns':fft_ns,
             **{key+'_us':float(np.median(r[key+'_ns'])/1000) for key in ['fit','closed','rotation','nine_query']},
             'aft_us':float(np.median(aft['aft_ns'])/1000),'fft_us':float(np.median(fft_ns)/1000)})
    supplemental_checks(report,first_pcm,binaries,ffmpeg,output)
    render(report,selected,output)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(output/'report.json')


if __name__=='__main__':main()
