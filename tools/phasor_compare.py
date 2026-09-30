#!/usr/bin/env python3
"""Measure the FFT phasor control (earlier "FAFT" prototype); keep data private.

Optional NumPy/Matplotlib/Pillow/FFmpeg; Rust core has no dependencies.
The analytic AFT remains a reference preset with a different transform target.
"""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from aft_compare import (ROOT, decode, periodic_hann, window_frames, timed_fft,
                         rust_frames as reference_frames, relative_l2)


def rust_frames(binary, frames, tolerance, minimum, repeats=1):
    if frames.ndim != 2 or not np.all(np.isfinite(frames)):
        raise ValueError('invalid frames')
    data = '\n'.join(' '.join(format(v, '.17g') for v in row) for row in frames)
    proc = subprocess.run([str(binary), str(frames.shape[1]), str(repeats),
                           str(minimum), str(tolerance)], input=data, text=True,
                          capture_output=True, check=True)
    rows = [json.loads(line) for line in proc.stdout.splitlines()]
    if len(rows) != len(frames):
        raise ValueError('frame count mismatch')
    for source, row in zip(frames, rows):
        validate_row(source, row, tolerance)
    return rows


def project_model(row, positions):
    """Independent synthesis from exported descriptors, not Rust's IFFT output."""
    positions = np.asarray(positions, dtype=float)
    values = np.zeros_like(positions)
    covered = np.zeros(positions.shape, dtype=int)
    for model in row['models']:
        mask = (positions >= model['start']) & (positions < model['end'])
        t = positions[mask] - model['start']
        n = model['end'] - model['start']
        fraction = t / (n-1)
        y = (1-fraction)*model['first'] + fraction*model['last']
        for c in model['circles']:
            phase = 2*np.pi*c['bin']*t/n
            y += c['re']*np.cos(phase)-c['im']*np.sin(phase)
        values[mask] = y*row['scale']
        covered[mask] += 1
    if not np.all(covered == 1):
        raise ValueError('model coverage')
    return values


def validate_row(source, row, tolerance):
    y = np.asarray(row['decoded'])
    if y.shape != source.shape or not np.all(np.isfinite(y)):
        raise ValueError('invalid decoded output')
    error = y-source
    rms = float(np.sqrt(np.mean(error**2)))
    if rms > tolerance or not math.isclose(rms, row['rmse'], abs_tol=1e-8, rel_tol=1e-8):
        raise ValueError('RMS budget or metric mismatch')
    if not math.isclose(float(np.max(np.abs(error))), row['max_error'], abs_tol=1e-8, rel_tol=1e-8):
        raise ValueError('max-error mismatch')
    # Every frame is checked using independent direct projected-circle synthesis.
    direct = project_model(row, np.arange(len(source)))
    if np.max(np.abs(direct-y)) > 1e-7*max(1.0, float(np.max(np.abs(source)))):
        raise ValueError('direct circle / inverse rotation mismatch')


def summary(rows, frames):
    counts = Counter(c['bits'] for r in rows for b in r['models'] for c in b['circles'])
    supports = Counter(b['end']-b['start'] for r in rows for b in r['models'])
    raw = np.asarray([r['spectrum'] for r in rows])
    spectrum = raw[...,0] + 1j*raw[...,1]
    actual = np.asarray([r['decoded'] for r in rows])
    return {'frames':len(rows), 'mean_circles':float(np.mean([r['circles'] for r in rows])),
            'mean_descriptors':float(np.mean([r['descriptors'] for r in rows])),
            'mean_blocks':float(np.mean([r['blocks'] for r in rows])),
            'mean_estimated_model_bits':float(np.mean([r['estimated_model_bits'] for r in rows])),
            'rms_error':float(np.sqrt(np.mean((actual-frames)**2))),
            'worst_frame_rmse':max(r['rmse'] for r in rows),
            'max_error':max(r['max_error'] for r in rows),
            'kernel_vs_numpy_relative_l2':relative_l2(spectrum,np.fft.rfft(frames,axis=1)),
            'precision_histogram':dict(sorted(counts.items())),
            'block_length_histogram':dict(sorted(supports.items())),
            **{key+'_median_us':float(np.median([r[key+'_ns'] for r in rows])/1000)
               for key in ('kernel','fit','decode')}}


def save(fig, output, name):
    fig.savefig(output/f'{name}.png', dpi=150, facecolor='white')
    path = output/f'{name}.svg'
    fig.savefig(path, facecolor='white')
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)


def plots(report, selected, output):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    sweep = report['scaling']; sizes = [r['n'] for r in sweep]
    for key,label in [('numpy_us','NumPy rFFT'),('kernel_us','Rust radix-2 rotations'),
                      ('fit_us','Phasor tree + precision + checks'),('aft_us','Integral AFT reference')]:
        axes[0].loglog(sizes,[r[key] for r in sweep],'o-',label=label)
    axes[0].set(xlabel='Samples N',ylabel='Median time (µs)',title='Transform and optimization costs are separate')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.2)
    for mode,style in [('root','--'),('tree','-')]:
        for excerpt in report['excerpts']:
            rows = [r for r in excerpt['trials'] if r['mode']==mode]
            axes[1].plot([r['rms_error'] for r in rows],[r['mean_descriptors'] for r in rows],
                         'o'+style,label=f'{excerpt["start"]:g} s · {mode}')
    axes[1].set(xscale='log',xlabel='Measured RMS error (PCM16 units)',ylabel='Mean baseline + circle descriptors',
                title='Fewer circles with an explicit sample-error budget')
    axes[1].legend(fontsize=8,ncol=2); axes[1].grid(alpha=.2)
    fig.suptitle('FFT phasor control · rotating-circle model search on Beethoven',fontsize=14)
    save(fig,output,'phasor-benchmark')

    source,row = selected
    n = len(source); t = np.arange(n)/report['rate']*1000
    y = np.asarray(row['decoded']); fft=np.fft.rfft(source); reconstructed=np.fft.rfft(y)
    fig, axes = plt.subplots(2,2,figsize=(12,7),layout='constrained')
    axes[0,0].plot(t,source/32768,label='Windowed source',lw=1.8)
    axes[0,0].plot(t,y/32768,label='Phasor reconstruction',lw=1,ls='--')
    axes[0,0].set(title='Selected frame at 60 s excerpt midpoint',xlabel='Local time (ms)',ylabel='Full-scale amplitude')
    axes[0,0].legend(fontsize=8)
    axes[1,0].plot(t,y-source,color='#b44b31')
    axes[1,0].set(title=f'RMS {row["rmse"]:.3f}; peak {row["max_error"]:.3f} PCM16',xlabel='Local time (ms)',ylabel='Sample error (PCM16 units)')
    f=np.fft.rfftfreq(n,1/report['rate']); factor=2/(periodic_hann(n).sum()*32768)
    for spectrum,label in [(fft,'FFT of source'),(reconstructed,'FFT of phasor decode')]:
        axes[0,1].semilogx(f[1:-1],20*np.log10(np.maximum(abs(spectrum[1:-1])*factor,1e-8)),label=label,lw=1)
    axes[0,1].set(title='Same discrete spectrum target',xlabel='Frequency (Hz)',ylabel='Magnitude (dBFS)',ylim=(-130,0))
    axes[0,1].legend(fontsize=8)
    circles=row['models'][0]['circles']
    radii=[math.hypot(c['re'],c['im'])*row['scale'] for c in circles]
    axes[1,1].scatter(radii,[c['bits'] for c in circles],s=22,c='#38745d')
    axes[1,1].set(xscale='log',xlabel='Projected circle radius (PCM16)',ylabel='Bits per Cartesian component',
                  title=f'{len(circles)} circles · independent per-circle precision')
    for ax in axes.flat: ax.grid(alpha=.2)
    fig.suptitle('FFT phasor root model · requested RMS ≤ 16 PCM16',fontsize=14)
    save(fig,output,'phasor-model')

    # A visible epicycle chain: only the largest 8 circles, explicitly distinguished
    # from the full model in the reconstruction panel. No invented coefficients.
    top=sorted(circles,key=lambda c:math.hypot(c['re'],c['im']),reverse=True)[:8]
    baseline=row['models'][0]
    fig, (ax, wave) = plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    phase=np.linspace(0,2*np.pi,100)
    limit=sum(math.hypot(c['re'],c['im']) for c in top)*row['scale']/32768*1.15
    limit=max(limit,1e-6)
    ax.set(xlim=(-limit,limit),ylim=(-limit,limit),aspect='equal',xlabel='Imaginary projection',ylabel='Real projection',
           title=f'Rotating chain: largest {len(top)} of {len(circles)} circles')
    rings=[ax.plot([],[],alpha=.45,lw=.7)[0] for _ in top]
    chain,=ax.plot([],[],'o-',ms=3,color='#244b73')
    partial=np.full(n,baseline['first'])
    partial+=(baseline['last']-baseline['first'])*np.arange(n)/(n-1)
    for c in top:
        a=2*np.pi*c['bin']*np.arange(n)/n
        partial+=c['re']*np.cos(a)-c['im']*np.sin(a)
    partial*=row['scale']/32768
    wave.plot(t,source/32768,label='Source',lw=1.4,color='#777777')
    wave.plot(t,y/32768,label='Full phasor model',ls='--',lw=1,color='#d65421')
    wave.plot(t,partial,label='Visible circles + baseline',lw=.9,color='#244b73')
    marker,=wave.plot([],[],'o',color='#244b73'); cursor=wave.axvline(0,color='#777777',lw=.6)
    wave.set(xlabel='Local time (ms)',ylabel='Full-scale amplitude',title='Projection builds the waveform')
    wave.legend(fontsize=7,loc='lower left'); wave.grid(alpha=.2)
    def update(frame):
        index=frame*(n-1)/47
        z=0j; xs=[0.0]; ys=[0.0]
        for ring,c in zip(rings,top):
            radius=math.hypot(c['re'],c['im'])*row['scale']/32768
            ring.set_data(z.imag+radius*np.sin(phase),z.real+radius*np.cos(phase))
            z+=complex(c['re'],c['im'])*np.exp(2j*np.pi*c['bin']*index/n)*row['scale']/32768
            xs.append(z.imag);ys.append(z.real)
        chain.set_data(xs,ys)
        base=((1-index/(n-1))*baseline['first']+index/(n-1)*baseline['last'])*row['scale']/32768
        marker.set_data([index/report['rate']*1000],[z.real+base]);cursor.set_xdata([index/report['rate']*1000]*2)
        fig.suptitle(f'Beethoven · frame-relative sample {index:.1f} · slowed, silent illustration')
        return [*rings,chain,marker,cursor]
    animation=FuncAnimation(fig,update,frames=48,interval=100)
    animation.save(output/'phasor-rotating-circles.gif',writer=PillowWriter(fps=10),dpi=100)
    plt.close(fig)


def build_examples():
    cargo=shutil.which('cargo')
    if not cargo: raise ValueError('Cargo missing; build release examples first')
    subprocess.run([cargo,'build','--offline','--release','--example','phasor_bench','--example','aft_bench'],cwd=ROOT,check=True)
    meta=subprocess.run([cargo,'metadata','--offline','--no-deps','--format-version','1'],cwd=ROOT,capture_output=True,text=True,check=True)
    return Path(json.loads(meta.stdout)['target_directory'])/'release/examples'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--ffmpeg',default=os.environ.get('FFMPEG','ffmpeg'))
    args=p.parse_args(); source=args.input.resolve(); output=args.output.resolve()
    if not source.is_file(): p.error('input recording missing')
    if output.exists() or output==ROOT or ROOT in output.parents: p.error('output must be new and outside repository')
    ffmpeg=shutil.which(args.ffmpeg)
    if not ffmpeg: p.error('FFmpeg missing')
    binary=build_examples();output.mkdir(parents=True)
    report={'experiment':'phasor-fft-control-v1-dyadic-rotating-circles','date':time.strftime('%Y-%m-%d'),
            'n':1024,'hop':512,'rate':44100,'seconds':2,'window':'periodic Hann before all methods; mono PCM16',
            'min_block':32,'tolerances':[1,16,128],
            'cost':'Minimize baseline+circle descriptors, then estimated model bits; NOT encoded bitrate.',
            'timing':'Warmed in-process calls; output allocation included; kernel excludes real-to-complex conversion; fit includes normalization, search, precision, final decode/error check; audio/window/text/JSON/plotting excluded.',
            'environment':{'platform':platform.platform(),'python':platform.python_version(),'numpy':np.__version__,
                           'rustc':subprocess.run(['rustc','--version'],capture_output=True,text=True,check=True).stdout.strip(),
                           'ffmpeg':subprocess.run([ffmpeg,'-version'],capture_output=True,text=True,check=True).stdout.splitlines()[0]},
            'excerpts':[],'scaling':[]}
    selected=None;first_pcm=None
    for start in [60,180,300]:
        print(f'FFT phasor control, Beethoven at {start} s',flush=True)
        pcm=decode(ffmpeg,source,start,2,44100)
        if first_pcm is None: first_pcm=pcm
        frames=window_frames(pcm,1024,512)
        excerpt={'start':start,'frames':len(frames),'trials':[]}
        for tolerance in report['tolerances']:
            for mode,minimum in [('root',1024),('tree',32)]:
                rows=rust_frames(binary/'phasor_bench',frames,tolerance,minimum)
                trial={'mode':mode,'tolerance':tolerance,**summary(rows,frames)}
                excerpt['trials'].append(trial)
                print(f'  {mode} eps={tolerance}: {trial["mean_circles"]:.1f} circles, {trial["mean_blocks"]:.2f} blocks, {trial["fit_median_us"]:.1f} µs',flush=True)
                if start==60 and mode=='root' and tolerance==16:
                    selected=(frames[len(frames)//2],rows[len(rows)//2])
                    # These invertible model parameters never belong in docs.
                    (output/'private-selected-model.json').write_text(json.dumps(rows[len(rows)//2]))
        report['excerpts'].append(excerpt)
    for n in [256,512,1024,2048,4096]:
        print(f'Timing N={n}',flush=True)
        begin=len(first_pcm)//2-n//2
        x=first_pcm[begin:begin+n]*periodic_hann(n)
        row=rust_frames(binary/'phasor_bench',x[None,:],16,32,11)[0]
        reference=reference_frames(binary/'aft_bench',x[None,:],repeats=11)[0]
        numpy_ns=timed_fft(x[None,:],101)[0].tolist()
        report['scaling'].append({'n':n,'numpy_ns':numpy_ns,'kernel_ns':row['kernel_ns'],
            'fit_ns':row['fit_ns'],'decode_ns':row['decode_ns'],'aft_ns':reference['aft_ns'],
            'aft_fit_ns':reference['fit_ns'],
            'numpy_us':float(np.median(numpy_ns)/1000),'aft_us':float(np.median(reference['aft_ns'])/1000),
            **{k+'_us':float(np.median(row[k+'_ns'])/1000) for k in ['kernel','fit','decode']}})
    print('Rendering measured plots and rotating-circle GIF',flush=True)
    plots(report,selected,output)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(output/'report.json')


if __name__=='__main__': main()
