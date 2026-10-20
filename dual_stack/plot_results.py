#!/usr/bin/env python3
"""Plot recorded samples without interpolation or synthetic measurements."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('results',type=Path);args=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=args.results
    telemetry=[json.loads(x) for x in (root/'telemetry.jsonl').read_text(encoding='utf-8').splitlines()]
    requests=[json.loads(x) for x in (root/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
    phases=json.loads((root/'phases.json').read_text(encoding='utf-8'))
    # Cleanup restores the old service after the last measured phase; omit it.
    end=max(phase['end_ns'] for phase in phases)
    telemetry=[row for row in telemetry if row['monotonic_ns']<=end]
    origin=min(x['monotonic_ns'] for x in telemetry)
    seconds=lambda value:(value-origin)/1e9
    x=[seconds(row['monotonic_ns']) for row in telemetry]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(4,1,figsize=(13,11),sharex=True,constrained_layout=True)
    axes[0].plot(x,[row.get('main_rails_sum_w') for row in telemetry],color='#2563eb',lw=1,label='3-rail sum (board sensors)')
    axes[0].set_ylabel('Power (W)');axes[0].legend(loc='upper left')
    for key,color in [('cpu','#ea580c'),('gpu','#16a34a'),('tj','#7c3aed')]:
        axes[1].plot(x,[row.get('temperatures_c',{}).get(key) for row in telemetry],color=color,lw=1,label=key)
    axes[1].set_ylabel('Temperature (C)');axes[1].legend(loc='upper left',ncol=3)
    axes[2].plot(x,[row['ram_used_mib']/1024 for row in telemetry],color='#0f766e',lw=1,label='System RAM used')
    axes[2].plot(x,[row['mem_available_bytes']/2**30 for row in telemetry],color='#64748b',lw=1,label='MemAvailable')
    axes[2].set_ylabel('Memory (GiB)');axes[2].legend(loc='upper left',ncol=2)
    for kind,key,color,label in [('vla','model_wall_ms','#0369a1','VLA model'),('vlm','ttft_ms','#c2410c','VLM TTFT')]:
        rows=[row for row in requests if row['kind']==kind and not row['warmup']]
        axes[3].scatter([seconds(row['start_ns']) for row in rows],
                        [(row['response']['timings'][key] if kind=='vla' else row[key]) for row in rows],
                        color=color,s=5,alpha=.7,label=label,rasterized=True)
    axes[3].set_ylabel('Latency (ms)');axes[3].legend(loc='upper left',ncol=2)
    labels={'vla_standalone':'VLA only','vlm_standalone':'VLM only','dual_resident_idle':'Both idle',
            'vla_dual_resident':'VLA active','vlm_dual_resident':'VLM active','concurrent_saturated':'Both active'}
    for phase in phases:
        name=phase['name'];a,b=seconds(phase['start_ns']),seconds(phase['end_ns'])
        if name.startswith('load_'):
            for ax in axes:ax.axvspan(a,b,color='#cbd5e1',alpha=.35,lw=0)
        if name in labels:
            for ax in axes:ax.axvline(a,color='#94a3b8',lw=.5,ls='--')
            axes[0].text((a+b)/2,1.02,labels[name],transform=axes[0].get_xaxis_transform(),ha='center',fontsize=8)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    axes[3].set_xlabel('Seconds since first telemetry sample (loading phases shaded)')
    fig.suptitle('AGX Orin 64GB | pi0.5 FlashRT BF16 + Qwen3.5-9B Q5_K_M / F16 vision',fontsize=14)
    fig.savefig(root/'timeline.png',dpi=160)
    fig.savefig(root/'timeline.svg')
    plt.close(fig)
    processing=json.loads((root/'postprocessing.json').read_text(encoding='utf-8'))
    processing.update(plot_generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      matplotlib_version=matplotlib.__version__,
                      plot_files_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                                         for name in ['timeline.png','timeline.svg']})
    (root/'postprocessing.json').write_text(json.dumps(processing,indent=2)+'\n',encoding='utf-8')
    print(root/'timeline.png')


if __name__=='__main__':main()
