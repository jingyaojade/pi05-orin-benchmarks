#!/usr/bin/env python3
"""Two-column scientific figure from recorded samples; no fabricated points."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);args=p.parse_args();root=args.results
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(4,2,figsize=(15,11),sharex='col',sharey='row',constrained_layout=True)
    labels={'vla_dual_resident':'VLA','vlm_dual_resident':'VLM','concurrent_saturated':'Saturated','periodic_2hz_0p2hz':'2Hz / 0.2Hz'}
    for column,(variant,title) in enumerate([('two_process','Two processes'),('shared_stream','One Python PID / shared CUDA context')]):
        data=root/variant
        phases=json.loads((data/'phases.json').read_text(encoding='utf-8'))
        end=max(r['end_ns'] for r in phases)
        telemetry=[json.loads(x) for x in (data/'telemetry.jsonl').read_text(encoding='utf-8').splitlines()]
        telemetry=[r for r in telemetry if r['monotonic_ns']<=end]
        requests=[json.loads(x) for x in (data/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
        start=telemetry[0]['monotonic_ns'];seconds=lambda v:(v-start)/1e9
        x=[seconds(r['monotonic_ns']) for r in telemetry]
        axes[0,column].plot(x,[r.get('main_rails_sum_w') for r in telemetry],lw=.9,color='#2563eb',label='3-rail board sensor sum')
        for sensor,color in [('cpu','#ea580c'),('gpu','#16a34a'),('tj','#7c3aed')]:
            axes[1,column].plot(x,[r['temperatures_c'].get(sensor) for r in telemetry],color=color,lw=.8,label=sensor)
        axes[2,column].plot(x,[r['ram_used_mib']/1024 for r in telemetry],color='#0f766e',lw=.8,label='System RAM used')
        axes[2,column].plot(x,[r['mem_available_bytes']/2**30 for r in telemetry],color='#64748b',lw=.8,label='MemAvailable')
        for kind,color,label in [('vla','#0369a1','VLA model'),('vlm','#c2410c','VLM TTFT')]:
            rows=[r for r in requests if r['kind']==kind and not r['warmup']]
            axes[3,column].scatter([seconds(r['start_ns']) for r in rows],
                [(r['response']['timings']['model_wall_ms'] if kind=='vla' else r['ttft_ms']) for r in rows],
                color=color,s=4,alpha=.65,label=label,rasterized=True)
        for phase in phases:
            a,b=seconds(phase['start_ns']),seconds(phase['end_ns'])
            if phase['name'].startswith('load_'):
                for row in range(4):axes[row,column].axvspan(a,b,color='#cbd5e1',alpha=.35,lw=0)
            if phase['name'] in labels:
                for row in range(4):axes[row,column].axvline(a,color='#94a3b8',ls='--',lw=.5)
                axes[0,column].text((a+b)/2,1.02,labels[phase['name']],ha='center',fontsize=8,
                                     transform=axes[0,column].get_xaxis_transform())
        axes[0,column].set_title(title,pad=23)
        for row in range(4):axes[row,column].grid(axis='y',alpha=.2)
        axes[3,column].set_xlabel('Elapsed seconds; loading shaded')
    for row,label in enumerate(['Power (W)','Temperature (C)','Memory (GiB)','Latency (ms)']):
        axes[row,0].set_ylabel(label);axes[row,0].legend(loc='upper left',fontsize=8,ncol=2 if row else 1)
    fig.suptitle('pi0.5 FlashRT BF16 + Qwen3.5-9B Q5_K_M / F16 vision | AGX Orin 64GB',fontsize=14)
    fig.savefig(root/'timeline.png',dpi=160);fig.savefig(root/'timeline.svg');plt.close(fig)
    meta=json.loads((root/'postprocessing.json').read_text(encoding='utf-8'))
    meta.update(plot_generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),matplotlib_version=matplotlib.__version__,
                plot_files_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['timeline.png','timeline.svg']})
    (root/'postprocessing.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    print(root/'timeline.png')


if __name__=='__main__':main()
