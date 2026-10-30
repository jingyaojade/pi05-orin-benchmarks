"""Layerwise numerical screening; shared residency is not deployment memory."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np

from worker import build_adapter, load_cases, model_sample, write_json
from metrics import accuracy, stats


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--flashrt-root',type=Path,default=Path('/home/nvidia/pi05-orin-flashrt/vendor/FlashRT'))
    p.add_argument('--checkpoint',type=Path,default=Path('/home/nvidia/pi05-orin/models/pi05_libero'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    cases,names=load_cases(a.inputs)
    # Use only the first four recorded images for selection; later images and
    # the alternate noises are held out from the layer selection decision.
    discovery=[0,1,3,5,7]
    holdout=[i for i in range(len(cases)) if i not in discovery]
    obj=build_adapter({'encoder_layers':list(range(18)),
                       'decoder_layers':list(range(18))},a,cases[1])
    torch=obj.torch;proxy=obj.quantization;pipeline=obj.pipe.pipeline
    pair=[torch.cuda.Event(enable_timing=True) for _ in range(2)]

    def select(enc=(),dec=()):
        proxy.set_active_selection(encoder_layers=enc,decoder_layers=dec)
        with torch.cuda.stream(obj.model_stream):
            pipeline.record_infer_graph(external_stream_int=obj.model_stream.cuda_stream)
        torch.cuda.synchronize()

    select()
    teacher=[]
    for case in cases:
        obj.prepare(case);teacher.append(obj.infer(case['noise']))
    teacher=np.stack(teacher);np.save(a.output/'bf16_same_residency_actions.npy',teacher)
    obj.prepare(cases[1]);obj.stage_model_inputs(cases[1]['noise'],include_images=True)
    base_wall=[model_sample(obj,cases[1]['noise'],pair)[0] for _ in range(10)]
    result={'discovery_indices':discovery,'holdout_indices':holdout,'case_names':names,
        'baseline_probe_latency':stats(base_wall),'layers':[],
        'selection_rule':{'minimum_per_case_cosine':.999,'maximum_rmse':.02,'maximum_abs':.1},
        'scope':'Single-layer precision screening against same-process BF16; five timing probes are diagnostic only. All quantized copies remain resident.'}
    for domain in ('encoder','decoder'):
        for layer in range(18):
            select([layer] if domain=='encoder' else [],[layer] if domain=='decoder' else [])
            predictions=[]
            for i in discovery:
                obj.prepare(cases[i]);predictions.append(obj.infer(cases[i]['noise']))
            metric=accuracy(np.stack(predictions),teacher[discovery])
            obj.prepare(cases[1]);obj.stage_model_inputs(cases[1]['noise'],include_images=True)
            walls=[model_sample(obj,cases[1]['noise'],pair)[0] for _ in range(5)]
            row={'domain':domain,'layer':layer,'accuracy':metric,
                 'min_case_cosine':min(v['cosine'] for v in metric['per_case']),
                 'probe_latency':stats(walls)}
            row['numerical_screen_pass']=(row['min_case_cosine']>=.999 and metric['rmse']<=.02 and metric['max_abs']<=.1)
            result['layers'].append(row)
            write_json(a.output/'screening.json',result)
            print(json.dumps({'event':'screen_layer','domain':domain,'layer':layer,
                'cosine':metric['cosine'],'min_case_cosine':row['min_case_cosine'],
                'rmse':metric['rmse'],'probe_ms':row['probe_latency']['mean_ms'],
                'screen_pass':row['numerical_screen_pass']}),flush=True)
    selected={d:[r['layer'] for r in result['layers'] if r['domain']==d and r['numerical_screen_pass']]
              for d in ('encoder','decoder')}
    result['selected_layers']=selected
    result['quantization_residency']=proxy.metadata()
    result['completed']=True
    write_json(a.output/'screening.json',result)
    with (a.output/'layers.csv').open('w',newline='') as f:
        fields=['domain','layer','cosine','min_case_cosine','rmse','max_abs','probe_mean_ms','numerical_screen_pass']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in result['layers']:
            writer.writerow({'domain':r['domain'],'layer':r['layer'],'cosine':r['accuracy']['cosine'],
                'min_case_cosine':r['min_case_cosine'],'rmse':r['accuracy']['rmse'],
                'max_abs':r['accuracy']['max_abs'],'probe_mean_ms':r['probe_latency']['mean_ms'],
                'numerical_screen_pass':r['numerical_screen_pass']})
    print(json.dumps({'event':'screen_complete','selected_layers':selected}),flush=True)


if __name__=='__main__':main()
