"""Generate official OpenPI 10-step action references in its original environment."""
import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deployment-root',type=Path,default=Path('/home/nvidia/pi05-orin'))
    a=p.parse_args()
    repo=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(repo/'baseline'))
    import benchmark as baseline
    policy, deployment, checkpoint, helper = baseline.load_policy(
        argparse.Namespace(deployment_root=a.deployment_root,checkpoint=None))
    import numpy as np
    import torch
    with np.load(a.inputs,allow_pickle=False) as f: data={k:f[k] for k in f.files}
    a.output.mkdir(parents=True,exist_ok=True)
    provenance={'checkpoint':str(checkpoint),'helper':str(helper),'cases':[]}
    predictions={2:[],3:[]}
    for index,name in enumerate(data['names']):
        observation={'observation/state':data['state'][index],
                     'observation/image':data['image'][index],
                     'observation/wrist_image':data['wrist_image'][index],
                     'prompt':str(data['prompt'][index])}
        cpu_inputs=baseline.prepare_cpu(policy,observation)
        tensors,obs,kwargs=baseline.transfer_and_observe(policy,cpu_inputs,data['noise'][index])
        shapes={k:list(v.shape) for k,v in obs.images.items()}
        valid={k:bool(v.item()) for k,v in obs.image_masks.items()}
        for views in (2,3):
            current=obs
            if views==3:
                if not np.array_equal(data['third_image'][index],data['image'][index]):
                    raise ValueError('This 3-view reference helper only supports the declared duplicate-front test')
                invalid=[k for k,v in obs.image_masks.items() if not bool(v.item())]
                if len(invalid)!=1: raise RuntimeError(f'Expected one masked camera: {valid}')
                first=next(k for k,v in obs.image_masks.items() if bool(v.item()))
                images=dict(obs.images); masks=dict(obs.image_masks)
                # The declared third view duplicates the transformed front image.
                images[invalid[0]]=images[first].clone()
                masks[invalid[0]]=torch.ones_like(masks[invalid[0]],dtype=torch.bool)
                current=dataclasses.replace(obs,images=images,image_masks=masks)
            torch.cuda.synchronize(); start=time.perf_counter()
            with torch.inference_mode(): actions=policy._sample_actions(policy._pytorch_device,current,**kwargs)
            torch.cuda.synchronize()
            result=baseline.finish_cpu(policy,tensors,actions)
            predictions[views].append(np.asarray(result['actions']).copy())
            print(json.dumps({'case':str(name),'valid_views':views,
                              'seconds':time.perf_counter()-start}),flush=True)
        provenance['cases'].append({'name':str(name),'image_shapes':shapes,'original_image_masks':valid})
    for views,values in predictions.items():
        np.save(a.output/f'openpi_{views}valid_10steps.npy',np.stack(values))
    provenance['scope']='Numerical action reference, not task success. 3-valid duplicates front image.'
    (a.output/'reference_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__=='__main__':main()
