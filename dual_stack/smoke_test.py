#!/usr/bin/env python3
"""Exercise the deployed APIs with finite output and observation-change checks."""
import argparse
import base64
import io
import json
from pathlib import Path
import sys
from clients import request_json, vla_request, vlm_request


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--skip-vlm',action='store_true')
    p.add_argument('--websocket',action='store_true')
    p.add_argument('--check-changed-prompt',action='store_true')
    args=p.parse_args()
    import numpy as np
    from PIL import Image
    repo=Path(__file__).resolve().parents[1]
    with np.load(repo/'reports/engine-comparison-2026-10-16/baseline-latency/fixed_inputs.npz') as f:
        fixture={k:f[k] for k in f.files}
    def image(value):
        out=io.BytesIO();Image.fromarray(value).save(out,format='PNG')
        return base64.b64encode(out.getvalue()).decode()
    body={'image_base64':image(fixture['image']),'wrist_image_base64':image(fixture['wrist_image']),
          'noise':fixture['noise'].tolist(),'state':fixture['state'].tolist(),
          'prompt':str(fixture['prompt'].item())}
    results={'health':request_json('http://127.0.0.1:8081/health')}
    def check(result):
        value=np.asarray(result['response']['actions'])
        assert value.shape==(10,7) and np.isfinite(value).all()
        return value
    fixed=[vla_request('http://127.0.0.1:8081',body) for _ in range(3)]
    actions=[check(x) for x in fixed]
    assert all(np.array_equal(actions[0],a) for a in actions[1:])
    changed=vla_request('http://127.0.0.1:8081',dict(body,image_base64=image(np.zeros_like(fixture['image']))))
    assert not np.array_equal(actions[0],check(changed)), 'Changing camera input had no effect'
    results.update(vla_fixed=fixed,vla_changed_image=changed)
    if args.check_changed_prompt:
        changed=vla_request('http://127.0.0.1:8081',dict(body,prompt='pick up the blue cube'))
        check(changed)
        assert changed['response']['prompt_reconfigured']
        assert not np.array_equal(actions[0],check(changed))
        restored=vla_request('http://127.0.0.1:8081',body)
        check(restored)
        results.update(vla_changed_prompt=changed,vla_restored_prompt=restored)
    if not args.skip_vlm:
        results['vlm_health']=request_json('http://127.0.0.1:8082/health')
        data=base64.b64encode((repo/'dual_stack/fixtures/scene-0.png').read_bytes()).decode()
        results['vlm']=vlm_request('http://127.0.0.1:8082',data,'从左到右描述桌面上三个物体的颜色和类别，只输出一句简短中文，不要解释。')
        assert results['vlm']['timings'].get('predicted_n',0)>0
    if args.websocket:
        sys.path.insert(0,'/home/nvidia/pi05-orin/openpi/packages/openpi-client/src')
        from openpi_client import msgpack_numpy
        from websockets.sync.client import connect
        with connect('ws://127.0.0.1:8000',compression=None) as socket:
            metadata=msgpack_numpy.unpackb(socket.recv())
            socket.send(msgpack_numpy.packb({'observation/image':fixture['image'],
                'observation/wrist_image':fixture['wrist_image'],'observation/state':fixture['state'],
                'prompt':str(fixture['prompt'].item())}))
            value=msgpack_numpy.unpackb(socket.recv())
            assert value['actions'].shape==(10,7) and np.isfinite(value['actions']).all()
            results['websocket']={'model':metadata['model'],'action_shape':list(value['actions'].shape),
                                  'all_finite':True,'server_timing':value['server_timing']}
    results['passed']=True
    args.output.write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':True,'output':str(args.output)}))


if __name__=='__main__':main()
