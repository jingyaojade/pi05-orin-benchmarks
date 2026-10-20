"""Preserve the deployed OpenPI msgpack protocol on port 8000 via the local VLA API."""
import asyncio
import base64
import http
import io
from pathlib import Path
import sys
import time


def serve(host, port, http_url):
    import numpy as np
    from PIL import Image
    import websockets.asyncio.server as ws
    # Reuse the pinned official serialization helper without importing OpenPI/JAX.
    sys.path.insert(0,'/home/nvidia/pi05-orin/openpi/packages/openpi-client/src')
    from openpi_client import msgpack_numpy
    from clients import request_json

    def encoded(value):
        value=np.asarray(value)
        if value.ndim==3 and value.shape[0]==3:value=value.transpose(1,2,0)
        if np.issubdtype(value.dtype,np.floating):value=(value*255).astype(np.uint8)
        if value.shape!=(224,224,3) or value.dtype!=np.uint8:
            raise ValueError('This FlashRT service requires RGB uint8 224x224')
        stream=io.BytesIO();Image.fromarray(value).save(stream,format='PNG')
        return base64.b64encode(stream.getvalue()).decode()

    async def handle(connection):
        metadata=request_json(http_url+'/metadata')
        await connection.send(msgpack_numpy.packb(metadata))
        async for message in connection:
            started=time.perf_counter()
            observation=msgpack_numpy.unpackb(message)
            body={'image_base64':encoded(observation['observation/image']),
                  'wrist_image_base64':encoded(observation['observation/wrist_image']),
                  'state':np.asarray(observation['observation/state']).tolist(),
                  'prompt':str(observation['prompt'])}
            result=await asyncio.to_thread(request_json,http_url+'/infer',body)
            await connection.send(msgpack_numpy.packb({'actions':np.asarray(result['actions']),
                'policy_timing':{'infer_ms':result['timings']['policy_wall_ms']},
                'server_timing':{'infer_ms':(time.perf_counter()-started)*1000}}))

    async def health(connection, request):
        if request.path=='/healthz':return connection.respond(http.HTTPStatus.OK,'OK\n')

    async def run():
        async with ws.serve(handle,host,port,compression=None,max_size=8*1024*1024,
                            ping_interval=None,ping_timeout=None,process_request=health) as server:
            print(f'OPENPI_WEBSOCKET_READY {host}:{port}',flush=True)
            await server.serve_forever()
    asyncio.run(run())
