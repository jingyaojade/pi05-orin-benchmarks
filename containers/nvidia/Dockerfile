# syntax=docker/dockerfile:1.7
# Source build candidate for x86_64 RTX 4090/5090. GPU validation is pending.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.03-py3
FROM ${BASE_IMAGE}
ARG BENCHMARK_COMMIT=source-archive
ARG BUILD_JOBS=2
ARG CUDA_ARCHITECTURES="89;120"
LABEL org.opencontainers.image.title="pi05 + Qwen NVIDIA RTX benchmark candidate"
LABEL org.opencontainers.image.revision=${BENCHMARK_COMMIT}
ENV DEBIAN_FRONTEND=noninteractive JAX_PLATFORMS=cpu OMP_NUM_THREADS=6 \
    PYTHONUNBUFFERED=1 NO_PROXY=127.0.0.1,localhost \
    OPENPI_DATA_HOME=/tmp/openpi-cache \
    PYTHONPATH=/opt/openpi/src:/opt/openpi/packages/openpi-client/src \
    PATH=/opt/bench-venv/bin:${PATH} BENCHMARK_COMMIT=${BENCHMARK_COMMIT} \
    BENCHMARK_TARGET=nvidia BENCHMARK_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
    CUDA_DEVICE_ORDER=PCI_BUS_ID
WORKDIR /workspace/benchmarks
COPY . /workspace/benchmarks/
RUN test "$(uname -m)" = x86_64 && \
    apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates build-essential cmake ninja-build python3-venv && \
    apt-get clean
# Keep NGC Torch; isolate the legacy OpenPI Python dependencies in a venv.
RUN python3 -m venv --system-site-packages /opt/bench-venv && \
    python -c 'import torch,json; open("/opt/base-torch.json","w").write(json.dumps({"version":torch.__version__,"file":torch.__file__}))' && \
    env -u PIP_CONSTRAINT python -m pip install --no-cache-dir -r deployment/requirements.txt
RUN python - <<'PY'
import json, pathlib, subprocess, torch, transformers, shutil
root=pathlib.Path('/workspace/benchmarks')
p=json.loads((root/'containers/nvidia/pins.json').read_text())
assert transformers.__version__ == '4.53.2'
assert torch.__version__ == json.load(open('/opt/base-torch.json'))['version'], 'NGC Torch was replaced'
for folder, url, pin in [('/opt/openpi','https://github.com/Physical-Intelligence/openpi.git',p['openpi_commit']),
                          ('/opt/llama.cpp','https://github.com/ggml-org/llama.cpp.git',p['llama_cpp_commit'])]:
    subprocess.run(['git','init',folder],check=True)
    subprocess.run(['git','-C',folder,'remote','add','origin',url],check=True)
    subprocess.run(['git','-C',folder,'fetch','--depth','1','origin',pin],check=True)
    subprocess.run(['git','-C',folder,'checkout','--detach','FETCH_HEAD'],check=True)
    assert subprocess.check_output(['git','-C',folder,'rev-parse','HEAD'],text=True).strip()==pin
patch=pathlib.Path('/opt/openpi/src/openpi/models_pytorch/transformers_replace')
shutil.copytree(patch,pathlib.Path(transformers.__file__).parent,dirs_exist_ok=True)
deployment=pathlib.Path('/opt/pi05-deployment'); deployment.mkdir()
(deployment/'openpi').symlink_to('/opt/openpi',target_is_directory=True)
(deployment/'serve.py').symlink_to(root/'deployment/serve.py')
PY
# Native rebuild. No Orin/Thor FlashRT binary is reused on RTX.
RUN cmake -S /opt/llama.cpp -B /opt/llama-build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" \
      -DGGML_NATIVE=OFF -DGGML_CCACHE=OFF -DLLAMA_BUILD_TESTS=OFF \
      -DGGML_CUDA_FA_QUANTS=f16-f16 -DGGML_CUDA_NCCL=OFF \
      -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF && \
    cmake --build /opt/llama-build --target llama-server -j${BUILD_JOBS}
RUN python -m pip freeze > /opt/requirements-built.txt && \
    python -c 'from openpi.models_pytorch.pi0_pytorch import PI0Pytorch; from transformers.models.gemma.modeling_gemma import GemmaRMSNorm; import inspect; assert "cond_dim" in inspect.signature(GemmaRMSNorm).parameters' && \
    python -m compileall -q containers
RUN python - <<'PY'
import hashlib,json,pathlib
root=pathlib.Path('/workspace/benchmarks')
files={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
       for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
pathlib.Path('/opt/benchmark-source-sha256.json').write_text(json.dumps(files,indent=2))
PY
EXPOSE 8000 8081 8082
ENTRYPOINT ["bash", "/workspace/benchmarks/containers/entrypoint.sh"]
CMD ["doctor"]
