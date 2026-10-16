# pi05-LIBERO Orin baseline profiling

Run with the existing deployment environment after stopping other inference
clients and the model service to avoid memory and compute contention. The script
itself does not manage services, change power settings, or modify deployment
files. It imports the deployed `serve.py` policy helper and preserves its official
BF16/FP32 parameter choices, preprocessing, quantile normalization, and action
decoding. Compilation stays disabled.

```bash
/home/nvidia/pi05-orin/.venv/bin/python benchmark.py \
  --deployment-root /home/nvidia/pi05-orin \
  --iterations 30 --warmup 10 \
  --output results/baseline
```

Use `--checkpoint` to override `DEPLOYMENT_ROOT/models/pi05_libero`. The output
directory must be empty. `--help` works without importing PyTorch or openpi.

The three sequential phases each have 10 excluded warmups and 30 measured
samples by default:

1. **Official policy, uninstrumented:** the public `Policy.infer()` call, timed
   with completed CUDA work at both ends. This is the primary local baseline;
   it includes CPU preprocessing, input transfers, model sampling, output transfer,
   and action decoding, and excludes network communication.
2. **Model only, uninstrumented:** prebuilt CUDA inputs to completed CUDA outputs,
   using the exact same model and sampling function. Wall time includes Python
   launch/control overhead as well as GPU execution. An outer CUDA event pair
   supplies a stream elapsed interval as an additional measurement.
3. **Instrumented decomposition:** CPU preprocessing, transfer plus
   `Observation.from_dict`, model, and output transfer plus postprocessing.
   Transfer and whole-model boundaries synchronize. Temporary instance wrappers
   record CUDA events for internal stages without synchronizing those stages.
   Event pairs are reused across requests; warmup excludes their initial setup.

Internal events cover model preprocessing, prefix embedding, each of three vision
encodings, language token lookup, prefix transformer KV prefill, all 10 denoise
iterations, suffix embedding, expert transformer, attention-mask conversion,
action input/output projections, and time MLP linear layers. The padded third
image is still encoded, matching upstream behavior. SiLU activations belong to
the full `time_mlp_total` interval and enclosing suffix embedding, rather than
the individual linear-layer rows.

CUDA events measure elapsed intervals on the current stream, including any host
launch gaps between recorded events. They do **not** measure GPU kernel busy
time or utilization. Nested intervals overlap: vision belongs to prefix embedding,
and the expert transformer belongs to each denoise iteration. Do not sum the
whole stage table. `model_remaining_cuda_stream_elapsed_ms` subtracts only the
nonoverlapping top-level intervals and includes model setup, Euler updates,
loop-control synchronization, and remaining gaps.

One seeded synthetic observation and explicit fixed noise are reused throughout.
The observation has two valid cameras and one padded masked camera; output is
`(10, 7)`. Every measured output must equal the initial reference exactly, and a
final public-policy check verifies wrapper restoration. Ten denoising iterations
and ten predicted actions are separate settings.

Artifacts:

- `raw.json`: all warmup durations, measured phase timings, detailed stage events,
  and numerical checks.
- `samples.csv`: long-form phase metrics for every measured sample.
- `stages.csv`: every instrumented interval, with denoise iteration, image, and
  parent stage.
- `summary.json` and `summary.csv`: means, medians, p95, range, and population
  standard deviation. JSON also includes per-call stage statistics and caveats.
- `provenance.json`: hardware and power/clock observations, package and CUDA
  versions, environment, Git identity/status when available, hashes of model
  weights, normalization/tokenizer assets, and relevant loaded source files.
- `fixed_inputs.npz` and `reference_actions.npy`: reproducible inputs, noise, and
  expected output. Load with `allow_pickle=False`.

The script runs read-only hardware commands when available. Missing permissions
are recorded in provenance. It does not use `sudo`. Sequential phases can drift
with temperature and clocks; these results characterize this deployment and
fixture rather than robot task success or control frequency.
