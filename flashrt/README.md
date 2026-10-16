# FlashRT Orin comparison

## Reproduce this repository's experiment

Run from this repository's root on the Orin. `prepare_orin.sh` uses an isolated
`/home/nvidia/pi05-orin-flashrt/.venv`, the already downloaded NVIDIA CUDA-enabled
PyTorch wheel, a pinned FlashRT checkout, and CUTLASS v4.4.2. It does not change
the baseline environment or JetPack. The build is resumable; completed objects
are reused. `FLASHRT_BUILD_JOBS` defaults to 2.

```bash
bash flashrt/prepare_orin.sh
bash flashrt/run.sh bf16 reports/reproduction-bf16
python reporting/report.py --results reports/baseline-2026-10-16 \
  --compare reports/reproduction-bf16 \
  --output reports/reproduction-bf16/report_zh.md
```

`run.sh` temporarily stops `openpi-pi05` and restores its previous running state
on exit. It records the pre-run process/clock/power snapshot, exact installed
packages, CUTLASS commit, and extension hashes alongside benchmark data. Other
users' processes are left running; snapshots disclose the observed host load.

The source-oriented commands below are an alternative manual build workflow.

This directory prepares a reproducible FlashRT comparison for the deployed
OpenPI `pi05_libero` service. Measured BF16 and INT8 results are recorded in [the comparison report](../reports/flashrt-comparison_zh.md). BF16 is the selected comparison; INT8 has large action deviations on the fixed fixture.

Upstream source: https://github.com/flashrt-project/FlashRT

Pinned commit: `6cb2f80caec2f561a1331c554a6d3bbf95d2ed55`.
Upstream code is Apache-2.0; the local patch is identified separately in output.

## Input and numerical contract

Use the baseline's `fixed_inputs.npz` and `reference_actions.npy`. The same
synthetic RGB camera frames, prompt, state, and explicit float32 noise values are
loaded without resampling. `pi05_libero` excludes state from the language prompt.

The isolated FlashRT environment does not need OpenPI/JAX to tokenize prompts.
`standalone_tokenizer.py` reads the official SentencePiece model and follows
OpenPI's no-state path exactly: strip, replace underscores/newlines with spaces,
encode cleaned text with BOS, encode the newline separator separately, then
truncate/pad token IDs and the boolean mask to 200. The adapter populates the
pinned frontend's existing `("openpi", 200)` tokenizer protocol cache before
`set_prompt`. This fixes model-file selection and avoids depending on OpenPI
imports or fallback model discovery in the isolated environment, without editing
upstream code or changing the three-file patch hashes.

Select the model with `--tokenizer /path/to/paligemma_tokenizer.model` or the
`PI05_TOKENIZER_PATH` or `FLASHRT_PALIGEMMA_TOKENIZER` environment variable
(CLI takes precedence, followed by those variables in that order). The default is
`/home/nvidia/pi05-orin/cache/big_vision/paligemma_tokenizer.model`. Provenance
records the exact file SHA256, full padded token IDs/mask, valid IDs and count.
No tokenizer download or implicit fallback occurs during a benchmark run.

The local patch retains three 224×224 SigLIP views, 27 vision layers, all
`3×256+200=968` encoder GEMM rows, a 200-token prompt buffer, ten predicted
actions, and ten ODE denoising steps. The third image is all-zero uint8 and is
masked. Pooling stays 1 and `cache_frames=1` runs a fresh vision/prefix pass for
every observation.

Stock FlashRT accepts camera images without image masks and normally shrinks
no-state prompts to their actual token count. Setting `max_prompt_len=200` alone
does not preserve that buffer. The explicit local patch adds
`valid_num_views=2, fixed_prompt_length=True`:

- Q and residual rows remain in the original 968-row layout. RoPE uses
  `cumsum(valid_mask)-1`, matching OpenPI's positions for valid tokens.
- Each encoder layer copies its entire 200-row prompt K/V capacity from rows
  `[768:968]` into `[512:712]`. Source and destination are disjoint. Two static
  D2D copy nodes per layer are captured; no new CUDA kernel is introduced.
- Existing BF16 FA2 `seqused_k` masks the compact valid prefix. Decoder K/V
  append at `512+valid_prompt_tokens`, using the existing device offset support.
- SigLIP and encoder GEMMs keep padded computation. FA2 can skip masked K/V
  ranges, so attention's work differs from a dense masked OpenPI softmax.

This supports a static contiguous suffix-camera mask and prompt capacity no
larger than the masked camera gap. It is not a general per-frame image-mask API.

FlashRT still uses different numerical boundaries: timestep modulations are
precomputed, most arithmetic and action-state accumulation use BF16, and kernels
fuse operations. The deployed OpenPI sampler retains some FP32 operations and
uses FP32 diffusion noise/update state. The harness reports cosine, RMSE, mean
and maximum absolute action error against OpenPI, without claiming bit equality
or robot-task success. Repeat calls and the diagnostic stage chain must exactly
match this FlashRT instance's first result.

## Prepare the source and build

The NVIDIA Orin deployment guide was tested with CUDA 12.6. A source audit found
no mandatory CUDA 12.6 API in the selected SM87 build, and the recorded SM87 build has now succeeded on CUDA 12.2. Use the existing NVIDIA CUDA-enabled PyTorch environment
(Python 3.10 is supported). Do not replace it with a generic PyPI torch wheel.

```bash
git clone https://github.com/flashrt-project/FlashRT.git FlashRT
git -C FlashRT checkout 6cb2f80caec2f561a1331c554a6d3bbf95d2ed55
git clone --depth 1 --branch v4.4.2 https://github.com/NVIDIA/cutlass.git FlashRT/third_party/cutlass

# Install build helpers in the isolated deployment environment.
python -m pip install 'cmake>=3.24' pybind11 pyyaml ml_dtypes safetensors sentencepiece
python flashrt/prepare_patch.py --flashrt-root FlashRT --apply
python -m pip install --no-deps -e FlashRT

cmake -S FlashRT -B FlashRT/build_orin_sm87 \
  -DGPU_ARCH=87 -DFA2_ARCH_NATIVE_ONLY=ON \
  -DFA2_HDIMS='96;128;256' -DFA2_DTYPES=bf16 \
  -DFLASHRT_SLIM_BUILD=ON \
  -DFLASHRT_ENABLE_MOTUS=OFF -DFLASHRT_ENABLE_LINGBOT=OFF \
  -DFLASHRT_ENABLE_AUDIO_CODEBOOK=OFF -DFLASHRT_BUILD_QWEN3_VL=OFF \
  -DFLASHRT_ENABLE_NATIVE_CPP=OFF \
  -DPython3_EXECUTABLE="$(command -v python)"
cmake --build FlashRT/build_orin_sm87 \
  --target flash_rt_kernels flash_rt_fa2 -j2
```

`FA2_ARCH_NATIVE_ONLY=ON` is necessary on CUDA 12.2: the default matrix includes
newer Blackwell architectures that this compiler cannot recognize. The explicit
targets avoid unrelated JAX FFI targets. Both modules communicate using raw
pointers and link Python/CUDA libraries, with no torch/ATen C++ ABI dependency.
The PyPI package does not ship compiled `.so` files. CUTLASS v4.4.2 is required
as a separate source checkout even for this Orin configuration.

`prepare_patch.py` verifies the three upstream file hashes before applying all
changes. It also emits the reviewable unified patch and manifest. Alternatively,
on the pinned source, run `git apply --check` and `git apply` with
`patches/0001-orin-fixed-prompt-suffix-camera-mask.patch`. Start from a clean
pinned source when reapplying; the helper intentionally rejects patched or
different upstream files. The adapter verifies all patched source hashes.

## Run after the baseline

Avoid other GPU clients during either benchmark; this harness does not stop
services, modify clocks, alter drivers, or change the system power mode.

```bash
python flashrt/benchmark.py \
  --flashrt-root FlashRT \
  --checkpoint /home/nvidia/pi05-orin/models/pi05_libero \
  --fixture results/baseline/fixed_inputs.npz \
  --tokenizer /home/nvidia/pi05-orin/cache/big_vision/paligemma_tokenizer.model \
  --precision bf16 --warmup 10 --iterations 30 \
  --output results/flashrt_bf16
```

The first measured phase times untouched full-graph observation-to-action calls,
with completed CUDA work at both ends. It includes image staging and action
transfer/unnormalization and excludes initial loading, conversion, calibration,
capture, and warmup. Text is static for this no-state fixture; its embedding is
prepared before measurement, matching FlashRT's persistent prompt API. OpenPI's
public policy phase also performs prompt processing, so compare its model-only
phase separately when discussing model acceleration.

The second phase, `model_uninstrumented`, directly replays that same original
full production graph from CUDA-ready inputs to completed CUDA outputs. Image
inputs are staged once before this phase. Fixture noise is restored before
**every** timed replay because the graph overwrites its noise slot with final
actions. A synchronization completes all staging before the timer starts. An
outer wall timer and one CUDA event pair surround the full graph launch, followed
by GPU completion. `model_wall_ms` and `model_cuda_stream_elapsed_ms` use the
same names as the baseline. There are 10 excluded warmups and 30 measured samples
by default. Output transfer, action unnormalization, and per-sample bitwise
comparison occur after both timed intervals have ended. No diagnostic graph is
captured or replayed in this phase.

Only after both production measurements, the diagnostic phase captures separate
vision and prefix graphs, reusing the already captured ten-step decoder graph.
CUDA events bracket their consecutive replay on the same stream; one terminal
synchronization retrieves timings. These stage graphs reuse the same weights,
buffers, mask patch, and GEMM tactics. They never replace the production graph.
Separate launches introduce boundaries absent from the full graph; stage times
are diagnostic costs, not an exact decomposition of its latency. The final full
graph replay must still reproduce the same actions.

Artifacts include `raw.json`, `summary.json`, `samples.csv`, `accuracy.json`,
`provenance.json`, the input fixture, and FlashRT reference actions. Provenance
records source/patch hashes, model and input hashes, package/CUDA versions,
hardware, read-only clock/power observations, and the actual shape settings.
The output directory must be empty. Optional `--min-cosine`/`--max-rmse` provide
explicit numerical gates when a comparison criterion has been chosen.

INT8 is a separate optional run using `--precision int8` and a fresh output
directory. It does not replace the first BF16 result. Dynamic INT8 applies to
encoder and decoder; vision INT8, frozen activation scales, and temporal K/V
reuse remain disabled. Accuracy against the same OpenPI reference is reported.

## Offline checks

```bash
python -m unittest discover -s flashrt/tests -v
python flashrt/benchmark.py --help
```

CPU tests verify masked-position construction, nonoverlapping prompt-K/V
compaction through two attention layers, pinned patch placement, and Python
syntax. Additional timing-boundary tests verify that the model phase selects the
original full graph, restores noise before timing, waits for GPU completion before
stopping its wall timer, and keeps D2H outside the timed helper. They do not
establish CUDA build success, GPU graph correctness, latency,
or policy accuracy. Set `FLASHRT_SOURCE` to the unpatched pinned source tree to
run the source-placement check outside this workspace.

Tokenizer tests execute the official OpenPI `tokenize` method body extracted via
AST, avoiding its unrelated JAX imports. They compare IDs, mask and dtypes for
normal/empty/Unicode/cleaned/long prompts and multiple capacities. With
`sentencepiece` installed and `PI05_TOKENIZER_PATH` set, the same checks use the
real official model; the source cache-protocol test also verifies that OpenPI
and fallback model-locator imports are bypassed when the explicit cache is set.
