# Serving Measurement and Scheduling Foundation

`nar_vae.serving` defines deterministic contracts for a future concurrent NAR-VAE service. It does
not run EchoDiT, decode DACVAE audio, open a network port, require a serving framework, or make the
current checkpoint streamable. The current global ODE and complete DACVAE decode remain
non-streaming.

## Request and bucket contract

`RequestMetadata` keeps the target-text language and reference-audio language independent. A
reference language is never copied into the target role, and the target role is never used to
infer source-language checkpoint coverage. The serving layer schedules already validated work; it
does not grant multilingual, speaker, cross-lingual, duration, or streaming capabilities.

`ShapeBucketKey` is an exact compatibility identity containing:

- checkpoint identity and generation profile;
- precision;
- padded text and reference buckets;
- padded latent and generation-block buckets;
- solver, step index, and step count;
- CFG layout.

Every field participates in equality. A batch contains one work class and one exact key, so a
precision, shape, step, checkpoint, or CFG difference isolates the request.

```python
from nar_vae.serving import RequestMetadata, ShapeBucketKey

key = ShapeBucketKey(
    checkpoint="checkpoint@sha256:...",
    generation_profile="fast",
    precision="bf16",
    text_bucket=64,
    reference_bucket=256,
    latent_bucket=512,
    block_bucket=32,
    solver="euler",
    step_index=0,
    step_count=16,
    cfg_layout="independent-fused",
)
request = RequestMetadata(
    request_id="request-0001",
    client_id="client-0001",
    arrival_time_s=0.0,
    first_audio_deadline_s=0.1,
    bucket_key=key,
    target_language="tr",
    reference_language="es",
)
```

## Admission and scheduling contract

`DeadlineBatchScheduler` accepts an injected clock and a `SchedulerConfig`. Admission rejects an
expired request or one beyond `max_active_requests`. The first-block dispatch deadline is the
earlier of the request deadline and `arrival_time_s + max_queue_delay_s`; waiting in the queue
beyond it records a timeout instead of hiding overload in an unbounded queue. Once dispatched,
completion is still checked against the request's absolute first-audio deadline.

Ready first blocks are preferred over continuations. To prevent starvation, one continuation batch
is selected after `max_first_block_batches` consecutive first-block batches when both classes are
ready. Within a class, earliest deadline wins; readiness, admission sequence, and request ID form
stable tie-breakers. `max_batch_size` is always enforced.

The scheduler is an execution-independent state machine. A later GPU integration must measure the
actual service time of each batch and call `complete_batch`; this package does not pretend that its
synthetic batch durations are model timings.

## Timing and existing benchmark compatibility

`StageTiming` records seconds for:

```text
TTFA = queue + conditioning + generation + decode + transfer + packetization
```

Some implementations may overlap stages, so TTFA is recorded independently rather than
recalculated from the stage sum. Load reports summarize every stage with count, mean, p50, p95,
p99, minimum, and maximum.

The existing `nar_vae.benchmark.run_benchmark` API and its timing fields remain compatible:

- `ttft` is still the first completed ODE step and is not playable audio;
- `ttfa` and `total` still end when the complete non-streaming waveform reaches CPU;
- `ode_sampling`, `decoding`, and `output_transfer` keep their values and meanings.

Each benchmark row now also contains a `stages` view. It maps generation, decode, and transfer to
the existing values, uses zero queue for the direct local call, and uses zero packetization because
no independently playable packet exists. The result carries
`result_kind=complete_waveform_benchmark`, `model_streaming=false`, and
`named_gpu_streaming_evidence=false`.

## Synthetic load suite

The standard suite runs both arrival patterns at 1, 8, 16, 32, and 50 clients:

```python
from nar_vae.serving import run_synthetic_load_suite, write_json_result

result = run_synthetic_load_suite(
    target_language="tr",
    reference_language="es",
    provenance={"experiment": "scheduler-contract-v1"},
)
write_json_result(result, "synthetic-load.json")
```

The synchronized-burst scenario admits all clients at one instant. The steady-stream scenario
staggers client admissions and gives each request continuation blocks. Both use `ManualClock` and
advance through an event loop without a wall-clock sleep. Callers can inject scheduler limits,
synthetic service times, client counts, and a clock factory.

Reports are JSON-compatible and contain:

- workload, exact bucket, language-pair, seed, and caller provenance;
- submitted, admitted, completed, rejected, timed-out, and failed counts;
- rejection/failure reason distributions;
- queue and TTFA p50/p95/p99 plus all other stage distributions;
- completed-request throughput and block counts;
- batch-size distribution and deterministic batch history;
- per-request lifecycle and first-audio timing records;
- null GPU memory/utilization fields.

Every suite and scenario report says `synthetic=true`, `audio_generated=false`,
`model_streaming=false`, `hardware_measured=false`, and `claim_eligible=false`. These records test
scheduling, timing, aggregation, and serialization. They cannot support a 50-user, sub-100-ms,
streaming, multilingual, or cross-lingual checkpoint claim.

## Remaining hardware and architecture gates

The next current-checkpoint cycle must connect actual compatible ODE work to this boundary and run
the load matrix on a named GPU while continuing to label complete-waveform results as
non-streaming. True first-playable-audio generation would require a different causal decoding
contract in addition to separately trained blockwise/local acoustic weights. Because this project
keeps DACVAE immutable, that redesign is out of scope; this serving boundary must continue to label
its output as complete-waveform, non-streaming inference.
