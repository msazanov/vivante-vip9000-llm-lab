# Profiling and inventory tooling

The tools in this directory are small entrypoints around the profiling and
board-policy contracts. Profiling and inventory tools are read-only or
append-only. The explicitly documented A733 fan-policy helper is the sole
privileged exception and changes only its validated runtime thermal trips.

Public weights, binaries, NBGs, custom kernels and source, SDK or kernel
patches, and NPU tools may be published when they are redistributable and
listed in `docs/experiments/public-artifact-manifest.json`. The manifest must
record SHA-256, byte size, origin, source commit, build/runtime/toolchain
provenance, and destination. Personal/sensitive data is prohibited, including
tokens, passwords, logins, private keys, identifiers, and credentials. Existing
payloads and raw evidence are not automatically tracked or copied merely
because a public artifact policy permits publication.
The authoritative artifact classes and explicitly safe canonical-source paths
are defined in `docs/experiments/public-artifact-classes.json`; its LFS globs
are checked against `.gitattributes`. A new non-canonical file, including an
arbitrarily named tooling script or SDK archive, requires a manifest entry or
an explicit safe-source policy row with provenance.

## Generate the public Q1 C0 fixture

Create the deterministic packed Q1/Q8 carrier fixture in a new directory:

```bash
python3 tooling/generate_q1_vip_fixture.py \
  --output-dir /tmp/q1-vip-c0-fixture
```

The generated fixture files are public synthetic data. A generated NBG is a
publishable artifact only when it is redistributable and first added to the
public-artifact manifest; this command does not automatically track it.

## Pin a real Bonsai Q1 workload manifest

Read the external GGUF with the pinned PrismML `gguf-py` reader and write only
metadata, payload offsets, and SHA-256 values to the repository:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tooling/q1_memory_accounting.py \
  --model /home/random/.local/share/orange-rag/models/Bonsai-27B-Q1_0.gguf \
  --gguf-python-root /home/random/src/llama-prismml/gguf-py \
  --output benchmarks/workloads/bonsai-27b-q1.json
```

Re-read the generated manifest through the pinned identity and accounting
validator:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tooling/q1_memory_accounting.py \
  --check benchmarks/workloads/bonsai-27b-q1.json
```

The command prints `PASS q1-memory-accounting/v1` only after the exact model
size/SHA-256, Q1 payload offsets, and pinned Bonsai totals validate. The source
GGUF and a future sidecar payload remain external until an intentional
redistributable manifest entry authorizes a public destination.
`sidecar.total_payload_bytes` is the binary sidecar span through the final
tensor end, including any alignment padding, rather than the sum of payload
bytes alone.

## Profile one command

Create a new result directory and retain the child output and telemetry:

```bash
python3 tooling/profile_command.py \
  --output-dir benchmarks/results/<run_id> \
  --run-id <run_id> \
  --interval-ms 100 \
  --label <experiment-label> \
  -- <command> <arg>...
```

`--run-id` is required and the output-directory basename must equal it. The
optional relative `--phase-file` defaults to `phases.jsonl`; its path inside
the run directory is passed to the direct child as `VIP9000_PHASE_FILE`. The
command writes `metadata.json`, `stdout.log`, `stderr.log`, `telemetry.jsonl`,
and the phase file. The profiler appends `profiler_start` and
`profiler_complete` lifecycle events with monotonic timestamps; the child may
append finer runtime events through `VIP9000_PHASE_FILE`. It samples the direct child plus board-wide readable
`/proc`/`sys` interfaces. This is not a claim of full process-tree accounting:
unobserved worker processes and vendor/kernel work may not be included in
process totals. Missing sensors are retained as missing and do not abort the
child command. Launch failures retain metadata and raw files. The output
directory must not already exist, a signal maps to exit status `128 + signal`,
and a collector failure cleans up the child process group after recording
`profiler_error`. The profiler also terminates any process-group descendants
left behind after the direct child exits, preventing late writes after the
`profiler_complete` boundary. Setup failures are retained in metadata whenever
the output directory itself remains writable.

The profiler does not automatically know VIP init, graph-load, allocation,
copy/flush, synchronization, or teardown boundaries. The benchmark harness
must record those phase timings and copy/byte counters when the runtime exposes
them, alongside the profiler artifacts.

## Summarize one target run

Read an immutable run bundle and emit a deterministic JSON summary. The bundle
must contain `metadata.json`, `telemetry.jsonl`, `thermal-guard.jsonl`, and the
non-empty phase JSONL named by the safe relative `metadata.files.phases` path:

```bash
python3 tooling/summarize_target_run.py benchmarks/results/<run_id>
python3 tooling/summarize_target_run.py benchmarks/results/<run_id> \
  --output /tmp/<run_id>-summary.json
```

The summary includes run status and elapsed time, sample counts, separate
profiler-child and guarded-workload peak RSS/HWM/thread counts, thermal
statistics by type, CPU-policy current/max frequency and governors,
cooling-state statistics by type, minimum available memory, and swap-use
delta/peak when the raw fields provide enough data. Phase events are validated
for nondecreasing monotonic timestamps and counted independently from thermal
guard events. If telemetry exposes only
`SwapFree`, the tool still derives the used-space delta but does not invent an
absolute used-space peak. JSONL is consumed as a stream. Exact min/median/max
statistics use a temporary on-disk SQLite spool with a bounded 2 MiB page
cache and 4096-value insert batch, so long traces are not retained in RAM; the
spool is removed after summarization and is never created inside the immutable
run bundle. Empty phase evidence is rejected; new profiler bundles always
contain the two lifecycle events unless phase recording itself failed. Inputs
are validated fail-closed and never modified; `--output` refuses to overwrite
an existing file.

## Profile VIPLite phases

`profiled_viplite_runner.c` keeps the NBG and buffers resident and prints H2D,
wall-clock `vip_run_network`, internal `VIP_NETWORK_PROP_PROFILING`, D2H, and
byte equality with the first output for each iteration. The runner accepts a
compiled NBG only; it does not parse GGUF or provide a host CPU implementation.
Device profiler cycles establish device execution, but a closed runtime may
still hide an internal fallback.

Build the runner on the target against the installed VIPLite:

```bash
gcc -O2 -std=c11 -Wall -Wextra -Werror \
  tooling/profiled_viplite_runner.c -I/usr/include -L/usr/lib \
  -Wl,-rpath,/usr/lib -lNBGlinker -o /tmp/profiled_viplite_runner

/tmp/profiled_viplite_runner --iterations 1000 \
  network_binary.nb input_0.dat output.bin

# Two input tensors and one output, as in packed Q1×Q8 E003:
/tmp/profiled_viplite_runner --iterations 100 \
  q1_q8.nb packed_weights.q1_0.bin activation.q8_0.bin output.f32.bin
```

Each input must have exactly the size reported by `vip_get_buffer_size`;
extra or missing bytes are an error. The output is written once after the loop
and is excluded from D2H timing.

Build the strict compact summary and chart as follows:

```bash
python3 tooling/summarize_viplite_profile.py <raw-run>/stdout.log \
  --run-id <run_id> \
  --thermal-guard <raw-run>/thermal-guard.jsonl \
  --telemetry <raw-run>/telemetry.jsonl \
  --output benchmarks/results/<run_id>/summary.json

python3 tooling/generate_viplite_phase_chart.py \
  benchmarks/results/<run_id>/summary.json \
  --output benchmarks/charts/<run_id>.svg
```

The parser requires contiguous iteration indices, `first` only for index 0, a
successful device profiler, and complete tensor/quantization metadata. Output
equality with the first run remains a repeatability check and is never renamed
as a golden. Neither command overwrites an existing output file.

Build the overall XY map only from saved JSON evidence:

```bash
python3 tooling/generate_experiment_xy_chart.py \
  --output benchmarks/charts/experiment-xy-overview.svg
```

The generator intentionally uses three panels. Bonsai shows milliseconds per
token and decode tokens/s; ShuffleNet shows milliseconds per inference and
inferences/s; the CPU golden shows PASS or no check. These units must not share
one axis. Failed or no-metric run IDs are listed separately and never receive a
synthetic `0`.

Build the packed Q1×Q8 EVIS chart from its machine summary:

```bash
python3 tooling/generate_q1_evis_chart.py \
  benchmarks/results/q1-vip9000-evis-bonsai-20260811/summary.json \
  --output benchmarks/charts/q1-vip9000-evis-bonsai-20260811.svg
```

The upper panel compares only the identical `16×128` shape. The lower panel
uses logarithmic XY axes for real Bonsai tiles. The generator fails closed if a
linear full-layer projection is presented as a measurement and never
overwrites an existing SVG.

## Fail closed on unsafe target state

Wrap every sustained target workload with the thermal execution guard inside
the profiler. The guard inventories and pins every readable thermal zone, CPU
frequency policy, and cooling device, samples them at 100 ms, and terminates
the command's process group at the 85 °C ceiling or when required sysfs state
disappears or changes identity:

```bash
python3 tooling/profile_command.py \
  --output-dir benchmarks/results/<run_id> \
  --run-id <run_id> \
  --interval-ms 100 \
  --label <experiment-label> \
  -- python3 tooling/thermal_exec_guard.py \
       --limit-mc 85000 \
       --interval-ms 100 \
       --trace benchmarks/results/<run_id>/thermal-guard.jsonl \
       -- <command> <arg>...
```

The guard is standard-library-only and leaves child stdout/stderr attached to
the profiler. Exit `86` means a runtime safety abort and exit `87` means
preflight or launch failure. Signals are returned as `128 + signal`. The
direct child and its descendants run in a separate process group so cleanup
also covers a descendant that ignores `SIGTERM`. The guard samples direct-child
RSS/HWM, threads, CPU ticks, context switches, and I/O, but external driver or
kernel work remains board-wide evidence rather than process accounting.

## Record one model result

After the profiled command and quality comparison produce a validated result
JSON, append it to the canonical ledger and selected card:

```bash
python3 tooling/record_model_result.py \
  --input benchmarks/results/<run_id>/result.json \
  --ledger benchmarks/results/model-runs.jsonl \
  --card benchmarks/models/<model-card>.md \
  --repo-root .
```

The recorder validates model identity, workload configuration, separate prompt
and decode metrics, quality evidence, status, software/workload/statistics
objects, and a safe raw-result path under
`benchmarks/results/<run_id>/*.json`. A qualified result needs non-null
headline metrics and positive repetitions, plus an existing compatible
qualified CPU reference or a self-referencing CPU baseline. It rejects
duplicate `run_id` values and requires exactly one result-marker pair in the
card. `--repo-root` defaults to `.`. Qualified recording additionally requires
the raw JSON, `metadata.json`, `stdout.log`, `stderr.log`, and `telemetry.jsonl`
under the resolved run directory. Metadata must match `run_id`, report
`exit_code: 0`, and contain null `launch_error` and `profiler_error`; the
safe-relative `metadata.files.phases` path must exist within that run directory
and may name a custom/nested phase file. Staged or missing raw files remain
allowed for unqualified, rejected, and failed records. POSIX `flock` serializes
the ledger/card pair; fsynced sibling temporary files and rollback/recovery
protect against a card replacement failure. This is not a cross-filesystem
atomic transaction. Failed and rejected records remain in both indexes.

## Inventory the target board

Run the read-only target inventory recipe on the A733/VIP9000 board:

```bash
bash tooling/target_inventory.sh \
  --output-dir benchmarks/inventory/<inventory_id>
```

It writes temporary `system.txt` and `runtime-files.sha256` first, covering
OS/CPU/memory, device-tree compatibility, VIP devices, NPU devfreq, CPU
frequency policies, thermals, and hashes of readable runtime files. It publishes
the output directory only after mandatory sections succeed; a failure leaves no
partial inventory. It records metadata and hashes, not an automatic artifact
publication. Redistributable files follow the public-artifact manifest and
privacy gate.

## Inspect the AcuityLite container

Run the host-side inspection against the pinned image (or explicitly supplied
image):

```bash
bash tooling/inspect_acuitylite.sh \
  --output benchmarks/inventory/<inventory_id>/acuitylite.txt \
  --image acuitylite:ready \
  --docker-bin docker
```

The container is run with no network, read-only root, dropped capabilities,
`no-new-privileges`, PID/memory limits, and a temporary `/tmp`. The host output
is temporary until Docker exits successfully. It records package/version
markers, SDK archive hash, a hash and sanitized marker for the license target,
prebuilt SDK version, library names, relevant headers, and selected strings.
An archive whose license does not permit redistribution remains external;
redistributable SDK or kernel patches may be published through the manifest.

## Persist the approved A733 PWM-fan trip

The reviewed A733 fan policy sets only the passive trip that is explicitly
bound to a `pwm-fan` cooling device in each of the `cpub_thermal_zone` and
`cpul_thermal_zone` zones. The helper completes type, binding, trip-count,
writeability, and numeric-original-value preflight before writing either
value, writes exactly `30000` millidegrees Celsius, and verifies readback. A
preflight failure therefore prevents partial writes. A write or readback
failure attempts to restore every captured original value; rollback is
best-effort and an incomplete rollback is reported as a warning.

Install the helper and unit as root, then enable the oneshot for future boots:

```bash
sudo install -o root -g root -m 0755 tooling/set_a733_fan_trip.sh \
  /usr/local/sbin/set_a733_fan_trip.sh
sudo install -o root -g root -m 0644 tooling/systemd/a733-fan-trip-30c.service \
  /etc/systemd/system/a733-fan-trip-30c.service
sudo systemctl daemon-reload
sudo systemctl enable --now a733-fan-trip-30c.service
```

Before applying the policy, the unit runs a binding-aware readiness check up
to five times with a one-second delay between failed checks. The complete
oneshot has an eight-second start timeout. This covers boot-time module/sysfs
readiness without creating an unbounded retry loop; a later apply failure is
reported as a failed unit and is not retried indefinitely.

Verify the persistent unit and both live values with the helper's binding-aware
read-only mode. It repeats zone and cooling-device discovery, but never opens a
trip temperature for writing:

```bash
systemctl is-enabled a733-fan-trip-30c.service
systemctl is-active a733-fan-trip-30c.service
sudo /usr/local/sbin/set_a733_fan_trip.sh --verify
```

The unit and installed helper are persistent files; the thermal trip values
are runtime sysfs state and are reapplied by the unit after boot. The helper
does not change governors, PWM values, fan mode, CPU-frequency trips, GPU or
NPU trips, or critical trips. Thirty degrees is an operator-approved fan
activation point, not a hardware safety claim and not a replacement for the
benchmark abort ceiling or kernel protection.

## How model-card rows are generated

`record_model_result.py` is the only row generator. It first validates the
input against the recorder’s required fields and status rules, checks the
ledger for a duplicate run ID, verifies the card has exactly one
`<!-- MODEL_RESULTS_START -->` and one `<!-- MODEL_RESULTS_END -->` marker,
then constructs both updated files using sibling temporary files and replaces
them in sequence; this is not a cross-file transaction. The row reports prompt
tok/s, decode tok/s, TTFT, peak RSS, quality, status, and a relative raw-result
link. Do not hand-edit rows or replace the model-card header.

## Current benchmark charts

The chart is built only from the saved canonical ledger and run summaries;
values are never copied manually into the SVG. For deterministic regeneration:

```bash
python3 tooling/generate_benchmark_chart.py \
  --ledger benchmarks/results/model-runs.jsonl \
  --results-dir benchmarks/results \
  --output benchmarks/charts/benchmark-overview.svg \
  --force
```

The `--force` flag publishes the SVG atomically after validating its inputs. To
check freshness without changing files:

```bash
python3 tooling/generate_benchmark_chart.py \
  --ledger benchmarks/results/model-runs.jsonl \
  --results-dir benchmarks/results \
  --output benchmarks/charts/benchmark-overview.svg \
  --check
```

Exit code `0` means that the SVG is current, `1` means that it is absent or
stale, and `2` means that a required input is absent or invalid. `--check` and
`--force` are mutually exclusive. The sources of truth are the JSONL ledger
`benchmarks/results/model-runs.jsonl` and
`benchmarks/results/*/summary.json`; the model card and SVG are derived views
of those saved records.
