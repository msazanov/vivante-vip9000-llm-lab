# Benchmark Chart Generator Design

## Problem

The canonical benchmark ledger is machine-readable, but its wide Markdown
table is difficult to read on GitHub and becomes increasingly fragile as more
experiments are added. The repository needs a deterministic chart generated
from saved evidence, not a manually maintained image or copied values.

## Goals

- Generate one GitHub-renderable SVG from canonical saved CPU and NPU result
  files using only the Python standard library.
- Keep CPU throughput and NPU latency in separate panels and units.
- Display medians and p10-p90 intervals without hiding result qualification.
- Exclude incomparable, failed, or metric-less attempts from plotted bars while
  reporting how many records were omitted and why.
- Make regeneration atomic, reproducible, testable, and suitable for a future
  CI freshness check.
- Preserve the full append-only Markdown table as a detailed fallback while
  making the generated chart the primary GitHub view.

## Non-goals

- No interactive JavaScript or HTML dashboard.
- No matplotlib, Pillow, pandas, Vega, or network dependency.
- No interpolation of missing metrics and no conversion of device latency into
  LLM token throughput.
- No claim that unqualified results establish model quality or numerical NPU
  correctness.
- No mutation of the canonical JSONL ledger or per-run summaries.

## Files and responsibilities

- `tooling/generate_benchmark_chart.py`: parse, validate, group, render, and
  atomically publish the SVG.
- `tests/test_generate_benchmark_chart.py`: unit and CLI behavior tests using
  temporary synthetic fixtures.
- `benchmarks/charts/benchmark-overview.svg`: generated deterministic artifact.
- `tooling/README.md`: regeneration and freshness-check commands.
- `benchmarks/models/bonsai-27b.md`: embed the chart and place the wide raw
  table inside a collapsed details block without changing result markers.
- `README.md`: link the generated benchmark overview.

The chart generator remains independent of `record_model_result.py`. The
recorder owns the ledger and table; the chart generator consumes their saved
outputs.

## CLI contract

```text
python3 tooling/generate_benchmark_chart.py \
  --ledger benchmarks/results/model-runs.jsonl \
  --results-dir benchmarks/results \
  --output benchmarks/charts/benchmark-overview.svg \
  --force
```

The three paths have the values above as repository-relative defaults.

- With no mode flag, generation refuses to replace an existing output.
- `--force` writes a sibling temporary file, flushes and fsyncs it, then uses
  `os.replace` to atomically publish the new SVG and fsyncs the parent
  directory. It creates a missing output parent only after every input has
  validated.
- `--check` performs a byte-for-byte comparison against the existing output,
  writes nothing, returns 0 when current, 1 when stale, and 2 for malformed or
  missing required inputs. A missing output is stale (1), not malformed.
- `--check` and `--force` are mutually exclusive.
- Diagnostics go to stderr and never include full raw result payloads.

The SVG contains no generation timestamp, host path, hostname, or random ID.
Identical inputs therefore produce identical bytes.

## CPU data selection and comparability

CPU rows come from `benchmarks/results/model-runs.jsonl`. A row is eligible for
plotting only when:

- `configuration.backend` is `cpu` case-insensitively;
- status does not case-insensitively start with `failed`;
- both prompt and decode statistics contain finite, non-negative p10, median,
  and p90 values ordered `p10 <= median <= p90`.

Eligible rows are grouped by the comparison signature:

```text
model.id
model.sha256
software.runtime
software.runtime_commit
software.compiler
software.driver
software.kernel
software.sdk
workload (the complete canonical object)
configuration.context
configuration.batch
configuration.ubatch
```

Thread count and partition are intentionally excluded because those are the
variables being compared. `software.command` is excluded because it encodes
those variables. `software.repository_commit` is evidence provenance rather
than the commit of the measured runtime and is therefore displayed in metadata
but excluded from comparison; `software.runtime_commit` remains mandatory.
Any newly introduced workload or execution-affecting configuration field must
be explicitly classified before the generator accepts it. A comparison cohort
is plotted only when it contains at least two eligible runs. Singleton cohorts,
failed rows, and rows with missing metrics are counted in the SVG caption but
do not influence scales. This keeps the bounded smoke test out of the
pp512/tg128 topology comparison without relying on run-name conventions.

Each eligible CPU row must also have a positive integer
`performance.repetitions`; each prompt/decode `samples` array must have exactly
that many finite, non-negative values; and p10, median, and p90 must be
consistent with that saved sample set. Quantiles use sorted samples and linear
interpolation at zero-based position `(count - 1) * q`; saved and recomputed
values must satisfy `math.isclose(rel_tol=1e-9, abs_tol=1e-12)`. Identity and
signature fields are mandatory rather than treated as empty strings.

Within a cohort, runs are sorted by `run_id`. Cohorts are sorted by their full
signature. Labels use a short model name plus the configured partition; the
full run ID appears in SVG metadata and the visible legend.

## NPU data selection

NPU rows are discovered as sorted
`benchmarks/results/*/summary.json` paths. A file is eligible only when:

- `schema_version` equals `vip9000-capability-run/v1`;
- `host_run_us` and `device_inference_us` contain ordered finite, non-negative
  p10, median, and p90 values;
- `configuration.measured_loops` is a positive integer;
- status does not case-insensitively start with `failed`.

For every NPU summary with that schema, `run_id` is required, unique, and must
equal the result-directory basename. Both latency statistics must report a
positive integer `samples` equal to `configuration.measured_loops`.
`configuration.loops` must equal `warmup_loops + measured_loops`.

Eligible NPU rows are grouped by an invariant comparison signature containing:

```text
asset_sha256.network_binary.nb
asset_sha256.input_0.dat
asset_sha256.vpm_run
asset_sha256.libNBGlinker.so
asset_sha256.libVIPhal.so
target.driver_abi
target.driver_software
target.board
target.kernel
target.device
target.module
target.cid
target.device_count
target.logical_core_count
configuration.device_core
configuration.loops
configuration.warmup_loops
configuration.measured_loops
configuration.preload
configuration.npd
configuration.bypass_output
```

The NPU frequency and host/profiler affinity are experimental variables, so
they are shown in the run label rather than included in the signature. Any new
memory, input-layout, output, or execution-mode field must be classified before
the row is accepted. Cohorts receive visible headers and separators so rows
from different network/input/runtime signatures are not presented as one
comparison. The current v1 evidence does not expose input shape/layout as
separate fields, so the exact network-binary and input-file hashes are its
identity anchors; a future schema that exposes those fields must add them to
the signature.

The plotted values are host and device latency converted from microseconds to
milliseconds. Process lifecycle time, initialization, output capture, and
device cycles remain in the result summary and are not plotted on the latency
axis. NPU cohorts are sorted by their full signature and rows within each
cohort by `run_id`.

The current resident100 result is therefore eligible; single-execution,
output-capture, and failed-prelaunch records are reported as omitted rather
than silently mixed into the steady-state panel.

## SVG presentation

The artifact is a 1200-pixel-wide SVG with a white background and two
vertically stacked panels so GitHub can scale it on desktop and mobile without
mixing units. Its minimum height is 900 pixels and its height grows
deterministically by 72 pixels for every plotted run beyond the initial three;
horizontal bars keep labels readable as the ledger grows.

### CPU panel

- Title: `CPU LLM throughput — higher is better`.
- X-axis: tokens per second.
- Each run has paired horizontal prompt and decode median bars.
- A horizontal whisker with vertical end caps shows p10-p90.
- Prompt and decode use distinct color-blind-safe colors.

### NPU panel

- Title: `VIP9000 resident latency — lower is better`.
- X-axis: milliseconds.
- Each run has paired horizontal host and device median bars.
- A horizontal whisker with vertical end caps shows p10-p90.
- The panel explicitly says that process/setup overhead and correctness are not
  represented by these bars.

Both panels use zero baselines, deterministic "nice" ticks, three-decimal value
labels, and XML-escaped text. The tick step is the first value in
`{1, 2, 5, 10} * 10^n` greater than or equal to `max(p90) / 5`; the axis ceiling
is the next multiple of that step. Formatting is locale-independent UTF-8 with
LF newlines and one trailing newline. Unqualified or rejected results receive
a visible status suffix and dashed bar outline; only status exactly
`qualified` uses a solid outline.

The footer states the saved-data sources, omitted-record counts, and that
unqualified results are performance observations rather than quality claims.

The SVG includes a `<title>` and `<desc>` plus text equivalents for every bar
to improve accessibility and make values inspectable without color alone. A
reader can verify each plotted median directly from the image: every bar has a
numeric label, every interval has visible end caps, and each panel has an
explicit legend and unit label.

The SVG exposes a stable structural interface for tests. Each metric is wrapped
in a group with `data-panel`, `data-cohort-id`, `data-run-id`, `data-metric`, and
`data-status`. Its children use `data-role="median-bar"`,
`data-role="whisker"`, `data-role="whisker-cap"`, and
`data-role="median-label"`. Cohort IDs are the first 12 lowercase hexadecimal
characters of SHA-256 over the canonical JSON comparison signature.

## Markdown integration

The Bonsai model card embeds the repository-relative chart immediately before
the recorded-results section. The existing wide table, header, and
`MODEL_RESULTS_START`/`MODEL_RESULTS_END` markers move unchanged inside:

```html
<details>
<summary>Full append-only result table</summary>

... existing table and markers ...

</details>
```

This preserves recorder compatibility while making the chart the default
GitHub presentation. README links to the SVG and the model card; it does not
duplicate benchmark values manually.

## Validation and error handling

The generator fails closed on malformed JSON, duplicate run IDs, identity/path
mismatches, unsafe or non-finite numeric values, inconsistent sample counts,
invalid quantile ordering, missing source files, and an eligible dataset that
cannot produce both required panels. Schema-valid records with a status that
starts with `failed`, or with intentionally absent resident statistics, are
omitted and counted. Wrong types, partial statistics, missing required identity
or signature fields, and internally inconsistent statistics are fatal and
return 2. Summary files with another schema version are out of scope rather
than NPU omissions because the directory also contains CPU summaries.

For CPU, a failed row is omitted before performance validation; a non-failed
row with zero repetitions and all throughput metrics absent is an expected
missing-metrics omission, while any partial metric set or mismatch with a
positive repetition count is fatal. For NPU, a failed row is omitted before
latency validation; a non-failed row with both resident latency blocks and
`measured_loops` absent is a non-resident omission, while a positive
`measured_loops` value with absent or partial latency statistics is fatal.

Omission reasons use a fixed display order: failed status, missing metrics,
non-resident execution, and singleton CPU cohort. Unsupported-schema summary
files are reported last as a separate out-of-scope count and do not count as
rejected NPU evidence.

All labels pass through XML escaping. The generator reads only explicit ledger,
result-directory, and output paths supplied by the caller. It does not follow
data-provided output paths or open model/raw artifact paths.

## Test strategy

Tests follow red-green-refactor and cover:

1. CPU comparison grouping excludes singleton smoke workloads and retains
   topology variants with the same comparison signature.
2. NPU discovery accepts resident statistics and omits single/output/failed
   capability records; NPU cohorts separate incompatible assets and modes.
3. Missing, non-finite, negative, or misordered p10/median/p90 values fail with
   actionable diagnostics.
4. XML-special labels are escaped and the generated document parses with
   `xml.etree.ElementTree`.
5. Unqualified status is visible; units and panel caveats are present.
6. Identical input produces byte-identical SVG.
7. Default output refuses overwrite; `--force` replaces atomically; `--check`
   distinguishes current, stale, and malformed inputs without writing.
8. The repository's real saved data generates an SVG containing the A76/A55
   CPU comparison and resident100 NPU row.
9. The existing `record_model_result.py` suite still passes after the table is
   wrapped in `<details>`.
10. Structural visual assertions prove that every plotted series has a visible
    median label, whisker end caps, legend entry, unit, status treatment, and
    omission caption; the checked-in real-data SVG is then rendered and
    inspected once as the GitHub-facing visual QA artifact.
11. Sample-count, identity/path, NPU cohort, newly introduced field, and
    malformed-versus-omitted boundaries fail closed as specified.
12. More than three plotted rows increases SVG height deterministically and
    preserves one non-overlapping labeled row per run.

## Acceptance criteria

- The standard-library CLI generates the checked-in SVG from current saved
  records and a second generation is byte-identical.
- `--check` succeeds against the checked-in artifact.
- The SVG parses as XML and includes both panels with correct current medians.
- A rendered visual inspection confirms that labels do not overlap, whiskers
  are visible, and both desktop-width and scaled-down views remain readable.
- The full repository test suite passes.
- A GPT-5.6 Luna review confirms data selection, unit separation, deterministic
  output, and recorder-marker compatibility.
