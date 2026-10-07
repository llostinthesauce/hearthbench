# Bounded local agent benchmark

`agentic` measures a complete tool loop and the files it leaves behind. It is a
small diagnostic comparison, not evidence of unattended real-world reliability.
Each trial starts from a deterministic private scratch fixture. No tool launches
a service, downloads weights, or executes model-written code. `run_tests` renders
static-check results; its pytest-shaped output is not an actual pytest run.

The five existing pricing tasks and their IDs remain available. Three added tasks
use an isolated local-serving fixture rather than the real machine configuration:

| Task ID | What must happen |
| --- | --- |
| `agent_local_config_repair` | Inspect current policy, actual configuration and consumer; record the cause before editing; repair two JSON defaults; verify after the final edit. |
| `agent_local_config_recovery` | Encounter a deliberately stale exact patch anchor; inspect real evidence and recover to the correct configuration. |
| `agent_doc_provenance` | Read both current and superseded documentation and inspect launcher/consumer evidence; write a structured audit file with current authority and readiness requirements. |

Configuration scoring checks the parsed final JSON, including field types and
preservation of every other field. The provenance task scores its written JSON,
not a claimed fix in the final answer. Protected files include generated inventory
and an unrelated pending operator note. New files outside the allowed target and
protected edits followed by restoration violate scope. Repair checks require successful README, current policy, configuration and
consumer reads before cause recording and editing; edits must use apply_patch,
with successful static verification after the last edit. Recovery requires the
exact requested initial failed anchor, investigation afterward, and no stale retry. A do-nothing trajectory earns no outcome credit.

## Short head-to-head

Serve the first candidate through the repository's existing lifecycle, then run
this command against its actual loaded model ID. Substitute your model ID and
quant label; do not select the first `/v1/models` entry as the loaded identity.
For llama.cpp use its stable alias and port 8080 instead of the MLX example.

```bash
MODEL_ID='actual-loaded-model-id'
QUANT='candidate-a'
python3 scripts/bench_quality.py \
  --model "$MODEL_ID" --url http://127.0.0.1:8085/v1 \
  --backend mlx --quant "$QUANT" \
  --evals agentic --agent-tasks local_short --repeats 1 \
  --output-dir results/agentic_head_to_head \
  --transcript-dir results/agentic_head_to_head/candidate-a
```

Switch to the second candidate using the existing lifecycle, then repeat with its
model ID, quant label and a `candidate-b` transcript directory. Use the same
backend, launcher defaults, context and thinking policy for a quant-only
comparison. The CLI disables thinking by default. `--enable-thinking` opts in;
reasoning text is preserved in trajectory transcripts and tool-call replay, while
only answer content and workspace evidence are scored. For a model that always
reasons, increase `--token-scale` (for example 4) and report that setting. It scales
the per-request budget for agentic turns too; the cumulative task token cap remains
12,000. An empty answer stopped by the request token limit is reported as
`truncated_before_answer` and excluded from scores. Partial answers that truncate
still receive task scoring and keep `stop_reason=truncated`.

This short preset runs **three local trajectories once each**. Omit
`--agent-tasks local_short` to run all eight tasks once. Exact task IDs are also
accepted by `--agent-tasks`, for example `agent_local_config_repair` for one
smoke trajectory. The Python API can select exact IDs through
`evals.agentic.eval.build_cases(agent_tasks=(...), repeats=1)`.
For a reliability follow-up use `--repeats 3`; one successful trial establishes
completion once, not reliability. The aggregate label `reliable` is inherited
from the existing report format and must be interpreted alongside trial count.

## Bounds and interpretation

Each new task allows 14 assistant turns, 600 seconds, 12,000 cumulative output
tokens and 64 tool calls; requests are capped at 1,024 tokens and the remaining
trajectory token budget. Real HTTP request timeouts are capped at the remaining
wall budget. Output usage enforcement depends on the server reporting token usage.
Repeated identical consecutive calls stop after three, including repetitions
inside a single batch. Alternating cycles eventually hit the step/tool/token
limits; they are not labeled as identical-call repetition.

A successful `finish` ends the batch immediately. A failed `finish` returns an
error to the model so it can recover. Unknown tools, bad arguments and patch
errors are tool results rather than harness crashes. A server request failure before the task deadline is a
harness error, excluded from model task scoring. Expiry of the task deadline
is a scored max_wall_clock failure, including a request timed out by that deadline. A late response cannot mutate
the workspace after the wall budget expires. A synchronous custom client that
ignores timeout cannot be preempted by this runner.

Compare task pass counts, named missed outcomes and violated constraints, plus
stop reasons and token/latency totals. Keep startup/server errors and truncated
answers separate from task failures. Preserve the raw CSV/JSON and JSONL
transcripts; tool transcripts retain inspected file content and reasoning. These
fixtures are synthetic and contain no real user files. Existing pricing checks
remain narrow static pattern checks; this suite does not establish arbitrary
Python repair correctness or actual serving readiness.
