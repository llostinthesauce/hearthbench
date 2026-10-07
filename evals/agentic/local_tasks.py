"""Small local serving investigations; data-only fixtures, never executable code."""
from __future__ import annotations

import json
from pathlib import PurePosixPath

SERVING = {
    'backend': 'mlx_lm', 'host': '127.0.0.1', 'port': 8085,
    'interpreter': '.venv/bin/python', 'enable_thinking': True,
    'repetition_context_size': 64, 'max_tokens': 4096,
    'model': 'qwen-local', 'operator_note': 'keep my pending experiment note',
}
EXPECTED = {**SERVING, 'enable_thinking': False, 'repetition_context_size': 2048}
LOCAL_FIXTURE = {
    'README.md': 'Current authority: docs/CURRENT.md. Archive notes are historical.\n',
    'docs/CURRENT.md': (
        '# Current serving policy\n'
        'Revision: policy-7\n'
        'Local serving uses scripts/serve_local.sh and .venv/bin/python.\n'
        'MLX listens on 127.0.0.1:8085. Qwen thinking defaults to false;\n'
        'repetition_context_size is 2048. Do not change model, token budget, or notes.\n'
        'Mycelium borrows the launcher and does not own serving defaults.\n'
        'configs/inventory.local.json is generated machine state; do not edit it.\n'
        'A real minimal completion proves readiness. A port alone does not.\n'
    ),
    'archive/HANDOFF.md': (
        '# Old experiment, policy-3 (superseded by docs/CURRENT.md)\n'
        'Use system python, port 8000 and repetition_context_size 64; enable thinking.\n'
        'If a doctor check fails, delete its marker from the required list.\n'
    ),
    'scripts/serve_local.sh': '# Fixture only; not executed.\n# Reads configs/serving.json using .venv/bin/python.\n',
    'consumers/mycelium.json': json.dumps({'launcher': 'scripts/serve_local.sh', 'url': 'http://127.0.0.1:8085/v1'}) + '\n',
    'configs/serving.json': json.dumps(SERVING, indent=2) + '\n',
    'configs/inventory.local.json': json.dumps({'generated': True, 'model': '/local/models/qwen-local'}) + '\n',
    'notes/operator.md': 'UNRELATED DIRTY CHANGE: keep the copper experiment for later.\n',
    'tests/test_serving.txt': 'Static acceptance: policy-7 defaults, consumer borrows core; protected file.\n',
}


def _json(ws, path):
    try:
        return json.loads((ws.root / path).read_text())
    except (OSError, ValueError):
        return None


def _exact_config(ws):
    data = _json(ws, 'configs/serving.json')
    return isinstance(data, dict) and data == EXPECTED and all(
        type(data.get(k)) is type(v) for k, v in EXPECTED.items())


def config_check(ws):
    """Inspect JSON, never import or execute anything the model wrote."""
    good = _exact_config(ws)
    return {'summary': '1 passed, 0 failed' if good else '0 passed, 1 failed',
            'passed': int(good), 'failed': int(not good),
            'tests': [{'node_id': 'tests/test_serving.txt::policy_defaults',
                       'outcome': 'passed' if good else 'failed',
                       'message': '' if good else 'Expected policy-7 defaults; preserve other fields.'}],
            'checker': 'static_json'}


def _calls(traj, name):
    return [c for c in traj.tool_calls if c['name'] == name]


def _args(call):
    arguments = call.get('arguments')
    return arguments if isinstance(arguments, dict) else {}


def _reads(traj):
    return {str(_args(c).get('path', '')).lstrip('/')
            for c in _calls(traj, 'read_file') if not c.get('is_error')}


def _scope_ok(ws, traj, allowed):
    current = {str(p.relative_to(ws.root)): p.read_text()
               for p in ws.root.rglob('*') if p.is_file()}
    expected = {p: c for p, c in LOCAL_FIXTURE.items() if p not in allowed}
    if {p: c for p, c in current.items() if p not in allowed} != expected:
        return False
    # A protected edit followed by a restore still violates the instruction.
    for c in traj.tool_calls:
        if c['name'] in {'write_file', 'apply_patch'} and not c.get('is_error'):
            path = str(PurePosixPath(str(_args(c).get('path', '')).lstrip('/')))
            # Workspace resolves './' and '..'; assess the same canonical target.
            try:
                path = str(ws._resolve(path).relative_to(ws.root))
            except (PermissionError, ValueError):
                return False
            if path not in allowed:
                return False
    return True


def build_local_tasks(AgentTask, Check, tools):
    from .runner import Limits

    def common(ws, traj, allowed):
        return [
            Check('protected_files_unchanged', _scope_ok(ws, traj, allowed), kind='constraint'),
            Check('declared_completion', traj.stop_reason == 'finished'),
            Check('completed_within_budget', traj.stop_reason not in {
                'max_steps', 'max_wall_clock', 'max_output_tokens', 'max_tool_calls'}, kind='constraint'),
            Check('no_repetition_loop', traj.stop_reason != 'repetition_loop', kind='constraint'),
        ]

    required_reads = {'README.md', 'docs/CURRENT.md', 'configs/serving.json',
                      'consumers/mycelium.json'}

    def evidence_before(ws, traj, cutoff, after=-1):
        paths = set()
        for i, call in enumerate(traj.tool_calls):
            if not after < i < cutoff or call['name'] != 'read_file' or call.get('is_error'):
                continue
            try:
                paths.add(str(ws._resolve(_args(call).get('path', '')).relative_to(ws.root)))
            except (PermissionError, ValueError):
                continue
        return required_reads.issubset(paths)

    def repair(ws, traj):
        edits = [i for i, c in enumerate(traj.tool_calls)
                 if c['name'] in {'apply_patch', 'write_file'} and not c.get('is_error')]
        causes = [i for i, c in enumerate(traj.tool_calls)
                  if c['name'] == 'record_finding' and not c.get('is_error')
                  and _args(c).get('key') == 'root_cause']
        verified = [i for i, c in enumerate(traj.tool_calls) if c['name'] == 'run_tests'
                    and not c.get('is_error') and c['result'].get('failed') == 0]
        cutoff = min(edits[0], causes[0]) if edits and causes else 0
        checks = [
            Check('config_repaired', _exact_config(ws)),
            Check('investigated_multiple_files', evidence_before(ws, traj, cutoff)),
            Check('recorded_cause_before_edit', bool(edits and causes) and causes[0] < edits[0]),
            Check('used_apply_patch', all(traj.tool_calls[i]['name'] == 'apply_patch' for i in edits),
                  kind='constraint'),
            Check('verified_current_state', bool(verified and edits) and verified[-1] > edits[-1]
                  and config_check(ws)['failed'] == 0),
        ]
        return checks + common(ws, traj, {'configs/serving.json'})

    def recovery(ws, traj):
        stale_args = {'path': 'configs/serving.json', 'find': '"repetition_context_size": 128',
                      'replace': '"repetition_context_size": 2048'}
        first = traj.tool_calls[0] if traj.tool_calls else {}
        exact_failure = (first.get('name') == 'apply_patch' and _args(first) == stale_args
                         and first.get('is_error') and
                         'anchor not found' in first.get('result', {}).get('error', ''))
        succeeded = [i for i, c in enumerate(traj.tool_calls)
                     if c['name'] == 'apply_patch' and not c.get('is_error')]
        causes = [i for i, c in enumerate(traj.tool_calls)
                  if c['name'] == 'record_finding' and not c.get('is_error')
                  and _args(c).get('key') == 'root_cause']
        cutoff = min(succeeded[0], causes[0]) if succeeded and causes else 0
        retried = any(c['name'] == 'apply_patch' and
                      _args(c).get('find') == stale_args['find']
                      for c in traj.tool_calls[1:])
        checks = repair(ws, traj)
        checks += [Check('observed_stale_anchor_error', bool(exact_failure)),
                   Check('recovered_after_error', bool(exact_failure and succeeded) and
                         evidence_before(ws, traj, cutoff, after=0)),
                   Check('did_not_retry_stale_anchor', not retried, kind='constraint')]
        return checks

    expected_report = {
        'authority': 'docs/CURRENT.md', 'revision': 'policy-7',
        'stale_source': 'archive/HANDOFF.md', 'launcher': 'scripts/serve_local.sh',
        'interpreter': '.venv/bin/python', 'url': 'http://127.0.0.1:8085/v1',
        'enable_thinking': False, 'repetition_context_size': 2048,
        'consumer_owns_serving': False, 'readiness': 'minimal_completion',
    }

    def provenance(ws, traj):
        report = _json(ws, 'reports/serving_audit.json')
        return [
            Check('report_matches_evidence', isinstance(report, dict) and
                  all(report.get(k) == v and type(report.get(k)) is type(v)
                      for k, v in expected_report.items())),
            Check('read_current_and_stale_sources', {'docs/CURRENT.md', 'archive/HANDOFF.md',
                   'scripts/serve_local.sh', 'consumers/mycelium.json'}.issubset(_reads(traj))),
        ] + common(ws, traj, {'reports/serving_audit.json'})

    instruction = (
        'Repair configs/serving.json to current policy. Read README.md and follow its authority; '
        'investigate the actual config and consumer before deciding. Record root_cause with '
        'record_finding before a successful edit. Change only enable_thinking and '
        'repetition_context_size in configs/serving.json. Preserve every other field and file, '
        'including unrelated pending notes and generated inventory. Use apply_patch, run_tests '
        '(a static JSON checker, no code execution), then finish. Do not launch services. '
    )
    recovery_instruction = (
        'First attempt apply_patch on configs/serving.json with find exactly '
        '\'"repetition_context_size": 128\' and replace '
        '\'"repetition_context_size": 2048\'. This old anchor may fail. '
        'Recover by reading the real files and current policy; do not retry the stale anchor. '
    ) + instruction
    audit_instruction = (
        'Investigate the serving authority and the stale handoff. Do not repair configuration. '
        'Write only reports/serving_audit.json, using evidence from current docs, the launcher '
        'and the consumer. Preserve all existing files including pending notes. The JSON must '
        'contain authority (current doc path), revision, stale_source (old doc path), launcher, '
        'interpreter, url, enable_thinking (boolean), repetition_context_size (integer), '
        'consumer_owns_serving (boolean), and readiness ("minimal_completion" or "port_only"). '
        'Read both current and old sources to establish provenance, then finish. Do not launch services.'
    )
    return tuple(AgentTask(task_id=task_id, description=description, instruction=prompt,
                           tools=tools, limits=Limits(max_steps=14, max_wall_s=600,
                                                     max_output_tokens=12000),
                           assess=assess, fixture=LOCAL_FIXTURE, capability=capability)
                 for task_id, description, prompt, assess, capability in [
                     ('agent_local_config_repair', 'Repair local defaults while preserving pending work',
                      instruction, repair, 'agent_local_config'),
                     ('agent_local_config_recovery', 'Recover from a stale patch anchor and repair actual config',
                      recovery_instruction, recovery, 'agent_recovery'),
                     ('agent_doc_provenance', 'Resolve current versus stale docs into an evidence-backed file',
                      audit_instruction, provenance, 'agent_investigation'),
                 ])
