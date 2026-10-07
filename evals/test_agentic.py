"""The agent harness must stay bounded, isolated, and deterministic."""
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.agentic import eval as agentic_eval
from evals.agentic.runner import Limits, Trajectory, run_trajectory
from evals.agentic.tasks import TASKS, by_id, summarize_checks
from evals.agentic.workspace import (
    BROKEN_PATTERN, FIXED_PATTERN, FIXTURE, Workspace, SCHEMAS, dispatch_table, schemas_for,
)
from evals.core import Case, Response


@pytest.fixture()
def workspace(tmp_path):
    ws = Workspace(tmp_path / "ws")
    ws.reset()
    yield ws
    ws.cleanup()


# --- isolation -------------------------------------------------------------

@pytest.mark.parametrize("escape", [
    "../../../etc/passwd", "../outside.txt", "src/../../escape.py",
])
def test_traversal_is_refused(workspace, escape):
    for result in (workspace.read_file(escape), workspace.write_file(escape, "x"),
                   workspace.list_directory(escape)):
        assert "escapes the workspace" in result.get("error", "")


def test_an_absolute_path_is_contained_not_followed(workspace, tmp_path):
    """`/etc/passwd` is reinterpreted relative to the workspace root.

    The security property is containment, not rejection: a leading slash is
    stripped, so the write lands at <workspace>/etc/passwd and the real file is
    neither read nor touched.
    """
    assert "error" in workspace.read_file("/etc/passwd")          # ENOENT inside the workspace
    assert "root:" not in json.dumps(workspace.read_file("/etc/passwd"))
    workspace.write_file("/etc/passwd", "contained")
    assert (workspace.root / "etc" / "passwd").read_text() == "contained"
    assert Path("/etc/passwd").read_text() != "contained"


def test_write_cannot_create_files_outside_the_root(workspace, tmp_path):
    workspace.write_file("../../pwned.txt", "nope")
    assert not (tmp_path / "pwned.txt").exists()
    assert not (tmp_path.parent / "pwned.txt").exists()


def test_reset_is_byte_identical_between_trials(workspace):
    workspace.apply_patch("src/pricing.py", "total += total * HEAVY_SURCHARGE_RATE", "pass")
    workspace.write_file("junk.txt", "left over")
    workspace.record_finding("k", "v")
    workspace.reset()
    assert (workspace.root / "src" / "pricing.py").read_text() == FIXTURE["src/pricing.py"]
    assert not (workspace.root / "junk.txt").exists()
    assert workspace.findings == {}
    assert workspace.finished is None


# --- the fixture actually encodes the bug the tasks assume -----------------

def test_fixture_starts_broken_and_the_documented_fix_repairs_it(workspace):
    assert BROKEN_PATTERN.search(FIXTURE["src/pricing.py"])
    assert workspace.run_tests()["failed"] == 1
    workspace.apply_patch("src/pricing.py",
                          "total += total * HEAVY_SURCHARGE_RATE",
                          "total += base * HEAVY_SURCHARGE_RATE")
    result = workspace.run_tests()
    assert result["failed"] == 0 and result["passed"] == 2
    assert FIXED_PATTERN.search((workspace.root / "src" / "pricing.py").read_text())


def test_run_tests_reports_whether_the_test_file_was_edited(workspace):
    assert workspace.run_tests()["test_file_modified"] is False
    workspace.write_file("tests/test_pricing.py", "def test_ok():\n    assert True\n")
    assert workspace.run_tests()["test_file_modified"] is True


def test_apply_patch_refuses_an_ambiguous_anchor(workspace):
    workspace.write_file("src/dup.py", "x = 1\nx = 1\n")
    assert "matches 2 times" in workspace.apply_patch("src/dup.py", "x = 1", "x = 2")["error"]


def test_tool_errors_are_returned_not_raised(workspace):
    assert "error" in workspace.read_file("nope.py")
    assert "error" in workspace.get_ticket("T-999")
    assert "error" in workspace.search_files("[unclosed")
    assert workspace.get_ticket("T-999")["available"] == ["T-411"]


def test_schemas_cover_every_dispatchable_tool(workspace):
    assert set(SCHEMAS) == set(dispatch_table(workspace))
    with pytest.raises(KeyError):
        schemas_for(("no_such_tool",))


# --- bounded execution -----------------------------------------------------

class FakeClient:
    """Replays canned assistant turns; records what it was sent."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []

    def complete(self, case: Case) -> Response:
        self.requests.append(case)
        if not self.turns:
            return Response(text="done", finish_reason="stop")
        return self.turns.pop(0)


def _call(name, arguments, call_id="c1"):
    return Response(text="", finish_reason="tool_calls", tool_calls=[
        {"id": call_id, "name": name, "arguments": arguments,
         "raw_arguments": json.dumps(arguments)}])


def test_max_steps_is_enforced(workspace):
    client = FakeClient([_call("list_directory", {"path": f"src{i}"}) for i in range(50)])
    traj = run_trajectory(client, workspace, task_id="t", trial=0, system="s", instruction="i",
                          tools=("list_directory", "finish"), limits=Limits(max_steps=4))
    assert traj.stop_reason == "max_steps"
    assert traj.steps == 4


def test_identical_repeated_calls_stop_the_loop(workspace):
    client = FakeClient([_call("read_file", {"path": "README.md"}) for _ in range(20)])
    traj = run_trajectory(client, workspace, task_id="t", trial=0, system="s", instruction="i",
                          tools=("read_file", "finish"), limits=Limits(max_steps=20, repeat_limit=3))
    assert traj.stop_reason == "repetition_loop"
    assert traj.steps < 20


def test_output_token_budget_is_enforced(workspace):
    turns = [Response(text="", finish_reason="tool_calls", completion_tokens=500,
                      tool_calls=[{"id": f"c{i}", "name": "list_directory",
                                   "arguments": {"path": f"p{i}"}, "raw_arguments": "{}"}])
             for i in range(30)]
    traj = run_trajectory(FakeClient(turns), workspace, task_id="t", trial=0, system="s",
                          instruction="i", tools=("list_directory", "finish"),
                          limits=Limits(max_steps=30, max_output_tokens=1200))
    assert traj.stop_reason == "max_output_tokens"


def test_wall_clock_is_enforced(workspace):
    class SlowClient(FakeClient):
        """Distinct arguments each turn, so the repetition breaker cannot fire
        first and mask whether the wall-clock bound works."""

        def __init__(self):
            super().__init__([])
            self.turn = 0

        def complete(self, case):
            time.sleep(0.4)
            self.turn += 1
            return _call("list_directory", {"path": f"dir{self.turn}"})

    started = time.monotonic()
    traj = run_trajectory(SlowClient(), workspace, task_id="t", trial=0, system="s",
                          instruction="i", tools=("list_directory",),
                          limits=Limits(max_steps=100, max_wall_s=1.0))
    assert traj.stop_reason == "max_wall_clock"
    assert time.monotonic() - started < 6


def test_a_tool_outside_the_allowlist_is_refused_not_executed(workspace):
    client = FakeClient([_call("write_file", {"path": "src/x.py", "content": "boom"}),
                         Response(text="stopping", finish_reason="stop")])
    traj = run_trajectory(client, workspace, task_id="t", trial=0, system="s", instruction="i",
                          tools=("read_file", "finish"), limits=Limits(max_steps=3))
    assert traj.unknown_tools == ["write_file"]
    assert not (workspace.root / "src" / "x.py").exists()


def test_unparseable_arguments_are_counted_and_not_executed(workspace):
    bad = Response(text="", finish_reason="tool_calls", arguments_unparseable=1, tool_calls=[
        {"id": "c1", "name": "write_file", "arguments": None, "raw_arguments": "{bad"}])
    traj = run_trajectory(FakeClient([bad, Response(text="stop", finish_reason="stop")]),
                          workspace, task_id="t", trial=0, system="s", instruction="i",
                          tools=("write_file", "finish"), limits=Limits(max_steps=3))
    assert traj.invalid_arguments == 1
    assert traj.tool_calls[0]["is_error"]


def test_server_failure_is_a_harness_error_not_a_task_failure(workspace):
    traj = run_trajectory(FakeClient([Response(text="", error="unreachable: refused")]),
                          workspace, task_id="t", trial=0, system="s", instruction="i",
                          tools=("finish",), limits=Limits(max_steps=3))
    assert traj.stop_reason == "server_error"
    assert not traj.ok


def test_finish_ends_the_trajectory(workspace):
    client = FakeClient([_call("finish", {"summary": "did the thing"}),
                         _call("read_file", {"path": "README.md"})])
    traj = run_trajectory(client, workspace, task_id="t", trial=0, system="s", instruction="i",
                          tools=("read_file", "finish"), limits=Limits(max_steps=6))
    assert traj.stop_reason == "finished"
    assert traj.finished_summary == "did the thing"
    assert traj.steps == 1


def test_assistant_tool_call_turn_is_replayed_to_the_server(workspace):
    """A tool result with no preceding tool_calls turn is rejected by strict servers."""
    client = FakeClient([_call("read_file", {"path": "README.md"}),
                         Response(text="ok", finish_reason="stop")])
    run_trajectory(client, workspace, task_id="t", trial=0, system="s", instruction="i",
                   tools=("read_file", "finish"), limits=Limits(max_steps=4))
    second = client.requests[1].messages
    assert second[-2]["role"] == "assistant" and second[-2]["tool_calls"]
    assert second[-1]["role"] == "tool"
    assert second[-1]["tool_call_id"] == second[-2]["tool_calls"][0]["id"]


def test_transcript_is_written(workspace, tmp_path):
    client = FakeClient([_call("finish", {"summary": "s"})])
    traj = run_trajectory(client, workspace, task_id="t", trial=2, system="s", instruction="i",
                          tools=("finish",), limits=Limits(max_steps=3),
                          transcript_dir=tmp_path / "tr")
    path = Path(traj.transcript_path)
    assert path.is_file() and path.name == "t_trial2.jsonl"
    assert [json.loads(l)["role"] for l in path.read_text().splitlines()][:2] == ["system", "user"]


# --- scoring ---------------------------------------------------------------

def test_a_do_nothing_trajectory_scores_zero_outcomes_on_every_task(workspace):
    """Constraint checks pass vacuously; they must not add up to credit."""
    for task in TASKS:
        verdict = summarize_checks(task.assess(workspace, Trajectory(
            task_id=task.task_id, trial=0, stop_reason="max_steps")))
        assert verdict["outcomes_passed"] == 0, task.task_id
        assert verdict["passed"] is False, task.task_id


def test_narrating_a_fix_without_applying_it_does_not_pass(workspace):
    traj = Trajectory(task_id="agent_fix_ticket", trial=0, stop_reason="finished",
                      final_text="I changed src/pricing.py to use the base rate.",
                      finished_summary="Fixed src/pricing.py")
    verdict = summarize_checks(by_id("agent_fix_ticket").assess(workspace, traj))
    assert verdict["passed"] is False
    assert "fix_is_correct" in verdict["outcomes_missed"]


def test_editing_the_test_file_violates_a_constraint(workspace):
    workspace.apply_patch("src/pricing.py", "total += total * HEAVY_SURCHARGE_RATE",
                          "total += base * HEAVY_SURCHARGE_RATE")
    workspace.write_file("tests/test_pricing.py", "def test_x():\n    assert True\n")
    traj = Trajectory(task_id="agent_fix_ticket", trial=0, stop_reason="finished",
                      finished_summary="Fixed src/pricing.py")
    verdict = summarize_checks(by_id("agent_fix_ticket").assess(workspace, traj))
    assert "tests_not_modified" in verdict["constraints_violated"]
    assert verdict["passed"] is False


def test_synthesis_task_distinguishes_the_compounded_answer(workspace):
    task = by_id("agent_multi_file_synthesis")
    correct = Trajectory(task_id=task.task_id, trial=0, stop_reason="answered",
                         final_text="base 384.00, remote 18.50, heavy 46.08\nTOTAL: 448.58")
    wrong = Trajectory(task_id=task.task_id, trial=0, stop_reason="answered",
                       final_text="base 384.00, remote 18.50\nTOTAL: 450.80")
    names = {c.name: c for c in task.assess(workspace, correct)}
    assert names["total_correct"].passed and names["format_obeyed"].passed
    bad = {c.name: c for c in task.assess(workspace, wrong)}
    assert not bad["total_correct"].passed
    assert "compounded" in bad["total_correct"].detail


def test_build_cases_respects_the_repeats_override():
    assert len(agentic_eval.build_cases(repeats=1)) == len(TASKS)
    assert len(agentic_eval.build_cases(repeats=4)) == len(TASKS) * 4
    only = agentic_eval.build_cases(agent_tasks=("agent_recover",), repeats=2)
    assert {c.meta["task_id"] for c in only} == {"agent_recover"}


def test_summarize_labels_intermittent_tasks():
    rows = [{"case_id": "agent_fix_ticket#t0", "passed": 1, "score": 1.0, "detail": "stop=finished"},
            {"case_id": "agent_fix_ticket#t1", "passed": 0, "score": 0.5, "detail": "stop=max_steps"},
            {"case_id": "agent_recover#t0", "passed": 0, "score": 0.0, "detail": "stop=repetition_loop"}]
    out = agentic_eval.summarize(rows)
    assert out["by_task"]["agent_fix_ticket"]["verdict"] == "intermittent"
    assert out["by_task"]["agent_recover"]["verdict"] == "never"
    assert out["intermittent_tasks"] == ["agent_fix_ticket"]
    assert out["stop_reasons"]["repetition_loop"] == 1


def test_failed_finish_allows_recovery(workspace):
    client = FakeClient([_call('finish', {}), _call('finish', {'summary': 'recovered'})])
    traj = run_trajectory(client, workspace, task_id='t', trial=0, system='s', instruction='i',
                          tools=('finish',), limits=Limits())
    assert traj.steps == 2
    assert traj.finished_summary == 'recovered'


def test_finish_stops_later_mutations_in_same_batch(workspace):
    response = _call('finish', {'summary': 'done'})
    response.tool_calls.extend(_call('write_file', {'path': 'oops', 'content': 'x'}).tool_calls)
    traj = run_trajectory(FakeClient([response]), workspace, task_id='t', trial=0,
                          system='s', instruction='i', tools=('finish', 'write_file'), limits=Limits())
    assert traj.stop_reason == 'finished'
    assert not (workspace.root / 'oops').exists()


def test_batch_repetition_is_stopped_at_threshold(workspace):
    response = _call('read_file', {'path': 'README.md'})
    response.tool_calls *= 4
    response.tool_calls.extend(_call('write_file', {'path': 'oops', 'content': 'x'}).tool_calls)
    traj = run_trajectory(FakeClient([response]), workspace, task_id='t', trial=0,
                          system='s', instruction='i', tools=('read_file', 'write_file'), limits=Limits())
    assert traj.stop_reason == 'repetition_loop'
    assert len(traj.tool_calls) == 3
    assert not (workspace.root / 'oops').exists()


def test_remaining_token_budget_caps_next_request(workspace):
    response = _call('read_file', {'path': 'README.md'})
    response.completion_tokens = 80
    client = FakeClient([response, Response(text='done')])
    run_trajectory(client, workspace, task_id='t', trial=0, system='s', instruction='i',
                   tools=('read_file',), limits=Limits(max_output_tokens=100))
    assert [c.max_tokens for c in client.requests] == [100, 20]


def test_reasoning_is_preserved_in_transcript_and_replay(workspace, tmp_path):
    response = _call('read_file', {'path': 'README.md'})
    response.reasoning_content = 'Inspect the repository first.'
    client = FakeClient([response, Response(text='done')])
    traj = run_trajectory(client, workspace, task_id='t', trial=0, system='s', instruction='i',
                          tools=('read_file',), limits=Limits(), transcript_dir=tmp_path / 'trace')
    entries = [json.loads(line) for line in Path(traj.transcript_path).read_text().splitlines()]
    assert entries[2]['reasoning_content'] == response.reasoning_content
    assert client.requests[1].messages[2]['reasoning_content'] == response.reasoning_content


def test_local_tasks_are_registered():
    assert {task.task_id for task in TASKS} >= {
        'agent_local_config_repair', 'agent_local_config_recovery', 'agent_doc_provenance'}


def _local_turns(recover=False):
    turns = []
    if recover:
        turns.append(_call('apply_patch', {'path': 'configs/serving.json',
                      'find': '"repetition_context_size": 128',
                      'replace': '"repetition_context_size": 2048'}))
    turns += [_call('read_file', {'path': path}) for path in
              ('README.md', 'docs/CURRENT.md', 'configs/serving.json', 'consumers/mycelium.json')]
    turns += [_call('record_finding', {'key': 'root_cause', 'value': 'Config drifted from policy-7.'}),
              _call('apply_patch', {'path': 'configs/serving.json', 'find': '"enable_thinking": true',
                                   'replace': '"enable_thinking": false'}),
              _call('apply_patch', {'path': 'configs/serving.json', 'find': '"repetition_context_size": 64',
                                   'replace': '"repetition_context_size": 2048'}),
              _call('run_tests', {}), _call('finish', {'summary': 'Repaired configs/serving.json.'})]
    return turns


@pytest.mark.parametrize('recover', [False, True])
def test_local_config_tasks_pass_actual_repair_and_recovery(tmp_path, recover):
    task = by_id('agent_local_config_recovery' if recover else 'agent_local_config_repair')
    ws = Workspace(tmp_path / 'local', fixture=task.fixture)
    traj = run_trajectory(FakeClient(_local_turns(recover)), ws, task_id=task.task_id,
                          trial=0, system=task.system, instruction=task.instruction,
                          tools=task.tools, limits=task.limits)
    verdict = summarize_checks(task.assess(ws, traj))
    assert verdict['passed'], verdict
    assert ws.run_tests()['checker'] == 'static_json'


@pytest.mark.parametrize('damage', ['note', 'inventory', 'extra_file', 'fake_test', 'restore'])
def test_local_config_scope_rejects_unrelated_changes(tmp_path, damage):
    task = by_id('agent_local_config_repair')
    ws = Workspace(tmp_path / 'local', fixture=task.fixture)
    turns = _local_turns()
    paths = {'note': 'notes/operator.md', 'inventory': 'configs/inventory.local.json',
             'extra_file': 'new.py', 'fake_test': 'tests/test_serving.txt', 'restore': 'notes/operator.md'}
    path = paths[damage]
    turns.insert(-1, _call('write_file', {'path': path, 'content': 'unrelated'}))
    if damage == 'restore':
        turns.insert(-1, _call('write_file', {'path': path, 'content': task.fixture[path]}))
    traj = run_trajectory(FakeClient(turns), ws, task_id=task.task_id, trial=0,
                          system=task.system, instruction=task.instruction, tools=task.tools, limits=task.limits)
    verdict = summarize_checks(task.assess(ws, traj))
    assert 'protected_files_unchanged' in verdict['constraints_violated']
    assert not verdict['passed']


def test_local_repair_requires_verification_after_last_edit(tmp_path):
    task = by_id('agent_local_config_repair')
    ws = Workspace(tmp_path / 'local', fixture=task.fixture)
    turns = _local_turns()
    # A passing check on intermediate state does not verify a later edit.
    turns.insert(-1, _call('apply_patch', {'path': 'configs/serving.json',
                 'find': '"enable_thinking": false', 'replace': '"enable_thinking": true'}))
    turns.insert(-1, _call('apply_patch', {'path': 'configs/serving.json',
                 'find': '"enable_thinking": true', 'replace': '"enable_thinking": false'}))
    traj = run_trajectory(FakeClient(turns), ws, task_id=task.task_id, trial=0,
                          system=task.system, instruction=task.instruction, tools=task.tools, limits=task.limits)
    assert 'verified_current_state' in summarize_checks(task.assess(ws, traj))['outcomes_missed']


def test_doc_provenance_scores_file_not_narration(tmp_path):
    task = by_id('agent_doc_provenance')
    ws = Workspace(tmp_path / 'local', fixture=task.fixture)
    report = {'authority': 'docs/CURRENT.md', 'revision': 'policy-7',
              'stale_source': 'archive/HANDOFF.md', 'launcher': 'scripts/serve_local.sh',
              'interpreter': '.venv/bin/python', 'url': 'http://127.0.0.1:8085/v1',
              'enable_thinking': False, 'repetition_context_size': 2048,
              'consumer_owns_serving': False, 'readiness': 'minimal_completion'}
    turns = [_call('read_file', {'path': p}) for p in ('docs/CURRENT.md', 'archive/HANDOFF.md',
             'scripts/serve_local.sh', 'consumers/mycelium.json')]
    turns += [_call('write_file', {'path': 'reports/serving_audit.json', 'content': json.dumps(report)}),
              _call('finish', {'summary': 'Recorded verified policy.'})]
    traj = run_trajectory(FakeClient(turns), ws, task_id=task.task_id, trial=0,
                          system=task.system, instruction=task.instruction, tools=task.tools, limits=task.limits)
    assert summarize_checks(task.assess(ws, traj))['passed']
    ws.write_file('reports/serving_audit.json', json.dumps({**report, 'authority': 'archive/HANDOFF.md'}))
    assert not summarize_checks(task.assess(ws, traj))['passed']
    ws.write_file('reports/serving_audit.json', 'I repaired it correctly.')
    assert not summarize_checks(task.assess(ws, traj))['passed']


def test_tool_call_cap_limits_a_large_batch(workspace):
    response = _call('list_directory', {'path': 'a'})
    response.tool_calls += [_call('list_directory', {'path': str(i)}).tool_calls[0] for i in range(20)]
    traj = run_trajectory(FakeClient([response]), workspace, task_id='t', trial=0,
                          system='s', instruction='i', tools=('list_directory',), limits=Limits(max_tool_calls=5))
    assert traj.stop_reason == 'max_tool_calls'
    assert len(traj.tool_calls) == 5


def test_external_ticket_path_is_refused(workspace):
    assert 'escapes the workspace' in workspace.get_ticket('../../../outside')['error']


def test_run_cases_preserves_preexisting_scratch_subfolder(tmp_path):
    root = tmp_path / 'scratch'
    preexisting = root / 'agent_local_config_repair_t0'
    preexisting.mkdir(parents=True)
    note = preexisting / 'mine.txt'
    note.write_text('owner data')
    cases = agentic_eval.build_cases(agent_tasks=('agent_local_config_repair',), repeats=1)
    results = agentic_eval.run_cases(cases, FakeClient(_local_turns()), workspace_root=root)
    assert results[0][0].passed
    assert note.read_text() == 'owner data'


def test_eval_client_preserves_returned_reasoning(monkeypatch):
    import io
    import urllib.request
    from evals.core import EvalClient
    data = {'choices': [{'message': {'content': 'answer', 'reasoning_content': 'reasoning trace'},
                         'finish_reason': 'stop'}], 'usage': {}}
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *args, **kwargs: io.BytesIO(json.dumps(data).encode()))
    response = EvalClient().complete(Case(case_id='r', prompt='test'))
    assert response.reasoning_content == 'reasoning trace'
    assert response.reasoning_chars == len(response.reasoning_content)


def test_trajectory_temporarily_caps_http_timeout(workspace):
    class TimedClient(FakeClient):
        timeout = 600
        def complete(self, case):
            assert 0 < self.timeout <= 1
            return super().complete(case)
    client = TimedClient([Response(text='done')])
    run_trajectory(client, workspace, task_id='t', trial=0, system='s', instruction='i',
                   tools=('finish',), limits=Limits(max_wall_s=1))
    assert client.timeout == 600


def test_agentic_honors_scaled_case_token_budget(tmp_path):
    from dataclasses import replace
    cases = agentic_eval.build_cases(agent_tasks=('agent_local_config_repair',), repeats=1)
    cases = [replace(c, max_tokens=4096) for c in cases]
    client = FakeClient([Response(text='done')])
    agentic_eval.run_cases(cases, client, workspace_root=tmp_path / 'ws')
    assert client.requests[0].max_tokens == 4096


def test_agentic_empty_reasoning_truncation_is_unscored(tmp_path):
    from scripts.bench_quality import _run_self_driven
    from evals.registry import EVALS
    cases = agentic_eval.build_cases(agent_tasks=('agent_local_config_repair',), repeats=1)
    response = Response(text='', reasoning_content='unfinished reasoning', finish_reason='length')
    result = _run_self_driven(EVALS['agentic'], cases, FakeClient([response]), {}, False,
                              {'workspace_root': tmp_path / 'ws'})
    assert result['rows'][0]['status'] == 'truncated_before_answer'
    assert result['rows'][0]['stop_reason'] == 'truncated'
    assert result['summary']['scored'] == 0
    assert result['summary']['truncated_before_answer'] == 1


def test_agent_task_preset_selects_only_local_tasks():
    cases = agentic_eval.build_cases(agent_tasks=('local_short',), repeats=1)
    assert [c.meta['task_id'] for c in cases] == ['agent_local_config_repair',
            'agent_local_config_recovery', 'agent_doc_provenance']


def test_cli_local_preset_dry_run_is_three_cases(monkeypatch, capsys):
    from scripts import bench_quality
    monkeypatch.setattr(sys, 'argv', ['bench_quality', '--model', 'fixture', '--evals', 'agentic',
                       '--agent-tasks', 'local_short', '--repeats', '1', '--dry-run'])
    bench_quality.main()
    assert 'agentic: 3 case(s)' in capsys.readouterr().out


@pytest.mark.parametrize('elapsed,expected', [(600, 'max_wall_clock'), (10, 'server_error')])
def test_deadline_timeout_is_distinct_from_early_server_failure(workspace, monkeypatch, elapsed, expected):
    from evals.agentic import runner
    now = [0.0]
    monkeypatch.setattr(runner.time, 'monotonic', lambda: now[0])
    class DeadlineClient:
        timeout = 600
        def complete(self, case):
            now[0] = elapsed
            return Response(text='', error='unreachable: timed out')
    trajectory = run_trajectory(DeadlineClient(), workspace, task_id='deadline', trial=0,
                                system='s', instruction='i', tools=('finish',),
                                limits=Limits(max_wall_s=600))
    assert trajectory.stop_reason == expected
    assert trajectory.ok is (expected == 'max_wall_clock')


@pytest.mark.parametrize('violation', ['blind_edit', 'no_readme', 'write_file', 'cause_before_reads'])
def test_local_repair_rejects_wrong_method_or_evidence_order(tmp_path, violation):
    from evals.agentic.local_tasks import EXPECTED
    task = by_id('agent_local_config_repair')
    turns = _local_turns()
    if violation == 'blind_edit':
        turns = [turns[4], turns[5], turns[6], *turns[:4], *turns[7:]]
    elif violation == 'no_readme':
        turns.pop(0)
    elif violation == 'write_file':
        turns[5:7] = [_call('write_file', {'path': 'configs/serving.json',
                                         'content': json.dumps(EXPECTED)})]
    elif violation == 'cause_before_reads':
        turns = [turns[4], *turns[:4], *turns[5:]]
    ws = Workspace(tmp_path / 'local', fixture=task.fixture)
    traj = run_trajectory(FakeClient(turns), ws, task_id=task.task_id, trial=0,
                          system=task.system, instruction=task.instruction, tools=task.tools, limits=task.limits)
    verdict = summarize_checks(task.assess(ws, traj))
    assert not verdict['passed'], (violation, verdict)
    assert ws.run_tests()['failed'] == 0  # Correct final state alone must not pass.


@pytest.mark.parametrize('violation', ['wrong_anchor', 'wrong_replacement', 'not_first', 'stale_retry',
                                      'read_before_failure'])
def test_local_recovery_requires_exact_first_failure_and_no_retry(tmp_path, violation):
    task = by_id('agent_local_config_recovery')
    turns = _local_turns(recover=True)
    if violation == 'wrong_anchor':
        turns[0] = _call('apply_patch', {'path': 'configs/serving.json', 'find': 'random missing text',
                                       'replace': '"repetition_context_size": 2048'})
    elif violation == 'wrong_replacement':
        turns[0] = _call('apply_patch', {'path': 'configs/serving.json',
                 'find': '"repetition_context_size": 128', 'replace': '"repetition_context_size": 64'})
    elif violation == 'not_first':
        turns.insert(0, _call('list_directory', {}))
    elif violation == 'stale_retry':
        turns.insert(1, turns[0])
    elif violation == 'read_before_failure':
        turns = [turns[1], turns[0], *turns[2:]]
    ws = Workspace(tmp_path / 'local', fixture=task.fixture)
    traj = run_trajectory(FakeClient(turns), ws, task_id=task.task_id, trial=0,
                          system=task.system, instruction=task.instruction, tools=task.tools, limits=task.limits)
    verdict = summarize_checks(task.assess(ws, traj))
    assert not verdict['passed'], (violation, verdict)
    assert ws.run_tests()['failed'] == 0
