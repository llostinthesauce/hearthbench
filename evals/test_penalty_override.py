"""Exercise the installed request-default block without importing GPU modules."""
import ast
from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import serving_runtime


def test_explicit_zero_penalties_override_server_defaults():
    path = serving_runtime.installed_server_source()
    tree = ast.parse(path.read_text())
    assignments = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self' and target.attr in {'repetition_penalty', 'presence_penalty', 'frequency_penalty'}:
                if 'self.body' in ast.unparse(node):
                    assignments.append(node)
    assert len(assignments) == 3
    cli = SimpleNamespace(repetition_penalty=1.05, presence_penalty=.2, frequency_penalty=.3)
    for body, expected in [({}, [1.05, .2, .3]), ({'repetition_penalty':0.,'presence_penalty':0.,'frequency_penalty':0.}, [0.,0.,0.])]:
        obj = SimpleNamespace(body=body, response_generator=SimpleNamespace(cli_args=cli))
        exec(compile(ast.Module(body=assignments, type_ignores=[]), '<penalties>', 'exec'), {'self':obj,'cli_args':cli})
        assert [obj.repetition_penalty,obj.presence_penalty,obj.frequency_penalty] == expected
