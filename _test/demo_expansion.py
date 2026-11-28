# coding: utf-8

"""terminal demonstration of the optional alias/merge expansion budget

run with:  PYTHONPATH=<repo root>:<repo root>/_test python -I demo_expansion.py
each numbered section corresponds to one of the requirements.
"""

import io
import tracemalloc

import ruamel.yaml
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.error import ExpansionBudgetError, ExpansionCycleError
from ruamel.yaml.expansion import count_expansion


def yaml_rt() -> ruamel.yaml.YAML:
    return ruamel.yaml.YAML(typ='rt')


def show(title: str, data: object) -> None:
    buf = io.StringIO()
    yaml_rt().dump(data, buf)
    print(f'--- {title} ---')
    print(buf.getvalue(), end='')


def section(n: int, title: str) -> None:
    print(f'\n==================== {n}. {title} ====================')


# 1. the budget itself ------------------------------------------------------
section(1, 'a positive integer budget, off by default')
print('default budget:', yaml_rt().expansion_budget)
for bad in (0, -1, True, 1.5, '10'):
    try:
        y = yaml_rt()
        y.expansion_budget = bad  # type: ignore
    except ValueError as exc:
        print(f'  {bad!r:5} -> ValueError: {exc}')

text = 'k: v\n'
y = yaml_rt()
y.expansion_budget = 2
print('count of "k: v" =', count_expansion(y.load(text)), '(key + value)')
y.expansion_budget = 1
try:
    y.load(text)
except ExpansionBudgetError as exc:
    print('one below the count ->', exc)

# 2. counting rules ---------------------------------------------------------
section(2, 'scalars count, containers do not; shared sub graphs count per path')
y = yaml_rt()
for src in ('k: v\n', '42\n', '- 1\n- 2\n'):
    print(f'  {src.splitlines()[0]!r:12} -> {count_expansion(y.load(src))}')
print('  empty containers ->', count_expansion(y.load('a: {}\nb: []\n')), '(only keys a/b)')
shared = y.load('a: &x\n  p: 1\n  q: 2\nb: *x\n')
print('  a: &x p/q, b: *x ->', count_expansion(shared), '(2 keys + 4 + 4 scalars)')

# 3. merge priority, masked values are not counted --------------------------
section(3, 'merge priority: explicit > first source; masked values not counted')
y = yaml_rt()
merge_yaml = (
    '- &mx\n  a: x1\n  b: x2\n  c: x3\n'
    '- &my\n  a: y1\n  b: y2\n  d: y4\n'
    '-\n  a: 1\n  <<: [*mx, *my]\n  m: 6\n'
)
data = y.load(merge_yaml)
for i, item in enumerate(data):
    print(f'  item {i}: {dict(item)}  count={count_expansion(item)}')
print('  in the merged map "a: 1" masks both sources, "b" comes from &mx')
print('  ->', dict(data[2]))

# 4. effective cycles are refused, masked ones are ignored ------------------
section(4, 'cycles in effective data vs. masked merge branches')
y = yaml_rt()
y.expansion_budget = 10 ** 6
for src in ('&a\n- 1\n- *a\n', '&a\nx: 1\nself: *a\n'):
    try:
        y.load(src)
    except ExpansionCycleError as exc:
        print(f'  {src.splitlines()[0]!r} ->', exc)
root = CommentedMap()
root['name'] = 'top'
root['selfref'] = root
try:
    y.dump(root, io.StringIO())
except ExpansionCycleError as exc:
    print('  programmatic cycle on dump ->', exc)
# masked cyclic merge branch
src = CommentedMap()
loop = CommentedSeq([1])
loop.append(loop)
src['loop'] = loop
src['keep'] = 'k'
merged = CommentedMap()
merged['loop'] = 0  # masks the cyclic branch
merged.add_yaml_merge([(0, src)])
good = CommentedMap()
good['m'] = merged
buf = io.StringIO()
y.dump(good, buf)
print('  masked cyclic merge branch is effective as only', dict(good['m']))
print('  and dumps fine:', repr(buf.getvalue()))

# 5. load and dump use the same effective facts -----------------------------
section(5, 'load and dump agree; after mutation dump uses the new facts')
y = yaml_rt()
data = y.load('a: &x\n  p: 1\n  q: 2\nb: *x\n')
y.expansion_budget = 10
buf = io.StringIO()
y.dump(data, buf)
print('  count 10 == budget 10 -> dumped', len(buf.getvalue()), 'bytes')
data['a']['extra'] = 9  # alias shared object
try:
    y.dump(data, io.StringIO())
except ExpansionBudgetError as exc:
    print('  after a.extra=9 on the alias-shared object ->', exc)
del data['a']['extra']
del data['a']['q']
buf = io.StringIO()
y.dump(data, buf)
print('  after deleting q it fits again:', repr(buf.getvalue()))

# 6. a failing dump does not write to the stream ----------------------------
section(6, 'no partial output: the caller stream is untouched on failure')


class RecordingStream:
    encoding = None

    def __init__(self) -> None:
        self.writes = []

    def write(self, data):
        self.writes.append(data)

    def flush(self) -> None:
        pass


y = yaml_rt()
y.expansion_budget = 2
bad = yaml_rt().load('x: 1\ny: 2\n')
stream = RecordingStream()
try:
    y.dump(bad, stream)
except ExpansionBudgetError:
    pass
print('  write() calls on a rejected dump:', len(stream.writes))
stream = RecordingStream()
y.dump(yaml_rt().load('x: 1\n'), stream)
print('  an accepted dump uses write():', stream.writes)

# 7. per document budgets, reuse after failure ------------------------------
section(7, 'per-document budgets and reusing the instance after a failure')
y = yaml_rt()
y.expansion_budget = 10
docs = list(y.load_all('---\nsmall: 1\n---\na: &x\n  p: 1\n  q: 2\nb: *x\n'))
print('  load_all with L=10 ->', [dict(d) if hasattr(d, 'items') else d for d in docs])
y.expansion_budget = 9
try:
    list(y.load_all('---\nsmall: 1\n---\na: &x\n  p: 1\n  q: 2\nb: *x\n'))
except ExpansionBudgetError as exc:
    print('  load_all with L=9 ->', exc)
print('  the same instance still loads the first doc:', dict(y.load('---\nsmall: 1\n')))

# 8. build_layered_shared_graph(30) -----------------------------------------
section(8, '30 layers, two references each: rejected under L=1000, bounded memory')
try:
    from test_redact import build_layered_shared_graph  # noqa: E402
except ImportError:
    # test_redact needs pytest; the demo should also run in a plain terminal,
    # so mirror its fixture (it is the fixture used by the tests)
    def build_layered_shared_graph(depth: int = 30) -> CommentedMap:
        layer0 = CommentedMap()
        layer0['secret'] = 'topsecret'
        layer0['mark'] = 0
        prev = layer0
        layers = [layer0]
        for i in range(1, depth + 1):
            m = CommentedMap()
            m['first'] = prev
            m['second'] = prev
            m['mark'] = i
            layers.append(m)
            prev = m
        root = CommentedMap()
        root['layers'] = layers
        return root

graph = build_layered_shared_graph(30)
text = io.StringIO()
yaml_rt().dump(graph, text)
text = text.getvalue()
print('  document size:', len(text), 'bytes')
print('  exact scalar occurrences in the effective data:', count_expansion(graph))
y = yaml_rt()
y.expansion_budget = 1000
for where, action in (
    ('dump', lambda: y.dump(graph, io.StringIO())),
    ('load', lambda: y.load(text)),
):
    tracemalloc.start()
    tracemalloc.reset_peak()
    base = tracemalloc.get_traced_memory()[0]
    try:
        action()
        print(f'  {where}: NOT rejected')
    except ExpansionBudgetError as exc:
        _, peak = tracemalloc.get_traced_memory()
        delta = (peak - base) / 1048576
        print(f'  {where}: rejected ({exc.used} counted), peak +{delta:.3f} MiB')
    finally:
        tracemalloc.stop()
