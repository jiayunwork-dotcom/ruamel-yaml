# coding: utf-8

"""terminal demonstration of YAML(typ='rt').redact()

run with:  PYTHONPATH=<repo root> python -I demo_redact.py
each numbered section corresponds to one of the redaction requirements.
"""

import io

import ruamel.yaml
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.redact import REDACTED

yml = ruamel.yaml.YAML(typ='rt')


def show(title: str, data: object) -> None:
    buf = io.StringIO()
    yml.dump(data, buf)
    print(f'--- {title} ---')
    print(buf.getvalue())


def section(n: int, title: str) -> None:
    print(f'\n==================== {n}. {title} ====================')


# 1. basic path based replacement ------------------------------------------
section(1, 'mapping keys and sequence indices')
data = yml.load(
    'list:\n- keep\n- secret\nmap:\n  k: v\nnested:\n  deep: value\n',
)
out = yml.redact(data, [['list', 1], ['map', 'k'], ['nested', 'deep']])
show('redacted round trip object', out)
print('selected values are the string', repr(REDACTED))

# 2. occurrences, not objects ------------------------------------------------
section(2, 'two references to one mapping; only one occurrence is masked')
data = yml.load(
    'a: &base\n  token: secret\n  keep: 1\n'
    'b: *base\nm:\n  <<: *base\n  token: own\n',
)
out = yml.redact(data, [['a', 'token']])
show('a.token redacted, b.token untouched', out)
print('a.token =', out['a']['token'], '| b.token =', out['b']['token'])
print('a is b in result?', out['a'] is out['b'], '(they had to be split)')

# 3. merge keys in both directions ------------------------------------------
section(3, '<< inherited field vs merge source field')
src = 'base: &b\n  token: secret\n  keep: 1\nm:\n  <<: *b\n  z: 9\n'
inherited = yml.redact(yml.load(src), [['m', 'token']])
show('redact the inherited position', inherited)
source = yml.redact(yml.load(src), [['base', 'token']])
show('redact the source position', source)

# 4. merge facts survive; source edits stay consistent after reload ---------
section(4, 'edit an unredacted source; direct access and reload agree')
data = yml.load(
    'base: &b\n  token: secret\n  extra: 7\nm:\n  <<: *b\n  z: 9\n',
)
out = yml.redact(data, [['m', 'z']])  # unrelated redaction keeps the alias
out['base']['newfield'] = 42
del out['base']['extra']
buf = io.StringIO()
yml.dump(out, buf)
text = buf.getvalue()
show('after assigning newfield and deleting extra on the source', out)
reloaded = yml.load(text)
print('direct m.newfield =', out['m']['newfield'],
      '| reloaded m.newfield =', reloaded['m']['newfield'])
print("'extra' gone directly?", 'extra' not in out['m'],
      '| after reload?', 'extra' not in reloaded['m'])

# 5. recursive aliases / cycles ---------------------------------------------
section(5, 'a finite path through a recursive alias')
root = CommentedMap()
root['name'] = 'top'
root['nested'] = CommentedMap()
root['nested']['v'] = 1
root['selfref'] = root
out = yml.redact(root, [['selfref', 'nested', 'v']])
show('cycle retained, nested value redacted', out)
cycle_buf = io.StringIO()
yml.dump(out, cycle_buf)
reloaded = yml.load(cycle_buf.getvalue())
print('selfref still a real cycle after reload?', reloaded['selfref'] is reloaded)

# 6. input and sibling results are never touched ----------------------------
section(6, 'input graph and a second result stay independent')
data = yml.load('a:\n  token: secret\nb:\n  token: secret\n')
first = yml.redact(data, [['a', 'token']])
second = yml.redact(data, [['b', 'token']])
first['a']['added'] = 99
print('input a.token =', data['a']['token'], '| second a.token =', second['a']['token'])
print("'added' leaked into input?", 'added' in data['a'],
      '| into second result?', 'added' in second['a'])

# 7. deep shared graph stays compact ----------------------------------------
section(7, '30 layers, two references each, one deep redaction (<= 64 KiB)')
layer0 = CommentedMap()
layer0['secret'] = 'topsecret'
layer0['mark'] = 0
prev = layer0
layers = [layer0]
for i in range(1, 31):
    m = CommentedMap()
    m['first'] = prev
    m['second'] = prev
    m['mark'] = i
    layers.append(m)
    prev = m
graph = CommentedMap()
graph['layers'] = layers
deep_path = ['layers', 30] + ['first'] * 30 + ['secret']
out = yml.redact(graph, [deep_path])
buf = io.StringIO()
yml.dump(out, buf)
text = buf.getvalue()
print('output size:', len(text.encode()), 'bytes (limit 65536)')
print('leaf copies (secret: count) =', text.count('secret:'),
      '-> linear, not 2**30 expansion')

# 8. invalid paths raise ValueError, no partial result -----------------------
section(8, 'invalid paths raise ValueError before anything is changed')
data = yml.load('a:\n  b: 1\nseq:\n- 1\n- 2\n')
snapshot_buf = io.StringIO()
yml.dump(data, snapshot_buf)
snapshot = snapshot_buf.getvalue()
for bad in ([['a', 'missing']], [['seq', 9]], [[]], [['a', 'b', 'x']], [['a', -1]]):
    try:
        yml.redact(data, bad)
    except ValueError as exc:
        print(f'{bad!r:24} -> ValueError: {exc}')
check_buf = io.StringIO()
yml.dump(data, check_buf)
print('input unchanged after the failures?', check_buf.getvalue() == snapshot)
