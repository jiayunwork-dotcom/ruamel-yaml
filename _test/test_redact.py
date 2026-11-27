# coding: utf-8

"""
testing of path based redaction of round trip data (YAML(typ='rt').redact)

The fixtures reuse the anchor/alias and merge (<<) shapes exercised in
test_anchor.py and the commented shapes from test_comments.py.
"""

import io

import pytest  # type: ignore  # NOQA

import ruamel.yaml  # type: ignore
from roundtrip import dedent  # type: ignore # NOQA
from ruamel.yaml.comments import CommentedMap, CommentedSeq, merge_attrib  # type: ignore
from ruamel.yaml.redact import REDACTED  # type: ignore


def yaml_rt() -> ruamel.yaml.YAML:
    return ruamel.yaml.YAML(typ='rt')


def dump(yml: "ruamel.yaml.YAML", data: object) -> str:
    buf = io.StringIO()
    yml.dump(data, buf)
    return buf.getvalue()


def build_layered_shared_graph(depth: int = 30) -> CommentedMap:
    """a shared graph of ``depth`` layers.

    layer 0 holds the secret; every higher layer references the previous layer
    *twice* (``first`` and ``second``).  Without alias handling this grows
    exponentially when copied.
    """
    layer0 = CommentedMap()
    layer0['secret'] = 'topsecret'
    layer0['mark'] = 0
    prev = layer0
    layers = [layer0]
    for i in range(1, depth + 1):
        m = CommentedMap()
        m['first'] = prev
        m['second'] = prev  # second reference to the previous layer
        m['mark'] = i
        layers.append(m)
        prev = m
    root = CommentedMap()
    root['layers'] = layers
    return root


def deep_first_path(depth: int = 30) -> list:
    return ['layers', depth] + ['first'] * depth + ['secret']


class TestBasicRedaction:
    def test_scalar_map_and_seq(self) -> None:
        yml = yaml_rt()
        data = yml.load(dedent("""
        list:
        - keep
        - secret
        map:
          k: v
        nested:
          deep: value
        """))
        out = yml.redact(data, [['list', 1], ['map', 'k'], ['nested', 'deep']])
        assert out['list'][0] == 'keep'
        assert out['list'][1] == REDACTED
        assert out['map']['k'] == REDACTED
        assert out['nested']['deep'] == REDACTED
        # the result is still a round trip object and dumps/loads cleanly
        text = dump(yml, out)
        assert REDACTED in text
        again = yml.load(text)
        assert again['list'][1] == REDACTED

    def test_redacted_value_is_plain_string(self) -> None:
        yml = yaml_rt()
        data = yml.load('a: 1\n')
        out = yml.redact(data, [['a']])
        assert out['a'] == '[REDACTED]'
        assert isinstance(out['a'], str)

    def test_unselected_values_unchanged(self) -> None:
        yml = yaml_rt()
        data = yml.load(dedent("""
        a: 1
        b: 2
        c: 3
        """))
        out = yml.redact(data, [['b']])
        assert out['a'] == 1 and out['b'] == REDACTED and out['c'] == 3


class TestOccurrenceNotObject:
    def test_anchor_two_references_only_one_redacted(self) -> None:
        yml = yaml_rt()
        data = yml.load(dedent("""
        a: &base
          token: secret
          keep: 1
        b: *base
        """))
        out = yml.redact(data, [['a', 'token']])
        assert out['a']['token'] == REDACTED
        # the other occurrence of the same mapping must stay untouched
        assert out['b']['token'] == 'secret'
        assert out['a']['keep'] == 1 and out['b']['keep'] == 1
        # the occurrences are now distinct objects in the result
        assert out['a'] is not out['b']

    def test_unselected_occurrence_still_shared(self) -> None:
        yml = yaml_rt()
        data = yml.load(dedent("""
        a: &base
          token: secret
          keep: 1
        b: *base
        c: *base
        """))
        out = yml.redact(data, [['a', 'token']])
        # b and c were both unselected occurrences: they keep sharing one clone
        assert out['b'] is out['c']
        assert out['b']['token'] == 'secret'

    def test_duplicate_and_reordered_paths_identical(self) -> None:
        yml = yaml_rt()
        src = dedent("""
        a: &b
          x: 1
          y: 2
        b: *b
        """)
        one = yml.redact(yml.load(src), [['a', 'x'], ['a', 'y'], ['a', 'x']])
        two = yml.redact(yml.load(src), [['a', 'y'], ['a', 'x']])
        three = yml.redact(yml.load(src), [['a', 'x'], ['a', 'y']])
        text = dump(yml, one)
        assert dump(yml, two) == text
        assert dump(yml, three) == text

    def test_seq_index_is_occurrence(self) -> None:
        yml = yaml_rt()
        shared = CommentedMap()
        shared['v'] = 'secret'
        seq = CommentedSeq([shared, shared])
        out = yml.redact(seq, [[0, 'v']])
        assert out[0]['v'] == REDACTED
        assert out[1]['v'] == 'secret'
        assert out[0] is not out[1]

    def test_redact_whole_object_keeps_alias_occurrence_intact(self) -> None:
        yml = yaml_rt()
        data = yml.load(dedent("""
        a: &b
          x: 1
          y: 2
        c: *b
        """))
        out = yml.redact(data, [['a']])
        assert out['a'] == REDACTED
        # the other occurrence is a full independent copy, not also redacted
        assert out['c']['x'] == 1 and out['c']['y'] == 2
        yml.load(dump(yml, out))


class TestMergeKeys:
    def test_redact_inherited_field_leaves_source_intact(self) -> None:
        yml = yaml_rt()
        data = yml.load(dedent("""
        base: &b
          token: secret
          keep: 1
        m:
          <<: *b
          z: 9
        """))
        out = yml.redact(data, [['m', 'token']])
        # the inherited value at this occurrence is masked ...
        assert out['m']['token'] == REDACTED
        # ... but the merge source must not change
        assert out['base']['token'] == 'secret'
        # a field not selected still resolves through the merge
        assert out['m']['keep'] == 1
        reloaded = yml.load(dump(yml, out))
        assert reloaded['m']['token'] == REDACTED
        assert reloaded['base']['token'] == 'secret'

    def test_redact_source_field_leaves_inherited_occurrence_intact(self) -> None:
        yml = yaml_rt()
        data = yml.load(dedent("""
        base: &b
          token: secret
          keep: 1
        m:
          <<: *b
          z: 9
        """))
        out = yml.redact(data, [['base', 'token']])
        assert out['base']['token'] == REDACTED
        # the unselected inherited position keeps its own value
        assert out['m']['token'] == 'secret'
        reloaded = yml.load(dump(yml, out))
        assert reloaded['base']['token'] == REDACTED
        assert reloaded['m']['token'] == 'secret'

    def test_multiple_sources_priority(self) -> None:
        yml = yaml_rt()
        data = yml.load(dedent("""
        - &CENTER {x: 1, y: 2}
        - &BIG {r: 10}
        - <<: [*CENTER, *BIG]
          label: center/big
        """))
        out = yml.redact(data, [[2, 'x'], [2, 'r']])
        assert out[2]['x'] == REDACTED
        assert out[2]['r'] == REDACTED
        # unselected inherited and explicit values keep the original semantics
        assert out[2]['y'] == 2
        assert out[2]['label'] == 'center/big'
        # the sources themselves are untouched
        assert out[0]['x'] == 1 and out[1]['r'] == 10

    def test_explicit_override_wins_over_merge(self) -> None:
        yml = yaml_rt()
        data = yml.load(dedent("""
        base: &b
          token: secret
        m:
          <<: *b
          token: own
        """))
        out = yml.redact(data, [['m', 'token']])
        # the explicit local value is the one that gets replaced
        assert out['m']['token'] == REDACTED
        assert out['base']['token'] == 'secret'

    def test_merge_relation_rebuilt_on_result(self) -> None:
        yml = yaml_rt()
        data = yml.load(dedent("""
        base: &b
          token: secret
          keep: 1
        m:
          <<: *b
          z: 9
        """))
        out = yml.redact(data, [['m', 'z']])  # unrelated redaction keeps sharing
        assert len(getattr(out['m'], merge_attrib)) == 1
        assert out['m'].merge[0][1] is out['base']

    def test_redact_whole_merge_source_keeps_inherited_values(self) -> None:
        yml = yaml_rt()
        data = yml.load(dedent("""
        base: &b
          token: secret
        m:
          <<: *b
          z: 9
        """))
        out = yml.redact(data, [['base']])
        assert out['base'] == REDACTED
        assert out['m']['token'] == 'secret' and out['m']['z'] == 9
        yml.load(dump(yml, out))

    def test_redact_whole_merged_map_leaves_source_intact(self) -> None:
        yml = yaml_rt()
        data = yml.load(dedent("""
        base: &b
          token: secret
        m:
          <<: *b
          z: 9
        """))
        out = yml.redact(data, [['m']])
        assert out['m'] == REDACTED
        assert out['base']['token'] == 'secret'
        yml.load(dump(yml, out))


class TestMergeSourceEdits:
    def test_assign_and_delete_source_reflected_via_merge_and_reload(self) -> None:
        yml = yaml_rt()
        data = yml.load(dedent("""
        base: &b
          token: secret
          keep: 1
          extra: 7
        m:
          <<: *b
          z: 9
        other:
          <<: *b
          w: 5
        """))
        out = yml.redact(data, [['m', 'z']])  # sharing of base must be preserved
        out['base']['newfield'] = 42
        del out['base']['extra']
        # direct effective-value access honours the merge relation
        assert out['m']['newfield'] == 42
        assert out['other']['newfield'] == 42
        assert 'extra' not in out['m']
        text = dump(yml, out)
        reloaded = yml.load(text)
        # the same facts hold after dump -> load
        assert reloaded['m']['newfield'] == 42
        assert reloaded['other']['newfield'] == 42
        assert 'extra' not in reloaded['m']
        assert reloaded['m']['z'] == REDACTED

    def test_local_override_still_effective(self) -> None:
        yml = yaml_rt()
        data = yml.load(dedent("""
        base: &b
          token: secret
        m:
          <<: *b
        """))
        out = yml.redact(data, [['base', 'token']])
        out['m']['token'] = 'local'
        assert out['m']['token'] == 'local'
        assert out['base']['token'] == REDACTED
        reloaded = yml.load(dump(yml, out))
        assert reloaded['m']['token'] == 'local'


class TestRecursiveAliases:
    def _recursive_map(self) -> CommentedMap:
        root = CommentedMap()
        root['name'] = 'top'
        root['nested'] = CommentedMap()
        root['nested']['v'] = 1
        root['selfref'] = root
        return root

    def test_cycle_preserved_when_redacting_inside(self) -> None:
        yml = yaml_rt()
        root = self._recursive_map()
        out = yml.redact(root, [['nested', 'v']])
        assert out['selfref'] is out  # the cycle is still a real cycle
        assert out['nested']['v'] == REDACTED
        reloaded = yml.load(dump(yml, out))
        assert reloaded['selfref'] is reloaded
        assert reloaded['nested']['v'] == REDACTED

    def test_finite_path_through_a_cycle(self) -> None:
        yml = yaml_rt()
        root = self._recursive_map()
        # the path goes through the alias back to the same object and onward
        out = yml.redact(root, [['selfref', 'nested', 'v']])
        assert out['selfref'] is out
        assert out['selfref']['nested']['v'] == REDACTED
        reloaded = yml.load(dump(yml, out))
        assert reloaded['selfref'] is reloaded
        assert reloaded['nested']['v'] == REDACTED

    def test_recursive_sequence(self) -> None:
        yml = yaml_rt()
        seq = CommentedSeq()
        seq.append('a')
        seq.append(1)
        seq.append(seq)
        out = yml.redact(seq, [[1]])
        assert out[2] is out
        assert out[1] == REDACTED and out[0] == 'a'
        reloaded = yml.load(dump(yml, out))
        assert reloaded[2] is reloaded

    def test_no_dangling_aliases_on_reload(self) -> None:
        yml = yaml_rt()
        data = yml.load(dedent("""
        a: &x {p: 1, q: 2}
        b: *x
        c:
          d: *x
          e: *x
        """))
        out = yml.redact(data, [['b', 'p']])
        text = dump(yml, out)
        # every alias must resolve to an anchor present in the document
        import re

        anchors = set(re.findall(r'&([A-Za-z0-9_]+)', text))
        for alias in re.findall(r'\*([A-Za-z0-9_]+)', text):
            assert alias in anchors
        reloaded = yml.load(text)
        assert reloaded['b']['p'] == REDACTED
        assert reloaded['a']['p'] == 1 and reloaded['c']['d']['p'] == 1


class TestIndependence:
    def test_input_graph_never_modified(self) -> None:
        yml = yaml_rt()
        src = dedent("""
        # header comment
        a: &b
          token: secret  # token comment
          keep: 1
        b: *b
        """)
        data = yml.load(src)
        before = dump(yml, data)
        yml.redact(data, [['a', 'token']])
        assert data['a']['token'] == 'secret'
        assert data['b']['token'] == 'secret'
        assert dump(yml, data) == before

    def test_two_results_are_independent(self) -> None:
        yml = yaml_rt()
        data = yml.load(dedent("""
        a:
          token: secret
        b:
          token: secret
        """))
        first = yml.redact(data, [['a', 'token']])
        second = yml.redact(data, [['b', 'token']])
        first['a']['added'] = 99
        assert 'added' not in data['a']
        assert 'added' not in second['a']
        assert second['a']['token'] == 'secret'

    def test_comments_preserved(self) -> None:
        yml = yaml_rt()
        data = yml.load(dedent("""
        # header
        a: &b
          token: secret  # token comment
          keep: 1
        m:
          <<: *b
          z: 9  # zed
        """))
        out = yml.redact(data, [['a', 'token']])
        text = dump(yml, out)
        assert '# header' in text
        assert '# token comment' in text
        assert '# zed' in text


class TestDeepSharedGraph:
    def test_thirty_layers_single_deep_path_under_64kib(self) -> None:
        yml = yaml_rt()
        root = build_layered_shared_graph(30)
        out = yml.redact(root, [deep_first_path(30)])
        text = dump(yml, out)
        size = len(text.encode('utf-8'))
        assert size <= 64 * 1024
        # the output is valid and reloads without dangling aliases
        reloaded = yml.load(text)
        node = reloaded['layers'][30]
        for _ in range(30):
            node = node['first']
        assert node['secret'] == REDACTED

    def test_sibling_chain_not_redacted(self) -> None:
        yml = yaml_rt()
        root = build_layered_shared_graph(30)
        out = yml.redact(root, [deep_first_path(30)])
        # the second-reference chain is a different occurrence and stays intact
        node = out['layers'][30]['second']
        for _ in range(29):
            node = node['first']
        assert node['secret'] == 'topsecret'

    def test_no_exponential_expansion(self) -> None:
        yml = yaml_rt()
        root = build_layered_shared_graph(30)
        out = yml.redact(root, [deep_first_path(30)])
        text = dump(yml, out)
        # each layer may appear at most twice; never 2**30 times
        assert text.count('mark:') <= 62
        # only the selected leaf and the untouched shared leaf carry 'secret'
        assert text.count('secret:') == 2


class TestErrors:
    @pytest.mark.parametrize(
        'path',
        [
            [['a', 'missing']],       # missing key
            [['seq', 9]],             # index out of range
            [[]],                     # empty path
            [['a', 'b', 'x']],        # scalar encountered mid path
            [['a', -1]],              # negative index
            [['a', True]],            # boolean is not a valid step
            ['ab'],                   # path itself is not a list of steps
            [['seq', '0']],           # string used to index a sequence
        ],
    )
    def test_invalid_paths_raise_valueerror(self, path: list) -> None:
        yml = yaml_rt()
        data = yml.load(dedent("""
        a:
          b: 1
        seq:
        - 1
        - 2
        """))
        with pytest.raises(ValueError):
            yml.redact(data, path)

    def test_no_partial_result_and_input_unchanged_on_error(self) -> None:
        yml = yaml_rt()
        data = yml.load(dedent("""
        a:
          b: 1
        seq:
        - 1
        - 2
        """))
        snapshot = dump(yml, data)
        with pytest.raises(ValueError):
            yml.redact(data, [['a', 'b'], ['seq', 99]])
        assert dump(yml, data) == snapshot

    def test_empty_paths_argument_raises(self) -> None:
        yml = yaml_rt()
        data = yml.load('a: 1\n')
        with pytest.raises(ValueError):
            yml.redact(data, [])

    def test_parent_path_replaces_child_selection(self) -> None:
        yml = yaml_rt()
        data = yml.load(dedent("""
        a:
          b:
            c: secret
        """))
        out = yml.redact(data, [['a'], ['a', 'b', 'c']])
        assert out['a'] == REDACTED

    def test_validation_runs_against_original_graph(self) -> None:
        # an inherited key is a valid mapping key for path resolution
        yml = yaml_rt()
        data = yml.load(dedent("""
        base: &b
          token: secret
        m:
          <<: *b
        """))
        out = yml.redact(data, [['m', 'token']])
        assert out['m']['token'] == REDACTED
        with pytest.raises(ValueError):
            yml.redact(data, [['m', 'absent']])

    def test_plain_programmatic_containers(self) -> None:
        # graphs built directly with dict/list are promoted to round trip types
        yml = yaml_rt()
        out = yml.redact({'a': 1, 'l': [1, {'k': 'secret'}]}, [['a'], ['l', 1, 'k']])
        assert out['a'] == REDACTED
        assert out['l'][0] == 1 and out['l'][1]['k'] == REDACTED
        yml.load(dump(yml, out))

    def test_only_roundtrip_type_supports_redact(self) -> None:
        # 'unsafe' is excluded because constructing it already emits a
        # PendingDeprecationWarning under this repository's pytest filters
        for typ in ('safe', 'base'):
            with pytest.raises(NotImplementedError):
                ruamel.yaml.YAML(typ=typ).redact({'a': 1}, [['a']])
