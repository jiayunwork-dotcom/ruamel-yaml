# coding: utf-8

"""
testing of the optional alias/merge expansion budget of YAML instances

The fixtures come from the repository's own anchor/merge tests
(``test_anchor.py``) and the shared graph / recursive graph fixtures of
``test_redact.py`` (``build_layered_shared_graph``, ``deep_first_path`` and
the recursive alias shapes).
"""

import io

import pytest  # type: ignore  # NOQA

import ruamel.yaml  # type: ignore
from roundtrip import dedent  # type: ignore # NOQA
from ruamel.yaml.comments import CommentedMap, CommentedSeq  # type: ignore
from ruamel.yaml.error import (  # type: ignore
    ExpansionBudgetError,
    ExpansionCycleError,
)
from ruamel.yaml.expansion import (  # type: ignore
    check_expansion,
    count_expansion,
)


def yaml_rt() -> ruamel.yaml.YAML:
    return ruamel.yaml.YAML(typ='rt')


def dump(yml: "ruamel.yaml.YAML", data: object) -> str:
    buf = io.StringIO()
    yml.dump(data, buf)
    return buf.getvalue()


class RecordingStream:
    """a write-only sink that records every call, without look-back support"""

    # emulate a text stream (StringIO has an ``encoding`` attribute) so the
    # emitter writes str rather than bytes to it
    encoding = None

    def __init__(self) -> None:
        self.writes: list = []

    def write(self, data: object) -> int:
        self.writes.append(data)
        return len(data)  # type: ignore

    def flush(self) -> None:
        pass

    @property
    def text(self) -> str:
        return ''.join(str(w) for w in self.writes)


# ---------------------------------------------------------------------------
# 1. the budget itself: positive int, default off
# ---------------------------------------------------------------------------


class TestBudgetConfiguration:
    def test_default_is_off(self) -> None:
        yml = yaml_rt()
        assert yml.expansion_budget is None
        # an exponential alias document loads happily without a budget
        data = yml.load('a: &x\n  p: 1\n  q: 2\nb: *x\nc: *x\n')
        assert data['c']['q'] == 2

    @pytest.mark.parametrize('bad', [0, -1, -100, True, False, 1.5, '10', [10]])
    def test_invalid_budgets_rejected(self, bad: object) -> None:
        yml = yaml_rt()
        with pytest.raises(ValueError):
            yml.expansion_budget = bad  # type: ignore
        assert yml.expansion_budget is None

    def test_positive_int_accepted(self) -> None:
        yml = yaml_rt()
        yml.expansion_budget = 1
        assert yml.expansion_budget == 1
        yml.expansion_budget = None
        assert yml.expansion_budget is None


# ---------------------------------------------------------------------------
# 2. counting rules
# ---------------------------------------------------------------------------


class TestCountingRules:
    def test_scalars_count_keys_and_values(self) -> None:
        yml = yaml_rt()
        assert count_expansion(yml.load('k: v\n')) == 2
        assert count_expansion(yml.load('- 1\n- 2\n')) == 2
        assert count_expansion(yml.load('42\n')) == 1

    def test_container_itself_counts_zero(self) -> None:
        yml = yaml_rt()
        assert count_expansion(yml.load('{}\n')) == 0
        assert count_expansion(yml.load('[]\n')) == 0
        assert count_expansion(yml.load('a: {}\nb: []\n')) == 2

    def test_equal_to_budget_is_allowed(self) -> None:
        yml = yaml_rt()
        yml.expansion_budget = 2
        assert yml.load('k: v\n') == {'k': 'v'}

    def test_one_over_budget_rejected_on_load(self) -> None:
        yml = yaml_rt()
        yml.expansion_budget = 1
        with pytest.raises(ExpansionBudgetError) as excinfo:
            yml.load('k: v\n')
        assert 'expansion budget' in str(excinfo.value)
        assert excinfo.value.budget == 1
        assert excinfo.value.where == 'load'

    def test_same_subgraph_on_two_paths_counts_twice(self) -> None:
        yml = yaml_rt()
        data = yml.load(dedent("""
        a: &x
          p: 1
          q: 2
        b: *x
        """))
        # 4 scalars in one occurrence, twice through two effective paths,
        # plus the two keys a/b
        assert count_expansion(data) == 2 + 2 * 4
        assert count_expansion(data['a']) == 4

    def test_check_expansion_helper(self) -> None:
        yml = yaml_rt()
        data = yml.load('k: v\n')
        assert check_expansion(data, 2) == 2
        assert check_expansion(data, 5) == 2
        with pytest.raises(ExpansionBudgetError):
            check_expansion(data, 1)


# ---------------------------------------------------------------------------
# 3. mapping merge priority, masked values do not count
# ---------------------------------------------------------------------------


class TestMergePriority:
    """fixtures adapted from test_anchor.TestMergeKeysValues"""

    merge_yaml = dedent("""\
    - &mx
      a: x1
      b: x2
      c: x3
    - &my
      a: y1
      b: y2  # masked by the one in &mx
      d: y4
    -
      a: 1
      <<: [*mx, *my]
      m: 6
    """)

    anchor_merge_yaml = dedent("""\
    - &CENTER {x: 1, y: 2}
    - &LEFT {x: 0, y: 2}
    - &BIG {r: 10}
    - &SMALL {r: 1}
    # All the following maps are equal:
    # Explicit keys
    - x: 1
      y: 2
      r: 10
      label: center/small
    # Merge one map
    - <<: *CENTER
      r: 10
      label: center/medium
    # Merge multiple maps
    - <<: [*CENTER, *BIG]
      label: center/big
    # Override
    - <<: [*BIG, *LEFT, *SMALL]
      x: 1
      label: center/huge
    """)

    def test_effective_keys_are_counted(self) -> None:
        yml = yaml_rt()
        data = yml.load(self.merge_yaml)
        assert [count_expansion(item) for item in data] == [6, 6, 10]
        # a sequence itself counts no scalars, only its members do
        assert count_expansion(data) == 6 + 6 + 10
        # the effective result is exactly what the safe loader sees (order of
        # the inherited keys may differ, the mapping itself is equal)
        safe = ruamel.yaml.YAML(typ='safe', pure=True).load(self.merge_yaml)
        assert dict(data[2]) == dict(safe[2])

    def test_masked_inherited_b_value_not_counted(self) -> None:
        yml = yaml_rt()
        data = yml.load(self.merge_yaml)
        merged = data[2]
        # explicit a: 1, m: 6, then b, c from &mx and only d from &my
        # (b from &my and a from &my are masked by higher priority keys)
        assert list(merged.keys()) == ['a', 'm', 'b', 'c', 'd']
        assert merged['b'] == 'x2'

    def test_masked_branch_with_large_subgraph_is_not_visited(self) -> None:
        yml = yaml_rt()
        src = dedent("""\
        - &src
          shared: &big
            a: 1
            b: 2
            c: 3
            d: 4
          keep: k
        - <<: *src
          shared: 0
        """)
        data = yml.load(src)
        # the four scalars under the masked 'shared' do not appear in the
        # effective data of the second map: it only has shared: 0 and keep: k
        assert data[1]['shared'] == 0
        assert count_expansion(data[1]) == 4
        # the source occurrence (first item) still contains &big:
        # keys shared/keep (2) + keep's value k (1) + nested map 4 keys and
        # 4 values (8) = 11
        assert count_expansion(data[0]) == 11
        assert count_expansion(data) == 11 + 4

    def test_merge_key_itself_not_counted(self) -> None:
        yml = yaml_rt()
        plain = yml.load('a: 1\n')
        merged = yml.load('base: &b {a: 1}\nm:\n  <<: *b\n')
        # base: key a + value 1 = 2 scalars; m: inherited a:1 = 2 scalars;
        # plus the two top level keys base and m; the '<<' key and its source
        # alias are not counted as an occurrence
        assert count_expansion(plain) == 2
        assert count_expansion(merged) == 2 + 2 + 2

    def test_anchor_merge_fixture_effective_counts(self) -> None:
        yml = yaml_rt()
        data = yml.load(self.anchor_merge_yaml)
        assert [count_expansion(x) for x in data] == [4, 4, 2, 2, 8, 8, 8, 8]
        assert count_expansion(data) == 44

    def test_priority_does_not_change_with_budget(self) -> None:
        yml = yaml_rt()
        yml.expansion_budget = 22
        data = yml.load(self.merge_yaml)
        assert data[2]['a'] == 1 and data[2]['b'] == 'x2'

    def test_override_priority_fixture_from_test_redact(self) -> None:
        yml = yaml_rt()
        data = yml.load(dedent("""
        - &CENTER {x: 1, y: 2}
        - &BIG {r: 10}
        - <<: [*CENTER, *BIG]
          label: center/big
        """))
        assert data[2]['x'] == 1 and data[2]['r'] == 10
        assert count_expansion(data) == 4 + 2 + 8 + 0


# ---------------------------------------------------------------------------
# 4. cycles in the effective data
# ---------------------------------------------------------------------------


class TestEffectiveCycles:
    @pytest.mark.parametrize(
        'source',
        [
            '&a\n- 1\n- *a\n',
            '&a\nx: 1\nself: *a\n',
            '- &a\n  - 1\n  - *a\n',
        ],
    )
    def test_cycle_rejected_even_with_huge_budget(self, source: str) -> None:
        yml = yaml_rt()
        yml.expansion_budget = 10 ** 6
        with pytest.raises(ExpansionCycleError) as excinfo:
            yml.load(source)
        assert 'cycle' in str(excinfo.value)

    def test_merge_cycle_yields_fixed_point(self) -> None:
        # a cyclic << relation does not recurse; its reachable own keys form
        # the finite fixed point.  Committing such a document also fails in
        # the existing constructor for an unrelated reason, so build the shape
        # programmatically.
        merged = CommentedMap()
        merged['a'] = 1
        merged.add_yaml_merge([(0, merged)])
        yml = yaml_rt()
        yml.expansion_budget = 1000
        assert count_expansion(merged) == 2
        assert merged['a'] == 1

    def test_programmatic_cycle_rejected_on_dump(self) -> None:
        # the shape used in test_redact.TestRecursiveAliases
        root = CommentedMap()
        root['name'] = 'top'
        root['nested'] = CommentedMap()
        root['nested']['v'] = 1
        root['selfref'] = root
        yml = yaml_rt()
        yml.expansion_budget = 1000
        with pytest.raises(ExpansionCycleError):
            yml.dump(root, io.StringIO())

    def test_masked_cycle_branch_does_not_falsely_reject(self) -> None:
        # the self referencing sequence lives in a merge source, but the key is
        # masked by an explicit value in the merged map, so it is not in the
        # effective data
        src = CommentedMap()
        loop = CommentedSeq([1])
        loop.append(loop)
        src['loop'] = loop
        src['keep'] = 'k'
        merged = CommentedMap()
        merged['loop'] = 0
        merged.add_yaml_merge([(0, src)])
        root = CommentedMap()
        root['m'] = merged
        yml = yaml_rt()
        yml.expansion_budget = 1000
        text = dump(yml, root)
        assert 'keep: k' in text
        assert yml.load(text)['m']['loop'] == 0

    def test_masked_cycle_from_yaml_is_accepted(self) -> None:
        # an anchored merge source containing a cycle; the key is overridden
        # in the merged map, so the cycle is not effective
        text = dedent("""
        base: &b
          guard: 1
          list: &l
          - 1
          - *l
        m:
          <<: *b
          list: 0
        """)
        yml = yaml_rt()
        yml.expansion_budget = 1000
        data = yml.load(text)
        assert data['m']['list'] == 0
        assert yml.load(dump(yml, data))['m']['guard'] == 1

    def test_unmasked_cycle_from_yaml_is_rejected(self) -> None:
        # ruamel's round trip constructor does not keep the back reference in
        # a merge source (it becomes null), so build the unmasked cyclic
        # merge branch programmatically and dump it
        src = CommentedMap()
        loop = CommentedSeq([1])
        loop.append(loop)
        src['loop'] = loop
        src['keep'] = 'k'
        merged = CommentedMap()
        merged.add_yaml_merge([(0, src)])
        root = CommentedMap()
        root['m'] = merged
        yml = yaml_rt()
        yml.expansion_budget = 1000
        with pytest.raises(ExpansionCycleError):
            yml.dump(root, io.StringIO())


# ---------------------------------------------------------------------------
# 5. composer/constructor and representer/serializer agree
# ---------------------------------------------------------------------------


class TestBothPathsAgree:
    merge_yaml = dedent("""\
    - &mx
      a: x1
      b: x2
      c: x3
    - &my
      a: y1
      b: y2
      d: y4
    -
      a: 1
      <<: [*mx, *my]
      m: 6
    """)

    def test_load_and_dump_use_same_budget(self) -> None:
        yml = yaml_rt()
        yml.expansion_budget = 22
        data = yml.load(self.merge_yaml)
        # dump of the just loaded data is accepted
        dump(yml, data)

        yml2 = yaml_rt()
        yml2.expansion_budget = 21
        with pytest.raises(ExpansionBudgetError):
            yml2.load(self.merge_yaml)
        # and the already constructed data is refused on dump as well
        data = yaml_rt().load(self.merge_yaml)
        with pytest.raises(ExpansionBudgetError):
            yml2.dump(data, io.StringIO())

    def test_constructed_data_dump_check(self) -> None:
        yml = yaml_rt()
        data = yml.load(self.merge_yaml)
        yml.expansion_budget = 22
        assert len(dump(yml, data)) > 0
        yml.expansion_budget = 21
        with pytest.raises(ExpansionBudgetError):
            yml.dump(data, io.StringIO())

    def test_dump_uses_modified_alias_shared_object(self) -> None:
        yml = yaml_rt()
        data = yml.load(dedent("""
        a: &x
          p: 1
          q: 2
        b: *x
        """))
        yml.expansion_budget = 10
        text = dump(yml, data)  # exactly at the budget
        assert text
        # mutate the alias-shared object: both occurrences now have one more
        # scalar, so the very same yaml instance refuses to write it
        data['a']['extra'] = 9
        with pytest.raises(ExpansionBudgetError):
            yml.dump(data, io.StringIO())
        # removing a value brings it back under the budget again
        del data['a']['extra']
        del data['a']['q']
        dump(yml, data)

    def test_dump_uses_modified_merge_source(self) -> None:
        yml = yaml_rt()
        data = yml.load(dedent("""
        base: &b
          token: secret
          keep: 1
        m:
          <<: *b
          z: 9
        """))
        yml.expansion_budget = 12
        assert dump(yml, data)
        # the inherited value is part of the effective data; editing the merge
        # source is reflected by the dump check (one new key + value)
        data['base']['newfield'] = 42
        assert data['m']['newfield'] == 42
        with pytest.raises(ExpansionBudgetError):
            yml.dump(data, io.StringIO())
        text = dump(yaml_rt(), data)
        assert 'newfield: 42' in text
        assert yaml_rt().load(text)['m']['newfield'] == 42

    def test_dump_uses_redacted_result(self) -> None:
        # the redact() entry point produces a fresh shared graph; dumping it
        # must apply the same rules to the redacted facts
        yml = yaml_rt()
        data = yml.load(dedent("""
        a: &base
          token: secret
          keep: 1
        b: *base
        """))
        out = yml.redact(data, [['a', 'token']])
        yml.expansion_budget = 10
        assert dump(yml, out)
        out['b']['added'] = 1
        with pytest.raises(ExpansionBudgetError):
            yml.dump(out, io.StringIO())


# ---------------------------------------------------------------------------
# 6. output stream is not written on a failing dump
# ---------------------------------------------------------------------------


class TestNoPartialOutput:
    def test_single_dump_failure_leaves_stream_empty(self) -> None:
        yml = yaml_rt()
        yml.expansion_budget = 2
        data = yaml_rt().load('a: 1\nb: 2\n')
        stream = RecordingStream()
        with pytest.raises(ExpansionBudgetError):
            yml.dump(data, stream)
        assert stream.writes == []

    def test_single_dump_success_writes(self) -> None:
        yml = yaml_rt()
        yml.expansion_budget = 2
        stream = RecordingStream()
        yml.dump(yaml_rt().load('a: 1\n'), stream)
        assert stream.text == 'a: 1\n'

    def test_dump_all_failure_leaves_stream_empty(self) -> None:
        yml = yaml_rt()
        yml.expansion_budget = 2
        good = yaml_rt().load('a: 1\n')
        bad = yaml_rt().load('x: 1\ny: 2\n')
        stream = RecordingStream()
        with pytest.raises(ExpansionBudgetError):
            yml.dump_all([good, bad], stream)
        # neither the good document nor a document separator was written
        assert stream.writes == []
        # reversed order: the failure is still found before any write
        stream = RecordingStream()
        with pytest.raises(ExpansionBudgetError):
            yml.dump_all([bad, good], stream)
        assert stream.writes == []

    def test_generator_documents_failure_leaves_stream_empty(self) -> None:
        yml = yaml_rt()
        yml.expansion_budget = 2
        good = yaml_rt().load('a: 1\n')
        bad = yaml_rt().load('x: 1\ny: 2\n')

        def docs() -> object:
            yield good
            yield bad

        stream = RecordingStream()
        with pytest.raises(ExpansionBudgetError):
            yml.dump_all(docs(), stream)
        assert stream.writes == []

    def test_stream_without_backtracking_is_supported(self) -> None:
        # RecordingStream has no seek/tell/getvalue; nothing relies on them
        yml = yaml_rt()
        yml.expansion_budget = 1000
        stream = RecordingStream()
        data = yml.load('a: &x\n  p: 1\nb: *x\n')
        yml.dump(data, stream)
        assert 'p: 1' in stream.text and 'b: *x' in stream.text

    def test_existing_partial_output_of_caller_untouched(self) -> None:
        yml = yaml_rt()
        yml.expansion_budget = 2
        bad = yaml_rt().load('x: 1\ny: 2\n')
        stream = RecordingStream()
        stream.write('previous: content\n')
        with pytest.raises(ExpansionBudgetError):
            yml.dump(bad, stream)
        assert stream.text == 'previous: content\n'


# ---------------------------------------------------------------------------
# 7. per document budgets, reuse of the YAML instance after a failure
# ---------------------------------------------------------------------------


class TestPerDocument:
    two_docs = dedent("""\
    ---
    small: 1
    ---
    a: &x
      p: 1
      q: 2
    b: *x
    """)

    def test_load_all_first_doc_does_not_affect_second(self) -> None:
        yml = yaml_rt()
        yml.expansion_budget = 10
        docs = list(yml.load_all(self.two_docs))
        assert docs[0] == {'small': 1}
        assert docs[1]['b']['q'] == 2

    def test_load_all_budget_is_per_document(self) -> None:
        yml = yaml_rt()
        yml.expansion_budget = 9
        with pytest.raises(ExpansionBudgetError):
            list(yml.load_all(self.two_docs))
        # same instance: the first document alone is fine
        first = yml.load('---\nsmall: 1\n')
        assert first == {'small': 1}

    def test_failure_then_good_document_same_instance(self) -> None:
        yml = yaml_rt()
        yml.expansion_budget = 2
        with pytest.raises(ExpansionBudgetError):
            yml.load('a: 1\nb: 2\n')
        assert yml.load('a: 1\n') == {'a': 1}

    def test_dump_all_second_doc_budget(self) -> None:
        yml = yaml_rt()
        yml.expansion_budget = 9
        small = yaml_rt().load('small: 1\n')
        big = yaml_rt().load('a: &x\n  p: 1\n  q: 2\nb: *x\n')
        stream = RecordingStream()
        with pytest.raises(ExpansionBudgetError):
            yml.dump_all([small, big], stream)
        assert stream.writes == []
        # the small document dumps fine on the same instance afterwards
        stream = RecordingStream()
        yml.dump_all([small], stream)
        assert stream.text == 'small: 1\n'

    def test_dump_all_all_documents_under_budget(self) -> None:
        yml = yaml_rt()
        yml.expansion_budget = 10
        small = yaml_rt().load('small: 1\n')
        big = yaml_rt().load('a: &x\n  p: 1\n  q: 2\nb: *x\n')
        stream = RecordingStream()
        yml.dump_all([small, big], stream)
        assert stream.text.count('---') >= 1
        assert 'q: 2' in stream.text


# ---------------------------------------------------------------------------
# 8. build_layered_shared_graph(30): L=1000 rejects, bounded memory
# ---------------------------------------------------------------------------


class TestLayeredSharedGraph:
    """reuses the fixtures from test_redact.build_layered_shared_graph"""

    def _graph(self) -> CommentedMap:
        from test_redact import build_layered_shared_graph  # type: ignore

        return build_layered_shared_graph(30)

    def test_exact_count_is_huge(self) -> None:
        graph = self._graph()
        # about 2**31 scalars, not 2**30 copies of the leaf only
        assert count_expansion(graph) > 2 ** 30

    def test_load_rejects_with_budget_1000(self) -> None:
        graph = self._graph()
        text = dump(yaml_rt(), graph)
        yml = yaml_rt()
        yml.expansion_budget = 1000
        with pytest.raises(ExpansionBudgetError) as excinfo:
            yml.load(text)
        assert excinfo.value.where == 'load'

    def test_dump_rejects_with_budget_1000(self) -> None:
        graph = self._graph()
        yml = yaml_rt()
        yml.expansion_budget = 1000
        stream = RecordingStream()
        with pytest.raises(ExpansionBudgetError) as excinfo:
            yml.dump(graph, stream)
        assert excinfo.value.where == 'dump'
        assert stream.writes == []

    def test_check_does_not_expand_the_tree(self) -> None:
        import tracemalloc

        graph = self._graph()
        text = dump(yaml_rt(), graph)
        yml = yaml_rt()
        yml.expansion_budget = 1000

        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            base = tracemalloc.get_traced_memory()[0]
            with pytest.raises(ExpansionBudgetError):
                yml.dump(graph, io.StringIO())
            _, dump_peak = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()
            base2 = tracemalloc.get_traced_memory()[0]
            with pytest.raises(ExpansionBudgetError):
                yml.load(text)
            _, load_peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        # both paths' per-request peak increase stays well under 32 MiB
        assert dump_peak - base <= 32 * 1024 * 1024
        assert load_peak - base2 <= 32 * 1024 * 1024

    def test_under_budget_layered_graph_dumps(self) -> None:
        from test_redact import build_layered_shared_graph

        graph = build_layered_shared_graph(3)
        yml = yaml_rt()
        yml.expansion_budget = count_expansion(graph)
        text = dump(yml, graph)
        assert text.count('mark:') <= 8


# ---------------------------------------------------------------------------
# extra edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_and_null_documents(self) -> None:
        for src in ('', '---\n', '# only comment\n', 'null\n', '~\n'):
            yml = yaml_rt()
            yml.expansion_budget = 1
            assert yml.load(src) is None

    def test_dumping_none(self) -> None:
        yml = yaml_rt()
        yml.expansion_budget = 1
        stream = RecordingStream()
        yml.dump(None, stream)
        assert 'null' in stream.text

    def test_budget_can_be_disabled_again(self) -> None:
        yml = yaml_rt()
        yml.expansion_budget = 1
        with pytest.raises(ExpansionBudgetError):
            yml.dump(yaml_rt().load('a: 1\nb: 2\n'), io.StringIO())
        yml.expansion_budget = None
        text = dump(yml, yaml_rt().load('a: 1\nb: 2\n'))
        assert 'b: 2' in text

    def test_set_members_count(self) -> None:
        yml = yaml_rt()
        data = yml.load('!!set\n? a\n? b\n')
        assert count_expansion(data) == 2

    def test_nested_mapping_key_counts(self) -> None:
        yml = yaml_rt()
        data = yml.load('? {a: 1}\n: value\n')
        # the nested key map has one key + one value, plus the outer value
        assert count_expansion(data) == 3

    @pytest.mark.parametrize('typ', ['rt', 'safe', 'base'])
    def test_all_pure_loaders_and_dumpers(self, typ: str) -> None:
        src = 'a: &x\n  p: 1\nb: *x\nc: *x\n'
        yml = ruamel.yaml.YAML(typ=typ, pure=True)
        yml.expansion_budget = 3
        with pytest.raises(ExpansionBudgetError):
            yml.load(src)
        other = ruamel.yaml.YAML(typ=typ, pure=True)
        data = other.load(src)
        stream = RecordingStream()
        with pytest.raises(ExpansionBudgetError):
            yml.dump(data, stream)
        assert stream.writes == []

    def test_full_typ_dump_only(self) -> None:
        yml = ruamel.yaml.YAML(typ='full')
        yml.expansion_budget = 1
        with pytest.raises(ExpansionBudgetError):
            yml.dump({'x': 1}, io.StringIO())

    def test_alternating_load_and_dump(self) -> None:
        yml = yaml_rt()
        yml.expansion_budget = 2
        for _ in range(3):
            assert yml.load('k: 1\n') == {'k': 1}
            stream = RecordingStream()
            yml.dump(yaml_rt().load('k: 1\n'), stream)
            assert stream.text == 'k: 1\n'

    def test_xdump_all_is_checked_before_writing(self) -> None:
        yml = yaml_rt()
        yml.expansion_budget = 2
        stream = RecordingStream()
        with pytest.raises(ExpansionBudgetError):
            yml.Xdump_all([yaml_rt().load('x: 1\ny: 2\n')], stream)
        assert stream.writes == []
        yml.expansion_budget = 2
        stream = RecordingStream()
        yml.Xdump_all([yaml_rt().load('x: 1\n')], stream)
        assert stream.text == 'x: 1\n'
