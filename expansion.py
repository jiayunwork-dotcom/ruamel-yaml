from __future__ import annotations

"""
optional expansion budget for ``YAML`` instances.

A small YAML document can use anchors/aliases and merge keys (``<<``) to
describe a *tree* that is exponentially larger than the document itself.  The
serializer deliberately keeps using aliases, but a consumer that walks the
"effective" data (merges resolved, every occurrence visited) can be made to
traverse a very large tree.  This module provides a cheap, opt-in guard that
counts the scalar occurrences of that effective tree and refuses documents
whose count exceeds a caller supplied budget.

The budget is set on the ``YAML`` instance with ``yaml.expansion_budget = L``
(a positive integer; ``None``, the default, disables the check).  The same
check runs on the two existing data paths:

* after the constructor has built a document (``load``/``load_all``), and
* before the representer turns a document into nodes (``dump``/``dump_all``).

Both use :func:`check_expansion` on the effective view of the very same object
graph with the same rules, so a document accepted on load is accepted on dump
and vice versa as long as the graph was not mutated in between.  The check is
performed on the data at hand every time; nothing about a previous load or
dump is cached, so edits to alias-shared objects or to merge sources are seen
immediately.

Counting rules
--------------
* every scalar occurrence counts one, whether it is a mapping key or a mapping
  value, a sequence member, or a scalar in a nested key;
* a container itself counts zero; an empty container therefore counts zero;
* merge keys (``<<``) are not counted, only the effective keys/values are: the
  own keys of a map mask inherited ones, and when several merge sources define
  the same key the first source in the ``<<`` list wins (ruamel.yaml's existing
  merge priority).  Values masked by an own key are never visited;
* the same sub graph reached through two different effective paths is counted
  twice, because it is two occurrences in the effective tree;
* members of a set count like keys (their ``null`` placeholders do not count).

Cycles
------
If the effective keys/values or sequence members form a cycle the expansion is
infinite, so the document is refused with :class:`ExpansionCycleError`, even
when the budget is huge.  Cycles that only live in a merge branch which is
masked by an explicit value are *not* part of the effective tree and therefore
do not cause a false rejection.  A cyclic ``<<`` relation itself yields its
finite least fixed point (the union of the reachable own keys), it does not
recurse forever.

Design: representation and reuse trade-off
------------------------------------------
The budget is a plain positive ``int`` on the ``YAML`` instance.  For one
check the walker computes, for every distinct container object it meets,

    f(obj) = number of scalar occurrences in the expansion of one occurrence
             of obj (mapping keys and values, sequence/set members, with the
             effective merges resolved and every alias child expanded again)

and derives ``f(root)`` bottom up.  It keeps:

* ``f`` (an int per distinct container; clamped to ``budget + 1`` while a
  budget is enforced, so astronomically shared graphs never build huge ints),
* the resolved effective key/value pairs of each distinct mapping (merge
  resolution is memoised for the duration of the same check),
* a DFS colour per distinct container (white/grey/black) for cycle detection,
  and one stack frame (a child iterator plus an accumulator) per open
  container.

All of it is discarded when the check returns.  This makes the check
O(number of distinct containers and their effective edges) in time and memory
-- independent of how large the *expanded* tree would be, and it never
materialises the expansion.  Because the values aborted/checked are derived
from the current objects, loading and dumping always see the current facts.

Why the memoised ``f`` is safe, unlike a cache that survives the check:

* within a check the graph cannot be mutated, so ``f`` per object is exactly
  the occurrence count, no matter through which path the object is reached
  (an alias reached twice adds ``f`` twice, so "same sub graph, different
  effective paths" still counts twice);
* a cross-check cache would go stale as soon as an alias-shared object or a
  merge source is edited between a load and a dump -- exactly what round
  tripping allows -- and a stale low count would let an over-budget document
  through.  Recomputing per check is what guarantees "dump uses the modified
  facts".

The alternatives traded away: a plain occurrence walk without per-object
memoisation repeats work for shared sub graphs (and could visit an
exponentially large effective tree made only of, say, empty containers, which
cost no scalars); a purely lazy early-abort walk also cannot report the exact
total for a document that stays under budget without walking every occurrence.
The DAG count costs one dict entry per distinct container, which for real
documents is linear in the document size and far below the 32 MiB budget used
in the acceptance test.

The helpers are usable directly too: :func:`count_expansion` returns the exact
count without a budget (for tests/diagnostics; it does not clamp, so the int
may be huge but no tree is materialised) and :func:`check_expansion` enforces
a budget.
"""

from collections.abc import Mapping
from typing import Any, Iterator

from ruamel.yaml.comments import (
    CommentedMap,
    CommentedSeq,
    CommentedSet,
    merge_attrib,
)
from ruamel.yaml.error import ExpansionBudgetError, ExpansionCycleError

if False:  # MYPY
    from typing import Dict, Iterable, List, Optional, Tuple  # NOQA


_CONTAINER_TYPES = (Mapping, list, tuple, set, frozenset, CommentedSet, CommentedSeq)


def _is_container(obj: Any) -> bool:
    return isinstance(obj, _CONTAINER_TYPES)


# DFS colours
_WHITE = 0
_GREY = 1
_BLACK = 2


class _MergeFrame:
    """one mapping whose effective pairs are being computed"""

    __slots__ = 'mapping', 'pairs', 'seen', 'entries', 'index', 'pending'

    def __init__(self, mapping: CommentedMap) -> None:
        self.mapping = mapping
        self.pairs = list(mapping.non_merged_items())
        self.seen = {key for key, _ in self.pairs}
        self.entries = list(getattr(mapping, merge_attrib, []))
        self.index = 0
        self.pending: Any = None

    def add(self, pairs: Any) -> None:
        for key, value in pairs:
            if key in self.seen:
                continue
            self.seen.add(key)
            self.pairs.append((key, value))


class _ExpansionWalker:
    def __init__(self, budget: Any, where: str) -> None:
        self.budget = budget
        self.where = where
        self.used = 0
        # id(mapping) -> resolved effective pairs, valid for this check only
        self._resolved: Dict[int, List[Any]] = {}

    # -- effective mapping view -------------------------------------------

    def _resolve(self, root: Any) -> Any:
        """effective ``(key, value)`` pairs of *root*, merges resolved.

        Own keys come first and mask everything below them; merge sources are
        consulted in ``<<`` order and the first source that provides a key
        wins.  A source's own keys again mask its inherited ones.  A mapping
        that is already being resolved (a cyclic ``<<`` relation) contributes
        nothing extra: its own keys were already collected on its own frame,
        which gives the least fixed point instead of infinite recursion.

        Only pairs that end up in the result are returned, so a value below a
        masked key is never visited (and can never cause a false rejection).
        """
        rid = id(root)
        cached = self._resolved.get(rid)
        if cached is not None:
            return cached
        stack = [_MergeFrame(root)]
        on_path = {rid}
        while stack:
            frame = stack[-1]
            if frame.pending is not None:
                source = frame.pending
                frame.pending = None
                frame.add(self._resolved[id(source)])
                continue
            if frame.index < len(frame.entries):
                source = frame.entries[frame.index][1]
                frame.index += 1
                if not isinstance(source, CommentedMap):
                    if isinstance(source, Mapping):
                        frame.add(source.items())
                    continue
                sid = id(source)
                if sid in on_path:
                    # cyclic merge relation: its own keys already contribute
                    # to the fixed point, there is nothing new to inherit
                    continue
                resolved = self._resolved.get(sid)
                if resolved is not None:
                    frame.add(resolved)
                else:
                    on_path.add(sid)
                    stack.append(_MergeFrame(source))
                    frame.pending = source
            else:
                self._resolved[id(frame.mapping)] = frame.pairs
                on_path.discard(id(frame.mapping))
                stack.pop()
        return self._resolved[rid]

    def _children(self, obj: Any) -> Iterator[Any]:
        """yield the children (keys, then values, or members) of one container"""
        if isinstance(obj, CommentedMap):
            for key, value in self._resolve(obj):
                yield key
                yield value
        elif isinstance(obj, Mapping):
            # plain dicts and CommentedKeyMap (nested mapping keys)
            for key, value in obj.items():
                yield key
                yield value
        elif isinstance(obj, (CommentedSet, set, frozenset)):
            for member in obj:
                yield member
        elif isinstance(obj, (CommentedSeq, list, tuple)):
            for member in obj:
                yield member
        # CommentedKeySeq is a tuple and handled above

    # -- counting and cycle detection -------------------------------------

    def _add(self, acc: int, value: int) -> int:
        acc += value
        if acc > self.budget:
            # keep the accumulator small, the exact number is irrelevant
            return self.budget + 1
        return acc

    def run(self, data: Any) -> int:
        if not _is_container(data):
            self.used = 1
            if self.used > self.budget:
                raise ExpansionBudgetError(self.budget, self.used, self.where)
            return self.used

        # iterative DFS, computing f(obj) on the way out; colours detect
        # effective cycles on the object graph, which is exactly "the
        # effective keys/values/members form a cycle".
        root_id = id(data)
        color: Dict[int, int] = {root_id: _GREY}
        f: Dict[int, int] = {}
        # frames: [container, child iterator, accumulator, id]
        stack: List[Any] = [[data, self._children(data), 0, root_id]]
        while stack:
            frame = stack[-1]
            obj, children, _acc, obj_id = frame
            try:
                child = next(children)
            except StopIteration:
                f[obj_id] = frame[2]
                color[obj_id] = _BLACK
                stack.pop()
                if stack:
                    parent = stack[-1]
                    parent[2] = self._add(parent[2], frame[2])
                continue
            if _is_container(child):
                cid = id(child)
                cstate = color.get(cid, _WHITE)
                if cstate == _GREY:
                    raise ExpansionCycleError(
                        self.where,
                        'effective mapping keys/values or sequence members '
                        'form a cycle, the expansion is not finite',
                    )
                if cstate == _BLACK:
                    frame[2] = self._add(frame[2], f[cid])
                else:
                    color[cid] = _GREY
                    stack.append([child, self._children(child), 0, cid])
            else:
                frame[2] = self._add(frame[2], 1)
        self.used = f[root_id]
        if self.used > self.budget:
            raise ExpansionBudgetError(self.budget, self.used, self.where)
        return self.used


class _Unlimited:
    """comparison target that is never exceeded, for exact counting"""

    def __lt__(self, other: Any) -> bool:
        return False

    def __ge__(self, other: Any) -> bool:
        return True


_UNLIMITED = _Unlimited()


def _validate_budget(budget: Any) -> None:
    if budget is None:
        return
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError(
            f'expansion budget must be a positive integer or None, got {budget!r}',
        )


validate_budget = _validate_budget


def count_expansion(data: Any, where: str = 'count') -> int:
    """return the exact number of scalar occurrences of the effective tree.

    Raises :class:`ExpansionCycleError` for cyclic effective data.  No budget
    is applied, so this is intended for tests and diagnostics, not as a guard
    for untrusted input.
    """
    return _ExpansionWalker(_UNLIMITED, where).run(data)


def check_expansion(data: Any, budget: int, where: str = 'load') -> int:
    """check *data* against *budget* and return the used scalar occurrence count.

    Raises :class:`ExpansionBudgetError` when the count exceeds the budget, or
    :class:`ExpansionCycleError` if the effective data is cyclic.  ``where`` is
    only used in the error message (``'load'`` or ``'dump'``).
    """
    _validate_budget(budget)
    return _ExpansionWalker(budget, where).run(data)
