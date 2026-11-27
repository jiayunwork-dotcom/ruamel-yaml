
from __future__ import annotations

"""
path based redaction of round trip (typ='rt') YAML data.

``YAML(typ='rt').redact(data, paths)`` returns a fully independent copy of the
round trip object graph in which the values at the given *occurrence* paths are
replaced by the string ``[REDACTED]``.

A path is a non-empty sequence of steps: strings address mapping keys (including
keys inherited through a ``<<`` merge), non-negative integers address sequence
indices.  Paths identify *occurrences* in the traversal tree, not objects: when
the same anchored object is referenced more than once, redacting through one
reference must not affect the other references.  Sharing that is not touched by
a redaction path is preserved (including cycles), so the output keeps using
aliases instead of expanding exponentially.

During the occurrence walk every object carries the set of *relative* redaction
suffixes that still apply below it (a frozenset of path tuples; the empty tuple
means the object itself is selected).  Two occurrences of one source object that
end up with the same relative set share one clone -- this is what keeps cycles
and untouched aliases intact -- while different sets force a split, so an anchor
can never carry a redaction to an occurrence that was not selected.

The input graph (values, anchors, merge relations, comments) is never modified.
"""

import copy

from ruamel.yaml.anchor import Anchor
from ruamel.yaml.comments import (
    CommentedKeyMap,
    CommentedKeySeq,
    CommentedMap,
    CommentedSeq,
    CommentedSet,
    TaggedScalar,
    comment_attrib,
    format_attrib,
    line_col_attrib,
    merge_attrib,
)
from ruamel.yaml.scalarbool import ScalarBoolean
from ruamel.yaml.scalarfloat import ScalarFloat
from ruamel.yaml.scalarint import ScalarInt
from ruamel.yaml.scalarstring import ScalarString
from ruamel.yaml.tag import Tag

if False:  # MYPY
    from typing import Any, Dict, Tuple, Set, Iterable, FrozenSet  # NOQA

REDACTED = '[REDACTED]'

# a redaction set is a frozenset of relative suffix tuples; the empty tuple
# selects the object reached at that point.
if False:  # MYPY
    RedSet = FrozenSet[Tuple[Any, ...]]
_NO_RED: Any = frozenset()
_SELF: Tuple[Any, ...] = ()


class RedactionError(ValueError):
    """raised when a redaction path cannot be resolved on the input graph"""


# ---------------------------------------------------------------------------
# path validation and normalisation, all against the *original* graph
# ---------------------------------------------------------------------------


def _validate_path(data: Any, path: Any) -> None:
    if not isinstance(path, (list, tuple)) or len(path) == 0:
        raise RedactionError(
            f'redaction path must be a non-empty list/tuple of steps, got {path!r}',
        )
    node = data
    for step in path:
        if isinstance(step, bool) or not isinstance(step, (str, int)):
            raise RedactionError(
                f'redaction path steps must be str keys or non-negative int '
                f'indices, got {step!r} in {list(path)!r}',
            )
        if isinstance(step, int) and step < 0:
            raise RedactionError(
                f'redaction path sequence indices must be non-negative, '
                f'got {step} in {list(path)!r}',
            )
        if isinstance(node, (CommentedMap, dict)):
            # __contains__ on a CommentedMap also honours inherited merge keys,
            # matching __getitem__
            if step not in node:
                raise RedactionError(
                    f'mapping has no key {step!r} while resolving {list(path)!r}',
                )
            node = node[step]
        elif isinstance(node, (CommentedSeq, list)):
            if not isinstance(step, int):
                raise RedactionError(
                    f'expected an integer index into a sequence, got {step!r} '
                    f'while resolving {list(path)!r}',
                )
            if step >= len(node):
                raise RedactionError(
                    f'sequence index {step} out of range (len {len(node)}) '
                    f'while resolving {list(path)!r}',
                )
            node = node[step]
        else:
            # ran into a scalar or another non traversable object
            raise RedactionError(
                f'cannot descend into {step!r}: value at that point is a '
                f'{type(node).__name__} while resolving {list(path)!r}',
            )


def _normalise_paths(paths: Any) -> RedSet:
    """return the set of root relative selected paths, parents masking children

    duplicate paths and order have no effect; a selected parent replaces any
    selection below it.
    """
    selected: Set[Tuple[Any, ...]] = set()
    for path in paths:
        t = tuple(path)
        # drop if an ancestor is already selected
        if any(t[:i] in selected for i in range(1, len(t))):
            continue
        # drop any strict descendants of this newly selected path
        selected = {p for p in selected if p[: len(t)] != t}
        selected.add(t)
    return frozenset(selected)


def _sub_red(red: RedSet, step: Any) -> RedSet:
    """the redaction set that applies one *step* below the current object"""
    out: Set[Tuple[Any, ...]] = set()
    for suffix in red:
        if suffix and suffix[0] == step:
            out.add(suffix[1:])
    return frozenset(out)


# ---------------------------------------------------------------------------
# occurrence aware cloner
# ---------------------------------------------------------------------------


class _Redactor:
    def __init__(self, root: Any, root_red: RedSet) -> None:
        self._root = root
        self._root_red = root_red
        # (id(source object), relative redaction set) -> clone
        # the same source object reached with the same remaining redaction
        # set yields the same clone, preserving untouched sharing and cycles;
        # a different redaction set forces a split so an anchor can never carry
        # a redaction to an occurrence that was not selected.
        self._memo: Dict[Tuple[int, RedSet], Any] = {}
        # source object ids currently being cloned on the recursion stack,
        # mapped to their (not yet finished) clone: used to recognise a back
        # edge that closes a cycle without mistaking separate occurrences for
        # one.
        self._active: Dict[int, Any] = {}
        # per source object the redaction set it must at least carry.  Grows
        # only when a finite path loops back (a back edge) and reaches an
        # already active frame with extra suffixes; a growth triggers a clean
        # rerun.  Sets only grow within the finite suffix universe, so this
        # reaches a fixed point.
        self._required: Dict[int, RedSet] = {}
        self._grew = False

    def redact(self) -> Any:
        self._required = {id(self._root): self._root_red}
        while True:
            self._memo = {}
            self._active = {}
            self._grew = False
            result = self._clone(self._root, self._root_red)
            if not self._grew:
                self._dedupe_anchors(result)
                return result

    # -- the core ----------------------------------------------------------

    def _clone(self, obj: Any, red: RedSet) -> Any:
        if _SELF in red:
            # parent path wins: the whole selected value is replaced, no descent
            return REDACTED
        if isinstance(obj, CommentedMap):
            return self._clone_container(obj, red, self._build_map)
        if isinstance(obj, CommentedSeq):
            return self._clone_container(obj, red, self._build_seq)
        if isinstance(obj, CommentedSet):
            return self._clone_container(obj, red, self._build_set)
        if isinstance(obj, CommentedKeyMap):
            return self._clone_container(obj, red, self._build_key_map)
        if isinstance(obj, CommentedKeySeq):
            return self._clone_key_seq(obj, red)
        if isinstance(obj, dict):
            # plain mapping made programmatically: promote to a round trip map
            promoted = CommentedMap(obj)
            return self._clone_container(promoted, red, self._build_map)
        if isinstance(obj, list):
            promoted = CommentedSeq(obj)
            return self._clone_container(promoted, red, self._build_seq)
        if isinstance(obj, set):
            promoted = CommentedSet(obj)
            return self._clone_container(promoted, red, self._build_set)
        if isinstance(obj, TaggedScalar):
            return self._clone_tagged_scalar(obj)
        if isinstance(obj, (ScalarString, ScalarInt, ScalarFloat, ScalarBoolean)):
            # scalar subclasses carry style/anchor/comment attributes: clone them
            return self._clone_scalar(obj)
        if isinstance(obj, (str, bytes, bool, int, float, tuple, frozenset, type(None))):
            # immutable leaf values can safely be shared with the source
            return obj
        # unknown mutable object: best effort independent copy
        return copy.deepcopy(obj)

    def _clone_container(self, obj: Any, red: RedSet, builder: Any) -> Any:
        obj_id = id(obj)
        active_clone = self._active.get(obj_id)
        if active_clone is not None:
            # back edge: this occurrence loops back to an object whose clone is
            # still being built on the same traversal path.  The cycle can only
            # be represented by one shared clone, so any suffixes reached via
            # this loop belong to that clone.  If they add redactions its frame
            # did not yet know about, grow the requirement and rerun the whole
            # pass so the children are rebuilt consistently.
            required = self._required.get(obj_id, _NO_RED)
            if not red <= required:
                self._required[obj_id] = required | red
                self._grew = True
            return active_clone
        # the effective set for this occurrence is its own suffix set union the
        # back-edge requirements accumulated for the object (empty unless a loop
        # reaches it).  Different effective sets stay distinct clones.
        effective = red | self._required.get(obj_id, _NO_RED)
        key = (obj_id, effective)
        cached = self._memo.get(key)
        if cached is not None:
            # an already finished clone with the exact same effective set:
            # reuse it, preserving ordinary sharing
            return cached
        new = self._new_empty(obj)
        self._memo[key] = new
        self._active[obj_id] = new
        try:
            builder(obj, effective, new)
        finally:
            self._active.pop(obj_id, None)
        return new

    def _new_empty(self, obj: Any) -> Any:
        if isinstance(obj, CommentedMap):
            return CommentedMap()
        if isinstance(obj, CommentedSeq):
            return CommentedSeq()
        if isinstance(obj, CommentedSet):
            return CommentedSet()
        if isinstance(obj, CommentedKeyMap):
            return CommentedKeyMap()
        raise TypeError(f'cannot create empty clone of {type(obj).__name__}')
    # -- container builders (obj already has a fresh clone registered) ------

    def _build_map(self, obj: CommentedMap, red: RedSet, new: CommentedMap) -> None:
        own_items = list(obj.non_merged_items())
        own_map = {k: v for k, v in own_items}
        merge_entries = list(getattr(obj, merge_attrib, []))
        inherited_keys = self._inherited_redaction_keys(red, own_map, obj)

        # own values, cloned with their remaining redaction sub trees
        for key, value in own_items:
            new_key = self._clone_key(key)
            new[new_key] = self._clone(value, _sub_red(red, key))

        # redacted inherited keys become explicit local overrides, inserted in
        # the original effective key position so ordering/semantics survive
        for key in obj.keys():
            if key in own_map or key not in inherited_keys:
                continue
            new_key = self._clone_key(key)
            sub = _sub_red(red, key)
            if _SELF in sub:
                new[new_key] = REDACTED
            else:
                new[new_key] = self._clone(obj[key], sub)

        # rebuild the merge relation on the new graph; sources are cloned with
        # an *empty* red set so redacting an inherited field never mutates the
        # shared source
        new_entries = []
        for pos, source in merge_entries:
            new_source = self._clone(source, _NO_RED)
            new_entries.append((pos, new_source))
        if new_entries:
            new.add_yaml_merge(new_entries)

        self._copy_attributes(obj, new)

    def _build_seq(self, obj: CommentedSeq, red: RedSet, new: CommentedSeq) -> None:
        for idx, value in enumerate(obj):
            new.append(self._clone(value, _sub_red(red, idx)))
        self._copy_attributes(obj, new)

    def _build_set(self, obj: CommentedSet, red: RedSet, new: CommentedSet) -> None:
        # sets can only hold hashable (scalar / immutable) members
        for value in obj:
            new.add(self._clone(value, _NO_RED))
        self._copy_attributes(obj, new)

    def _build_key_map(self, obj: CommentedKeyMap, red: RedSet, new: Any) -> None:
        seeded = CommentedMap()
        for k, v in obj._od.items():
            seeded[self._clone_key(k)] = self._clone(v, _NO_RED)
        new._od = seeded
        self._copy_attributes(obj, new)

    def _clone_key_seq(self, obj: CommentedKeySeq, red: RedSet) -> Any:
        # an immutable tuple wrapper; build a fresh instance (rare, only used as
        # a mapping key, so redaction below a key never applies)
        key = (id(obj), red)
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        new = CommentedKeySeq(tuple(self._clone(v, _NO_RED) for v in obj))
        self._memo[key] = new
        self._copy_attributes(obj, new)
        return new

    def _inherited_redaction_keys(
        self, red: RedSet, own_map: Dict[Any, Any], obj: CommentedMap,
    ) -> Set[Any]:
        result: Set[Any] = set()
        for suffix in red:
            if not suffix:
                continue
            key = suffix[0]
            if key in own_map:
                continue
            if key in obj:  # inherited through one of the merge sources
                result.add(key)
        return result

    def _clone_key(self, key: Any) -> Any:
        if isinstance(key, (CommentedKeySeq, CommentedKeyMap)):
            return self._clone(key, _NO_RED)
        return key

    def _clone_scalar(self, obj: Any) -> Any:
        key = (id(obj), _NO_RED)
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        clone = copy.copy(obj)
        self._memo[key] = clone
        # deep copy the YAML attributes so edits cannot reach the input
        for attr in (Anchor.attrib, Tag.attrib):
            if hasattr(obj, attr):
                setattr(clone, attr, copy.deepcopy(getattr(obj, attr)))
        for extra in ('_width', '_underscore', 'comment'):
            if hasattr(obj, extra):
                setattr(clone, extra, copy.deepcopy(getattr(obj, extra)))
        return clone

    def _clone_tagged_scalar(self, obj: Any) -> Any:
        key = (id(obj), _NO_RED)
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        clone = copy.copy(obj)
        self._memo[key] = clone
        self._copy_attributes(obj, clone)
        return clone

    # -- attribute plumbing ------------------------------------------------

    def _copy_attributes(self, source: Any, target: Any) -> None:
        for attr in (
            comment_attrib,
            format_attrib,
            line_col_attrib,
            Anchor.attrib,
            Tag.attrib,
        ):
            if hasattr(source, attr):
                setattr(target, attr, copy.deepcopy(getattr(source, attr)))

    # -- anchor de-duplication ---------------------------------------------

    def _dedupe_anchors(self, root: Any) -> None:
        """an explicit anchor name may only appear once in the output.

        Cloning keeps anchor names for clarity, but when an anchored source is
        split into several clones (or two distinct sources happen to share a
        name) only the first occurrence keeps the name; the others rely on
        serializer generated anchors.  Repeated objects still alias correctly
        because the serializer tracks object identity.
        """
        seen: Dict[Any, None] = {}
        visited: Set[int] = set()
        self._walk_anchors(root, seen, visited)

    def _walk_anchors(self, obj: Any, seen: Dict[Any, None], visited: Set[int]) -> None:
        scalar_types = (ScalarString, ScalarInt, ScalarFloat, ScalarBoolean, TaggedScalar)
        if isinstance(obj, scalar_types):
            self._clear_duplicate_anchor(obj, seen)
            return
        if not isinstance(obj, (CommentedMap, CommentedSeq, CommentedSet,
                                CommentedKeyMap, CommentedKeySeq)):
            return
        if id(obj) in visited:
            return
        visited.add(id(obj))
        self._clear_duplicate_anchor(obj, seen)
        if isinstance(obj, CommentedMap):
            for key, value in obj.non_merged_items():
                self._walk_anchors(key, seen, visited)
                self._walk_anchors(value, seen, visited)
            for _pos, source in getattr(obj, merge_attrib, []):
                self._walk_anchors(source, seen, visited)
        elif isinstance(obj, CommentedSeq):
            for value in obj:
                self._walk_anchors(value, seen, visited)
        elif isinstance(obj, CommentedSet):
            for value in obj:
                self._walk_anchors(value, seen, visited)
        elif isinstance(obj, CommentedKeyMap):
            for key, value in obj._od.items():
                self._walk_anchors(key, seen, visited)
                self._walk_anchors(value, seen, visited)
        elif isinstance(obj, CommentedKeySeq):
            for value in obj:
                self._walk_anchors(value, seen, visited)

    def _clear_duplicate_anchor(self, obj: Any, seen: Dict[Any, None]) -> None:
        anchor = getattr(obj, Anchor.attrib, None)
        if anchor is None or anchor.value is None:
            return
        if anchor.value in seen:
            anchor.value = None
            anchor.always_dump = False
        else:
            seen[anchor.value] = None


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def redact(data: Any, paths: Iterable[Any]) -> Any:
    """return an independent redacted copy of a round trip object graph.

    See the module doc string for the path semantics.  Raises
    :class:`ValueError` (``RedactionError``) if any path cannot be resolved.
    No partial result is ever returned: every path is validated against the
    untouched input graph before cloning starts.
    """
    path_list = list(paths)
    if not path_list:
        raise RedactionError('redact() needs at least one non-empty path')
    for path in path_list:
        _validate_path(data, path)
    root_red = _normalise_paths(path_list)
    return _Redactor(data, root_red).redact()
