"""Load a serialized mjlab env.yaml / agent.yaml without importing mjlab.

mjlab dumps its training cfg with ``yaml.dump`` (default Dumper), which tags
non-trivial leaves (tuples, enums, function refs) as ``!!python/...``. None of
the stock loaders can read that here:

* ``safe_load``   -- rejects every ``!!python/...`` tag (ConstructorError).
* ``full_load``   -- blocks ``!!python/object/apply`` (enums), fails even with
                     mjlab installed.
* ``unsafe_load`` -- needs the exact mjlab/hand_mjlab code that produced the dump.
                     That code is absent in the policy container, and it drifts
                     over time (e.g. a removed ``OriginType`` enum breaks the load).

This loader subclasses ``SafeLoader`` and degrades every ``!!python/...`` tag to
its plain dict/list/scalar content -- no imports, no object construction. Plain
values (the scalars we actually read, e.g. ``arm_vel_scale: 0.7``) are exact;
tagged leaves we don't need become their underlying structure.
"""

from pathlib import Path
from typing import Any

import yaml


class _IgnorePyTagLoader(yaml.SafeLoader):
    """SafeLoader that maps any ``!!python/...`` tag to plain dict/list/scalar."""


def _ignore_py_tag(loader: yaml.Loader, tag_suffix: str, node: yaml.Node) -> Any:
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


# One rule catches every python tag: !!python/name, /object, /object/apply, /tuple
# all expand to the prefix below.
_IgnorePyTagLoader.add_multi_constructor("tag:yaml.org,2002:python/", _ignore_py_tag)


def load_mjlab_yaml(path: "str | Path") -> dict:
    """Load a serialized mjlab YAML (env.yaml / agent.yaml) as nested plain dicts.

    Args:
        path: Path to the dumped YAML file.

    Returns:
        (dict): Config as nested plain dicts/lists/scalars. ``!!python/...`` tagged
            leaves are degraded (tuple->list, name/object->their content); plain
            scalars (e.g. vel_scale, ema) are exact. No mjlab import required.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_IgnorePyTagLoader)
    if not isinstance(data, dict):
        raise ValueError(
            f"mjlab YAML root must be a dict, got {type(data).__name__}: {path}"
        )
    return data
