"""Parse OpenStudio SWIG Python wrappers into method signatures.

The Ruby bindings are compiled C with no introspectable source, and dir()/getattr()
on the live ``openstudio.model`` module recovers method *names* only — not parameter
names or return types. But SWIG also emits Python proxy files (``openstudio*.py``) that
carry the full class tree, parameter names, and return-type annotations. We read those
as text (never execute them) to recover signatures, then ``search_api`` decorates its
output with them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# SWIG internal classes to skip
SKIP_CLASSES = {"SwigPyIterator", "_SwigNonDynamicMeta"}

_CLASS_RE = re.compile(r"^class (\w+)\((\w+(?:\.\w+)?)\):")
_METHOD_RE = re.compile(r"^\s{4}def (\w+)\(([^)]*)\)(?:\s*->\s*([\w.]+))?\s*:")
_MODFUNC_RE = re.compile(r"^def (_\w+)\(([^)]*)\)\s*(?:->\s*([\w.]+))?\s*:")
_ASSIGN_RE = re.compile(r"^(?:\w+\.)*(\w+)\.(\w+)\s*=\s*(_\w+)\s*$")
_STATIC_RE = re.compile(r"^\s+@staticmethod")

_SKIP_METHODS = {"__init__", "__repr__", "__str__", "__del__"}

_PRIMITIVE_TYPES = {
    "str": "String",
    "String": "String",
    "bool": "Boolean",
    "int": "Integer",
    "float": "Float",
    "double": "Float",
}

# Module-level cache (built once per process). PLW0603 is allowed in mcp_server.
_cache: dict[str, dict[str, dict]] | None = None


@dataclass
class ParsedMethod:
    name: str
    params: list[str]
    return_type: str | None


@dataclass
class ParsedClass:
    name: str
    instance_methods: dict[str, ParsedMethod] = field(default_factory=dict)
    static_methods: dict[str, ParsedMethod] = field(default_factory=dict)


def _clean_params(raw: str) -> list[str]:
    """Strip self/defaults/annotations, lowercase a leading capital.

    SWIG renders overloaded methods (a setter with several C++ signatures) as
    ``(self, *args)``, losing the real names. We keep that as ``...`` rather than an
    empty list so the rendered signature signals "takes arguments, names unavailable"
    instead of falsely reading as a no-arg call.
    """
    params = []
    for raw_part in raw.split(","):
        part = raw_part.strip()
        if not part or part == "self":
            continue
        if part.startswith("*"):
            if "..." not in params:
                params.append("...")
            continue
        name = re.split(r"[=:]", part)[0].strip().replace("[", "").replace("]", "")
        if name and name[0].isupper():
            name = name[0].lower() + name[1:]
        if name:
            params.append(name)
    return params


def _last_segment(type_str: str | None) -> str | None:
    return type_str.split(".")[-1] if type_str else None


def _parse_module_level_types(path: Path) -> dict[str, dict[str, ParsedMethod]]:
    """Capture ``def _fn(self, x) -> Ret:`` and cross-file ``Class.method = _fn``."""
    func_info: dict[str, ParsedMethod] = {}
    class_methods: dict[str, dict[str, ParsedMethod]] = {}

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        func_match = _MODFUNC_RE.match(line)
        if func_match:
            func_info[func_match.group(1)] = ParsedMethod(
                name=func_match.group(1),
                params=_clean_params(func_match.group(2)),
                return_type=_last_segment(func_match.group(3)),
            )
            continue
        assign_match = _ASSIGN_RE.match(line)
        if assign_match:
            target_class, method_name, func_name = assign_match.groups()
            info = func_info.get(func_name)
            if info is None:
                continue
            class_methods.setdefault(target_class, {})[method_name] = ParsedMethod(
                name=method_name,
                params=info.params,
                return_type=info.return_type,
            )
    return class_methods


def _parse_python_file(path: Path) -> list[ParsedClass]:
    """Capture ``class X(Parent):`` and its 4-space-indented method definitions."""
    classes: list[ParsedClass] = []
    current: ParsedClass | None = None
    in_static = False

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        class_match = _CLASS_RE.match(line)
        if class_match:
            name, parent = class_match.group(1), class_match.group(2)
            # Skip SWIG internals / STL container wrapper types / Python-only lowercase
            # classes. *Vector/Optional*/*Set/*Map with parent `object` are SWIG wrappers
            # around std::vector/optional/set/map — collection plumbing, not domain API,
            # and their Python dict/set helper methods (add/iteritems/has_key/...) have no
            # Ruby equivalent. Real domain classes inherit a model parent, never `object`,
            # so this never catches them. current=None so a skipped body can't leak onto
            # the prior class.
            if (
                name in SKIP_CLASSES
                or (name.endswith(("Vector", "Set", "Map")) and parent == "object")
                or (name.startswith("Optional") and parent == "object")
                or name[0].islower()
            ):
                current = None
                in_static = False
                continue
            current = ParsedClass(name=name)
            classes.append(current)
            in_static = False
            continue

        if current is None:
            continue

        if _STATIC_RE.match(line):
            in_static = True
            continue

        method_match = _METHOD_RE.match(line)
        if not method_match:
            continue

        method_name = method_match.group(1)
        if method_name.startswith("_") or method_name in _SKIP_METHODS:
            in_static = False
            continue

        method = ParsedMethod(
            name=method_name,
            params=_clean_params(method_match.group(2)),
            return_type=_last_segment(method_match.group(3)),
        )
        if in_static:
            current.static_methods[method_name] = method
        else:
            current.instance_methods[method_name] = method
        in_static = False

    return classes


def _resolve_return_type(type_str: str | None, all_class_names: set[str]) -> str:
    """Render a Python annotation as a Ruby-style return type string."""
    if not type_str:
        return "void"
    type_name = type_str.split(".")[-1]

    optional = re.fullmatch(r"Optional(\w+)", type_name)
    if optional:
        inner = optional.group(1)
        return f"{inner}, nil" if inner in all_class_names else "Object, nil"

    vector = re.fullmatch(r"(\w+)Vector", type_name)
    if vector:
        inner = vector.group(1)
        return f"Array<{inner}>" if inner in all_class_names else "Array"

    if type_name in _PRIMITIVE_TYPES:
        return _PRIMITIVE_TYPES[type_name]

    return type_name if type_name in all_class_names else "Object"


def _infer_return_type(method_name: str, all_class_names: set[str]) -> str:
    """Guess a return type from the method name when no annotation exists."""
    if re.match(r"(?:is|has)[A-Z]", method_name):
        return "Boolean"
    if re.match(r"set[A-Z]", method_name):
        return "Boolean"
    if re.match(r"(?:reset|remove|delete|autocalculate|autosize)[A-Z]", method_name):
        return "void"
    to_match = re.match(r"to_(\w+)$", method_name)
    if to_match:
        target = to_match.group(1)
        return target if target in all_class_names else "Object"
    if method_name == "name" or re.search(r"(?:Name|Type|String)$", method_name):
        return "String"
    if re.search(
        r"(?:Value|Number|Count|Size|Index|Area|Volume|Height|Width|Length|Ratio|Efficiency|COP)$",
        method_name,
    ):
        return "Float"
    return "Object"


def _wrapper_files(wrapper_dir: Path) -> list[Path]:
    return sorted(p for p in wrapper_dir.glob("openstudio*.py") if p.stem != "openstudio")


def _build(wrapper_dir: Path) -> dict[str, dict[str, dict]]:
    """Parse all wrapper files into ``{class: {method: {params, returns, static}}}``."""
    files = _wrapper_files(wrapper_dir)

    # Pass 1: cross-file module-level method assignments (carry return types).
    module_types: dict[str, dict[str, ParsedMethod]] = {}
    for path in files:
        for class_name, methods in _parse_module_level_types(path).items():
            module_types.setdefault(class_name, {}).update(methods)

    # Pass 2: class definitions. Keep the first occurrence of a class name.
    parsed: dict[str, ParsedClass] = {}
    for path in files:
        for cls in _parse_python_file(path):
            parsed.setdefault(cls.name, cls)

    all_class_names = set(parsed)

    # Apply cross-file methods: fill missing return types, add unseen methods.
    for class_name, methods in module_types.items():
        cls = parsed.get(class_name)
        if cls is None:
            continue
        for method_name, patched in methods.items():
            existing = cls.instance_methods.get(method_name)
            if existing is None:
                cls.instance_methods[method_name] = patched
            elif existing.return_type is None:
                existing.return_type = patched.return_type

    # Render to the public shape.
    result: dict[str, dict[str, dict]] = {}
    for class_name, cls in parsed.items():
        rendered: dict[str, dict] = {}
        for method in cls.static_methods.values():
            rendered[method.name] = _render(method, all_class_names, static=True)
        for method in cls.instance_methods.values():
            rendered[method.name] = _render(method, all_class_names, static=False)
        result[class_name] = rendered
    return result


def _render(method: ParsedMethod, all_class_names: set[str], *, static: bool) -> dict:
    returns = (
        _resolve_return_type(method.return_type, all_class_names)
        if method.return_type
        else _infer_return_type(method.name, all_class_names)
    )
    return {"params": method.params, "returns": returns, "static": static}


def _locate_wrapper_dir() -> Path:
    import openstudio

    return Path(openstudio.__file__).parent


def signatures() -> dict[str, dict[str, dict]]:
    """Return ``{class_name: {method_name: {params, returns, static}}}``.

    Locates the wrapper files via the installed ``openstudio`` package
    (``openstudio.__file__`` — the SDK shipped inside the MCP image) and caches the
    parse for the process. Called by ``search_api`` to decorate methods with their
    parameter names and return types.
    """
    global _cache
    if _cache is None:
        _cache = _build(_locate_wrapper_dir())
    return _cache
