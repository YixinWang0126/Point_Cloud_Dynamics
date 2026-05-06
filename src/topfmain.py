"""Compatibility wrapper around the vendored ``tools/topf`` module.

This keeps notebook imports stable while avoiding direct edits inside the git
submodule. It patches the author-specific Julia paths at import time and then
re-exports the original module API.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_TOPF_SRC = REPO_ROOT / "tools" / "topf" / "src"
VENDOR_TOPF_MAIN = VENDOR_TOPF_SRC / "topf" / "topfmain.py"

ABS_PROJECT = (
    '"--project=/Users/yxzuji/Downloads/工程/TpOT/tpot/tools/topf/src/topf/JuliaEnvironment"'
)
ABS_SCRIPT = (
    '"/Users/yxzuji/Downloads/工程/TpOT/tpot/tools/topf/src/topf/HomologyGeneratorsMultiD.jl"'
)
REL_PROJECT = '"--project=.topf/JuliaEnvironment"'
REL_SCRIPT = '".topf/HomologyGeneratorsMultiD.jl"'


if str(VENDOR_TOPF_SRC) not in sys.path:
    sys.path.append(str(VENDOR_TOPF_SRC))

import topf as _vendor_topf_package  # noqa: F401


def _load_vendor_module() -> types.ModuleType:
    source = VENDOR_TOPF_MAIN.read_text(encoding="utf-8")
    source = source.replace(ABS_PROJECT, REL_PROJECT)
    source = source.replace(ABS_SCRIPT, REL_SCRIPT)
    source = source.replace(
        "import ast\n",
        "import ast\nimport re\n",
        1,
    )
    source = source.replace(
        "def topf(\n",
        """def _literal_eval_julia(value):\n"""
        """    if not isinstance(value, str):\n"""
        """        return ast.literal_eval(value)\n"""
        """    cleaned = value.strip()\n"""
        """    cleaned = re.sub(r'^[A-Za-z_][A-Za-z0-9_]*(?:\\{[^\\[]*\\})?\\[', '[', cleaned)\n"""
        """    cleaned = cleaned.replace('nothing', 'None')\n"""
        """    return ast.literal_eval(cleaned)\n\n\n"""
        """def topf(\n""",
        1,
    )
    source = source.replace(
        "ast.literal_eval(data[0]), ast.literal_eval(data[1])",
        "_literal_eval_julia(data[0]), _literal_eval_julia(data[1])",
    )

    module = types.ModuleType("_tpot_vendor_topfmain")
    module.__file__ = str(VENDOR_TOPF_MAIN)
    module.__package__ = "topf"
    exec(compile(source, str(VENDOR_TOPF_MAIN), "exec"), module.__dict__)
    return module


_VENDOR_MODULE = _load_vendor_module()

for _name, _value in _VENDOR_MODULE.__dict__.items():
    if _name.startswith("__") and _name not in {"__doc__", "__all__"}:
        continue
    globals()[_name] = _value
