# MIT License
#
# Copyright (c) 2020-2025 CNRS
# Copyright (c) 2025- pyannoteAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Validate class names loaded from pipeline config.yaml files."""

from __future__ import annotations

# Official configs resolve pipeline and preprocessor classes from these packages.
# Anything else would let a malicious config import and instantiate arbitrary code.
_ALLOWED_PREFIXES = ("pyannote.", "pyannoteai.")


def qualified_config_class_name(name: str, default_module_name: str) -> str:
    """Return a dotted path, applying default_module_name when name has no dots."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Pipeline config class name must be a non-empty string.")
    name = name.strip()
    return name if "." in name else f"{default_module_name}.{name}"


def assert_trusted_config_class(
    name: str,
    default_module_name: str,
    *,
    trust_remote_code: bool = False,
) -> str:
    """Return the qualified class name, or raise if it is outside the allowlist.

    Parameters
    ----------
    name : str
        Class name from a pipeline config (`pipeline.name` or `preprocessors.*.name`).
    default_module_name : str
        Module used when `name` is unqualified, matching `get_class_by_name`.
    trust_remote_code : bool, optional
        When True, skip the allowlist. Default is False.
    """
    qualified = qualified_config_class_name(name, default_module_name)
    if trust_remote_code:
        return qualified
    if not qualified.startswith(_ALLOWED_PREFIXES):
        raise ValueError(
            f"Refusing to import {qualified!r} from a pipeline config. "
            "Only classes under the pyannote.* or pyannoteai.* packages "
            "are allowed when trust_remote_code is False."
        )
    return qualified
