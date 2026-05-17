"""Vendored third-party packages used by leela-chat.

The Fudan group's TransformerLens fork (with BT4/Leela-specific components)
is not published to PyPI — we depend on the source-tree directly. Rather
than (a) pinning a git URL and re-cloning on every environment setup, or
(b) maintaining an editable install of a checkout at a hardcoded filesystem
path, we vendor the library in-tree.

This follows the same convention pip itself uses (``pip/_vendor/``): the
package directory ``transformer_lens/`` lives unchanged under here, and we
prepend this directory to ``sys.path`` so that

    from transformer_lens import HookedTransformer

resolves to the vendored copy *before* any pip-installed ``transformer_lens``
in the environment (we don't want a conflicting upstream install to win).

Provenance and local modifications are recorded in ``VENDOR.md``.

This module's only side effect is the ``sys.path`` insertion, performed
once on first import. Importing it is cheap and idempotent.
"""
from __future__ import annotations

import os
import sys

_VENDOR_DIR = os.path.dirname(__file__)
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)
