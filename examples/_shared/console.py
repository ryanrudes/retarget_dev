"""A single shared :mod:`rich` console for all example output.

Routing every message through one console keeps styling consistent and lets the
video renderer share it with its progress bar.
"""

from __future__ import annotations

from rich.console import Console

console = Console()
error_console = Console(stderr=True, style="red")
