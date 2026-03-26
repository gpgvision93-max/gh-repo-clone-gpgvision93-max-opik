"""
Utility to force OpenBCI configuration files to use text/console output.

It scans common config locations for the OpenBCI GUI and Python projects,
applies text-friendly values, and keeps a .bak backup before writing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple


DEFAULT_CANDIDATES: Tuple[str, ...] = (
    "./config.json",
    "~/.config/OpenBCI/config.json",
    "~/Library/Application Support/OpenBCI/config.json",
    r"C:\Users\%USERNAME%\AppData\Local\OpenBCI\config.json",
)

TEXT_VALUES = {
    "display_format": "text",
    "output_mode": "text",
    "surface_format": "text",
    "ui_display": "text",
    "data_view": "text",
}

DISABLE_FLAGS = {
    "visual_mode": False,
}


def expand_path(path_str: str) -> Path:
    """Expand user and environment variables and return an absolute Path."""
    expanded = os.path.expanduser(os.path.expandvars(path_str))
    return Path(expanded).expanduser().resolve()


def find_candidate_files(extra: Iterable[str] | None = None) -> List[Path]:
    """Return existing config files from default and user-supplied paths."""
    paths: List[Path] = []
    seen = set()
    candidates = list(extra or []) + list(DEFAULT_CANDIDATES)
    for cand in candidates:
        candidate_path = expand_path(cand)
        if candidate_path.exists() and candidate_path.is_file():
            if candidate_path not in seen:
                paths.append(candidate_path)
                seen.add(candidate_path)
    return paths


def apply_text_settings(path: Path, dry_run: bool = False) -> Tuple[bool, str]:
    """
    Apply text-friendly keys to a JSON config file.

    Returns (changed, message).
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            content = json.load(f)
    except json.JSONDecodeError:
        return False, "Skipped: not valid JSON"
    except OSError as exc:
        return False, f"Skipped: cannot read file ({exc})"

    if not isinstance(content, dict):
        return False, "Skipped: config root is not an object"

    changed = False
    for key, value in TEXT_VALUES.items():
        if content.get(key) != value:
            content[key] = value
            changed = True
    for key, value in DISABLE_FLAGS.items():
        if content.get(key) != value:
            content[key] = value
            changed = True

    if not changed:
        return False, "No changes needed"

    if dry_run:
        return True, "Would update (dry-run)"

    backup_path = path.with_suffix(path.suffix + ".bak")
    try:
        shutil.copy2(path, backup_path)
    except OSError as exc:
        return False, f"Failed to create backup: {exc}"

    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(content, f, indent=2)
    except OSError as exc:
        return False, f"Failed to write updated config: {exc}"

    return True, f"Updated (backup: {backup_path.name})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Force OpenBCI config files to output text instead of visual plots."
    )
    parser.add_argument(
        "--path",
        action="append",
        help="Additional config file path(s) to check/update (can be repeated).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = find_candidate_files(args.path)
    if not candidates:
        print("No config files found in common locations.")
        return 1

    exit_code = 0
    for cfg in candidates:
        changed, message = apply_text_settings(cfg, dry_run=args.dry_run)
        status = "CHANGED" if changed else "SKIPPED"
        print(f"[{status}] {cfg} -> {message}")
        if "Failed" in message:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
