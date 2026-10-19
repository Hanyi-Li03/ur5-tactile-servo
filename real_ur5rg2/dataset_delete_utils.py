from pathlib import Path
import shutil


RED = "\033[91m"
BOLD = "\033[1m"
DOUBLE_WIDTH = "\033#6"
RESET = "\033[0m"


def _count_files(path: Path, pattern: str = "*") -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob(pattern) if item.is_file())


def _count_dirs(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_dir())


def _top_level_entries(path: Path, limit: int = 20) -> list[str]:
    entries = sorted(path.iterdir(), key=lambda item: item.name)[:limit]
    lines = []
    for entry in entries:
        kind = "dir" if entry.is_dir() else "file"
        lines.append(f"  - [{kind}] {entry.name}")
    remaining = max(0, len(list(path.iterdir())) - limit)
    if remaining:
        lines.append(f"  - ... and {remaining} more entries")
    return lines


def confirm_and_rmtree(path: str | Path, reason: str = "overwrite") -> None:
    path = Path(path).expanduser()
    if not path.exists():
        return

    print(f"{RED}{BOLD}WARNING: DATASET DELETE OPERATION{RESET}")
    print(f"{RED}{BOLD}WARNING: This action is permanent and cannot be undone.{RESET}")
    print(f"{RED}{BOLD}WARNING: Check the data path carefully before confirming.{RESET}")
    print(f"{RED}{BOLD}WARNING: {reason} will permanently delete this folder:{RESET}")
    print(f"{RED}{BOLD}{DOUBLE_WIDTH} DATA PATH: {path.resolve()}{RESET}")
    print(f"{RED}Data summary:{RESET}")
    print(f"{RED}  - meta files: {_count_files(path / 'meta')}{RESET}")
    print(f"{RED}  - parquet episodes: {_count_files(path / 'data', '*.parquet')}{RESET}")
    print(f"{RED}  - video files: {_count_files(path / 'videos')}{RESET}")
    print(f"{RED}  - image files: {_count_files(path / 'images')}{RESET}")
    print(f"{RED}  - total files: {_count_files(path)}{RESET}")
    print(f"{RED}  - total subfolders: {_count_dirs(path)}{RESET}")
    print(f"{RED}Top-level contents:{RESET}")
    for line in _top_level_entries(path):
        print(f"{RED}{line}{RESET}")

    ans = input(f"{RED}{BOLD}Type confirm to delete, anything else to abort: {RESET}").strip()
    if ans != "confirm":
        raise SystemExit("Quit without changing the dataset.")
    shutil.rmtree(path)
