"""Admit known desktop contexts without changing the whole-card run limit."""
import math
import subprocess
from pathlib import Path

DESKTOP_EXECUTABLES = frozenset({
    "/usr/bin/ptyxis", "/usr/bin/nautilus", "/usr/bin/resources",
    "/home/jake-k/apps/google-chrome/chrome",
    "/usr/lib/claude-desktop/claude-desktop",
})


def desktop_ready(used_mib, executables, limit_mib=1536):
    if type(used_mib) not in (int, float) or not math.isfinite(used_mib):
        return False
    if type(limit_mib) not in (int, float) or not math.isfinite(limit_mib) or not 0 < limit_mib <= 1536:
        return False
    return (0 <= used_mib <= limit_mib and isinstance(executables, list)
            and all(type(path) is str and path in DESKTOP_EXECUTABLES for path in executables))


def snapshot():
    used = float(subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
    ], text=True, timeout=10).strip())
    raw = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits",
    ], text=True, timeout=10)
    processes = []
    for line in raw.splitlines():
        pid = int(line.strip())
        if pid <= 0:
            raise ValueError("invalid GPU process identity")
        executable = str(Path(f"/proc/{pid}/exe").resolve(strict=True))
        processes.append(dict(pid=pid, executable=executable))
    return dict(ready=desktop_ready(used, [row["executable"] for row in processes]),
                used_mib=used, processes=processes, idle_limit_mib=1536,
                running_cap_mib=23552)
