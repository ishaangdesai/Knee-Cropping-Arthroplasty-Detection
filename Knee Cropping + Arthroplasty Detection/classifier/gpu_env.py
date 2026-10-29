"""
Must be the first import in any entry-point script (train.py, evaluate_folder.py,
app.py) - before torch is imported, directly or transitively.

This conda env's PyTorch/cuDNN DLLs live under <env>\\Library\\bin, and are only
found if `conda activate` correctly prepended that folder to PATH. If activation
didn't fully take (e.g. a shell that never ran `conda init` for its shell type,
or a stale terminal), Windows' DLL search can resolve a different, incompatible
cudnn64_9.dll from elsewhere on PATH first - which shows up as a crash like
"Invalid handle. Cannot load symbol cudnnGetVersion" instead of a clean
"torch not found" error, since a same-named DLL was found, just the wrong one.

os.add_dll_directory() sidesteps PATH order entirely: it points Windows at this
env's own Library/bin directly, based on sys.prefix (i.e. whichever python.exe
actually launched this process), regardless of what else is on PATH.
"""

import os
import sys
from pathlib import Path

if sys.platform == "win32":
    # NOTE: os.add_dll_directory() does NOT fix this - tested empirically. Torch's
    # cuDNN loading on Windows only honors the classic PATH-based DLL search, not
    # the AddDllDirectory mechanism. So this prepends PATH directly instead.
    _env_root = Path(sys.prefix)
    # Mirrors exactly what conda's activate script prepends to PATH for a Windows
    # env - not just Library/bin. cudnn64_9.dll's own runtime-loaded dependencies
    # (cudnn_ops64_9.dll etc.) live alongside it, but other conda packages'
    # native deps can live in any of these.
    _candidate_dirs = [
        _env_root,
        _env_root / "Library" / "mingw-w64" / "bin",
        _env_root / "Library" / "usr" / "bin",
        _env_root / "Library" / "bin",
        _env_root / "Scripts",
        _env_root / "bin",
    ]
    _existing = [p for p in _candidate_dirs if p.is_dir()]
    if _existing:
        os.environ["PATH"] = os.pathsep.join(str(p) for p in _existing) + os.pathsep + os.environ.get("PATH", "")
