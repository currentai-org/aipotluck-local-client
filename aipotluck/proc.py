"""Subprocess options shared by everything that starts a helper binary (llama-server, llama-bench,
newt, nvidia-smi).

On Windows the background service runs under pythonw.exe (a GUI-subsystem process, so no console
window opens at logon). A console program started from a process with no console gets a brand-new,
visible console window of its own -- one per llama-server, llama-bench and newt start, popping up
on the desktop and killing the process if someone closes it. CREATE_NO_WINDOW gives the child a
console with no window instead. Elsewhere this is empty, so call sites stay one line:

    subprocess.Popen(cmd, ..., **NO_WINDOW)
"""

from __future__ import annotations

import sys

CREATE_NO_WINDOW = 0x08000000

NO_WINDOW: dict = {"creationflags": CREATE_NO_WINDOW} if sys.platform == "win32" else {}
