"""Background key listener for the demo's arm toggle.

Reads single keystrokes from stdin without waiting for Enter — raw termios on
POSIX, msvcrt on Windows. When stdin is not a TTY (piped/redirected runs) the
listener degrades to a no-op and the demo runs on the ``--arm`` flag only.
The listener never touches audio state; the main loop polls ``poll()``.
"""

from __future__ import annotations

import queue
import sys
import threading


class KeyboardListener:
    """Poll single keystrokes in the background; ``poll()`` returns them one at a time."""

    def __init__(self) -> None:
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._restore = None
        if sys.stdin is not None and sys.stdin.isatty():
            self._start()

    @property
    def active(self) -> bool:
        return self._thread is not None

    def _start(self) -> None:
        try:
            if sys.platform == "win32":
                self._thread = threading.Thread(target=self._loop_windows, daemon=True)
            else:
                self._thread = threading.Thread(target=self._loop_posix, daemon=True)
            self._thread.start()
        except Exception:
            # No key toggles available (exotic terminal) — flag-only demo still works.
            self._thread = None

    def _loop_windows(self) -> None:
        import msvcrt

        while True:
            key = msvcrt.getwch()
            self._queue.put(key)
            if key in ("q", "\x03"):  # quit / Ctrl-C — stop reading
                break

    def _loop_posix(self) -> None:
        import termios
        import tty

        fd = sys.stdin.fileno()
        self._restore = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                key = sys.stdin.read(1)
                self._queue.put(key)
                if key in ("q", "\x03"):  # quit / Ctrl-C — stop reading
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, self._restore)
            self._restore = None

    def poll(self) -> str | None:
        """Return one pending keystroke, or None."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        if self._restore is not None:
            try:
                import termios

                fd = sys.stdin.fileno()
                termios.tcsetattr(fd, termios.TCSADRAIN, self._restore)
            except Exception:  # terminal already gone at shutdown
                pass
            self._restore = None
