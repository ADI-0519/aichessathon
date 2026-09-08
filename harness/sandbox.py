import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import IO

from harness.rules import STDERR_HEAD, STDERR_TAIL, STDOUT_CAP, WATCHDOG_GRACE_MS

RUNNER = Path(__file__).resolve().parent / "runner.py"
DRAIN_GRACE_S = 0.2
SUSPENDS = hasattr(signal, "SIGSTOP")
# The container points these at a /tmp wiped between games.
SCRATCH_VARS = ("HOME", "TMPDIR", "XDG_CACHE_HOME", "TORCH_HOME", "HF_HOME", "NUMBA_CACHE_DIR")


class AgentFailure(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def local(directory: Path, seed: int = 0) -> "Agent":
    """Run an agent as a process on this machine, through the platform's runner."""
    resolved = directory.resolve()
    return Agent([sys.executable, str(RUNNER), str(resolved)], resolved.name, seed)


class Agent:
    """One agent process, spoken to exactly as the platform speaks to a container."""

    def __init__(self, command: list[str], name: str, seed: int = 0) -> None:
        self.command = command
        self.name = name
        self.seed = seed
        self.stderr_log = ""
        self._process: subprocess.Popen[bytes] | None = None
        self._scratch: str | None = None
        self._chunks: queue.Queue[tuple[str, bytes]] = queue.Queue()
        self._readers: list[threading.Thread] = []
        self._buffer = b""
        self._head = b""
        self._tail = b""
        self._written = 0

    def start(self, init_budget_s: float) -> None:
        self._scratch = tempfile.mkdtemp(prefix="agent-")
        process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=self._environment(self._scratch),
            start_new_session=SUSPENDS,
        )
        self._process = process
        self._readers = [
            self._reader(_pipe(process.stdout), "stdout"),
            self._reader(_pipe(process.stderr), "stderr"),
        ]
        ready = self._await_line(time.monotonic() + init_budget_s)
        if ready is None:
            raise AgentFailure("init" if process.poll() is None else "crash")
        if not _is_ready(ready):
            raise AgentFailure("init")

    # The platform freezes you while the opponent thinks.
    def suspend(self) -> None:
        if SUSPENDS:
            self._signal(signal.SIGSTOP)

    def resume(self) -> None:
        if SUSPENDS:
            self._signal(signal.SIGCONT)

    def move(self, fen: str, time_left_ms: int) -> str:
        if self._process is None:
            raise RuntimeError("Agent moved before start")
        request = json.dumps({"fen": fen, "time_left_ms": time_left_ms}).encode()
        try:
            _pipe(self._process.stdin).write(request + b"\n")
        except BrokenPipeError:
            raise AgentFailure("crash") from None
        line = self._await_line(time.monotonic() + (time_left_ms + WATCHDOG_GRACE_MS) / 1000.0)
        if line is None:
            raise AgentFailure("flag")
        return _parse_move(line)

    def stop(self) -> None:
        if self._process is None:
            return
        if SUSPENDS:
            self._signal(signal.SIGKILL)
        self._process.kill()
        for reader in self._readers:
            reader.join(DRAIN_GRACE_S)
        self._drain()
        self.stderr_log = self._output()
        _pipe(self._process.stdin).close()
        self._process.wait()
        self._process = None
        self._readers = []
        if self._scratch is not None:
            shutil.rmtree(self._scratch, ignore_errors=True)
            self._scratch = None

    def _environment(self, scratch: str) -> dict[str, str]:
        environment = dict(os.environ)
        environment.update(dict.fromkeys(SCRATCH_VARS, scratch))
        environment["OMP_NUM_THREADS"] = "1"
        environment["HARNESS_SEED"] = str(self.seed)
        return environment

    def _signal(self, number: signal.Signals) -> None:
        if self._process is None:
            return
        try:
            os.killpg(self._process.pid, number)
        except ProcessLookupError:
            return

    def _reader(self, stream: IO[bytes], name: str) -> threading.Thread:
        reader = threading.Thread(target=self._forward, args=(stream, name), daemon=True)
        reader.start()
        return reader

    # The reader owns its pipe, so a thread outliving stop() never reads a closed stream.
    def _forward(self, stream: IO[bytes], name: str) -> None:
        with stream:
            while True:
                chunk = stream.read(STDOUT_CAP)
                self._chunks.put((name, chunk))
                if not chunk:
                    return

    def _await_line(self, deadline: float) -> bytes | None:
        while b"\n" not in self._buffer:
            if len(self._buffer) >= STDOUT_CAP:
                raise AgentFailure("illegal")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                name, chunk = self._chunks.get(timeout=remaining)
            except queue.Empty:
                return None
            if name == "stderr":
                self._keep(chunk)
            elif not chunk:
                raise AgentFailure("crash")
            else:
                self._buffer += chunk
        line, _, self._buffer = self._buffer.partition(b"\n")
        return line

    def _drain(self) -> None:
        while True:
            try:
                name, chunk = self._chunks.get_nowait()
            except queue.Empty:
                return
            if name == "stderr":
                self._keep(chunk)

    def _keep(self, chunk: bytes) -> None:
        if not chunk:
            return
        self._written += len(chunk)
        room = STDERR_HEAD - len(self._head)
        if room > 0:
            self._head += chunk[:room]
            chunk = chunk[room:]
        if chunk:
            self._tail = (self._tail + chunk)[-STDERR_TAIL:]

    def _output(self) -> str:
        dropped = self._written - len(self._head) - len(self._tail)
        middle = f"\n[{dropped:,} bytes dropped]\n".encode() if dropped > 0 else b""
        return (self._head + middle + self._tail).decode("utf-8", "replace")


def _pipe(stream: IO[bytes] | None) -> IO[bytes]:
    if stream is None:
        raise RuntimeError("The agent process exposed no pipe")
    return stream


def _is_ready(line: bytes) -> bool:
    try:
        payload = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("ready") is True


def _parse_move(line: bytes) -> str:
    try:
        payload = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise AgentFailure("illegal") from None
    move = payload.get("move") if isinstance(payload, dict) else None
    if not isinstance(move, str):
        raise AgentFailure("illegal")
    return move
