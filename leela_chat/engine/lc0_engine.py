"""Lc0 subprocess wrapper for MCTS analysis with verbose-move-stats capture.

Wraps ``chess.engine.SimpleEngine.popen_uci`` to spawn a long-lived Lc0
process and capture the per-move-candidate ``info string`` lines Lc0 emits
when started with ``--verbose-move-stats``. Each such line carries one
move's MCTS statistics (N, P, Q, V, U, WL, D, M, S) — Chunk-2's parser
turns these strings into Pydantic ``MCTSFindings``; this module is purely
the I/O boundary.

Why a custom wrapper instead of using ``analyse()`` directly: python-chess
stores ``info string ...`` payloads on ``InfoDict["string"]`` per line, but
each new line overwrites the prior value if you wait for the final
aggregated dict. We use ``engine.analysis(...)`` (streaming iterator) and
collect every ``info["string"]`` as it arrives, so all candidate-move
lines survive.

Lifecycle conventions match :mod:`leela_chat.extractors.stockfish`:
- caller owns the lifecycle (``with Lc0Engine(...) as eng:`` or explicit
  ``close()``); spawning per call is wasteful (~2-3 s BT4 load)
- the engine spawns with ``--verbose-move-stats`` + ``--weights=<path>``
  on the command line; further UCI options can be passed via ``extra_args``
- analysis uses ``chess.engine.Limit(nodes=N)`` — Lc0 honours this as
  ``go nodes N`` internally
"""
from __future__ import annotations

import time
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

import chess
import chess.engine
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    pass


class Lc0SearchResult(BaseModel):
    """Raw output of one Lc0 search: info-string lines + telemetry.

    This is the I/O-layer return type. Chunk 2's parser
    (``leela_chat.extractors.mcts_parser``) consumes
    ``info_string_lines`` and emits the structured ``MCTSFindings``.

    ``info_string_lines`` is the post-prefix content of each ``info
    string ...`` UCI line Lc0 emitted during the search — i.e., literally
    what came after the words ``info string`` on the wire. Includes both
    per-move-candidate stat lines and (in Lc0 v0.32.1) miscellaneous
    diagnostic ``info string`` chatter; filtering is the parser's job.
    """

    info_string_lines: list[str] = Field(
        ...,
        description=(
            "Each ``info string ...`` payload Lc0 emitted, in arrival order. "
            "Stripped of the literal ``info string`` prefix."
        ),
    )
    bestmove_uci: str | None = Field(
        ...,
        description=(
            "Lc0's chosen best move in UCI form (e.g. ``e2e4``). None if "
            "the engine returned no bestmove (search interrupted or "
            "engine error)."
        ),
    )
    nodes_requested: int = Field(
        ..., ge=0,
        description="Node budget the caller asked Lc0 to spend on the search.",
    )
    wall_seconds: float = Field(
        ..., ge=0.0,
        description="Wall-clock duration of ``analyse_verbose`` from request to bestmove.",
    )
    search_complete: bool = Field(
        ...,
        description=(
            "True iff the engine returned a bestmove. False on timeout / "
            "crash / interruption — partial info-string lines are still "
            "present, but downstream handlers should treat the data as "
            "incomplete."
        ),
    )

    model_config = ConfigDict(frozen=True)


class Lc0Engine:
    """Long-lived Lc0 subprocess for MCTS analysis with verbose-move-stats capture.

    Spawns Lc0 once and reuses the process across many ``analyse_verbose``
    calls — the BT4 network load (~2-3 s) is amortised across the lifetime
    of the instance.

    Args:
        binary: Path to the Lc0 executable (e.g.
            ``/workspace/bin/lc0`` on the pod, built from Lc0 v0.32.1 per
            ADR-0004).
        weights: Path to the Lc0 weights file. For BT4, this is the
            ``.pb.gz`` form Lc0 ingests directly. Note: distinct from the
            ``.onnx`` form used by lczerolens / our network extractor.
        extra_args: Additional command-line arguments to pass to Lc0
            (e.g. ``["--backend=cuda-fp16", "--threads=2"]``). Defaults
            to none; Lc0 chooses sensible backend / thread settings
            automatically.

    Example:
        ::

            with Lc0Engine(binary=Path("/workspace/bin/lc0"),
                           weights=Path("/workspace/weights/BT4.../BT4-...pb.gz")) as eng:
                result = eng.analyse_verbose(board, nodes=10000)
                for line in result.info_string_lines:
                    ...
    """

    def __init__(
        self,
        binary: Path,
        weights: Path,
        *,
        extra_args: list[str] | None = None,
    ) -> None:
        if not Path(binary).exists():
            raise FileNotFoundError(f"Lc0 binary not found: {binary}")
        if not Path(weights).exists():
            raise FileNotFoundError(f"Lc0 weights not found: {weights}")

        command: list[str] = [
            str(binary),
            "--verbose-move-stats",
            f"--weights={weights}",
        ]
        if extra_args:
            command.extend(extra_args)

        logger.info(f"Starting Lc0 from {binary}...")
        self._engine: chess.engine.SimpleEngine = chess.engine.SimpleEngine.popen_uci(command)
        engine_name = self._engine.id.get("name", "Lc0")
        logger.info(f"Lc0 ready: id_name={engine_name}")
        self._engine_name = engine_name

    @property
    def name(self) -> str:
        """Engine identifier as advertised by ``id name`` over UCI."""
        return self._engine_name

    def analyse_verbose(
        self,
        board: chess.Board,
        *,
        nodes: int = 10000,
    ) -> Lc0SearchResult:
        """Run an MCTS search; return the raw info-string lines.

        Streams the analysis via ``engine.analysis(...)`` rather than
        ``engine.analyse(...)`` so that each ``info string ...`` line
        survives — the streaming context manager surfaces every InfoDict
        as it arrives, and we pull ``info["string"]`` from each one. With
        the non-streaming form, the final aggregate retains only the
        *last* ``info["string"]`` value, losing all the per-candidate data.

        Args:
            board: Position to search.
            nodes: Node budget. Default 10000 — comfortable trade-off
                between depth and latency on the pod (~5-10 s per call).

        Returns:
            :class:`Lc0SearchResult` with the captured lines, the engine's
            bestmove, and telemetry. Caller passes
            ``result.info_string_lines`` to the Chunk-2 parser.
        """
        info_string_lines: list[str] = []
        bestmove_uci: str | None = None
        search_complete: bool = False
        start = time.perf_counter()

        with self._engine.analysis(board, chess.engine.Limit(nodes=nodes)) as analysis:
            for info in analysis:
                s = info.get("string")
                if isinstance(s, str) and s:
                    info_string_lines.append(s)
            # When the context exits, the engine has emitted bestmove and
            # analysis._finished is set. Wait() retrieves the BestMove.
            try:
                best = analysis.wait()
                if best.move is not None:
                    bestmove_uci = best.move.uci()
                    search_complete = True
            except chess.engine.EngineError as exc:
                logger.warning(f"Lc0 analysis ended without bestmove: {exc}")

        wall = time.perf_counter() - start
        logger.debug(
            f"lc0 analyse_verbose: nodes={nodes} lines={len(info_string_lines)} "
            f"bestmove={bestmove_uci} wall={wall:.2f}s complete={search_complete}"
        )
        return Lc0SearchResult(
            info_string_lines=info_string_lines,
            bestmove_uci=bestmove_uci,
            nodes_requested=nodes,
            wall_seconds=wall,
            search_complete=search_complete,
        )

    def close(self) -> None:
        """Quit the Lc0 process cleanly."""
        try:
            self._engine.quit()
        except chess.engine.EngineError as exc:
            logger.warning(f"Lc0 quit raised (ignored): {exc}")

    def __enter__(self) -> "Lc0Engine":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
