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

import re
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


# Matches Lc0's verbose-move-stats "node ..." summary header line. We
# only need a cheap prefix check, not the full _HEADER_RE; the parser
# does the real parsing downstream. Used by ``_keep_last_block`` to
# split a multi-block live-stats stream into its final emit cycle.
_NODE_SUMMARY_RE = re.compile(r"^\s*node\s+\(\s*\d+\s*\)")


def _keep_last_block(info_string_lines: list[str]) -> list[str]:
    """Return only the lines belonging to the last verbose-move-stats
    emit cycle.

    With ``--log-live-stats`` on (the engine's default — see
    ``Lc0Engine.__init__``), Lc0 emits a verbose-move-stats block on
    every UCI ``info`` update — potentially 20+ blocks per 10k-node
    search. Single-snapshot callers (Chunks 1-5 extractors that don't
    want trajectory data) need only the final block. Trajectory
    callers (``extract_mcts_trajectory``, 6a v2+) consume the whole
    stream and group it themselves.

    Implementation: find the last ``node`` summary line; walk backward
    to find the previous ``node`` boundary (or list start); return the
    inclusive slice. If no ``node`` summary is found, returns the
    input unchanged (no live-stats emissions; the input is whatever
    Lc0 produced).
    """
    last_node_idx: int | None = None
    for i, line in enumerate(info_string_lines):
        if _NODE_SUMMARY_RE.match(line):
            last_node_idx = i
    if last_node_idx is None:
        return info_string_lines
    start = 0
    for j in range(last_node_idx - 1, -1, -1):
        if _NODE_SUMMARY_RE.match(info_string_lines[j]):
            start = j + 1
            break
    return info_string_lines[start:last_node_idx + 1]


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
    multipv_requested: int = Field(
        default=1, ge=1,
        description=(
            "MultiPv value passed to the engine for this search. "
            "Added in Chunk 7b. 1 = single-PV (default; quick mode). "
            ">1 = N principal variations reported via ``principal_variations`` "
            "below (used by deep mode for cross-PV cross-check)."
        ),
    )
    principal_variations: dict[int, list[str]] = Field(
        default_factory=dict,
        description=(
            "Per-multipv-index principal variations. Key is the 1-based PV "
            "index Lc0 reports in info['multipv']; value is the PV move "
            "sequence in UCI form (info['pv']). Empty dict when "
            "``multipv_requested == 1`` AND the engine didn't report a PV. "
            "When multipv > 1, key range is 1..multipv_requested. "
            "Last-emitted PV per index wins (Lc0 refines PVs across the "
            "search; we keep the final state). Used by extract_mcts to "
            "populate MCTSChild.pv_san on the matching root candidates."
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

        # ``--log-live-stats`` is a ProOnly UCI option in Lc0
        # v0.32.1 — accessible only via the CLI flag, NOT via UCI
        # ``setoption``. We spawn with it always-on so trajectory callers
        # (Chunk 6a v2+ ``extract_mcts_trajectory``) see the stream of
        # per-info-update verbose-move-stats blocks; single-snapshot
        # callers filter to the last block in ``analyse_verbose``.
        # Source: ``update_research/repos/lc0-master/src/search/classic/params.cc:294``.
        command: list[str] = [
            str(binary),
            "--verbose-move-stats",
            "--log-live-stats",
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
        log_live_stats: bool = False,
        multipv: int = 1,
        root_moves: "list[chess.Move] | None" = None,
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
            log_live_stats: If True, returned ``info_string_lines``
                contain the FULL multi-block stream (one verbose-move-
                stats block per UCI info update, each terminated by a
                ``node`` summary line) — consumed by
                ``extract_mcts_trajectory`` for intra-search Q-trajectory
                data. If False (default), the stream is filtered down to
                the final emit cycle so single-snapshot extractors
                (Chunks 1-5) see only the end-of-search snapshot. The
                ``--log-live-stats`` CLI flag is always set on the
                engine (it's a ProOnly Lc0 option not accessible via
                UCI ``setoption``); this kwarg only controls whether
                the wrapper filters the stream before returning.
            multipv: MultiPV value to pass to the engine (Chunk 7b). ``1``
                (default) = single-PV behavior, byte-identical with pre-7b.
                ``>1`` = N principal variations reported via
                ``info["multipv"]`` + ``info["pv"]``; the wrapper captures
                them into ``principal_variations`` on the returned
                :class:`Lc0SearchResult`. Note: verbose-move-stats output
                (``info["string"]`` lines) is IDENTICAL regardless of
                multipv — Lc0 always emits one line per root candidate
                + a ``node`` summary line. The multipv-distinguishing
                data lives in ``info["pv"]`` / ``info["multipv"]``.
                Verified against the v0.32.1 build via the
                ``lc0_multipv_probe.py`` script during 7b design.
            root_moves: Optional list of moves to constrain the search to
                at the ROOT level (Chunk 7c). Threaded through to
                python-chess's ``engine.analysis(..., root_moves=...)``
                which sends the UCI ``go searchmoves m1 m2 ...`` command.
                Lc0 honours this — visits accumulate only below the named
                moves; forbidden root candidates stay at 0 visits. Use case
                for Chunk 7c: focused MCTS on a single candidate move for
                ``did_you_consider``'s deep-dive prose section. Note:
                verbose-move-stats output is IDENTICAL in shape (all root
                candidates still emit one line each); forbidden moves
                just have visits ≤ 1. The deep PV from the allowed move
                lives in ``info["pv"]`` and gets captured into
                ``principal_variations`` per the existing multipv path.
                Verified against the v0.32.1 build via the
                ``lc0_searchmoves_probe.py`` script during 7c design.

        Returns:
            :class:`Lc0SearchResult` with the captured lines, the engine's
            bestmove, per-multipv-index principal variations (when
            multipv > 1), and telemetry. Caller passes
            ``result.info_string_lines`` to the Chunk-2 parser and uses
            ``result.principal_variations`` to populate
            :attr:`MCTSChild.pv_san` on the matching root candidates.
        """
        info_string_lines: list[str] = []
        # Per-multipv-index PV: key = multipv index (1-based per UCI),
        # value = PV move sequence in UCI form. Last-write-wins per index
        # because Lc0 refines PVs across the search and we want the final
        # state.
        principal_variations: dict[int, list[str]] = {}
        bestmove_uci: str | None = None
        search_complete: bool = False
        start = time.perf_counter()

        # Pass multipv + root_moves directly to analysis() — python-chess
        # refuses configure({"MultiPv": ...}) with "automatically managed"
        # and the analysis()/analyse() multipv= + root_moves= kwargs are
        # the supported path. root_moves is python-chess's name for the
        # UCI ``searchmoves`` go-command parameter (Chunk 7c addition).
        analysis_kwargs: dict = {"multipv": multipv}
        if root_moves is not None:
            analysis_kwargs["root_moves"] = root_moves
        with self._engine.analysis(
            board, chess.engine.Limit(nodes=nodes), **analysis_kwargs,
        ) as analysis:
            for info in analysis:
                s = info.get("string")
                if isinstance(s, str) and s:
                    info_string_lines.append(s)
                # Capture per-PV move sequence. info["multipv"] is 1-based
                # when present, but python-chess sometimes OMITS the
                # multipv field on single-PV calls (multipv=1 implied as
                # default). info["pv"] is a list of chess.Move objects.
                # 7c gate relaxation: capture PVs whenever info["pv"] is
                # present, defaulting to multipv index 1 when the field
                # is absent. This is what makes focused-pass single-PV
                # cases (multipv=1 + root_moves) populate pv_san on the
                # focal child. Last value per index wins (Lc0 refines PVs
                # across the search).
                pv_moves = info.get("pv")
                if pv_moves:
                    pv_idx = info.get("multipv")
                    idx = int(pv_idx) if pv_idx is not None else 1
                    principal_variations[idx] = [m.uci() for m in pv_moves]
            try:
                best = analysis.wait()
                if best.move is not None:
                    bestmove_uci = best.move.uci()
                    search_complete = True
            except chess.engine.EngineError as exc:
                logger.warning(f"Lc0 analysis ended without bestmove: {exc}")

        raw_line_count = len(info_string_lines)
        if not log_live_stats:
            info_string_lines = _keep_last_block(info_string_lines)

        wall = time.perf_counter() - start
        logger.debug(
            f"lc0 analyse_verbose: nodes={nodes} live_stats={log_live_stats} "
            f"multipv={multipv} root_moves={len(root_moves) if root_moves else 0} "
            f"raw_lines={raw_line_count} "
            f"returned_lines={len(info_string_lines)} pvs={len(principal_variations)} "
            f"bestmove={bestmove_uci} wall={wall:.2f}s complete={search_complete}"
        )
        return Lc0SearchResult(
            info_string_lines=info_string_lines,
            bestmove_uci=bestmove_uci,
            nodes_requested=nodes,
            wall_seconds=wall,
            search_complete=search_complete,
            multipv_requested=multipv,
            principal_variations=principal_variations,
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
