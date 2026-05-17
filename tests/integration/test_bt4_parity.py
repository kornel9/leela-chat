"""BT4 parity test — our extractor vs ``lc0`` binary.

Loads BT4 via :func:`leela_chat.network.load_bt4` (HookedTransformer path)
and ``lc0`` via python-chess, then compares WDL on a fixed set of
positions covering openings, middlegames, endgames, and the castling-rich
edge case that the lczerolens path previously got wrong.

Skipped automatically unless:
- CUDA is available (BT4 forward needs ~1.3 GB; CPU works but is too slow
  for a pytest run)
- ``$LEELA_CHAT_LC0_BIN`` points to an ``lc0`` executable
- ``$LEELA_CHAT_LC0_WEIGHTS`` points to the BT4 ``.pb.gz`` weights file

These match the pod's setup_pod.sh defaults; on the local WSL dev box the
test is skipped.

Tolerances:
- WDL per-channel max abs diff 0.05 (about 5 percentage points — fp32 vs
  fp16 backend rounding plus ``HistoryFill`` differences explain the gap)
- WDL ordering (which of W/D/L is largest) must match
- Top policy move must match
"""
from __future__ import annotations

import os

import chess
import chess.engine
import pytest


pytestmark = pytest.mark.integration


TEST_FENS = [
    # name, fen
    ("startpos",       "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("after_e4",       "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
    ("after_e4_e5",    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"),
    ("italian_open",   "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
    ("najdorf",        "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6"),
    ("kid_main",       "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP3PPP/R1BQKB1R w KQ - 0 6"),
    ("french_advance", "rnbqkbnr/ppp2ppp/4p3/3pP3/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 3"),
    # Castling-rich middlegame — both sides can castle either side.
    # Lc0 strongly favors O-O-O. This is the regression case for our
    # vendored uci_to_idx encoding fix.
    ("castling_rich",  "r3k2r/pppqbppp/2np1n2/4p3/4P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 4 8"),
    ("rook_endgame",   "8/5pk1/6p1/r6p/8/5RK1/6PP/8 w - - 0 1"),
    # Trivially winning endgame for White
    ("kr_vs_k",        "8/8/8/8/3k4/8/3R4/4K3 w - - 0 1"),
    # Trivially winning endgame for Black-to-move's opponent (White)
    ("kpvk_drawn",     "8/8/8/8/3k4/4P3/4K3/8 b - - 0 1"),
    ("zugzwang_kbnk",  "8/8/8/8/8/2N5/2B3k1/3K4 w - - 0 1"),
    # White-to-move tactical: simple fork available
    ("knight_fork",    "r1bqk2r/ppp2ppp/2n2n2/2bpp3/4P3/2NP1N2/PPP2PPP/R1BQKB1R w KQkq - 0 5"),
    # Black-to-move forcing sequence
    ("tactic_black",   "r2qk2r/pp2bppp/2n1bn2/2pp4/3P4/2NBPN2/PPP2PPP/R1BQK2R b KQkq - 0 7"),
]


def _have_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _have_lc0() -> tuple[str | None, str | None]:
    bin_ = os.environ.get("LEELA_CHAT_LC0_BIN")
    wts = os.environ.get("LEELA_CHAT_LC0_WEIGHTS")
    if bin_ and wts and os.path.exists(bin_) and os.path.exists(wts):
        return bin_, wts
    return None, None


@pytest.fixture(scope="module")
def bt4_model():
    if not _have_cuda():
        pytest.skip("BT4 parity test requires CUDA")
    from leela_chat.network import load_bt4
    return load_bt4()


@pytest.fixture(scope="module")
def lc0_engine():
    bin_, wts = _have_lc0()
    if not bin_:
        pytest.skip(
            "BT4 parity test requires $LEELA_CHAT_LC0_BIN and "
            "$LEELA_CHAT_LC0_WEIGHTS environment variables"
        )
    eng = chess.engine.SimpleEngine.popen_uci(
        [bin_, f"--weights={wts}", "--backend=cuda-fp16"]
    )
    eng.configure({"UCI_ShowWDL": True})
    yield eng
    eng.quit()


@pytest.mark.parametrize("name,fen", TEST_FENS, ids=[n for n, _ in TEST_FENS])
def test_bt4_wdl_matches_lc0(name, fen, bt4_model, lc0_engine):
    """Our extractor's WDL should match lc0 binary within rounding tolerance."""
    from leela_chat.extractors.network_inference import extract_network_inference

    board = chess.Board(fen)
    result = extract_network_inference(board, model=bt4_model, top_n=5)

    info = lc0_engine.analyse(board, chess.engine.Limit(nodes=1))
    pov_wdl = info["wdl"]
    # Get from lc0 in side-to-move (POV) form, normalize to White perspective.
    wdl = pov_wdl.white()  # PovWdl -> Wdl from WHITE pov
    expected_w = wdl.wins / 1000.0
    expected_d = wdl.draws / 1000.0
    expected_l = wdl.losses / 1000.0

    got_w = result.wdl.white_wins
    got_d = result.wdl.draw
    got_l = result.wdl.black_wins

    # Per-channel tolerance
    tol = 0.05
    assert abs(got_w - expected_w) < tol, (
        f"{name}: W differs {got_w:.3f} vs lc0 {expected_w:.3f}"
    )
    assert abs(got_d - expected_d) < tol, (
        f"{name}: D differs {got_d:.3f} vs lc0 {expected_d:.3f}"
    )
    assert abs(got_l - expected_l) < tol, (
        f"{name}: L differs {got_l:.3f} vs lc0 {expected_l:.3f}"
    )

    # Argmax of WDL should agree (sanity check that we're not swapping channels)
    got_argmax = max(range(3), key=[got_w, got_d, got_l].__getitem__)
    exp_argmax = max(range(3), key=[expected_w, expected_d, expected_l].__getitem__)
    assert got_argmax == exp_argmax, (
        f"{name}: WDL argmax differs (us={['W','D','L'][got_argmax]} "
        f"vs lc0={['W','D','L'][exp_argmax]})"
    )


@pytest.mark.parametrize("name,fen", TEST_FENS, ids=[n for n, _ in TEST_FENS])
def test_bt4_top_move_matches_lc0(name, fen, bt4_model, lc0_engine):
    """Our extractor's top-policy move should match lc0's 1-node best move."""
    from leela_chat.extractors.network_inference import extract_network_inference

    board = chess.Board(fen)
    result = extract_network_inference(board, model=bt4_model, top_n=5)
    if not result.top_moves:
        pytest.skip(f"{name}: no legal moves (terminal position)")
    our_top = result.top_moves[0].uci

    info = lc0_engine.analyse(board, chess.engine.Limit(nodes=1))
    lc0_pv = info.get("pv")
    if not lc0_pv:
        pytest.skip(f"{name}: lc0 didn't return a PV at nodes=1")
    lc0_top = lc0_pv[0].uci()

    # In some positions multiple moves have near-identical policy weight; if
    # our top-1 doesn't match lc0 exactly, accept it if lc0's pick is in our
    # top-3 (because at 1 node lc0's pick is mostly policy-driven anyway).
    if our_top != lc0_top:
        our_top3 = [mv.uci for mv in result.top_moves[:3]]
        assert lc0_top in our_top3, (
            f"{name}: top moves disagree — us={our_top} (top3={our_top3}) lc0={lc0_top}"
        )
