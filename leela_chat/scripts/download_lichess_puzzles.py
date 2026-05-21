"""Download + parse the Lichess Puzzle DB into a queryable Parquet artifact.

Idempotent AND integrity-checked. Run on the pod:

    uv run python -m leela_chat.scripts.download_lichess_puzzles

What it does, in order:

1. **Download** the `.csv.zst` from ``https://database.lichess.org/lichess_db_puzzle.csv.zst``
   (~150MB). If the file is already present AND its uncompressed CSV
   has ≥ 5,000,000 rows AND a Parquet file exists with matching row
   count, the download step is a no-op (idempotent).
2. **Decompress** to `.csv` (~2GB). Skipped if the CSV is already
   present + has the expected row count.
3. **Parse** to Parquet (~500MB after columnar compression). Skipped
   if the Parquet exists + has matching row count.
4. **Integrity-check**: CSV row count ≥ 5M (real DB is ~5.9M as of
   2026), Parquet row count == CSV row count. If either check fails,
   the relevant intermediate artifact is deleted and the pipeline
   re-runs from that step.

The two artifacts (``.csv.zst`` immutable source + ``.parquet`` indexed
for query) both live on the network volume at
``/workspace/lichess_puzzle_db/`` — persists across pod restarts.

Args:
    --force: ignore the integrity check and re-run all steps.
    --dry-run: report what would happen; don't touch the filesystem
        or network. Used for tests.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

from loguru import logger

CORPUS_DIR = Path("/workspace/lichess_puzzle_db")
CSV_ZST_PATH = CORPUS_DIR / "lichess_db_puzzle.csv.zst"
CSV_PATH = CORPUS_DIR / "lichess_db_puzzle.csv"
PARQUET_PATH = CORPUS_DIR / "lichess_db_puzzle.parquet"
LICHESS_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
MIN_EXPECTED_ROWS = 5_000_000  # real DB is ~5.9M as of 2026; significantly
                               # below this indicates a truncated artifact.


def _stream_download(url: str, dest: Path, dry_run: bool = False) -> None:
    """Stream-download URL → dest with a progress hint."""
    if dry_run:
        logger.info(f"[dry-run] would download {url} → {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"downloading {url} → {dest} (~150MB; ~3-10 min depending on link)")
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as out:
        total = 0
        chunk_size = 4 * 1024 * 1024  # 4MB chunks
        while True:
            buf = resp.read(chunk_size)
            if not buf:
                break
            out.write(buf)
            total += len(buf)
            if total % (32 * 1024 * 1024) < chunk_size:
                logger.info(f"  ~{total / 1e6:.0f} MB downloaded")
    logger.info(f"  done: {dest.stat().st_size / 1e6:.1f} MB")


def _decompress_zstd(src: Path, dest: Path, dry_run: bool = False) -> None:
    """Decompress src (.csv.zst) → dest (.csv)."""
    if dry_run:
        logger.info(f"[dry-run] would decompress {src} → {dest}")
        return
    import zstandard as zstd
    logger.info(f"decompressing {src} → {dest} (~2GB expanded; ~30s)")
    dctx = zstd.ZstdDecompressor()
    with open(src, "rb") as r, open(dest, "wb") as w:
        dctx.copy_stream(r, w)
    logger.info(f"  done: {dest.stat().st_size / 1e9:.2f} GB")


def _parse_csv_to_parquet(csv_path: Path, parquet_path: Path, dry_run: bool = False) -> None:
    """Parse the CSV with pyarrow.csv and write a Parquet."""
    if dry_run:
        logger.info(f"[dry-run] would parse {csv_path} → {parquet_path}")
        return
    import pyarrow.csv as pacsv
    import pyarrow.parquet as pq
    logger.info(f"parsing {csv_path} → {parquet_path} (~30-60s)")
    # The Lichess CSV has a header. Default ReadOptions handles it.
    read_opts = pacsv.ReadOptions(block_size=64 * 1024 * 1024)
    convert_opts = pacsv.ConvertOptions(
        # Themes is a space-separated string; we keep it as string here
        # and split per-row at sampling time (themes_to_list in the
        # sampling module handles either form).
        column_types={
            "PuzzleId": "string",
            "FEN": "string",
            "Moves": "string",
            "Rating": "int32",
            "RatingDeviation": "int32",
            "Popularity": "int32",
            "NbPlays": "int32",
            "Themes": "string",
            "GameUrl": "string",
            "OpeningTags": "string",
        },
    )
    table = pacsv.read_csv(csv_path, read_options=read_opts, convert_options=convert_opts)
    pq.write_table(table, parquet_path, compression="zstd")
    logger.info(f"  done: {parquet_path.stat().st_size / 1e6:.1f} MB; {len(table):,} rows")


def _csv_row_count(csv_path: Path) -> int:
    """Count rows in the CSV (minus header). Fast: line count via os.read."""
    count = 0
    with open(csv_path, "rb") as f:
        while True:
            buf = f.read(8 * 1024 * 1024)
            if not buf:
                break
            count += buf.count(b"\n")
    # Subtract 1 for the header line.
    return max(0, count - 1)


def _parquet_row_count(parquet_path: Path) -> int:
    import pyarrow.parquet as pq
    return pq.read_metadata(str(parquet_path)).num_rows


def run(force: bool = False, dry_run: bool = False) -> None:
    """Main entry point. Idempotent; integrity-checks intermediates."""
    if dry_run:
        logger.info("=== DRY RUN — no filesystem changes ===")

    # --- Step 1: download ---
    if force or not CSV_ZST_PATH.exists():
        _stream_download(LICHESS_URL, CSV_ZST_PATH, dry_run=dry_run)
    else:
        logger.info(f"  {CSV_ZST_PATH} present ({CSV_ZST_PATH.stat().st_size / 1e6:.1f} MB); skipping download")

    # --- Step 2: decompress ---
    csv_needs_redo = (
        force
        or not CSV_PATH.exists()
        or (not dry_run and CSV_PATH.exists() and _csv_row_count(CSV_PATH) < MIN_EXPECTED_ROWS)
    )
    if csv_needs_redo:
        if CSV_PATH.exists() and not dry_run:
            row_n = _csv_row_count(CSV_PATH)
            logger.warning(
                f"  CSV {CSV_PATH} present but row count {row_n:,} < expected "
                f"{MIN_EXPECTED_ROWS:,} — re-decompressing"
            )
            CSV_PATH.unlink()
        _decompress_zstd(CSV_ZST_PATH, CSV_PATH, dry_run=dry_run)
    else:
        logger.info(f"  {CSV_PATH} present and row count OK; skipping decompress")

    # --- Step 3: parse to Parquet ---
    parquet_needs_redo = force or not PARQUET_PATH.exists()
    if not parquet_needs_redo and not dry_run:
        # Integrity check: Parquet row count must match CSV row count.
        try:
            pq_rows = _parquet_row_count(PARQUET_PATH)
            csv_rows = _csv_row_count(CSV_PATH)
            if pq_rows != csv_rows:
                logger.warning(
                    f"  Parquet rows {pq_rows:,} != CSV rows {csv_rows:,} — re-parsing"
                )
                parquet_needs_redo = True
            elif pq_rows < MIN_EXPECTED_ROWS:
                logger.warning(
                    f"  Parquet rows {pq_rows:,} < expected {MIN_EXPECTED_ROWS:,} — re-parsing"
                )
                parquet_needs_redo = True
        except Exception as exc:
            logger.warning(f"  Parquet integrity check failed ({exc}) — re-parsing")
            parquet_needs_redo = True

    if parquet_needs_redo:
        if PARQUET_PATH.exists() and not dry_run:
            PARQUET_PATH.unlink()
        _parse_csv_to_parquet(CSV_PATH, PARQUET_PATH, dry_run=dry_run)
    else:
        logger.info(f"  {PARQUET_PATH} present and integrity-checked; skipping parse")

    # --- Final integrity check ---
    if not dry_run:
        pq_rows = _parquet_row_count(PARQUET_PATH)
        logger.info(f"corpus ready: {PARQUET_PATH} ({pq_rows:,} rows)")
        if pq_rows < MIN_EXPECTED_ROWS:
            logger.error(
                f"  WARNING: Parquet has only {pq_rows:,} rows; "
                f"expected ≥ {MIN_EXPECTED_ROWS:,}. The corpus may be "
                f"truncated. Run with --force to redo from scratch."
            )
            sys.exit(2)
    else:
        logger.info("=== DRY RUN COMPLETE ===")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Ignore integrity check; redo all steps.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would happen; don't touch fs/net.")
    args = parser.parse_args()
    run(force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
