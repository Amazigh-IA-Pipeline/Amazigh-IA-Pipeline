"""Safe read/write of reference/reference.xlsx and reference/fingerprints.parquet.

reference.xlsx is the pipeline's ONLY permanent memory: one row per video ever
downloaded (or failed), key = video_id. Everything else can be rebuilt from it.

Writes are atomic (temp file + rename) and one backup copy
(reference.backup.xlsx) is taken at the start of every step that writes it.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd

from common.utils import PATHS, read_excel, write_excel

# Column groups from the spec (section 10). A few extra columns are added at
# the end of each group (marked +) because the pipeline needs them.
REFERENCE_COLUMNS = [
    # identity
    'video_id', 'url', 'title', 'upload_date', 'duration_sec', 'view_count', 'is_short', 'has_captions',
    # channel
    'channel_id', 'channel_name', 'channel_folder', 'dialect',
    # download
    'download_status', 'fail_reason', 'file_path', 'file_size_mb', 'actual_duration_sec', 'downloaded_at',
    # dedup
    'dedup_status', 'duplicate_of',
    # segment
    'segment_status', 'n_segments_kept', 'hours_kept', 'segment_error',      # + segment_error
    # push  (pushed_at is split per source, since a video can be pushed as audio AND as segments)
    'audio_repo', 'audio_pushed_at', 'segments_repo', 'segments_pushed_at',
]

NUMERIC_COLUMNS = {'duration_sec', 'view_count', 'is_short', 'has_captions', 'file_size_mb',
                   'actual_duration_sec', 'n_segments_kept', 'hours_kept'}

FINGERPRINT_COLUMNS = ['video_id', 'duration_sec', 'window_sec', 'fp_start', 'fp_middle', 'fp_end']


# ── reference.xlsx ──────────────────────────────────────────────────────────

def empty_reference() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object) for c in REFERENCE_COLUMNS})


def load_reference(path: Path | None = None) -> pd.DataFrame:
    """Load the reference as a DataFrame indexed by nothing (plain rows), with
    every spec column present. Text columns are '' when empty, numeric
    columns are numbers (NaN when empty)."""
    path = Path(path or PATHS['reference'])
    if not path.exists():
        return empty_reference()
    df = read_excel(path)
    for col in REFERENCE_COLUMNS:
        if col not in df.columns:
            df[col] = ''
    extra = [c for c in df.columns if c not in REFERENCE_COLUMNS]
    df = df[REFERENCE_COLUMNS + extra].astype(object)
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col].replace('', None), errors='coerce').astype(object)
    if df['video_id'].duplicated().any():
        raise ValueError(f'{path}: video_id is duplicated — the reference is corrupt, restore the backup')
    return df.reset_index(drop=True)


def save_reference(df: pd.DataFrame, path: Path | None = None) -> None:
    path = Path(path or PATHS['reference'])
    out = df.copy()
    for col in REFERENCE_COLUMNS:
        if col not in out.columns:
            out[col] = ''
    extra = [c for c in out.columns if c not in REFERENCE_COLUMNS]
    out = out[REFERENCE_COLUMNS + extra]
    write_excel(out, path)


def backup_reference() -> None:
    """Keep ONE backup copy of reference/ files, taken before a step writes them."""
    for key, suffix in (('reference', '.backup.xlsx'), ('fingerprints', '.backup.parquet')):
        src = PATHS[key]
        if src.exists():
            shutil.copy2(src, src.with_name(src.stem + suffix))


def known_video_ids() -> set[str]:
    ref = load_reference()
    return set(ref['video_id'].astype(str))


class Reference:
    """Convenience wrapper: load once, update rows by video_id, save periodically."""

    def __init__(self):
        self.df = load_reference()
        self._index = {vid: i for i, vid in enumerate(self.df['video_id'].astype(str))}

    def __contains__(self, video_id) -> bool:
        return str(video_id) in self._index

    def get(self, video_id) -> dict | None:
        i = self._index.get(str(video_id))
        return None if i is None else self.df.loc[i].to_dict()

    def upsert(self, video_id, **fields) -> None:
        video_id = str(video_id)
        for col in fields:
            if col not in self.df.columns:
                self.df[col] = ''
                self.df[col] = self.df[col].astype(object)
        if video_id in self._index:
            i = self._index[video_id]
            for col, value in fields.items():
                self.df.at[i, col] = value
        else:
            row = {c: '' for c in self.df.columns}
            for c in NUMERIC_COLUMNS:
                row[c] = float('nan')
            row.update(fields)
            row['video_id'] = video_id
            self.df = pd.concat([self.df, pd.DataFrame([row]).astype(object)], ignore_index=True)
            self._index[video_id] = len(self.df) - 1

    def rows(self, **conditions) -> pd.DataFrame:
        mask = pd.Series(True, index=self.df.index)
        for col, expected in conditions.items():
            values = self.df[col].astype(str)
            if isinstance(expected, (list, tuple, set)):
                mask &= values.isin([str(e) for e in expected])
            else:
                mask &= values == str(expected)
        return self.df[mask]

    def save(self) -> None:
        save_reference(self.df)


# ── fingerprints.parquet ────────────────────────────────────────────────────

def load_fingerprints(path: Path | None = None) -> dict[str, dict]:
    """video_id -> {'duration_sec', 'window_sec', 'fp_start', 'fp_middle', 'fp_end'}."""
    path = Path(path or PATHS['fingerprints'])
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    out = {}
    for rec in df.to_dict('records'):
        out[str(rec['video_id'])] = {
            'duration_sec': float(rec['duration_sec']),
            'window_sec': float(rec['window_sec']),
            'fp_start': list(rec['fp_start']),
            'fp_middle': list(rec['fp_middle']),
            'fp_end': list(rec['fp_end']),
        }
    return out


def save_fingerprints(fingerprints: dict[str, dict], path: Path | None = None) -> None:
    path = Path(path or PATHS['fingerprints'])
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{'video_id': vid, **{k: rec[k] for k in FINGERPRINT_COLUMNS[1:]}}
            for vid, rec in sorted(fingerprints.items())]
    df = pd.DataFrame(rows, columns=FINGERPRINT_COLUMNS)
    tmp = path.with_name(path.stem + '.tmp.parquet')
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
