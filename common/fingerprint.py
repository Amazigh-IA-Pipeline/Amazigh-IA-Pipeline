"""Chromaprint audio fingerprints at three positions (start, middle, end).

Why three windows: two different episodes of a show often share the same
intro jingle, so matching only the first minutes wrongly flags them as
duplicates. A video is a duplicate only if start, middle AND end all match.

Two ways to compute a fingerprint (config fingerprint.tool):
  - ffmpeg  : ffmpeg's built-in chromaprint muxer (no temp files). Needs an
              ffmpeg build with --enable-chromaprint (Linux packages, the
              gyan.dev "full" Windows build).
  - fpcalc  : Chromaprint's fpcalc, on a temporary WAV cut with ffmpeg.
  - auto    : ffmpeg if it supports chromaprint, else fpcalc.
Both use Chromaprint's default algorithm, so their outputs are comparable.

Used by: download (computes them right after a download) and dedup_audio
(compares them, and computes any that are missing).
"""
from __future__ import annotations

import functools
import os
import shutil
import subprocess
import tempfile

import numpy as np

FRAME_SEC = 4096 / 3 / 11025          # Chromaprint hop: ~0.124 s per fingerprint frame


@functools.lru_cache(maxsize=1)
def ffmpeg_has_chromaprint() -> bool:
    if shutil.which('ffmpeg') is None:
        return False
    out = subprocess.run(['ffmpeg', '-hide_banner', '-muxers'], capture_output=True, text=True)
    return any(line.split()[1:2] == ['chromaprint'] for line in out.stdout.splitlines() if len(line.split()) > 1)


def resolve_tool(tool: str = 'auto') -> str:
    tool = (tool or 'auto').lower()
    if tool == 'auto':
        if ffmpeg_has_chromaprint():
            return 'ffmpeg'
        if shutil.which('fpcalc'):
            return 'fpcalc'
        raise RuntimeError('No fingerprint tool: install an ffmpeg build with chromaprint, '
                           'or Chromaprint\'s fpcalc (Windows: winget install AcoustID.Chromaprint)')
    if tool == 'ffmpeg' and not ffmpeg_has_chromaprint():
        raise RuntimeError('fingerprint.tool=ffmpeg but this ffmpeg has no chromaprint muxer')
    if tool == 'fpcalc' and not shutil.which('fpcalc'):
        raise RuntimeError('fingerprint.tool=fpcalc but fpcalc is not on PATH')
    return tool


def _fp_ffmpeg(path: str, start: float, length: float) -> list[int]:
    cmd = ['ffmpeg', '-nostdin', '-v', 'error', '-ss', f'{start:.3f}', '-t', f'{length:.3f}', '-i', path,
           '-vn', '-ac', '1', '-f', 'chromaprint', '-fp_format', 'raw', '-']
    res = subprocess.run(cmd, capture_output=True, timeout=600)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.decode('utf-8', 'replace').strip()[:300])
    return np.frombuffer(res.stdout, dtype='<u4').astype(np.int64).tolist()


def _fp_fpcalc(path: str, start: float, length: float) -> list[int]:
    fd, tmp = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    try:
        cut = ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-ss', f'{start:.3f}', '-t', f'{length:.3f}',
               '-i', path, '-vn', '-ac', '1', '-ar', '11025', tmp]
        res = subprocess.run(cut, capture_output=True, timeout=600)
        if res.returncode != 0:
            raise RuntimeError(res.stderr.decode('utf-8', 'replace').strip()[:300])
        res = subprocess.run(['fpcalc', '-raw', '-length', str(int(length) + 1), tmp],
                             capture_output=True, text=True, timeout=600)
        for line in res.stdout.splitlines():
            if line.startswith('FINGERPRINT='):
                # fpcalc may print signed ints: mask to unsigned 32-bit (the bug in the old script)
                return [int(x) & 0xFFFFFFFF for x in line.split('=', 1)[1].split(',') if x]
        raise RuntimeError(f'fpcalc gave no fingerprint: {res.stderr.strip()[:300]}')
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def window_starts(duration: float, window: float) -> dict[str, float]:
    if duration <= window:
        return {'fp_start': 0.0, 'fp_middle': 0.0, 'fp_end': 0.0}
    return {
        'fp_start': 0.0,
        'fp_middle': max(0.0, duration / 2 - window / 2),
        'fp_end': max(0.0, duration - window),
    }


def compute_fingerprints(path, duration: float, window: float = 60.0, tool: str = 'auto') -> dict:
    """Return {'duration_sec', 'window_sec', 'fp_start', 'fp_middle', 'fp_end'}."""
    tool = resolve_tool(tool)
    func = _fp_ffmpeg if tool == 'ffmpeg' else _fp_fpcalc
    record = {'duration_sec': float(duration), 'window_sec': float(window)}
    cache = {}
    for name, start in window_starts(duration, window).items():
        if start not in cache:
            cache[start] = func(str(path), start, min(window, duration) if duration > 0 else window)
        record[name] = cache[start]
        if not record[name]:
            raise RuntimeError(f'empty fingerprint for {name}')
    return record


# ── comparison ──────────────────────────────────────────────────────────────

def _popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, 'bitwise_count'):
        return np.bitwise_count(x)
    return np.unpackbits(x.view(np.uint8).reshape(-1, 4), axis=1).sum(axis=1)


def similarity(a, b, max_offset_sec: float = 3.0) -> float:
    """Bit similarity = 1 - (differing bits / 32), averaged over frames, best
    over shifts of up to ±max_offset_sec.

    ~1.0 for the same recording (even re-encoded in another codec), ~0.5-0.65
    for unrelated audio. Two changes from the old script, both measured on
    test audio:
      - shifting: the old script compared at offset 0 only, so a re-upload
        with a 1.5 s longer intro scored 0.0;
      - bit similarity instead of "share of frames within 2 bits": with a
        re-encode plus a fractional-frame offset the old rule gave only
        0.55-0.70 on a true duplicate, while bit similarity gave 0.92-0.98
        (unrelated audio: 0.56-0.67).
    """
    a = np.asarray(a, dtype=np.int64).astype(np.uint32)
    b = np.asarray(b, dtype=np.int64).astype(np.uint32)
    if a.size == 0 or b.size == 0:
        return 0.0
    max_shift = int(round(max_offset_sec / FRAME_SEC))
    min_overlap = max(10, int(0.5 * min(a.size, b.size)))
    best = 0.0
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            x, y = a[shift:], b
        else:
            x, y = a, b[-shift:]
        n = min(x.size, y.size)
        if n < min_overlap:
            continue
        diff_bits = _popcount(np.bitwise_xor(x[:n], y[:n]))
        score = 1.0 - float(diff_bits.mean()) / 32.0
        if score > best:
            best = score
    return best


def compare(fp1: dict, fp2: dict, max_offset_sec: float) -> dict[str, float]:
    return {name: similarity(fp1[name], fp2[name], max_offset_sec)
            for name in ('fp_start', 'fp_middle', 'fp_end')}
