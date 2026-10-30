"""Step 3 — download.

INPUT   dedup_url/output/deduplicated.xlsx

OUTPUT  download/downloads/<channel_folder>/audios/<video_id>.<ext>     best audio-only stream, not re-encoded
        download/downloads/<channel_folder>/metadata/<video_id>.json
        reference/reference.xlsx       one new row per video (downloaded OR failed)
        reference/fingerprints.parquet one fingerprint (start/middle/end) per downloaded video

For each video:
  1. download the best audio-only stream (no re-encoding)
  2. check it: size, file header, and real duration vs. expected (tolerance in config);
     if the check fails the file is deleted and the video is marked failed
  3. compute the audio fingerprint -> fingerprints.parquet
  4. write the metadata file and the reference row

Order: smaller channels first, so an interrupted run still gives a balanced sample.
Channel folder: chosen the first time a channel_id is seen, stored in reference.xlsx,
and reused for every later video of that channel (even if it is renamed).
Failures that belong to the VIDEO (private, removed, age-restricted, broken file...)
are written to the reference so they are not retried every run. Failures that belong
to the SESSION (YouTube bot check, network) are NOT written, so the next run retries
them; after 5 in a row the step stops (add cookies in config, or wait).

Resumable: videos already in the reference are skipped. Reference saved every N videos.
Options: --retry-failed  also retry videos marked failed in the reference
         --limit N        only process N videos (testing)

Run alone:  python download/download.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common.fingerprint import compute_fingerprints
from common.reference_io import Reference, backup_reference, load_fingerprints, save_fingerprints
from common.utils import (PATHS, get_logger, load_config, now_iso, probe_duration, read_excel, rel_path,
                          require_ffmpeg, safe_folder_name, setup_logging, to_float, to_int, ytdlp_base_options)

STEP = 'download'
log = get_logger(STEP)

MAGIC_BYTES = {
    'webm': (0, b'\x1a\x45\xdf\xa3'), 'mka': (0, b'\x1a\x45\xdf\xa3'),
    'm4a': (4, b'ftyp'), 'mp4': (4, b'ftyp'),
    'ogg': (0, b'OggS'), 'opus': (0, b'OggS'),
    'wav': (0, b'RIFF'), 'flac': (0, b'fLaC'),
}
AUDIO_EXTS = {'.webm', '.m4a', '.mp4', '.opus', '.ogg', '.mp3', '.mka', '.aac', '.wav', '.flac'}
SESSION_ERRORS = ('bot_check', 'network_error')
MAX_CONSECUTIVE_SESSION_ERRORS = 5


class SessionError(RuntimeError):
    pass


# ── YouTube access (the only function that touches the network) ────────────

def yt_download(url: str, audios_dir: Path, video_id: str, config: dict) -> dict:
    import yt_dlp
    dcfg = config['download']
    opts = ytdlp_base_options(config)
    opts.update({
        'format': 'bestaudio/best',
        'outtmpl': str(audios_dir / f'{video_id}.%(ext)s'),
        'postprocessors': [],
        'retries': 10, 'fragment_retries': 10, 'extractor_retries': 3,
        'sleep_interval': to_float(dcfg.get('sleep_min_sec'), 1.0),
        'max_sleep_interval': to_float(dcfg.get('sleep_max_sec'), 3.0),
        'overwrites': True,
    })
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=True)


def classify_error(message: str) -> str:
    msg = str(message).lower()
    if 'private video' in msg or 'video is private' in msg:
        return 'private'
    if 'members-only' in msg or 'join this channel' in msg:
        return 'members_only'
    if 'confirm your age' in msg or 'age-restricted' in msg or 'age restricted' in msg or 'inappropriate' in msg:
        return 'age_restricted'
    if 'not a bot' in msg or 'confirm you' in msg or 'sign in' in msg or 'cookies' in msg or 'http error 429' in msg:
        return 'bot_check'
    if any(k in msg for k in ('video unavailable', 'has been removed', 'no longer available', 'account associated',
                              'does not exist', 'copyright')):
        return 'unavailable'
    if 'country' in msg or 'geo' in msg:
        return 'region_blocked'
    if any(k in msg for k in ('timed out', 'connection', 'temporary failure', 'network', 'http error 5')):
        return 'network_error'
    return 'download_error'


# ── checks ──────────────────────────────────────────────────────────────────

def find_audio(audios_dir: Path, video_id: str) -> Path | None:
    for p in sorted(audios_dir.glob(f'{video_id}.*')):
        if p.suffix.lower() in AUDIO_EXTS:
            return p
    return None


def remove_partials(audios_dir: Path, video_id: str) -> None:
    for p in audios_dir.glob(f'{video_id}.*'):
        try:
            p.unlink()
        except OSError:
            pass


def check_file(path: Path, expected_sec: float, config: dict) -> tuple[bool, str, float]:
    dcfg = config['download']
    size = path.stat().st_size
    if size < to_float(dcfg.get('min_size_kb'), 100) * 1024:
        return False, f'file too small ({size / 1024:.0f} KB)', 0.0
    ext = path.suffix.lower().lstrip('.')
    if ext in MAGIC_BYTES:
        offset, magic = MAGIC_BYTES[ext]
        with open(path, 'rb') as f:
            f.seek(offset)
            if f.read(len(magic)) != magic:
                return False, f'bad file header for .{ext}', 0.0
    actual = probe_duration(path)
    if actual is None:
        return False, 'cannot read the audio duration (file broken or not audio)', 0.0
    if expected_sec > 0:
        tolerance = max(to_float(dcfg.get('duration_tolerance_sec'), 5),
                        expected_sec * to_float(dcfg.get('duration_tolerance_pct'), 2) / 100)
        if abs(actual - expected_sec) > tolerance:
            return False, f'duration {actual:.1f}s vs expected {expected_sec:.1f}s (tolerance {tolerance:.1f}s)', actual
    return True, '', actual


# ── channel folders ─────────────────────────────────────────────────────────

class ChannelFolders:
    """channel_id -> folder name, read from and written to reference.xlsx."""

    def __init__(self, ref: Reference):
        self.by_channel: dict[str, str] = {}
        for _, row in ref.df.iterrows():
            if row['channel_id'] and row['channel_folder'] and row['channel_id'] not in self.by_channel:
                self.by_channel[str(row['channel_id'])] = str(row['channel_folder'])
        self.used = {f.lower() for f in self.by_channel.values()}

    def folder_for(self, channel_id: str, channel_name: str) -> str:
        key = channel_id or f'name:{channel_name}'
        if key in self.by_channel:
            return self.by_channel[key]
        base = safe_folder_name(channel_name or channel_id)
        name = base
        if name.lower() in self.used or name.startswith('_'):
            name = f'{base}_{(channel_id or "x")[-6:]}'
        n = 2
        while name.lower() in self.used:
            name = f'{base}_{n}'
            n += 1
        self.by_channel[key] = name
        self.used.add(name.lower())
        return name


# ── prerequisite / run ─────────────────────────────────────────────────────

def prerequisite(config: dict, **options) -> tuple[str, str]:
    if not PATHS['deduplicated'].exists():
        return 'fail', f'missing {PATHS["deduplicated"]} — run dedup_url first'
    n = len(read_excel(PATHS['deduplicated']))
    if n == 0 and not options.get('retry_failed'):
        return 'skip', 'deduplicated.xlsx has no rows'
    return 'ok', f'{n} row(s) in deduplicated.xlsx'


def build_jobs(ref: Reference, retry_failed: bool) -> list[dict]:
    jobs = []
    for _, row in read_excel(PATHS['deduplicated']).iterrows():
        if row['video_id'] and row['video_id'] not in ref:
            jobs.append(row.to_dict())
    if retry_failed:
        for _, row in ref.rows(download_status='failed').iterrows():
            jobs.append(row.to_dict())
    # smaller channels first (total expected hours in this batch), keep order inside a channel
    totals: dict[str, float] = {}
    for j in jobs:
        totals[j.get('channel_id') or j.get('channel_name', '')] = \
            totals.get(j.get('channel_id') or j.get('channel_name', ''), 0) + to_float(j.get('duration_sec'))
    order = {k: i for i, k in enumerate(sorted(totals, key=lambda k: totals[k]))}
    jobs.sort(key=lambda j: order[j.get('channel_id') or j.get('channel_name', '')])
    return jobs


def folder_size_bytes(folder: Path) -> int:
    total = 0
    for dirpath, _, files in os.walk(folder):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return total


def process_video(job: dict, folder: str, config: dict) -> dict:
    """Download + check one video. Returns the reference fields to write."""
    vid = job['video_id']
    channel_dir = PATHS['downloads'] / folder
    audios_dir = channel_dir / 'audios'
    audios_dir.mkdir(parents=True, exist_ok=True)
    expected = to_float(job.get('duration_sec'))

    existing = find_audio(audios_dir, vid)       # left over from a crash before the reference was saved
    if existing is not None:
        ok, _, actual = check_file(existing, expected, config)
        if ok:
            log.info('  already on disk, verified: %s', existing.name)
            return {'ok': True, 'path': existing, 'actual': actual}
    remove_partials(audios_dir, vid)

    try:
        yt_download(job.get('url') or f'https://www.youtube.com/watch?v={vid}', audios_dir, vid, config)
    except Exception as exc:
        remove_partials(audios_dir, vid)
        reason = classify_error(exc)
        return {'ok': False, 'reason': reason, 'detail': str(exc).replace('\n', ' ')[:300]}

    path = find_audio(audios_dir, vid)
    if path is None:
        remove_partials(audios_dir, vid)
        return {'ok': False, 'reason': 'download_error', 'detail': 'no audio file after download'}
    ok, problem, actual = check_file(path, expected, config)
    if not ok:
        remove_partials(audios_dir, vid)
        return {'ok': False, 'reason': 'check_failed', 'detail': problem}
    return {'ok': True, 'path': path, 'actual': actual}


def run(config: dict, **options) -> dict:
    retry_failed = bool(options.get('retry_failed'))
    limit = options.get('limit')
    dcfg = config['download']
    fp_cfg = config.get('fingerprint') or {}
    save_every = max(1, to_int(dcfg.get('save_every'), 10))
    max_bytes = to_float(dcfg.get('max_gb'), 500) * 1024 ** 3
    require_ffmpeg()          # stop BEFORE downloading, instead of marking every video failed

    backup_reference()
    ref = Reference()
    fingerprints = load_fingerprints()
    folders = ChannelFolders(ref)
    jobs = build_jobs(ref, retry_failed)
    if limit:
        jobs = jobs[:int(limit)]
    log.info('%d video(s) to download%s', len(jobs), ' (incl. retries)' if retry_failed else '')
    if not jobs:
        return {'status': 'skipped', 'rows_in': 0, 'rows_out': 0, 'message': 'nothing new to download'}

    PATHS['downloads'].mkdir(parents=True, exist_ok=True)
    used_bytes = folder_size_bytes(PATHS['downloads'])
    counts = {'downloaded': 0, 'failed': 0, 'retry_later': 0}
    consecutive_session_errors = 0
    pending_save = 0

    def save():
        ref.save()
        save_fingerprints(fingerprints)

    try:
        for n, job in enumerate(jobs, 1):
            if used_bytes >= max_bytes:
                log.warning('storage cap of %.0f GB reached — stopping', max_bytes / 1024 ** 3)
                break
            vid = job['video_id']
            folder = folders.folder_for(str(job.get('channel_id') or ''), str(job.get('channel_name') or ''))
            log.info('[%d/%d] %s  %s / %s', n, len(jobs), vid, folder, str(job.get('title'))[:70])
            result = process_video(job, folder, config)

            identity = {
                'url': job.get('url', ''), 'title': job.get('title', ''), 'upload_date': job.get('upload_date', ''),
                'duration_sec': to_float(job.get('duration_sec')), 'view_count': to_int(job.get('view_count')),
                'is_short': to_int(job.get('is_short')), 'has_captions': to_int(job.get('has_captions')),
                'channel_id': job.get('channel_id', ''), 'channel_name': job.get('channel_name', ''),
                'channel_folder': folder, 'dialect': job.get('dialect', ''),
            }
            if result['ok']:
                consecutive_session_errors = 0
                path: Path = result['path']
                size = path.stat().st_size
                used_bytes += size
                meta = {k: job.get(k) for k in job}
                meta.update({'channel_folder': folder, 'file_path': rel_path(path), 'file_size_mb': round(size / 1048576, 2),
                             'actual_duration_sec': round(result['actual'], 2), 'downloaded_at': now_iso()})
                meta_dir = PATHS['downloads'] / folder / 'metadata'
                meta_dir.mkdir(parents=True, exist_ok=True)
                (meta_dir / f'{vid}.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str),
                                                     encoding='utf-8')
                try:
                    fingerprints[vid] = compute_fingerprints(path, result['actual'], to_float(fp_cfg.get('window_sec'), 60),
                                                             fp_cfg.get('tool', 'auto'))
                except Exception as exc:     # not fatal: dedup_audio computes missing fingerprints
                    log.warning('  fingerprint failed (dedup_audio will retry): %s', exc)
                ref.upsert(vid, **identity, download_status='downloaded', fail_reason='', file_path=rel_path(path),
                           file_size_mb=round(size / 1048576, 2), actual_duration_sec=round(result['actual'], 2),
                           downloaded_at=now_iso(), dedup_status='pending', duplicate_of='')
                counts['downloaded'] += 1
                log.info('  ok  %.1f MB, %.0f s', size / 1048576, result['actual'])
            elif result['reason'] in SESSION_ERRORS:
                consecutive_session_errors += 1
                counts['retry_later'] += 1
                log.warning('  %s (not recorded, will retry next run): %s', result['reason'], result['detail'])
                if consecutive_session_errors >= MAX_CONSECUTIVE_SESSION_ERRORS:
                    raise SessionError(f'{consecutive_session_errors} session errors in a row ({result["reason"]}). '
                                       'YouTube is blocking this session: set youtube.cookies_file in config, '
                                       'or wait and run again.')
                continue
            else:
                consecutive_session_errors = 0
                ref.upsert(vid, **identity, download_status='failed', fail_reason=f"{result['reason']}: {result['detail']}",
                           file_path='', file_size_mb=float('nan'), actual_duration_sec=float('nan'),
                           downloaded_at=now_iso(), dedup_status='', duplicate_of='')
                counts['failed'] += 1
                log.warning('  FAILED %s: %s', result['reason'], result['detail'])
            pending_save += 1
            if pending_save >= save_every:
                save()
                pending_save = 0
    finally:
        save()

    log.info('download summary: %s', counts)
    return {'status': 'done', 'rows_in': len(jobs), 'rows_out': counts['downloaded'],
            'message': f"{counts['downloaded']} downloaded, {counts['failed']} failed, "
                       f"{counts['retry_later']} to retry next run"}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default=None)
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    setup_logging(STEP)
    cfg = load_config(args.config)
    state, msg = prerequisite(cfg, retry_failed=args.retry_failed)
    log.info('prerequisite: %s — %s', state, msg)
    if state == 'fail':
        sys.exit(1)
    if state == 'ok':
        print(run(cfg, retry_failed=args.retry_failed, limit=args.limit))
