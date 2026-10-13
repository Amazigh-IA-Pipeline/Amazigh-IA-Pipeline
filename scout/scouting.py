"""Step 1 — scout.

INPUT   scout/input/*.xlsx — one row per source, columns:
            channel_url | playlist_url | video_url | short_url | dialect
        Fill ONE of the four URL columns per row. The dialect is copied onto
        every video found for that row.

OUTPUT  scout/output/scouted.xlsx        one row per video (replaced every run)
        scout/output/scout_removed.xlsx  every skipped video / row, with the reason
        scout/output/cache/<video_id>.json  per-video metadata cache, so an
                                            interrupted run does not refetch everything

What it does
  channel  -> lists the Videos, Live/Streams and Shorts tabs, then fetches full metadata per video
  playlist -> lists its videos; each video is credited to its REAL uploading channel
  video / short -> fetches the metadata directly

Skipped (and logged): live/upcoming, banned channels, likely music (title only),
shorter than scout.min_duration_sec, metadata errors, repeats.

Run alone:  python scout/scouting.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common.utils import (PATHS, get_logger, load_config, read_excel, setup_logging, to_float, to_int,
                          write_excel, ytdlp_base_options)
from common.video_id import extract_video_id, watch_url

STEP = 'scout'
log = get_logger(STEP)

URL_COLUMNS = ['channel_url', 'playlist_url', 'video_url', 'short_url']
SOURCE_NAMES = {'channel_url': 'channel', 'playlist_url': 'playlist', 'video_url': 'video', 'short_url': 'short'}

OUTPUT_COLUMNS = ['video_id', 'url', 'title', 'description', 'upload_date', 'duration_sec', 'view_count',
                  'like_count', 'channel_id', 'channel_name', 'channel_url', 'dialect', 'is_short',
                  'has_captions', 'caption_languages', 'source', 'dialect_conflict', 'input_file']
REMOVED_COLUMNS = ['input_file', 'input_row', 'source', 'source_url', 'video_id', 'title', 'channel_name',
                   'reason', 'detail']


# ── YouTube access (the only functions that touch the network) ─────────────

def yt_extract(url: str, config: dict, flat: bool) -> dict:
    import yt_dlp
    opts = ytdlp_base_options(config)
    opts['skip_download'] = True
    if flat:
        opts['extract_flat'] = 'in_playlist'
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=False)


def _flatten_entries(info: dict) -> list[dict]:
    out = []
    for entry in info.get('entries') or []:
        if not entry:
            continue
        if entry.get('_type') == 'playlist' or (entry.get('entries') and not entry.get('duration')):
            out.extend(_flatten_entries(entry))
        else:
            out.append(entry)
    return out


def channel_base_url(url: str) -> str:
    url = url.strip().rstrip('/')
    if not re.match(r'^https?://', url, re.I):
        url = 'https://' + url
    return re.sub(r'/(videos|shorts|streams|featured|playlists|live|community|about)$', '', url)


def list_channel(url: str, config: dict) -> tuple[dict, list[dict]]:
    """Returns (channel_info, [{'video_id', 'tab', 'live_status'}])."""
    base = channel_base_url(url)
    channel = {'channel_id': '', 'channel_name': '', 'channel_url': base}
    videos = []
    for tab in config['scout'].get('tabs') or ['videos', 'streams', 'shorts']:
        try:
            info = yt_extract(f'{base}/{tab}', config, flat=True)
        except Exception as exc:  # a channel without a Shorts/Live tab raises here
            if 'does not have' in str(exc).lower():
                log.info('  %s tab: none', tab)
                continue
            raise
        channel['channel_id'] = channel['channel_id'] or info.get('channel_id') or info.get('id') or ''
        channel['channel_name'] = channel['channel_name'] or info.get('channel') or info.get('uploader') or ''
        entries = _flatten_entries(info)
        log.info('  %s tab: %d video(s)', tab, len(entries))
        for entry in entries:
            vid = extract_video_id(entry.get('id')) or extract_video_id(entry.get('url'))
            videos.append({'video_id': vid, 'tab': tab, 'live_status': entry.get('live_status') or '',
                           'url': entry.get('url') or ''})
    return channel, videos


def list_playlist(url: str, config: dict) -> list[dict]:
    info = yt_extract(url, config, flat=True)
    return [{'video_id': extract_video_id(e.get('id')) or extract_video_id(e.get('url')),
             'tab': '', 'live_status': e.get('live_status') or '', 'url': e.get('url') or ''}
            for e in _flatten_entries(info)]


def fetch_video(video_id: str, config: dict) -> dict:
    info = yt_extract(watch_url(video_id), config, flat=False)
    subs = sorted((info.get('subtitles') or {}).keys())
    auto = sorted((info.get('automatic_captions') or {}).keys())
    return {
        'video_id': info.get('id') or video_id,
        'title': info.get('title') or '',
        'description': info.get('description') or '',
        'upload_date': info.get('upload_date') or '',
        'duration_sec': info.get('duration'),
        'view_count': info.get('view_count'),
        'like_count': info.get('like_count'),
        'channel_id': info.get('channel_id') or '',
        'channel_name': info.get('channel') or info.get('uploader') or '',
        'channel_url': info.get('channel_url') or '',
        'live_status': info.get('live_status') or '',
        'has_captions': bool(subs or auto),
        'caption_languages': ', '.join(subs + [f'{lang}(auto)' for lang in auto if lang not in subs][:5]),
    }


# ── helpers ─────────────────────────────────────────────────────────────────

def input_files() -> list[Path]:
    folder = PATHS['scout_input']
    if not folder.exists():
        return []
    return sorted(p for p in folder.glob('*.xlsx') if not p.name.startswith('~$'))


def prerequisite(config: dict, **options) -> tuple[str, str]:
    files = input_files()
    if not files:
        return 'fail', f'no .xlsx file in {PATHS["scout_input"]}'
    return 'ok', f'{len(files)} input file(s)'


def read_sources(removed: list) -> list[dict]:
    sources = []
    for path in input_files():
        df = read_excel(path)
        df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
        if not any(c in df.columns for c in URL_COLUMNS):
            removed.append(_removed(path.name, '', '', '', reason='invalid_input_file',
                                    detail=f'none of the columns {URL_COLUMNS} found'))
            continue
        for i, row in df.iterrows():
            filled = [c for c in URL_COLUMNS if c in df.columns and row.get(c)]
            excel_row = i + 2  # header is row 1
            if not filled:
                continue  # empty row
            if len(filled) > 1:
                removed.append(_removed(path.name, excel_row, '', ', '.join(row[c] for c in filled),
                                        reason='invalid_row', detail='more than one URL column filled'))
                continue
            col = filled[0]
            sources.append({'input_file': path.name, 'input_row': excel_row, 'source': SOURCE_NAMES[col],
                            'url': row[col], 'dialect': row.get('dialect', '')})
    return sources


def _removed(input_file, input_row, source, source_url, video_id='', title='', channel_name='', reason='',
             detail=''):
    return {'input_file': input_file, 'input_row': input_row, 'source': source, 'source_url': source_url,
            'video_id': video_id, 'title': title, 'channel_name': channel_name, 'reason': reason,
            'detail': str(detail)[:500]}


def _keyword_matches(keyword: str, text: str) -> bool:
    kw = keyword.lower()
    if re.search(r'[؀-ۿ]', kw):         # Arabic script: substring (prefixes like ال attach)
        return kw in text
    return re.search(r'(?<![a-z؀-ۿ])' + re.escape(kw) + r'(?![a-z؀-ۿ])', text) is not None


def music_keywords_in(title: str, keywords: list[str]) -> list[str]:
    text = str(title or '').lower()
    return [kw for kw in keywords if _keyword_matches(kw, text)]


def banned_reason(meta: dict, banned: dict) -> str:
    if meta.get('channel_id') and meta['channel_id'] in banned['channel_ids']:
        return f"channel_id {meta['channel_id']}"
    url = str(meta.get('channel_url') or '').rstrip('/').lower()
    if url and url in banned['channel_urls']:
        return f'channel_url {url}'
    name = str(meta.get('channel_name') or '').strip().lower()
    if name and name in banned['channel_names']:
        return f'channel_name {meta.get("channel_name")}'
    return ''


def format_date(value) -> str:
    text = str(value or '')
    if re.fullmatch(r'\d{8}', text):
        return f'{text[:4]}-{text[4:6]}-{text[6:]}'
    return text


def cached_fetch(video_id: str, config: dict) -> tuple[dict, bool]:
    cache = PATHS['scout_cache'] / f'{video_id}.json'
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding='utf-8')), True
        except Exception:
            pass
    meta = fetch_video(video_id, config)
    if meta.get('live_status') not in ('is_live', 'is_upcoming'):   # a live may end: don't cache it
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')
    return meta, False


# ── main ────────────────────────────────────────────────────────────────────

def run(config: dict, **options) -> dict:
    scfg = config['scout']
    min_dur = to_float(scfg.get('min_duration_sec'), 0)
    delay = to_float(scfg.get('delay_sec'), 0.3)
    removed: list[dict] = []
    kept: dict[str, dict] = {}            # video_id -> output row (insertion order = first seen)

    sources = read_sources(removed)
    log.info('%d source row(s) in %d file(s)', len(sources), len(input_files()))

    for src in sources:
        ctx = (src['input_file'], src['input_row'], src['source'], src['url'])
        log.info('[%s] %s  (%s)', src['source'], src['url'], src['input_file'])

        # 1. list the videos of this source
        try:
            if src['source'] == 'channel':
                channel, items = list_channel(src['url'], config)
                reason = banned_reason(channel, config['banned'])
                if reason:
                    removed.append(_removed(*ctx, channel_name=channel['channel_name'], reason='banned_channel',
                                            detail=f'{reason} — whole channel skipped ({len(items)} videos)'))
                    continue
            elif src['source'] == 'playlist':
                items = list_playlist(src['url'], config)
            else:
                vid = extract_video_id(src['url'])
                if not vid:
                    removed.append(_removed(*ctx, reason='no_video_id', detail='cannot read a video ID from the URL'))
                    continue
                items = [{'video_id': vid, 'tab': 'shorts' if src['source'] == 'short' else '', 'live_status': '',
                          'url': src['url']}]
        except Exception as exc:
            removed.append(_removed(*ctx, reason='listing_error', detail=exc))
            log.warning('  listing failed: %s', exc)
            continue

        # 2. per video: filter, fetch metadata, filter again
        fetched = 0
        for n, item in enumerate(items, 1):
            vid = item['video_id']
            if not vid:
                removed.append(_removed(*ctx, reason='no_video_id', detail=item.get('url')))
                continue
            if vid in kept:
                row = kept[vid]
                if src['dialect'] and row['dialect'] != src['dialect']:
                    row['dialect_conflict'] = True
                    detail = f"kept dialect '{row['dialect']}', this row says '{src['dialect']}'"
                else:
                    detail = 'already found earlier in this run'
                removed.append(_removed(*ctx, video_id=vid, title=row['title'], channel_name=row['channel_name'],
                                        reason='repeat_in_scout', detail=detail))
                continue
            if item.get('live_status') in ('is_live', 'is_upcoming'):
                removed.append(_removed(*ctx, video_id=vid, reason=item['live_status'].replace('is_', ''),
                                        detail='live or upcoming'))
                continue
            try:
                meta, from_cache = cached_fetch(vid, config)
                if not from_cache:
                    fetched += 1
                    time.sleep(delay)
            except Exception as exc:
                removed.append(_removed(*ctx, video_id=vid, reason='metadata_error', detail=exc))
                continue

            info = (ctx[0], ctx[1], ctx[2], ctx[3], vid, meta['title'], meta['channel_name'])
            if meta.get('live_status') in ('is_live', 'is_upcoming'):
                removed.append(_removed(*info, reason=meta['live_status'].replace('is_', ''), detail='live or upcoming'))
                continue
            reason = banned_reason(meta, config['banned'])
            if reason:
                removed.append(_removed(*info, reason='banned_channel', detail=reason))
                continue
            hits = music_keywords_in(meta['title'], config['music_keywords'])
            if hits:
                removed.append(_removed(*info, reason='likely_music', detail='title keywords: ' + ' | '.join(hits)))
                continue
            duration = to_float(meta.get('duration_sec'), 0)
            if duration <= 0:
                removed.append(_removed(*info, reason='unknown_duration', detail='YouTube gave no duration'))
                continue
            if duration < min_dur:
                removed.append(_removed(*info, reason='too_short', detail=f'{duration:.0f}s < {min_dur:.0f}s'))
                continue

            is_short = src['source'] == 'short' or item.get('tab') == 'shorts' or '/shorts/' in str(item.get('url'))
            kept[vid] = {
                'video_id': vid, 'url': watch_url(vid), 'title': meta['title'], 'description': meta['description'],
                'upload_date': format_date(meta['upload_date']), 'duration_sec': duration,
                'view_count': to_int(meta.get('view_count')), 'like_count': to_int(meta.get('like_count')),
                'channel_id': meta['channel_id'], 'channel_name': meta['channel_name'],
                'channel_url': meta['channel_url'], 'dialect': src['dialect'], 'is_short': int(is_short),
                'has_captions': int(bool(meta.get('has_captions'))), 'caption_languages': meta['caption_languages'],
                'source': src['source'], 'dialect_conflict': False, 'input_file': src['input_file'],
            }
            if n % 25 == 0:
                log.info('  %d/%d checked', n, len(items))
        log.info('  done: %d listed, %d fetched from YouTube', len(items), fetched)

    scouted = pd.DataFrame(list(kept.values()), columns=OUTPUT_COLUMNS)
    removed_df = pd.DataFrame(removed, columns=REMOVED_COLUMNS)
    write_excel(scouted, PATHS['scouted'])
    write_excel(removed_df, PATHS['scout_removed'])

    by_reason = removed_df['reason'].value_counts().to_dict() if len(removed_df) else {}
    conflicts = int(scouted['dialect_conflict'].sum()) if len(scouted) else 0
    log.info('scouted.xlsx: %d video(s), %.1f h | removed: %d %s | dialect conflicts: %d',
             len(scouted), scouted['duration_sec'].sum() / 3600 if len(scouted) else 0, len(removed_df),
             by_reason, conflicts)
    return {'status': 'done', 'rows_in': len(sources), 'rows_out': len(scouted),
            'message': f'{len(scouted)} video(s) scouted, {len(removed_df)} removed'}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default=None)
    args = parser.parse_args()
    setup_logging(STEP)
    cfg = load_config(args.config)
    state, msg = prerequisite(cfg)
    if state == 'fail':
        log.error(msg)
        sys.exit(1)
    print(run(cfg))
