"""Step 2 — dedup_url.

INPUT   scout/output/scouted.xlsx
        reference/reference.xlsx          (every video ever downloaded, failed or marked duplicate)

OUTPUT  dedup_url/output/deduplicated.xlsx   only videos never seen before
        dedup_url/removed_url.xlsx           every removed row, with the reason

Rule: every URL form (watch?v=, youtu.be/, /shorts/, mobile, tracking params...)
is reduced to the 11-character video ID and compared on that. A row is removed if
  - its video_id is already in reference.xlsx (whatever its status)
  - it appears twice in the same batch (the first one is kept)
  - no video ID can be extracted (excluded and flagged, never kept silently)

Run alone:  python dedup_url/dedup_url.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common.reference_io import load_reference
from common.utils import PATHS, get_logger, load_config, read_excel, setup_logging, write_excel
from common.video_id import extract_video_id

STEP = 'dedup_url'
log = get_logger(STEP)


def prerequisite(config: dict, **options) -> tuple[str, str]:
    if not PATHS['scouted'].exists():
        return 'fail', f'missing {PATHS["scouted"]} — run scout first'
    n = len(read_excel(PATHS['scouted']))
    if n == 0:
        return 'skip', 'scouted.xlsx has no rows'
    return 'ok', f'{n} scouted row(s)'


def run(config: dict, **options) -> dict:
    scouted = read_excel(PATHS['scouted'])
    reference = load_reference()
    status_by_id = dict(zip(reference['video_id'].astype(str),
                            reference['download_status'].astype(str) + '/' + reference['dedup_status'].astype(str)))
    log.info('%d scouted row(s); reference holds %d video(s)', len(scouted), len(status_by_id))

    ids = [extract_video_id(row.get('video_id')) or extract_video_id(row.get('url')) for _, row in scouted.iterrows()]
    scouted = scouted.assign(video_id=[i or '' for i in ids])

    keep_mask, reasons, details = [], [], []
    seen = set()
    for vid in scouted['video_id']:
        if not vid:
            keep_mask.append(False); reasons.append('no_video_id'); details.append('cannot read a video ID from video_id or url')
        elif vid in status_by_id:
            keep_mask.append(False); reasons.append('already_in_reference'); details.append(f'status {status_by_id[vid]}')
        elif vid in seen:
            keep_mask.append(False); reasons.append('duplicate_in_batch'); details.append('same video ID earlier in scouted.xlsx')
        else:
            keep_mask.append(True); reasons.append(''); details.append('')
            seen.add(vid)

    keep_mask = pd.Series(keep_mask, index=scouted.index)
    deduplicated = scouted[keep_mask].reset_index(drop=True)
    removed = scouted[~keep_mask].copy()
    removed.insert(0, 'reason', [r for r, k in zip(reasons, keep_mask) if not k])
    removed.insert(1, 'detail', [d for d, k in zip(details, keep_mask) if not k])

    write_excel(deduplicated, PATHS['deduplicated'])
    write_excel(removed.reset_index(drop=True), PATHS['removed_url'])
    counts = removed['reason'].value_counts().to_dict()
    log.info('deduplicated.xlsx: %d new video(s) | removed %d %s', len(deduplicated), len(removed), counts)
    return {'status': 'done', 'rows_in': len(scouted), 'rows_out': len(deduplicated),
            'message': f'{len(deduplicated)} new, {len(removed)} removed {counts}'}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default=None)
    args = parser.parse_args()
    setup_logging(STEP)
    cfg = load_config(args.config)
    state, msg = prerequisite(cfg)
    log.info('prerequisite: %s — %s', state, msg)
    if state == 'fail':
        sys.exit(1)
    if state == 'ok':
        print(run(cfg))
