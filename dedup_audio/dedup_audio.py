"""Step 4 — dedup_audio.

The problem: the same recording re-uploaded under a DIFFERENT video ID passes the
URL check. This step finds those by comparing audio fingerprints.

INPUT   reference/reference.xlsx      rows with dedup_status = pending
        reference/fingerprints.parquet  (fingerprints of OLD videos too, even if their
                                         audio was pushed and deleted locally)

OUTPUT  reference.xlsx: dedup_status pending -> unique | duplicate, duplicate_of = kept video_id
        dedup_audio/duplicates.xlsx: every duplicate cluster found (appended each run)
        No audio is touched, unless --move (moves duplicates to downloads/_duplicates/,
        a rename on the same disk, not a copy).

How it compares
  - only videos whose durations differ by <= dedup_audio.duration_tolerance_sec
  - only pairs involving at least one pending video (old unique-vs-unique pairs were judged before)
  - duplicate only if the fingerprints match at the START, the MIDDLE and the END
    (a shared intro jingle alone is not enough), allowing a small time shift
  - which copy is kept: the one already kept earlier, else the highest view count,
    else the earliest upload date

Run alone:  python dedup_audio/dedup_audio.py [--move]
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common.fingerprint import compare, compute_fingerprints
from common.reference_io import Reference, backup_reference, load_fingerprints, save_fingerprints
from common.utils import (PATHS, abs_path, get_logger, load_config, now_iso, read_excel, rel_path, setup_logging,
                          to_float, to_int, write_excel)

STEP = 'dedup_audio'
log = get_logger(STEP)


def prerequisite(config: dict, **options) -> tuple[str, str]:
    if not PATHS['reference'].exists():
        return 'fail', 'reference.xlsx does not exist yet — run download first'
    ref = Reference()
    n = len(ref.rows(dedup_status='pending'))
    if n == 0:
        return 'skip', 'no video with dedup_status = pending'
    return 'ok', f'{n} pending video(s)'


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def keep_order(row: dict):
    """Sort key: highest views first, then earliest upload date, then video_id."""
    return (-to_int(row.get('view_count')), str(row.get('upload_date') or '9999'), str(row['video_id']))


def run(config: dict, **options) -> dict:
    move = bool(options.get('move'))
    dcfg = config['dedup_audio']
    fp_cfg = config.get('fingerprint') or {}
    tolerance = to_float(dcfg.get('duration_tolerance_sec'), 3)
    threshold = to_float(dcfg.get('similarity_threshold'), 0.85)
    max_offset = to_float(dcfg.get('max_offset_sec'), 3)
    run_at = now_iso()

    backup_reference()
    ref = Reference()
    fps = load_fingerprints()

    pending = ref.rows(dedup_status='pending')
    log.info('%d pending video(s); %d fingerprint(s) stored', len(pending), len(fps))

    # 1. fingerprints missing for pending videos (e.g. download could not compute them)
    no_fp = []
    for _, row in pending.iterrows():
        vid = row['video_id']
        if vid in fps:
            continue
        path = abs_path(row['file_path']) if row['file_path'] else None
        if path is None or not path.exists():
            no_fp.append(vid)
            continue
        try:
            duration = to_float(row['actual_duration_sec']) or to_float(row['duration_sec'])
            fps[vid] = compute_fingerprints(path, duration, to_float(fp_cfg.get('window_sec'), 60), fp_cfg.get('tool', 'auto'))
            log.info('  computed missing fingerprint: %s', vid)
        except Exception as exc:
            log.warning('  cannot fingerprint %s: %s', vid, exc)
            no_fp.append(vid)
    if no_fp:
        log.warning('%d pending video(s) have no fingerprint and no audio — left pending: %s', len(no_fp), no_fp[:10])

    # 2. candidates: pending + unique, with a fingerprint
    cand = ref.df[ref.df['dedup_status'].isin(['pending', 'unique']) & ref.df['video_id'].isin(list(fps))]
    records = []
    for _, row in cand.iterrows():
        rec = row.to_dict()
        rec['_dur'] = to_float(row['actual_duration_sec']) or to_float(row['duration_sec']) or fps[row['video_id']]['duration_sec']
        rec['_pending'] = row['dedup_status'] == 'pending'
        records.append(rec)
    records.sort(key=lambda r: r['_dur'])

    uf = UnionFind()
    pair_scores = {}
    comparisons = 0
    for i, a in enumerate(records):
        j = i + 1
        while j < len(records) and records[j]['_dur'] - a['_dur'] <= tolerance:
            b = records[j]
            j += 1
            if not (a['_pending'] or b['_pending']):
                continue
            scores = compare(fps[a['video_id']], fps[b['video_id']], max_offset)
            comparisons += 1
            if min(scores.values()) >= threshold:
                uf.union(a['video_id'], b['video_id'])
                pair_scores[frozenset((a['video_id'], b['video_id']))] = scores
    log.info('%d comparison(s) within ±%.0fs duration windows', comparisons, tolerance)

    # 3. clusters -> statuses
    clusters: dict[str, list[dict]] = {}
    for rec in records:
        clusters.setdefault(uf.find(rec['video_id']), []).append(rec)

    report_rows = []
    counts = {'unique': 0, 'duplicate': 0, 'clusters': 0}
    cluster_n = 0
    for members in clusters.values():
        if len(members) == 1:
            rec = members[0]
            if rec['_pending']:
                ref.upsert(rec['video_id'], dedup_status='unique', duplicate_of='')
                counts['unique'] += 1
            continue
        cluster_n += 1
        counts['clusters'] += 1
        cluster_id = f"{run_at[:19].replace(':', '').replace('-', '')}-{cluster_n:03d}"
        earlier = sorted([m for m in members if not m['_pending']], key=keep_order)
        keeper = earlier[0] if earlier else sorted(members, key=keep_order)[0]
        rule = 'kept earlier' if earlier else 'highest views / earliest upload'
        for m in members:
            vid = m['video_id']
            if vid == keeper['video_id']:
                role = 'keep'
                if m['_pending']:
                    ref.upsert(vid, dedup_status='unique', duplicate_of='')
                    counts['unique'] += 1
            elif m['_pending']:
                role = 'duplicate'
                ref.upsert(vid, dedup_status='duplicate', duplicate_of=keeper['video_id'])
                counts['duplicate'] += 1
                if move and m['file_path']:
                    src = abs_path(m['file_path'])
                    if src.exists():
                        dest = PATHS['downloads'] / '_duplicates' / m['channel_folder'] / 'audios' / src.name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(src), str(dest))
                        ref.upsert(vid, file_path=rel_path(dest))
            else:
                role = 'unique_earlier_unchanged'   # two old "unique" videos linked through a new one
            direct = pair_scores.get(frozenset((vid, keeper['video_id'])))
            if direct is None and vid != keeper['video_id']:
                direct = compare(fps[vid], fps[keeper['video_id']], max_offset)
                linked = 'via another member' if min(direct.values()) < threshold else 'direct'
            else:
                linked = 'direct' if vid != keeper['video_id'] else ''
            report_rows.append({
                'run_at': run_at, 'cluster_id': cluster_id, 'video_id': vid, 'role': role,
                'duplicate_of': '' if vid == keeper['video_id'] else keeper['video_id'], 'keep_rule': rule,
                'sim_start': round(direct['fp_start'], 3) if direct else '',
                'sim_middle': round(direct['fp_middle'], 3) if direct else '',
                'sim_end': round(direct['fp_end'], 3) if direct else '',
                'match': linked, 'duration_sec': round(m['_dur'], 1), 'view_count': to_int(m.get('view_count')),
                'upload_date': m.get('upload_date', ''), 'channel_name': m.get('channel_name', ''),
                'title': m.get('title', ''), 'url': m.get('url', ''),
            })

    ref.save()
    save_fingerprints(fps)
    if report_rows:
        new = pd.DataFrame(report_rows)
        if PATHS['duplicates_report'].exists():
            new = pd.concat([read_excel(PATHS['duplicates_report']), new.astype(str)], ignore_index=True)
        write_excel(new, PATHS['duplicates_report'])

    log.info('dedup_audio: %d cluster(s), %d marked duplicate, %d marked unique%s', counts['clusters'],
             counts['duplicate'], counts['unique'], ' (duplicates moved to downloads/_duplicates/)' if move else '')
    return {'status': 'done', 'rows_in': len(pending), 'rows_out': counts['unique'],
            'message': f"{counts['unique']} unique, {counts['duplicate']} duplicate, {len(no_fp)} left pending"}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default=None)
    parser.add_argument('--move', action='store_true', help='move duplicate audio to downloads/_duplicates/')
    args = parser.parse_args()
    setup_logging(STEP)
    cfg = load_config(args.config)
    state, msg = prerequisite(cfg)
    log.info('prerequisite: %s — %s', state, msg)
    if state == 'fail':
        sys.exit(1)
    if state == 'ok':
        print(run(cfg, move=args.move))
