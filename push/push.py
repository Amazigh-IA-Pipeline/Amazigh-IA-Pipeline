"""Step 6 — push.

Command:  python push/push.py --source downloads|segments|both [--delete-local]

INPUT   downloads : rows downloaded + unique, audio_pushed_at empty    -> raw audio files
        segments  : rows segment_status = done, segments_pushed_at empty -> segments/<channel>/<video_id>/

OUTPUT  private Hugging Face dataset repos <org>/<prefix>-audio-001, -002 ...
                                       and <org>/<prefix>-segments-001, -002 ...
        reference.xlsx: audio_repo / audio_pushed_at, segments_repo / segments_pushed_at

The 80 GB rule: whole videos are grouped into repos so each stays under push.max_repo_gb
(default 75, leaving room for README/metadata). A video is never split across repos.
When a repo is full, the next one is created.

Segments are packed into parquet shards (~push.shard_mb each) at upload time, because
Hugging Face handles ~1 million small files badly; the parquet carries an HF Audio
feature, so datasets.load_dataset(repo) works directly. The local segments/ folder
stays as individual files. Raw audio is uploaded as is (files are already large).

Each repo also gets a README with real statistics (hours, files, per channel, per
dialect, duration distribution) and one consolidated metadata file
(metadata.csv for audio repos, videos.csv for segments repos).

Safety: after each upload the script lists the files on the hub and checks paths and
sizes against the local data. Only then is *_pushed_at written. --delete-local deletes
local copies of VERIFIED videos only (and never raw audio that still waits for segmenting).

Resumable: only videos with no *_pushed_at are pushed.
push.backend: local writes to push/_local_hub/ instead of Hugging Face (testing).
"""
from __future__ import annotations

import csv
import io
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common.reference_io import Reference, backup_reference
from common.utils import PATHS, abs_path, get_logger, load_config, now_iso, setup_logging, to_float, to_int

STEP = 'push'
log = get_logger(STEP)

KIND = {'downloads': 'audio', 'segments': 'segments'}
REPO_COL = {'downloads': 'audio_repo', 'segments': 'segments_repo'}
PUSHED_COL = {'downloads': 'audio_pushed_at', 'segments': 'segments_pushed_at'}
CLIP_BUCKETS = [(0, 15), (15, 18), (18, 21), (21, 24), (24, 27), (27, 10 ** 9)]
VIDEO_BUCKETS = [(0, 300), (300, 900), (900, 1800), (1800, 3600), (3600, 10 ** 9)]


# ── backends ────────────────────────────────────────────────────────────────

class HFBackend:
    def __init__(self, private: bool, files_per_commit: int):
        from huggingface_hub import HfApi
        self.api = HfApi()
        self.private = private
        self.files_per_commit = files_per_commit

    def ensure_repo(self, repo_id: str):
        self.api.create_repo(repo_id, repo_type='dataset', private=self.private, exist_ok=True)

    def upload(self, repo_id: str, files: list[tuple[Path | bytes, str]], message: str):
        from huggingface_hub import CommitOperationAdd
        for i in range(0, len(files), self.files_per_commit):
            chunk = files[i:i + self.files_per_commit]
            ops = [CommitOperationAdd(path_in_repo=dest, path_or_fileobj=src if isinstance(src, bytes) else str(src))
                   for src, dest in chunk]
            self.api.create_commit(repo_id, operations=ops, commit_message=f'{message} ({i // self.files_per_commit + 1})',
                                   repo_type='dataset')

    def list_files(self, repo_id: str) -> dict[str, int]:
        try:
            items = self.api.list_repo_tree(repo_id, repo_type='dataset', recursive=True)
            return {it.path: int(getattr(it, 'size', 0) or 0) for it in items if hasattr(it, 'size')}
        except Exception as exc:
            if 'not found' in str(exc).lower() or '404' in str(exc):
                return {}
            raise


class LocalBackend:
    """Mimics a hub on disk (push/_local_hub/<org>/<repo>/) to test without Hugging Face."""

    def __init__(self, root: Path):
        self.root = root

    def ensure_repo(self, repo_id: str):
        (self.root / repo_id).mkdir(parents=True, exist_ok=True)

    def upload(self, repo_id: str, files, message: str):
        for src, dest in files:
            target = self.root / repo_id / dest
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(src, bytes):
                target.write_bytes(src)
            else:
                shutil.copy2(src, target)

    def list_files(self, repo_id: str) -> dict[str, int]:
        base = self.root / repo_id
        if not base.exists():
            return {}
        return {p.relative_to(base).as_posix(): p.stat().st_size for p in base.rglob('*') if p.is_file()}


def make_backend(config: dict):
    pcfg = config['push']
    if str(pcfg.get('backend', 'hf')).lower() == 'local':
        return LocalBackend(PATHS['local_hub'])
    return HFBackend(bool(pcfg.get('private', True)), to_int(pcfg.get('files_per_commit'), 100))


# ── what to push ────────────────────────────────────────────────────────────

def candidates(ref: Reference, source: str) -> pd.DataFrame:
    df = ref.df
    if source == 'downloads':
        mask = (df['download_status'] == 'downloaded') & (df['dedup_status'] == 'unique')
    else:
        mask = df['segment_status'] == 'done'
    return df[mask & (df[PUSHED_COL[source]].astype(str) == '')]


def prerequisite(config: dict, **options) -> tuple[str, str]:
    if not PATHS['reference'].exists():
        return 'fail', 'reference.xlsx does not exist yet'
    source = options.get('push_source') or options.get('source') or 'segments'
    sources = ['downloads', 'segments'] if source == 'both' else [source]
    ref = Reference()
    n = sum(len(candidates(ref, s)) for s in sources)
    if n == 0:
        if options.get('delete_local'):
            n_del = sum(len(deletable(ref, s)) for s in sources)
            if n_del:
                return 'ok', f'nothing new to push; {n_del} verified video(s) to delete locally ({source})'
        return 'skip', f'nothing left to push for {source}'
    return 'ok', f'{n} video(s) to push ({source})'


def deletable(ref: Reference, source: str) -> list[dict]:
    """Verified (pushed) videos that still have a local copy that may be deleted."""
    out = []
    for _, row in ref.df[ref.df[PUSHED_COL[source]].astype(str) != ''].iterrows():
        if source == 'downloads':
            if row['segment_status'] == 'pending' and row['dedup_status'] == 'unique':
                continue        # still needed by the segment step
            p = abs_path(row['file_path']) if row['file_path'] else None
            if p is not None and p.exists():
                out.append({'video_id': row['video_id'], 'path': p})
        else:
            d = PATHS['segments'] / str(row['channel_folder']) / str(row['video_id'])
            if d.exists():
                out.append({'video_id': row['video_id'], 'path': d})
    return out


def segment_files(row: dict) -> tuple[Path, list[dict]]:
    video_dir = PATHS['segments'] / row['channel_folder'] / row['video_id']
    lines = []
    meta = video_dir / 'metadata.jsonl'
    if meta.exists():
        with open(meta, 'r', encoding='utf-8') as f:
            lines = [json.loads(l) for l in f if l.strip()]
    return video_dir, [l for l in lines if l.get('kept') and l.get('file_name')]


def local_size(row: dict, source: str) -> int:
    if source == 'downloads':
        p = abs_path(row['file_path'])
        return p.stat().st_size if p.exists() else -1
    video_dir, kept = segment_files(row)
    if not video_dir.exists():
        return -1
    return sum((video_dir / k['file_name']).stat().st_size for k in kept if (video_dir / k['file_name']).exists())


# ── shards ──────────────────────────────────────────────────────────────────

SHARD_FIELDS = ['segment_id', 'video_id', 'channel_id', 'channel_name', 'channel_folder', 'dialect',
                'duration_seconds', 'start_seconds', 'end_seconds', 'speech_ratio', 'speech_confidence',
                'modulation_ratio', 'forced_boundary_review', 'title', 'url', 'upload_date']


def write_shard(videos: list[dict], path: Path) -> int:
    """One parquet shard with an HF Audio column; returns the number of clips."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    audio_col, cols = [], {f: [] for f in SHARD_FIELDS}
    for v in videos:
        for line in v['kept']:
            data = (v['dir'] / line['file_name']).read_bytes()
            audio_col.append({'bytes': data, 'path': line['file_name']})
            for f in SHARD_FIELDS:
                value = line.get(f)
                cols[f].append(None if value is None else (value if f in ('duration_seconds', 'start_seconds', 'end_seconds', 'speech_ratio', 'speech_confidence', 'modulation_ratio', 'forced_boundary_review') else str(value)))
    types = {f: pa.string() for f in SHARD_FIELDS}
    types.update({f: pa.float64() for f in ('duration_seconds', 'start_seconds', 'end_seconds', 'speech_ratio',
                                            'speech_confidence', 'modulation_ratio')})
    types['forced_boundary_review'] = pa.bool_()
    audio_type = pa.struct([('bytes', pa.binary()), ('path', pa.string())])
    arrays = [pa.array(audio_col, type=audio_type)] + [pa.array(cols[f], type=types[f]) for f in SHARD_FIELDS]
    features = {'audio': {'_type': 'Audio'}}
    features.update({f: {'dtype': 'float64' if types[f] == pa.float64() else ('bool' if f == 'forced_boundary_review' else 'string'),
                         '_type': 'Value'} for f in SHARD_FIELDS})
    schema = pa.schema([pa.field('audio', audio_type)] + [pa.field(f, types[f]) for f in SHARD_FIELDS],
                       metadata={'huggingface': json.dumps({'info': {'features': features}})})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_arrays(arrays, schema=schema), path)
    return len(audio_col)


# ── README / consolidated metadata ─────────────────────────────────────────

def _bucket_label(lo, hi, unit):
    if hi >= 10 ** 9:
        return f'{lo}{unit}+'
    return f'{lo}-{hi}{unit}'


def build_readme(repo_id: str, kind: str, rows: pd.DataFrame, clip_hist: dict | None) -> str:
    if kind == 'audio':
        rows = rows.assign(_hours=rows['actual_duration_sec'].map(to_float) / 3600, _files=1)
    else:
        rows = rows.assign(_hours=rows['hours_kept'].map(to_float), _files=rows['n_segments_kept'].map(to_int))
    total_h, total_f = rows['_hours'].sum(), int(rows['_files'].sum())

    def table(col, label):
        g = rows.groupby(rows[col].replace('', '(none)'))[['_hours', '_files']].sum().sort_values('_hours', ascending=False)
        g['videos'] = rows.groupby(rows[col].replace('', '(none)')).size()
        lines = [f'| {label} | videos | files | hours |', '|---|---:|---:|---:|']
        lines += [f'| {k} | {int(r.videos)} | {int(r._files)} | {r._hours:.2f} |' for k, r in g.iterrows()]
        return '\n'.join(lines)

    if kind == 'audio':
        dur = rows['actual_duration_sec'].map(to_float)
        dist = [(_bucket_label(lo // 60, hi // 60 if hi < 10 ** 9 else hi, ' min'),
                 int(((dur >= lo) & (dur < hi)).sum())) for lo, hi in VIDEO_BUCKETS]
        dist_title, files_word, meta_file = 'Video duration distribution', 'audio files', 'metadata.csv'
        yaml_head = ''
    else:
        dist = [(_bucket_label(lo, hi, 's'), int(clip_hist.get(_bucket_label(lo, hi, 's'), 0))) for lo, hi in CLIP_BUCKETS]
        dist_title, files_word, meta_file = 'Clip duration distribution', 'clips', 'videos.csv'
        yaml_head = '---\nconfigs:\n- config_name: default\n  data_files:\n  - split: train\n    path: data/*.parquet\n---\n\n'
    dist_lines = ['| range | count |', '|---|---:|'] + [f'| {k} | {v} |' for k, v in dist]
    usage = ('```python\nfrom datasets import load_dataset\nds = load_dataset("%s", split="train")\n```' % repo_id
             if kind == 'segments' else 'Raw best-quality audio streams from YouTube, not re-encoded, '
             'one file per video under `audio/<channel>/<video_id>.<ext>`.')
    return (f'{yaml_head}# {repo_id}\n\nAmazigh-IA — Moroccan Tamazight speech from YouTube '
            f'({"raw audio" if kind == "audio" else "16 kHz mono speech clips"}).\n\n'
            f'**{total_h:.2f} hours**, **{total_f} {files_word}**, **{len(rows)} videos**, '
            f'**{rows["channel_id"].nunique()} channels**. Updated {now_iso()}.\n\n{usage}\n\n'
            f'Per-video metadata: `{meta_file}`.\n\n## Per dialect\n\n{table("dialect", "dialect")}\n\n'
            f'## Per channel\n\n{table("channel_folder", "channel")}\n\n## {dist_title}\n\n' + '\n'.join(dist_lines) + '\n')


def csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False, quoting=csv.QUOTE_MINIMAL)
    return buf.getvalue().encode('utf-8')


def clip_histogram(rows: pd.DataFrame) -> dict:
    hist = {}
    for col in rows.columns:
        if col.startswith('clips_'):
            hist[col[len('clips_'):]] = int(rows[col].map(to_int).sum())
    return hist


# ── run ─────────────────────────────────────────────────────────────────────

def repo_name(config: dict, kind: str, number: int) -> str:
    pcfg = config['push']
    org = str(pcfg.get('hf_org') or '').strip()
    name = f"{pcfg.get('repo_prefix', 'amazigh-ia')}-{kind}-{number:03d}"
    return f'{org}/{name}' if org else name


def push_source(config: dict, source: str, backend, ref: Reference, delete_local: bool) -> dict:
    kind = KIND[source]
    pcfg = config['push']
    max_bytes = to_float(pcfg.get('max_repo_gb'), 75) * 1024 ** 3
    shard_bytes = to_float(pcfg.get('shard_mb'), 500) * 1024 ** 2
    repo_col, pushed_col = REPO_COL[source], PUSHED_COL[source]

    todo = candidates(ref, source).sort_values(['channel_folder', 'upload_date', 'video_id'])
    log.info('[%s] %d video(s) to push', source, len(todo))

    # current repo = highest number already used; its real size is read from the hub
    used = sorted({r for r in ref.df[repo_col].astype(str) if r})
    number = max([int(r.rsplit('-', 1)[-1]) for r in used if r.rsplit('-', 1)[-1].isdigit()] or [1])
    repo_id = repo_name(config, kind, number)
    repo_used = sum(backend.list_files(repo_id).values())

    # 1. assign whole videos to repos
    batches: dict[str, list[dict]] = {}
    for _, row in todo.iterrows():
        r = row.to_dict()
        size = local_size(r, source)
        if size < 0:
            log.warning('  %s: local data missing, skipped', r['video_id'])
            continue
        if size > max_bytes:
            log.error('  %s: %.1f MB is larger than a whole repo (max_repo_gb), skipped', r['video_id'], size / 1048576)
            continue
        if repo_used + size > max_bytes:
            number += 1
            repo_id = repo_name(config, kind, number)
            repo_used = sum(backend.list_files(repo_id).values())
            log.info('  repo full -> next repo %s', repo_id)
        r['_size'] = size
        batches.setdefault(repo_id, []).append(r)
        repo_used += size

    pushed, failed, deleted = 0, 0, 0
    tag = datetime.now().strftime('%Y%m%d-%H%M%S')
    for repo_id, videos in batches.items():
        backend.ensure_repo(repo_id)
        files: list[tuple] = []
        expected: dict[str, list[tuple[str, int]]] = {}       # video_id -> [(path_in_repo, size)]
        staging = PATHS['push_staging'] / repo_id.replace('/', '__')
        shutil.rmtree(staging, ignore_errors=True)

        if source == 'downloads':
            for v in videos:
                src = abs_path(v['file_path'])
                dest = f"audio/{v['channel_folder']}/{src.name}"
                files.append((src, dest))
                expected[v['video_id']] = [(dest, src.stat().st_size)]
        else:
            shard, shard_size, shard_n = [], 0, 0

            def flush():
                nonlocal shard, shard_size, shard_n
                if not shard:
                    return
                shard_n += 1
                dest = f'data/train-{tag}-{shard_n:05d}.parquet'
                path = staging / dest
                write_shard(shard, path)
                files.append((path, dest))
                for v in shard:
                    expected[v['video_id']] = [(dest, path.stat().st_size)]
                shard, shard_size = [], 0

            for v in videos:
                video_dir, kept = segment_files(v)
                hist = {}
                for line in kept:
                    d = to_float(line.get('duration_seconds'))
                    for lo, hi in CLIP_BUCKETS:
                        if lo <= d < hi:
                            key = _bucket_label(lo, hi, 's')
                            hist[key] = hist.get(key, 0) + 1
                for key, n in hist.items():
                    ref.upsert(v['video_id'], **{f'clips_{key}': n})
                if not kept:
                    expected[v['video_id']] = []           # nothing to upload, still marked as pushed
                    continue
                if shard and shard_size + v['_size'] > shard_bytes:
                    flush()
                shard.append({'dir': video_dir, 'kept': kept, 'video_id': v['video_id']})
                shard_size += v['_size']
            flush()

        # consolidated metadata + README, rebuilt from the reference (this batch + earlier pushes)
        in_repo = ref.df[(ref.df[repo_col].astype(str) == repo_id) | ref.df['video_id'].isin(list(expected))].copy()
        in_repo[repo_col] = repo_id
        meta_cols = ['video_id', 'url', 'title', 'upload_date', 'duration_sec', 'actual_duration_sec', 'view_count',
                     'is_short', 'has_captions', 'channel_id', 'channel_name', 'channel_folder', 'dialect']
        if source == 'downloads':
            meta = in_repo[meta_cols].copy()
            meta.insert(0, 'file_name', [f"audio/{r.channel_folder}/{Path(str(r.file_path)).name}" for r in in_repo.itertuples()])
            files.append((csv_bytes(meta), 'metadata.csv'))
            hist = None
        else:
            clip_cols = sorted(c for c in in_repo.columns if c.startswith('clips_'))
            meta = in_repo[meta_cols + ['n_segments_kept', 'hours_kept'] + clip_cols].copy()
            files.append((csv_bytes(meta), 'videos.csv'))
            hist = clip_histogram(in_repo)
        files.append((build_readme(repo_id, kind, in_repo, hist).encode('utf-8'), 'README.md'))

        n_bytes = sum(len(s) if isinstance(s, bytes) else Path(s).stat().st_size for s, _ in files)
        log.info('  uploading %d file(s), %.2f GB, %d video(s) -> %s', len(files), n_bytes / 1024 ** 3, len(videos), repo_id)
        try:
            backend.upload(repo_id, files, f'Add {len(videos)} video(s) [{source}]')
        except Exception as exc:
            log.error('  upload to %s failed: %s', repo_id, exc)
            failed += len(videos)
            continue

        # 2. verify on the hub, then mark
        remote = backend.list_files(repo_id)
        stamp = now_iso()
        for v in videos:
            want = expected.get(v['video_id'])
            if want is None:
                continue
            ok = all(remote.get(path) == size for path, size in want)
            if ok:
                ref.upsert(v['video_id'], **{repo_col: repo_id, pushed_col: stamp})
                pushed += 1
            else:
                failed += 1
                log.error('  verification failed for %s: %s', v['video_id'],
                          [(p, s, remote.get(p)) for p, s in want])
        ref.save()
        shutil.rmtree(staging, ignore_errors=True)

    # 3. optional local deletion — verified videos only
    if delete_local:
        for item in deletable(ref, source):
            if item['path'].is_dir():
                shutil.rmtree(item['path'])
            else:
                item['path'].unlink()
            deleted += 1
        log.info('[%s] deleted local copies of %d verified video(s)', source, deleted)

    log.info('[%s] pushed and verified %d video(s), %d failed', source, pushed, failed)
    return {'pushed': pushed, 'failed': failed, 'deleted': deleted, 'repos': list(batches)}


def run(config: dict, **options) -> dict:
    source = options.get('push_source') or options.get('source') or 'segments'
    delete_local = bool(options.get('delete_local'))
    sources = ['downloads', 'segments'] if source == 'both' else [source]
    backend = make_backend(config)
    backup_reference()
    ref = Reference()
    total = {'pushed': 0, 'failed': 0, 'repos': []}
    for s in sources:
        res = push_source(config, s, backend, ref, delete_local)
        total['pushed'] += res['pushed']
        total['failed'] += res['failed']
        total['repos'] += res['repos']
    ref.save()
    if total['failed']:
        raise RuntimeError(f"{total['failed']} video(s) failed to push or verify (see log); "
                           f"{total['pushed']} pushed")
    return {'status': 'done', 'rows_in': total['pushed'] + total['failed'], 'rows_out': total['pushed'],
            'message': f"{total['pushed']} video(s) pushed to {sorted(set(total['repos']))}"}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default=None)
    parser.add_argument('--source', choices=['downloads', 'segments', 'both'], default='segments')
    parser.add_argument('--delete-local', action='store_true')
    args = parser.parse_args()
    setup_logging(STEP)
    cfg = load_config(args.config)
    state, msg = prerequisite(cfg, source=args.source)
    log.info('prerequisite: %s — %s', state, msg)
    if state == 'fail':
        sys.exit(1)
    if state == 'ok':
        print(run(cfg, source=args.source, delete_local=args.delete_local))
