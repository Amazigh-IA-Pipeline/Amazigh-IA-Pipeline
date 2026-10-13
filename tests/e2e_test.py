"""End-to-end test of the whole pipeline WITHOUT YouTube or Hugging Face.

Run it on a COPY of the repo (it writes into the repo folders):

    cp -r Amazigh-IA-Pipeline /tmp/pipeline_copy
    python /tmp/pipeline_copy/tests/e2e_test.py

What it does
  - generates speech audio offline (espeak-ng through `pip install espeakng-loader`)
  - replaces the two network functions (scout.yt_extract, download.yt_download)
    with a fake YouTube holding one channel + two video links, like the real test
  - pushes to a local fake hub (push.backend: local)
  - runs main.py several times and checks every step's input/output

The fake YouTube contains, on purpose:
  channel UC_TACH (Tachelhit): v1 normal | v2 music title | v3 = re-upload of v1 (other ID,
      other codec, 1.5 s longer intro) | v4 upcoming live | v5 past live stream |
      v8 private (download fails) | v9 & v10 two episodes sharing the same 60 s intro |
      shorts s1 (20 s, too short) and s2 (40 s)
  video link: v6 from channel UC_RIF (Tarifit)
  video link: v1 again with another dialect (dialect conflict)
  video link: a broken URL; a video from a banned channel
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import yaml

FIX = ROOT / 'tests' / '_fixtures'
SENTENCES = [
    "Azul fellawen, welcome to this program about the history of the Souss valley.",
    "Today we talk with our guest about agriculture, water and the old irrigation channels.",
    "The farmers of the region have worked the land for many generations with great care.",
    "In the mountains the winters are cold, and the families prepare their food in advance.",
    "Our guest explains how the weekly market brings people together from many villages.",
    "Children learn the language at home, from their grandmothers and grandfathers.",
    "The argan tree is very important for the economy and for the women's cooperatives.",
    "We will now take a short pause and then continue the discussion with new questions.",
    "Many listeners have sent messages asking about the traditional houses made of earth.",
    "These houses keep the inside cool in summer and warm during the long winter nights.",
    "Thank you for following us, and see you next week for another conversation.",
    "The poet recited verses about the sea, the wind and the people who left for the city.",
]


# ── audio fixtures ──────────────────────────────────────────────────────────

def speech_track(path: Path, seed: int, seconds: float, voice='en'):
    """Concatenate TTS sentences with pauses until `seconds` is reached."""
    from tts import synth
    import random
    rnd = random.Random(seed)
    rate = None
    samples = []
    while rate is None or len(samples) / rate < seconds:
        s, rate = synth(rnd.choice(SENTENCES), voice=voice, speed=rnd.choice([140, 155, 170]))
        samples.extend(s)
        samples.extend([0] * int(rate * rnd.choice([0.3, 0.5, 0.8, 1.2])))
    samples = samples[:int(seconds * rate)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(b''.join(int(x).to_bytes(2, 'little', signed=True) for x in samples))


def ff(*args):
    subprocess.run(['ffmpeg', '-v', 'error', '-y', *args], check=True)


def duration(path: Path) -> float:
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(path)],
                         capture_output=True, text=True).stdout
    return float(out)


def build_fixtures() -> dict[str, Path]:
    FIX.mkdir(parents=True, exist_ok=True)
    w = {}
    for name, seed, sec, voice in [('v1', 1, 180, 'en'), ('v5', 5, 150, 'en'), ('v6', 6, 130, 'fr'),
                                   ('intro', 9, 60, 'en'), ('v9b', 91, 120, 'en'), ('v10b', 101, 120, 'en'),
                                   ('s2', 12, 40, 'en'), ('v11', 11, 100, 'en')]:
        w[name] = FIX / f'{name}.wav'
        if not w[name].exists():
            speech_track(w[name], seed, sec, voice)
    files = {}
    files['v1'] = FIX / 'v1.webm'
    ff('-i', str(w['v1']), '-c:a', 'libopus', '-b:a', '64k', str(files['v1']))
    files['v3'] = FIX / 'v3.m4a'                                   # re-upload: 1.5 s extra, AAC
    ff('-f', 'lavfi', '-i', 'anullsrc=r=22050:cl=mono:d=1.5', '-i', str(w['v1']),
       '-filter_complex', '[0][1]concat=n=2:v=0:a=1', '-c:a', 'aac', '-b:a', '96k', str(files['v3']))
    files['v5'] = FIX / 'v5.webm'
    ff('-i', str(w['v5']), '-c:a', 'libopus', str(files['v5']))
    files['v6'] = FIX / 'v6.webm'
    ff('-i', str(w['v6']), '-c:a', 'libopus', str(files['v6']))
    for vid, rest in (('v9', 'v9b'), ('v10', 'v10b')):              # same 60 s intro, different episode
        files[vid] = FIX / f'{vid}.webm'
        ff('-i', str(w['intro']), '-i', str(w[rest]), '-filter_complex', '[0][1]concat=n=2:v=0:a=1',
           '-c:a', 'libopus', str(files[vid]))
    files['s2'] = FIX / 's2.webm'
    ff('-i', str(w['s2']), '-c:a', 'libopus', str(files['s2']))
    files['v12'] = FIX / 'v12.webm'                                  # later re-upload of v6 (after v6 was pushed+deleted)
    ff('-i', str(w['v6']), '-c:a', 'libopus', '-b:a', '32k', str(files['v12']))
    files['v11'] = FIX / 'v11.webm'
    ff('-i', str(w['v11']), '-c:a', 'libopus', str(files['v11']))
    return files


# ── fake YouTube ────────────────────────────────────────────────────────────

def vid_id(short: str) -> str:
    return (short + '_' * 11)[:11]                                  # 'v1' -> 'v1_________'


def make_fake_youtube(files: dict[str, Path]):
    from common.video_id import extract_video_id
    durations = {k: duration(p) for k, p in files.items()}
    durations['v2'] = 200.0
    durations['s1'] = 20.0
    durations['v8'] = 160.0
    durations['vb'] = 300.0
    videos = {
        'v1': ('Asays n tmazirt — episode 1', 'UC_TACH', 'Tachelhit TV', 5000, '20240101', ''),
        'v2': ('Ahwach clip 2024', 'UC_TACH', 'Tachelhit TV', 900, '20240102', ''),
        'v3': ('Asays n tmazirt ep1 (reupload)', 'UC_COPY', 'Copy Channel', 300, '20240301', ''),
        'v4': ('Live tonight', 'UC_TACH', 'Tachelhit TV', 0, '', 'is_upcoming'),
        'v5': ('Live debate (replay)', 'UC_TACH', 'Tachelhit TV', 1200, '20240110', 'was_live'),
        'v8': ('Private episode', 'UC_TACH', 'Tachelhit TV', 10, '20240111', ''),
        'v9': ('Talk show — episode 9', 'UC_TACH', 'Tachelhit TV', 700, '20240201', ''),
        'v10': ('Talk show — episode 10', 'UC_TACH', 'Tachelhit TV', 650, '20240208', ''),
        's1': ('tiny short', 'UC_TACH', 'Tachelhit TV', 40, '20240301', ''),
        's2': ('short story', 'UC_TACH', 'Tachelhit TV', 80, '20240302', ''),
        'v6': ('Arif: tamedyazt', 'UC_RIF', 'Rif Channel : Tarifit', 2500, '20231201', ''),
        'vb': ('banned talk', 'UC_BANNED', 'Banned One', 10, '20231201', ''),
        'v11': ('New episode', 'UC_TACH', 'Tachelhit TV', 100, '20240401', ''),
        'v12': ('Arif tamedyazt (copy)', 'UC_OTHER', 'Other Uploader', 50, '20240402', ''),
    }
    channel_tabs = {
        'UC_TACH': {'videos': ['v1', 'v2', 'v3', 'v4', 'v8', 'v9', 'v10'], 'streams': ['v5'], 'shorts': ['s1', 's2']},
    }
    by_id = {vid_id(k): k for k in videos}
    calls = {'metadata': 0}

    def yt_extract(url, config, flat):
        if '/channel/' in url or '/@' in url:
            base, tab = url.rsplit('/', 1)
            cid = 'UC_TACH'
            if tab not in channel_tabs[cid]:
                raise Exception(f'ERROR: This channel does not have a {tab} tab')
            entries = [{'id': vid_id(k), 'url': f'https://www.youtube.com/watch?v={vid_id(k)}',
                        'live_status': videos[k][5] or None} for k in channel_tabs[cid][tab]]
            if tab == 'videos':   # tab listing also shows v3 as if it were on this channel (realistic: mixed)
                pass
            return {'channel_id': cid, 'channel': 'Tachelhit TV', 'entries': entries}
        vid = extract_video_id(url)
        key = by_id.get(vid)
        if key is None:
            raise Exception('ERROR: Video unavailable')
        calls['metadata'] += 1
        title, cid, cname, views, date, live = videos[key]
        return {'id': vid, 'title': title, 'description': f'description of {key}', 'upload_date': date,
                'duration': None if live == 'is_upcoming' else durations[key], 'view_count': views, 'like_count': 1,
                'channel_id': cid, 'channel': cname, 'channel_url': f'https://www.youtube.com/channel/{cid}',
                'live_status': live or 'not_live', 'subtitles': {}, 'automatic_captions': {}}

    def yt_download(url, audios_dir, video_id, config):
        key = by_id[video_id]
        if key == 'v8':
            raise Exception("ERROR: [youtube] v8: Private video. Sign in if you've been granted access to this video")
        src = files[key]
        shutil.copy(src, Path(audios_dir) / f'{video_id}{src.suffix}')
        return {'id': video_id}

    return yt_extract, yt_download, calls


# ── helpers ─────────────────────────────────────────────────────────────────

def set_config(**overrides):
    path = ROOT / 'config' / 'config.yaml'
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    for dotted, value in overrides.items():
        section, key = dotted.split('.')
        cfg[section][key] = value
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding='utf-8')


def write_input(rows: list[dict], name='test_sources.xlsx'):
    df = pd.DataFrame(rows, columns=['channel_url', 'playlist_url', 'video_url', 'short_url', 'dialect'])
    (ROOT / 'scout' / 'input').mkdir(parents=True, exist_ok=True)
    df.to_excel(ROOT / 'scout' / 'input' / name, index=False)


def run_main(*args) -> int:
    import main
    sys.argv = ['main.py', *args, '--yes']
    return main.main()


def ref() -> pd.DataFrame:
    from common.reference_io import load_reference
    return load_reference().set_index('video_id')


CHECKS = []


def check(name, condition, detail=''):
    CHECKS.append((name, bool(condition)))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f'   [{detail}]' if detail and not condition else ''))


# ── scenario ────────────────────────────────────────────────────────────────

def main_test():
    for d in ('scout/output', 'dedup_url/output', 'download/downloads', 'segment/segments', 'segment/.vad_cache',
              'push/_local_hub', 'reference'):
        shutil.rmtree(ROOT / d, ignore_errors=True)
    for f in ('dedup_url/removed_url.xlsx', 'dedup_audio/duplicates.xlsx'):
        (ROOT / f).unlink(missing_ok=True)
    for f in (ROOT / 'scout' / 'input').glob('*.xlsx'):
        f.unlink()

    print('building audio fixtures (offline TTS)...')
    files = build_fixtures()
    yt_extract, yt_download, calls = make_fake_youtube(files)
    import download.download as dl
    import scout.scouting as sc
    sc.yt_extract = yt_extract
    dl.yt_download = yt_download

    (ROOT / 'config' / 'banned_channels.yaml').write_text('channel_ids: [UC_BANNED]\nchannel_urls: []\nchannel_names: []\n')
    set_config(**{'scout.delay_sec': 0, 'download.sleep_min_sec': 0, 'download.sleep_max_sec': 0,
                  'push.backend': 'local', 'push.hf_org': 'test-org', 'segment.workers': 2})
    write_input([
        {'channel_url': 'https://www.youtube.com/@TachelhitTV', 'dialect': 'Tachelhit'},
        {'video_url': f'https://youtu.be/{vid_id("v6")}?si=track', 'dialect': 'Tarifit'},
        {'video_url': f'https://m.youtube.com/watch?v={vid_id("v1")}&t=3s', 'dialect': 'Central Atlas Tamazight'},
        {'video_url': 'https://www.youtube.com/watch?v=broken', 'dialect': 'Tachelhit'},
        {'video_url': f'https://www.youtube.com/watch?v={vid_id("vb")}', 'dialect': 'Tachelhit'},
        {'video_url': f'https://www.youtube.com/watch?v={vid_id("v3")}', 'dialect': 'Tachelhit'},
    ])

    print('\n=== RUN 1: python main.py --all ===')
    code = run_main('--all')
    check('run 1 exit code 0', code == 0, code)

    scouted = pd.read_excel(ROOT / 'scout/output/scouted.xlsx', dtype=str)
    removed = pd.read_excel(ROOT / 'scout/output/scout_removed.xlsx', dtype=str)
    print('\n--- scout ---')
    print(scouted[['video_id', 'title', 'dialect', 'duration_sec', 'is_short', 'source', 'dialect_conflict']].to_string())
    print(removed[['source', 'video_id', 'reason', 'detail']].to_string())
    reasons = dict(zip(removed['video_id'].fillna(''), removed['reason']))
    check('scout keeps 9 videos (v1 v3 v5 v6 v8 v9 v10 s2 + ...)', set(scouted['video_id']) ==
          {vid_id(k) for k in ['v1', 'v3', 'v5', 'v6', 'v8', 'v9', 'v10', 's2']}, sorted(scouted['video_id']))
    check('music title removed', reasons.get(vid_id('v2')) == 'likely_music')
    check('upcoming live removed', reasons.get(vid_id('v4')) == 'upcoming')
    check('too-short short removed', reasons.get(vid_id('s1')) == 'too_short')
    check('banned channel removed', reasons.get(vid_id('vb')) == 'banned_channel')
    check('broken URL flagged', 'no_video_id' in set(removed['reason']))
    check('dialect conflict flagged on v1', scouted.set_index('video_id').loc[vid_id('v1'), 'dialect_conflict'] == 'True')
    check('past live stream kept', vid_id('v5') in set(scouted['video_id']))
    check('short marked is_short', scouted.set_index('video_id').loc[vid_id('s2'), 'is_short'] == '1')

    r = ref()
    print('\n--- reference ---')
    print(r[['channel_folder', 'download_status', 'fail_reason', 'dedup_status', 'duplicate_of', 'segment_status',
             'n_segments_kept', 'hours_kept', 'audio_pushed_at', 'segments_repo']].to_string())
    check('private video recorded as failed', r.loc[vid_id('v8'), 'download_status'] == 'failed'
          and r.loc[vid_id('v8'), 'fail_reason'].startswith('private'))
    check('re-upload v3 marked duplicate of v1', r.loc[vid_id('v3'), 'dedup_status'] == 'duplicate'
          and r.loc[vid_id('v3'), 'duplicate_of'] == vid_id('v1'), r.loc[vid_id('v3'), ['dedup_status', 'duplicate_of']].tolist())
    check('shared intro episodes v9 / v10 both unique', r.loc[vid_id('v9'), 'dedup_status'] == 'unique'
          and r.loc[vid_id('v10'), 'dedup_status'] == 'unique')
    check('channel folder sanitised', r.loc[vid_id('v6'), 'channel_folder'] == 'Rif Channel _ Tarifit')
    check('audio stored as audios/<video_id>.<ext>',
          (ROOT / 'download/downloads/Tachelhit TV/audios' / f'{vid_id("v1")}.webm').exists())
    check('metadata json written', (ROOT / 'download/downloads/Tachelhit TV/metadata' / f'{vid_id("v1")}.json').exists())
    fps = pd.read_parquet(ROOT / 'reference/fingerprints.parquet')
    check('one fingerprint per downloaded video', len(fps) == (r['download_status'] == 'downloaded').sum())
    dup_report = pd.read_excel(ROOT / 'dedup_audio/duplicates.xlsx', dtype=str)
    print(dup_report[['cluster_id', 'video_id', 'role', 'duplicate_of', 'sim_start', 'sim_middle', 'sim_end']].to_string())

    uniques = r[r['dedup_status'] == 'unique']
    check('every unique video segmented', (uniques['segment_status'] == 'done').all())
    check('duplicate NOT segmented', r.loc[vid_id('v3'), 'segment_status'] == 'pending')
    seg_dir = ROOT / 'segment/segments/Tachelhit TV' / vid_id('v1')
    lines = [json.loads(l) for l in (seg_dir / 'metadata.jsonl').read_text(encoding='utf-8').splitlines()]
    kept = [l for l in lines if l['kept']]
    check('v1 has kept clips', len(kept) > 0, len(kept))
    check('kept clips are 15-27 s', all(15 <= l['duration_seconds'] <= 27 for l in kept),
          [l['duration_seconds'] for l in kept])
    check('every kept clip file exists', all((seg_dir / l['file_name']).exists() for l in kept))
    check('no rejected audio stored in segments/', len(list(seg_dir.glob('*.flac'))) == len(kept))
    check('no .partial folders left', not list((ROOT / 'segment/segments').rglob('*.partial')))
    clip = next(seg_dir.glob('*.flac'))
    probe = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=sample_rate,channels', '-of', 'csv=p=0',
                            str(clip)], capture_output=True, text=True).stdout.strip()
    check('clips are 16 kHz mono FLAC', probe == '16000,1', probe)

    hub = ROOT / 'push/_local_hub/test-org'
    check('segments repo created', (hub / 'amazigh-ia-segments-001/README.md').exists())
    shards = list((hub / 'amazigh-ia-segments-001/data').glob('*.parquet'))
    import pyarrow.parquet as pq
    table = pq.read_table(shards[0])
    check('shard has an HF Audio column', b'Audio' in table.schema.metadata[b'huggingface'])
    total_rows = sum(pq.read_metadata(s).num_rows for s in shards)
    check('all kept clips are in the shards', total_rows == int(uniques['n_segments_kept'].astype(float).sum()),
          (total_rows, uniques['n_segments_kept'].tolist()))
    check('segments_pushed_at set for all segmented videos', (r[r['segment_status'] == 'done']['segments_pushed_at'] != '').all())
    print('\n--- README of the segments repo ---')
    print((hub / 'amazigh-ia-segments-001/README.md').read_text(encoding='utf-8')[:1500])

    print('\n=== RUN 2: same command again (must redo nothing) ===')
    before = (ROOT / 'reference/reference.xlsx').read_bytes()
    meta_calls = calls['metadata']
    code = run_main('--all')
    check('run 2 exit code 0', code == 0)
    check('run 2: scout used its cache (no new metadata calls)', calls['metadata'] == meta_calls,
          calls['metadata'] - meta_calls)
    dd = pd.read_excel(ROOT / 'dedup_url/output/deduplicated.xlsx')
    check('run 2: dedup_url finds nothing new', len(dd) == 0, len(dd))
    check('run 2: reference unchanged', ref()[['download_status', 'dedup_status', 'segment_status']].equals(
        r[['download_status', 'dedup_status', 'segment_status']]))

    print('\n=== RUN 3: push raw audio, then delete local copies ===')
    code = run_main('--steps', 'push', '--push-source', 'downloads', '--delete-local')
    r = ref()
    check('run 3 exit code 0', code == 0)
    check('audio repo has the unique audio files',
          len(list((hub / 'amazigh-ia-audio-001/audio').rglob('*.*'))) == (r['dedup_status'] == 'unique').sum())
    check('duplicates are not pushed', not list((hub / 'amazigh-ia-audio-001/audio').rglob(f'{vid_id("v3")}*')))
    check('local audio of v6 deleted after verified push',
          not (ROOT / 'download/downloads/Rif Channel _ Tarifit/audios' / f'{vid_id("v6")}.webm').exists())
    code = run_main('--steps', 'push', '--push-source', 'segments', '--delete-local')
    check('local segments deleted after verified push', not (ROOT / 'segment/segments/Tachelhit TV' / vid_id('v1')).exists())

    print('\n=== RUN 4: new batch — a re-upload of v6 (whose audio is gone locally) + a new video ===')
    write_input([{'video_url': f'https://youtu.be/{vid_id("v12")}', 'dialect': 'Tarifit'},
                 {'video_url': f'https://youtu.be/{vid_id("v11")}', 'dialect': 'Tachelhit'},
                 {'video_url': f'https://youtu.be/{vid_id("v1")}', 'dialect': 'Tachelhit'}], name='batch2.xlsx')
    (ROOT / 'scout/input/test_sources.xlsx').unlink()
    code = run_main('--all')
    r = ref()
    removed_url = pd.read_excel(ROOT / 'dedup_url/removed_url.xlsx', dtype=str)
    check('run 4 exit code 0', code == 0)
    check('v1 removed by dedup_url (already in reference)',
          removed_url.set_index('video_id').loc[vid_id('v1'), 'reason'] == 'already_in_reference')
    check('v12 matched against v6 using stored fingerprint only',
          r.loc[vid_id('v12'), 'dedup_status'] == 'duplicate' and r.loc[vid_id('v12'), 'duplicate_of'] == vid_id('v6'),
          r.loc[vid_id('v12'), ['dedup_status', 'duplicate_of']].tolist())
    check('v11 unique, segmented and pushed', r.loc[vid_id('v11'), 'segments_pushed_at'] != '')

    print('\n=== RUN 5: 80 GB rule with a tiny repo limit (5 MB) ===')
    set_config(**{'push.max_repo_gb': 0.005})
    code = run_main('--steps', 'push', '--push-source', 'both')    # nothing left -> skipped
    check('run 5: leftover push exit 0', code == 0)
    from common.reference_io import Reference
    rr = Reference()
    for vid in rr.df['video_id']:
        rr.upsert(vid, audio_repo='', audio_pushed_at='')
    rr.save()
    shutil.rmtree(hub, ignore_errors=True)
    # restore audio of unique videos for re-push
    for _, row in ref().reset_index().iterrows():
        if row['dedup_status'] == 'unique':
            key = [k for k in files if vid_id(k) == row['video_id']][0]
            dest = ROOT / row['file_path']
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(files[key], dest)
    code = run_main('--steps', 'push', '--push-source', 'downloads')
    r = ref()
    repos = sorted(set(r['audio_repo']) - {''})
    print('repos used:', repos)
    check('several audio repos created when one is full', len(repos) >= 2, repos)
    sizes = {p.name: sum(f.stat().st_size for f in p.rglob('audio/**/*') if f.is_file()) for p in hub.iterdir()}
    check('each repo holds whole videos only (each file in exactly one repo)',
          sum(len(list((hub / p / 'audio').rglob('*.*'))) for p in sizes) == (r['dedup_status'] == 'unique').sum())

    print('\n=== RUN 6: dry run ===')
    check('dry run exit 0', run_main('--all', '--dry-run') == 0)

    failed = [n for n, ok in CHECKS if not ok]
    print(f'\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed')
    if failed:
        print('FAILED:', *failed, sep='\n  ')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main_test())
