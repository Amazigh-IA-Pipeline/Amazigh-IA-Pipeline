# Amazigh-IA Pipeline

Builds a speech dataset of Moroccan Tamazight (Tachelhit, Tarifit, Central Atlas Tamazight) from YouTube, for ASR training.

```
scout  →  dedup_url  →  download  →  dedup_audio  →  segment  →  push
```

One folder per step, one script per folder. `reference/` is the only permanent memory, and everything else can be deleted and rebuilt. **Back up `reference/`.**

## Setup

```bash
pip install -r requirements.txt
```

- **ffmpeg** must be on PATH.
- **Fingerprinting** needs either an ffmpeg build with Chromaprint (Linux packages, or the gyan.dev *full* build on Windows) or Chromaprint's `fpcalc`. On Windows: `winget install AcoustID.Chromaprint`. The default `fingerprint.tool: auto` picks whichever is available.
- **Hugging Face:** run `huggingface-cli login` once, or set `HF_TOKEN`, then set `push.hf_org` in `config/config.yaml`.

## Quick test (one channel + two videos)

1. Copy `scout/sources_template.xlsx` into `scout/input/` and fill it in. Use one row per source and fill **one** URL column per row, plus `dialect`.
2. Optionally, set `push.backend: local` in `config/config.yaml` to push to `push/_local_hub/` instead of Hugging Face.
3. Run:
   ```bash
   python main.py --all --dry-run   # check what will run
   python main.py --all
   ```
4. Look at `scout/output/scouted.xlsx`, `scout/output/scout_removed.xlsx`, `reference/reference.xlsx`, `dedup_audio/duplicates.xlsx`, `segment/segments/`, and the run log in `logs/`.

`tests/e2e_test.py` runs the whole pipeline against a fake YouTube and a local hub (46 checks). Run it on a **copy** of the repo, because it writes into the repo folders. It needs `pip install espeakng-loader` for offline speech.

## Steps: input and output

| Step | Input | Output |
|---|---|---|
| **1. scout** `scout/scouting.py` | `scout/input/*.xlsx` with columns `channel_url \| playlist_url \| video_url \| short_url \| dialect` | `scout/output/scouted.xlsx` (one row per video), `scout/output/scout_removed.xlsx` (skipped, with the reason) |
| **2. dedup_url** `dedup_url/dedup_url.py` | `scouted.xlsx` + `reference.xlsx` | `dedup_url/output/deduplicated.xlsx` (new videos only), `dedup_url/removed_url.xlsx` |
| **3. download** `download/download.py` | `deduplicated.xlsx` | `download/downloads/<channel_folder>/audios/<video_id>.<ext>` and `.../metadata/<video_id>.json`; new rows in `reference.xlsx`; fingerprints in `fingerprints.parquet` |
| **4. dedup_audio** `dedup_audio/dedup_audio.py` | rows with `dedup_status = pending` + `fingerprints.parquet` | `dedup_status` = unique / duplicate and `duplicate_of` in the reference; `dedup_audio/duplicates.xlsx` |
| **5. segment** `segment/segment.py` | rows downloaded + unique + `segment_status = pending` | `segment/segments/<channel_folder>/<video_id>/<video_id>__00001.flac` + `metadata.jsonl`; `segment_status`, `n_segments_kept`, `hours_kept` |
| **6. push** `push/push.py` | rows without `*_pushed_at` | private HF repos `<org>/amazigh-ia-audio-001…` / `amazigh-ia-segments-001…`; `audio_repo`/`audio_pushed_at`, `segments_repo`/`segments_pushed_at` |

### What each step does

**scout**
- A **channel** row lists the Videos, Live/Streams and Shorts tabs.
- A **playlist** row lists its videos, and credits each one to its *real* uploading channel.
- A **video** or **short** row is fetched directly.

It skips the following and logs each one with a reason: live or upcoming videos, banned channels (`config/banned_channels.yaml`), likely music (title only, using `config/music_keywords.yaml`), videos shorter than `scout.min_duration_sec`, metadata errors, and repeats. When the same video comes from two rows with different dialects, the first row's dialect is kept and the video gets `dialect_conflict = True`. `scouted.xlsx` is replaced on every run. Per-video metadata is cached in `scout/output/cache/`, so an interrupted scout of a big channel doesn't refetch everything.

**dedup_url**
- Every URL form is reduced to the 11-character video ID before comparing.
- A row is removed if its ID is already in the reference, is repeated in the batch, or can't be read. Rows without a readable ID are flagged, never kept.

**download**
- Downloads the best audio-only stream, with no re-encoding.
- Checks size, file header and **real duration vs. expected**, then fingerprints the file, writes its metadata and adds the reference row. The reference is saved every N videos.
- Smaller channels go first.
- A channel's folder name is fixed the first time its `channel_id` is seen, so renaming the channel doesn't change it.
- Failures caused by the **video** (private, removed, age-restricted, broken file) are recorded as `failed` and are not retried.
- Failures caused by the **session** (YouTube's bot check, network errors) are *not* recorded, so the next run retries them. After 5 in a row the step stops and tells you to add cookies (`youtube.cookies_file`).
- `--retry-failed` retries the videos marked failed.

**dedup_audio**
- Compares only videos whose durations differ by at most 3 s, and only pairs that involve at least one new video.
- A pair is a duplicate only if the fingerprints match at the **start, middle and end**, so a shared intro jingle is not enough. A time shift of up to 3 s is allowed.
- Copy kept: the one kept earlier, then the most views, then the earliest upload.
- Works against old videos whose audio has already been pushed and deleted, because their fingerprints stay in `reference/`.
- `--move` moves duplicate audio to `downloads/_duplicates/`. It is off by default.

**segment**
- Silero VAD, with cuts at real pauses aiming for 20 s clips (15 s minimum, 27 s maximum). Speech that runs too long without a pause is cut at its quietest point and flagged for review.
- Clips with too little speech are rejected and only logged. `segment.save_rejected_audio: true` keeps their audio in `segment/rejected/` for listening.
- A video is written to `<video_id>.partial/` and renamed only when complete. The VAD results are cached, so retuning only the durations or ratios doesn't run Silero again.
- The cutting logic is carried over unchanged from `generate_segments.py`.

**push**
- Whole videos are grouped into repos of at most `push.max_repo_gb` (75 GB), and a video is never split. When a repo is full, the next one is created.
- Segments are packed into parquet shards of about 500 MB with an HF Audio column, so `datasets.load_dataset("<repo>")` works. Decoding the audio on the reader's side needs `pip install torchcodec`.
- Raw audio is uploaded as is.
- Each repo gets a README with real statistics and a consolidated metadata file.
- After uploading, the step **lists the hub files and checks paths and sizes** before writing `*_pushed_at`.
- `--delete-local` deletes only verified videos, and never raw audio that still waits for segmenting.

## main.py

```bash
python main.py                                   # interactive menu (1 2 3, 3-5, all)
python main.py --steps scout dedup_url download  # chosen steps, always in pipeline order
python main.py --from download --to segment      # a range
python main.py --all                             # everything
python main.py --all --dry-run                   # prerequisite checks only, executes nothing
```

Options passed to the steps: `--push-source downloads|segments|both` (default `segments`), `--delete-local`, `--move`, `--retry-failed`, `--limit N` (download and segment, for testing).

- **Prerequisites:** before each step `main.py` checks its input. A missing input means the step **fails**; nothing to do means it is **skipped**.
- **First failure:** the run stops and the summary says which step failed and why.
- **Stale input:** running `download` alone when `deduplicated.xlsx` is older than `scouted.xlsx` gives a warning.
- **Log:** `logs/run_<timestamp>.log` records each step, its duration, and rows in / rows out.

Each script also runs alone, for example `python download/download.py --limit 5`.

| You want to | Command |
|---|---|
| Process a new batch up to clean audio | `python main.py --steps scout dedup_url download dedup_audio` |
| Resume an interrupted download | `python main.py --steps download` |
| Segment and push what is ready | `python main.py --steps segment push` |
| Push the raw audio only | `python main.py --steps push --push-source downloads` |
| Check what would happen | `python main.py --all --dry-run` |

## reference/reference.xlsx

One row per video, keyed on `video_id`. Writes are atomic, and one backup (`reference.backup.xlsx`) is taken before each step that writes it.

| Group | Columns |
|---|---|
| Identity | `video_id`, `url`, `title`, `upload_date`, `duration_sec`, `view_count`, `is_short`, `has_captions` |
| Channel | `channel_id`, `channel_name`, `channel_folder`, `dialect` |
| Download | `download_status` (downloaded / failed), `fail_reason`, `file_path`, `file_size_mb`, `actual_duration_sec`, `downloaded_at` |
| Dedup | `dedup_status` (pending / unique / duplicate), `duplicate_of` |
| Segment | `segment_status` (pending / done), `n_segments_kept`, `hours_kept`, `segment_error` |
| Push | `audio_repo`, `audio_pushed_at`, `segments_repo`, `segments_pushed_at` |

`fingerprints.parquet` holds `video_id`, `duration_sec`, `window_sec`, `fp_start`, `fp_middle` and `fp_end`.

## Differences from the functional spec (v2)

- **`common/` holds four small files** instead of two: `video_id.py`, `reference_io.py`, `fingerprint.py` (used by download and dedup_audio) and `utils.py` (paths, config, logging, Excel).
- **`pushed_at` is split** into `audio_pushed_at` and `segments_pushed_at`, because a video can be pushed both as raw audio and as segments.
- **Extra reference columns:** `segment_error`, plus `clips_*` duration counts added by push for the README.
- **New fingerprint similarity.** It is now bit similarity (1 − differing bits/32), with a time-shift search. The old rule ("share of frames within 2 bits", no shift) scored a real re-encoded re-upload at only 0.55–0.70, and at 0.0 when it had a 1.5 s longer intro. The new rule scores duplicates at 0.92–0.99 and unrelated audio at about 0.6. The threshold is 0.85, for each of start, middle and end. The old script also had a sign bug (no `& 0xFFFFFFFF`), fixed here.
- **Session errors are not recorded as failures** (see download).

## Open points to confirm

1. Hugging Face organization name (`push.hf_org`).
2. Whether the banned-channels list is up to date (`config/banned_channels.yaml`).
3. Segment format: 16 kHz mono FLAC, 15–27 s clips (taken from `generate_segments.py`).
4. Minimum duration for shorts (`scout.min_duration_sec`, currently 30).
5. `--move` for duplicates, or the status column only (the default).
