"""Step 5 — segment.

INPUT   rows of reference.xlsx with download_status = downloaded, dedup_status = unique,
        segment_status = pending (their audio in download/downloads/)

OUTPUT  segment/segments/<channel_folder>/<video_id>/<video_id>__00001.flac   16 kHz mono clips (KEPT only)
        segment/segments/<channel_folder>/<video_id>/metadata.jsonl           one line per planned clip,
                                                                              kept AND rejected, with the reason
        reference.xlsx: segment_status = done, n_segments_kept, hours_kept

How it cuts: Silero VAD finds the speech; cuts are placed at real pauses and grouped
towards target_duration (min/max in config) by dynamic programming; a speech run too
long to fit gets a forced cut at its quietest point (flagged forced_boundary_review).
A clip is kept only if it passes the quality thresholds in config (length, speech
ratio, optional speech confidence and modulation ratio). Rejected clips are only
logged in metadata.jsonl (their audio is not saved, unless save_rejected_audio: true,
which writes them to segment/rejected/ — never inside segments/).

A video is atomic: it is written to <video_id>.partial/ and renamed to <video_id>/
only when complete, so a crash never leaves a half-segmented video.
The VAD result is cached in segment/.vad_cache/, so changing only the cheap settings
(durations, ratios) does not re-run Silero.

The cutting/VAD core is carried over unchanged from generate_segments.py.

Run alone:  python segment/segment.py [--limit N]
"""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import shutil
import signal
import subprocess
import sys
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.reference_io import Reference, backup_reference
from common.utils import PATHS, abs_path, get_logger, load_config, setup_logging, to_float, to_int

STEP = 'segment'
log = get_logger(STEP)

SAMPLE_RATE = 16_000


# ============================================================================
# Core carried over from generate_segments.py (planning, VAD backends, audio
# I/O, VAD cache, speech confidence, modulation ratio)
# ============================================================================

@dataclass
class SpeechUnit:
    start: float
    end: float
    hard_boundary_after: str | None = None


@dataclass
class PlannedSegment:
    start: float
    end: float
    start_boundary: str
    end_boundary: str


@dataclass
class Settings:
    target_duration: float
    min_duration: float
    max_duration: float
    threshold: float
    neg_threshold: float
    min_speech_ms: int
    min_silence_ms: int
    pad_ms: int
    max_internal_silence: float
    forced_cut_search: float
    min_speech_ratio: float
    min_speech_confidence: float
    min_modulation_ratio: float
    audio_format: str


def find_low_energy_cut(audio, low: float, high: float, preferred: float) -> float:
    import numpy as np
    low_sample = max(1, int(round(low * SAMPLE_RATE)))
    high_sample = min(len(audio) - 1, int(round(high * SAMPLE_RATE)))
    if high_sample <= low_sample:
        return max(low, min(high, preferred))

    frame = max(1, int(0.030 * SAMPLE_RATE))
    step = max(1, int(0.010 * SAMPLE_RATE))
    centers = np.arange(low_sample, high_sample + 1, step, dtype=np.int64)
    if centers.size == 0:
        return preferred

    scores = []
    half = frame // 2
    for center in centers:
        a = max(0, int(center) - half)
        b = min(len(audio), int(center) + half)
        scores.append(float(np.mean(np.abs(audio[a:b]))))
    quiet_center = int(centers[int(np.argmin(scores))])

    zero_radius = int(0.010 * SAMPLE_RATE)
    a = max(low_sample, quiet_center - zero_radius)
    b = min(high_sample + 1, quiet_center + zero_radius + 1)
    local = np.abs(audio[a:b])
    cut_sample = a + int(np.argmin(local)) if local.size else quiet_center
    return cut_sample / SAMPLE_RATE


def split_long_speech(spans: Sequence[tuple[float, float]], audio, settings: Settings) -> list[SpeechUnit]:
    units: list[SpeechUnit] = []
    padding = settings.pad_ms / 1000.0
    safe_piece_max = min(
        settings.max_duration - 2 * padding,
        settings.max_duration - settings.max_internal_silence,
    )
    for span_start, span_end in spans:
        cursor = span_start
        while span_end - cursor > safe_piece_max:
            remaining = span_end - cursor
            preferred = cursor + min(settings.target_duration, safe_piece_max)
            low = max(cursor + settings.min_duration, preferred - settings.forced_cut_search)
            high = min(cursor + safe_piece_max, preferred + settings.forced_cut_search)
            if remaining <= 2 * settings.min_duration:
                low = high = cursor + remaining / 2.0
            elif span_end - high < settings.min_duration:
                high = span_end - settings.min_duration
                low = min(low, high)
            cut = find_low_energy_cut(audio, low, high, preferred)
            cut = max(cursor + 0.25, min(cut, cursor + safe_piece_max))
            units.append(SpeechUnit(cursor, cut, "forced_low_energy"))
            cursor = cut
        units.append(SpeechUnit(cursor, span_end))

    for i in range(len(units) - 1):
        if units[i].hard_boundary_after:
            continue
        gap = units[i + 1].start - units[i].end
        if gap >= settings.max_internal_silence:
            units[i].hard_boundary_after = "long_pause"
    return units


def segment_cost(duration: float, settings: Settings, at_block_end: bool) -> float:
    deviation = ((duration - settings.target_duration) / settings.target_duration) ** 2
    if duration < settings.min_duration:
        deviation += 2.0 if at_block_end else 4.0
    return deviation


def partition_block(block, block_start, block_end, start_method, end_method, settings: Settings) -> list[PlannedSegment]:
    cuts = [block_start]
    methods = [start_method]
    for left, right in zip(block, block[1:]):
        cuts.append((left.end + right.start) / 2.0)
        methods.append("vad_pause")
    cuts.append(block_end)
    methods.append(end_method)

    count = len(cuts)
    best = [math.inf] * count
    previous: list[int | None] = [None] * count
    best[0] = 0.0
    for end_index in range(1, count):
        for start_index in range(end_index - 1, -1, -1):
            duration = cuts[end_index] - cuts[start_index]
            if duration > settings.max_duration + 1e-6:
                continue
            if best[start_index] == math.inf:
                continue
            candidate = best[start_index] + segment_cost(duration, settings, end_index == count - 1)
            if candidate < best[end_index]:
                best[end_index] = candidate
                previous[end_index] = start_index

    if previous[-1] is None:
        raise RuntimeError("Could not satisfy max-duration using available boundaries; "
                            "reduce padding or increase max-duration.")

    ranges: list[tuple[int, int]] = []
    cursor = count - 1
    while cursor > 0:
        prior = previous[cursor]
        ranges.append((prior, cursor))
        cursor = prior
    ranges.reverse()

    return [PlannedSegment(cuts[a], cuts[b], methods[a], methods[b]) for a, b in ranges]


def plan_segments(spans: Sequence[tuple[float, float]], audio, settings: Settings) -> list[PlannedSegment]:
    if not spans:
        return []
    units = split_long_speech(spans, audio, settings)
    padding = settings.pad_ms / 1000.0
    duration = len(audio) / SAMPLE_RATE
    result: list[PlannedSegment] = []

    block_start_index = 0
    incoming_method = "source_start"
    for i, unit in enumerate(units):
        is_last = i == len(units) - 1
        if not unit.hard_boundary_after and not is_last:
            continue
        block = units[block_start_index:i + 1]
        first, last = block[0], block[-1]
        block_start = first.start if incoming_method == "forced_low_energy" else max(0.0, first.start - padding)

        if is_last and not unit.hard_boundary_after:
            end_method, block_end = "source_end", min(duration, last.end + padding)
        elif unit.hard_boundary_after == "long_pause":
            end_method, block_end = "long_pause", min(duration, last.end + padding)
        else:
            end_method, block_end = "forced_low_energy", last.end

        result.extend(partition_block(block, block_start, block_end, incoming_method, end_method, settings))
        incoming_method = end_method
        block_start_index = i + 1
    return result


def interval_overlap(start: float, end: float, spans) -> float:
    return sum(max(0.0, min(end, b) - max(start, a)) for a, b in spans)


# ============================================================================
# VAD backends -- real Silero (lazy-imported) and a torch-free fake for
# --self-test, both exposing the same .detect(audio) -> [(start,end),...]
# ============================================================================

class SileroVADBackend:
    WINDOW_SAMPLES = 512  # Silero's own default window_size_samples

    def __init__(self, threads: int):
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad
        torch.set_num_threads(threads)
        self.torch = torch
        self.model = load_silero_vad()
        self._get_speech_timestamps = get_speech_timestamps
        import importlib.metadata
        self.version = importlib.metadata.version("silero-vad")

    def detect(self, audio, settings: Settings) -> list[tuple[float, float]]:
        timestamps = self._get_speech_timestamps(
            self.torch.from_numpy(audio), self.model,
            threshold=settings.threshold, neg_threshold=settings.neg_threshold,
            sampling_rate=SAMPLE_RATE,
            min_speech_duration_ms=settings.min_speech_ms,
            min_silence_duration_ms=settings.min_silence_ms,
            speech_pad_ms=0, return_seconds=False,
        )
        spans = []
        for item in timestamps:
            start = max(0.0, float(item["start"]) / SAMPLE_RATE)
            end = min(len(audio) / SAMPLE_RATE, float(item["end"]) / SAMPLE_RATE)
            if end > start:
                spans.append((start, end))
        return spans

    def frame_probs(self, audio, settings: Settings):
        """Raw per-window speech probability across the whole source, via a
        second, independent pass over the same loaded model (get_speech_timestamps
        resets model state on its own next call, so this doesn't leak state
        into a later detect() call). window_sec = WINDOW_SAMPLES / SAMPLE_RATE."""
        self.model.reset_states()
        n = len(audio)
        window_sec = self.WINDOW_SAMPLES / SAMPLE_RATE
        probs = []
        for i in range(0, n, self.WINDOW_SAMPLES):
            chunk = audio[i:i + self.WINDOW_SAMPLES]
            if len(chunk) < self.WINDOW_SAMPLES:
                pad = self.torch.zeros(self.WINDOW_SAMPLES - len(chunk), dtype=self.torch.float32)
                chunk_t = self.torch.cat([self.torch.from_numpy(chunk), pad])
            else:
                chunk_t = self.torch.from_numpy(chunk)
            probs.append(float(self.model(chunk_t, SAMPLE_RATE).item()))
        return probs, window_sec


class EnergyVADBackend:
    """--self-test stand-in: crude RMS-threshold speech detector, no torch."""
    version = 'self-test-energy-vad'

    def detect(self, audio, settings: Settings) -> list[tuple[float, float]]:
        import numpy as np
        window = max(1, int(SAMPLE_RATE * settings.min_speech_ms / 1000 / 2))
        loud = np.zeros(len(audio), dtype=bool)
        for i in range(0, len(audio), window):
            chunk = audio[i:i + window]
            if chunk.size and float(np.sqrt(np.mean(chunk.astype('float64') ** 2))) > 0.05:
                loud[i:i + window] = True
        spans = []
        in_span = False
        start = 0
        min_silence = int(SAMPLE_RATE * settings.min_silence_ms / 1000)
        i = 0
        n = len(loud)
        while i < n:
            if loud[i] and not in_span:
                start, in_span = i, True
            elif not loud[i] and in_span:
                j = i
                while j < n and not loud[j] and (j - i) < min_silence:
                    j += 1
                if j >= n or loud[j]:
                    i = j
                    continue
                spans.append((start / SAMPLE_RATE, i / SAMPLE_RATE))
                in_span = False
            i += 1
        if in_span:
            spans.append((start / SAMPLE_RATE, n / SAMPLE_RATE))
        min_speech = SAMPLE_RATE * settings.min_speech_ms / 1000 / SAMPLE_RATE
        return [(a, b) for a, b in spans if b - a >= min_speech]

    def frame_probs(self, audio, settings: Settings):
        """--self-test stand-in for SileroVADBackend.frame_probs: a crude
        RMS-energy-derived pseudo-probability at the same 32ms window size,
        purely so --self-test exercises the same speech_confidence code path."""
        import numpy as np
        window = 512
        probs = []
        for i in range(0, len(audio), window):
            chunk = audio[i:i + window]
            rms = float(np.sqrt(np.mean(chunk.astype('float64') ** 2))) if chunk.size else 0.0
            probs.append(min(1.0, rms / 0.1))
        return probs, window / SAMPLE_RATE


# ============================================================================
# Audio I/O
# ============================================================================

def require_ffmpeg():
    if shutil.which('ffmpeg') is None:
        raise RuntimeError("ffmpeg was not found on PATH. Install FFmpeg before running this tool.")


def decode_audio(path: Path):
    import numpy as np
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path), '-map', '0:a:0',
               '-ac', '1', '-ar', str(SAMPLE_RATE), '-f', 'f32le', '-acodec', 'pcm_f32le', 'pipe:1']
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        message = result.stderr.decode('utf-8', errors='replace').strip()
        raise RuntimeError(f"FFmpeg could not decode the audio: {message}")
    audio = np.frombuffer(result.stdout, dtype='<f4').copy()
    if audio.size == 0:
        raise RuntimeError("Decoded audio is empty")
    return audio


def write_segment_audio(path: Path, audio, start: float, end: float, fmt: str):
    import numpy as np
    first = max(0, int(round(start * SAMPLE_RATE)))
    last = min(len(audio), int(round(end * SAMPLE_RATE)))
    samples = np.clip(audio[first:last], -1.0, 1.0)
    pcm16 = (samples * 32767.0).astype('<i2', copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == 'wav':
        with wave.open(str(path), 'wb') as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE)
            handle.writeframes(pcm16.tobytes())
    elif fmt == 'flac':
        cmd = ['ffmpeg', '-y', '-v', 'error', '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', '1',
               '-i', 'pipe:0', '-c:a', 'flac', '-compression_level', '5', str(path)]
        result = subprocess.run(cmd, input=pcm16.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg could not encode FLAC: {result.stderr.decode('utf-8', errors='replace')}")
    else:
        raise ValueError(f"unknown format: {fmt}")


# ============================================================================
# Caching
# ============================================================================

def vad_settings_hash(settings: Settings) -> str:
    key = f"{settings.threshold}|{settings.neg_threshold}|{settings.min_speech_ms}|{settings.min_silence_ms}"
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def load_vad_cache(cache_path: Path, settings_hash: str):
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding='utf-8'))
    except Exception:
        return None
    if data.get('settings_hash') != settings_hash:
        return None
    if 'probs' not in data or 'probs_window_sec' not in data:
        # cache written before speech_confidence support was added -- fall
        # through to a recompute rather than crashing on old cache files.
        return None
    return data.get('spans'), data.get('duration'), data.get('probs'), data.get('probs_window_sec')


def save_vad_cache(cache_path: Path, spans, duration: float, probs, probs_window_sec: float,
                    settings_hash: str, vad_version: str):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix('.tmp')
    tmp.write_text(json.dumps({
        'settings_hash': settings_hash, 'spans': spans, 'duration': duration,
        'probs': probs, 'probs_window_sec': probs_window_sec, 'vad_version': vad_version,
    }), encoding='utf-8')
    tmp.replace(cache_path)


def speech_confidence(probs, window_sec: float, start: float, end: float) -> float:
    """Overlap-weighted mean of per-window speech probability over [start,end)."""
    import numpy as np
    if probs is None or len(probs) == 0 or end <= start:
        return 0.0
    probs = np.asarray(probs, dtype='float64')
    n = len(probs)
    first = max(0, int(start / window_sec))
    last = min(n - 1, int(end / window_sec))
    if first > last:
        return 0.0
    weights = []
    values = []
    for i in range(first, last + 1):
        w_start, w_end = i * window_sec, (i + 1) * window_sec
        overlap = max(0.0, min(end, w_end) - max(start, w_start))
        if overlap <= 0:
            continue
        weights.append(overlap)
        values.append(probs[i])
    if not weights:
        return 0.0
    weights = np.asarray(weights)
    values = np.asarray(values)
    return float(np.sum(values * weights) / np.sum(weights))


def speech_modulation_ratio(segment_audio, sr: int = SAMPLE_RATE) -> float:
    """Cheap speech/music discriminator (Scheirer & Slaney-style): fraction
    of the segment's amplitude-envelope spectral energy that falls in the
    ~2-8Hz syllabic-rate band. Real speech concentrates energy there; a
    sustained tone or music bed with no syllabic-rate amplitude modulation
    mostly doesn't. Secondary signal, not a full music classifier -- see
    the module docstring."""
    import numpy as np
    frame_len = max(1, int(0.010 * sr))  # 10ms envelope frames -> 100Hz envelope rate
    n_frames = len(segment_audio) // frame_len
    if n_frames < 8:
        return 0.0
    frames = segment_audio[:n_frames * frame_len].reshape(n_frames, frame_len).astype('float64')
    env = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    env = env - env.mean()
    if not np.any(np.abs(env) > 1e-9):
        return 0.0
    windowed = env * np.hanning(len(env))
    spec = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(env), d=frame_len / sr)
    total = float(spec.sum())
    if total <= 1e-12:
        return 0.0
    band = (freqs >= 2.0) & (freqs <= 8.0)
    return float(spec[band].sum() / total)



# ============================================================================
# Pipeline wiring (new): reference.xlsx in, segments/<channel>/<video_id>/ out
# ============================================================================

def settings_from_config(config: dict) -> Settings:
    s = config['segment']
    threshold = to_float(s.get('threshold'), 0.5)
    neg = s.get('neg_threshold')
    settings = Settings(
        target_duration=to_float(s.get('target_duration'), 20), min_duration=to_float(s.get('min_duration'), 15),
        max_duration=to_float(s.get('max_duration'), 27), threshold=threshold,
        neg_threshold=to_float(neg) if neg not in (None, '') else max(threshold - 0.15, 0.01),
        min_speech_ms=to_int(s.get('min_speech_ms'), 300), min_silence_ms=to_int(s.get('min_silence_ms'), 600),
        pad_ms=to_int(s.get('pad_ms'), 180), max_internal_silence=to_float(s.get('max_internal_silence'), 2.0),
        forced_cut_search=to_float(s.get('forced_cut_search'), 5.0),
        min_speech_ratio=to_float(s.get('min_speech_ratio'), 0.6),
        min_speech_confidence=to_float(s.get('min_speech_confidence'), 0.0),
        min_modulation_ratio=to_float(s.get('min_modulation_ratio'), 0.0),
        audio_format=str(s.get('format') or 'flac'),
    )
    if not (0 < settings.min_duration <= settings.target_duration <= settings.max_duration):
        raise ValueError('segment: expected 0 < min_duration <= target_duration <= max_duration')
    return settings


def ready_rows(ref: Reference):
    return ref.rows(download_status='downloaded', dedup_status='unique', segment_status='pending')


def prerequisite(config: dict, **options) -> tuple[str, str]:
    if not PATHS['reference'].exists():
        return 'fail', 'reference.xlsx does not exist yet — run download first'
    n = len(ready_rows(Reference()))
    if n == 0:
        return 'skip', 'no video with dedup_status = unique and segment_status = pending'
    return 'ok', f'{n} video(s) to segment'


_WORKER_VAD = None


def _worker_init():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _get_vad(backend: str, threads: int):
    global _WORKER_VAD
    if _WORKER_VAD is None:
        _WORKER_VAD = EnergyVADBackend() if backend == 'energy' else SileroVADBackend(threads)
    return _WORKER_VAD


def _summarise_existing(metadata_path: Path) -> dict:
    kept, rejected, kept_sec = 0, 0, 0.0
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('kept'):
                kept += 1
                kept_sec += float(row.get('duration_seconds') or 0)
            else:
                rejected += 1
    return {'n_kept': kept, 'n_rejected': rejected, 'kept_sec': kept_sec}


def segment_video(job: dict, settings: Settings, backend: str, threads: int, save_rejected: bool) -> dict:
    """Segment ONE video. Never raises: errors are returned in the result."""
    vid = job['video_id']
    folder = job['channel_folder']
    final_dir = PATHS['segments'] / folder / vid
    tmp_dir = PATHS['segments'] / folder / f'{vid}.partial'
    result = {'video_id': vid, 'status': 'done', 'error': '', 'n_kept': 0, 'n_rejected': 0, 'kept_sec': 0.0}

    if (final_dir / 'metadata.jsonl').exists():     # finished before, reference not saved yet
        result.update(_summarise_existing(final_dir / 'metadata.jsonl'))
        result['status'] = 'done_earlier'
        return result
    try:
        source = abs_path(job['file_path'])
        if not source.exists():
            raise FileNotFoundError(f'audio not found: {source}')
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        vad = _get_vad(backend, threads)
        audio = decode_audio(source)
        duration = len(audio) / SAMPLE_RATE
        cache_path = PATHS['vad_cache'] / folder / f'{vid}.json'
        settings_hash = vad_settings_hash(settings) + ('' if backend == 'silero' else f'-{backend}')
        cached = load_vad_cache(cache_path, settings_hash)
        if cached is not None:
            spans_list, _, probs, probs_window_sec = cached
            spans = [(a, b) for a, b in spans_list]
        else:
            spans = vad.detect(audio, settings)
            probs, probs_window_sec = vad.frame_probs(audio, settings)
            save_vad_cache(cache_path, [[a, b] for a, b in spans], duration, probs, probs_window_sec,
                           settings_hash, vad.version)

        planned = plan_segments(spans, audio, settings)
        ext = '.wav' if settings.audio_format == 'wav' else '.flac'
        video_fields = {k: job.get(k, '') for k in ('video_id', 'channel_id', 'channel_name', 'channel_folder',
                                                     'dialect', 'title', 'url', 'upload_date')}
        video_fields['source_duration_sec'] = round(duration, 2)
        lines = []
        for seq, item in enumerate(planned, start=1):
            segment_id = f'{vid}__{seq:05d}'
            clip = item.end - item.start
            speech_sec = interval_overlap(item.start, item.end, spans)
            ratio = speech_sec / clip if clip > 0 else 0.0
            conf = speech_confidence(probs, probs_window_sec, item.start, item.end)
            a = max(0, int(round(item.start * SAMPLE_RATE)))
            b = min(len(audio), int(round(item.end * SAMPLE_RATE)))
            mod = speech_modulation_ratio(audio[a:b])

            reasons = []
            if clip < settings.min_duration:
                reasons.append('short_segment')
            if clip > settings.max_duration + 1e-6:
                reasons.append('long_segment')
            if ratio < settings.min_speech_ratio:
                reasons.append('low_speech_ratio')
            if settings.min_speech_confidence > 0 and conf < settings.min_speech_confidence:
                reasons.append('low_speech_confidence')
            if settings.min_modulation_ratio > 0 and mod < settings.min_modulation_ratio:
                reasons.append('low_modulation_ratio')
            kept = not reasons

            file_name, review_file = None, None
            if kept:
                file_name = f'{segment_id}{ext}'
                write_segment_audio(tmp_dir / file_name, audio, item.start, item.end, settings.audio_format)
            elif save_rejected:
                review = PATHS['segments_rejected'] / folder / vid / f'{segment_id}{ext}'
                write_segment_audio(review, audio, item.start, item.end, settings.audio_format)
                review_file = review.relative_to(PATHS['segments_rejected']).as_posix()

            lines.append({
                'segment_id': segment_id, 'file_name': file_name, 'kept': kept,
                'rejection_reason': ';'.join(reasons), 'review_file_name': review_file,
                'start_seconds': round(item.start, 3), 'end_seconds': round(item.end, 3),
                'duration_seconds': round(clip, 3), 'speech_seconds': round(speech_sec, 3),
                'speech_ratio': round(ratio, 4), 'speech_confidence': round(conf, 4),
                'modulation_ratio': round(mod, 4), 'start_boundary': item.start_boundary,
                'end_boundary': item.end_boundary,
                'forced_boundary_review': item.start_boundary == 'forced_low_energy' or item.end_boundary == 'forced_low_energy',
                **video_fields,
            })
            if kept:
                result['n_kept'] += 1
                result['kept_sec'] += clip
            else:
                result['n_rejected'] += 1

        with open(tmp_dir / 'metadata.jsonl', 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(json.dumps(line, ensure_ascii=False, default=str) + '\n')
        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(tmp_dir, final_dir)          # the video becomes visible only when complete
        if not planned:
            result['status'] = 'done_no_speech'
    except Exception as exc:
        result['status'] = 'failed'
        result['error'] = f'{type(exc).__name__}: {exc}'[:400]
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return result


def _worker(args):
    return segment_video(*args)


def run(config: dict, **options) -> dict:
    scfg = config['segment']
    settings = settings_from_config(config)
    backend = str(scfg.get('vad_backend') or 'silero')
    threads = max(1, to_int(scfg.get('threads'), 1))
    workers = max(1, to_int(scfg.get('workers'), 2))
    save_every = max(1, to_int(scfg.get('save_every'), 5))
    save_rejected = bool(scfg.get('save_rejected_audio'))
    max_tasks = to_int(scfg.get('max_tasks_per_worker'), 40) or None

    if shutil.which('ffmpeg') is None:
        raise RuntimeError('ffmpeg is not on PATH')
    if backend == 'silero':
        try:
            import silero_vad  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError('Silero VAD is not installed: pip install torch silero-vad') from exc

    backup_reference()
    ref = Reference()
    rows = ready_rows(ref)
    limit = options.get('limit')
    if limit:
        rows = rows.head(int(limit))
    jobs = [r.to_dict() for _, r in rows.iterrows()]
    log.info('%d video(s) to segment | target %.0fs (min %.0f, max %.0f) | min speech ratio %.2f | VAD %s | %d worker(s)',
             len(jobs), settings.target_duration, settings.min_duration, settings.max_duration,
             settings.min_speech_ratio, backend, workers)
    if not jobs:
        return {'status': 'skipped', 'rows_in': 0, 'rows_out': 0, 'message': 'nothing to segment'}

    counts = {'done': 0, 'failed': 0, 'kept': 0, 'rejected': 0, 'hours': 0.0}
    pending_save = 0
    started = time.monotonic()

    def apply(result: dict, n: int):
        nonlocal pending_save
        vid = result['video_id']
        if result['status'] == 'failed':
            counts['failed'] += 1
            ref.upsert(vid, segment_error=result['error'])        # stays pending -> retried next run
            log.warning('[%d/%d] %s FAILED: %s', n, len(jobs), vid, result['error'])
        else:
            counts['done'] += 1
            counts['kept'] += result['n_kept']
            counts['rejected'] += result['n_rejected']
            counts['hours'] += result['kept_sec'] / 3600
            ref.upsert(vid, segment_status='done', n_segments_kept=result['n_kept'],
                       hours_kept=round(result['kept_sec'] / 3600, 4), segment_error='')
            log.info('[%d/%d] %s %s: %d kept (%.1f min), %d rejected', n, len(jobs), vid, result['status'],
                     result['n_kept'], result['kept_sec'] / 60, result['n_rejected'])
        pending_save += 1
        if pending_save >= save_every:
            ref.save()
            pending_save = 0

    args = [(job, settings, backend, threads, save_rejected) for job in jobs]
    try:
        if workers <= 1 or len(jobs) == 1:
            for n, a in enumerate(args, 1):
                apply(segment_video(*a), n)
        else:
            ctx = multiprocessing.get_context('spawn')
            kwargs = {'processes': min(workers, len(jobs)), 'initializer': _worker_init}
            if max_tasks:
                kwargs['maxtasksperchild'] = max_tasks
            with ctx.Pool(**kwargs) as pool:
                for n, result in enumerate(pool.imap_unordered(_worker, args, chunksize=1), 1):
                    apply(result, n)
    finally:
        ref.save()

    log.info('segment summary: %d video(s) done, %d failed, %d clips kept (%.2f h), %d rejected, %.0fs',
             counts['done'], counts['failed'], counts['kept'], counts['hours'], counts['rejected'],
             time.monotonic() - started)
    return {'status': 'done', 'rows_in': len(jobs), 'rows_out': counts['done'],
            'message': f"{counts['kept']} clips kept ({counts['hours']:.2f} h), {counts['failed']} video(s) failed"}


if __name__ == '__main__':
    import argparse
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default=None)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    setup_logging(STEP)
    cfg = load_config(args.config)
    state, msg = prerequisite(cfg)
    log.info('prerequisite: %s — %s', state, msg)
    if state == 'fail':
        sys.exit(1)
    if state == 'ok':
        print(run(cfg, limit=args.limit))
