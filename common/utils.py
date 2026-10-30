"""Small helpers shared by the steps: repo paths, config, logging, safe Excel I/O."""
from __future__ import annotations

import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

PATHS = {
    'config': ROOT / 'config' / 'config.yaml',
    'banned': ROOT / 'config' / 'banned_channels.yaml',
    'music': ROOT / 'config' / 'music_keywords.yaml',
    'scout_input': ROOT / 'scout' / 'input',
    'scout_output': ROOT / 'scout' / 'output',
    'scouted': ROOT / 'scout' / 'output' / 'scouted.xlsx',
    'scout_removed': ROOT / 'scout' / 'output' / 'scout_removed.xlsx',
    'scout_cache': ROOT / 'scout' / 'output' / 'cache',
    'reference_dir': ROOT / 'reference',
    'reference': ROOT / 'reference' / 'reference.xlsx',
    'reference_backup': ROOT / 'reference' / 'reference.backup.xlsx',
    'fingerprints': ROOT / 'reference' / 'fingerprints.parquet',
    'deduplicated': ROOT / 'dedup_url' / 'output' / 'deduplicated.xlsx',
    'removed_url': ROOT / 'dedup_url' / 'removed_url.xlsx',
    'downloads': ROOT / 'download' / 'downloads',
    'duplicates_report': ROOT / 'dedup_audio' / 'duplicates.xlsx',
    'push_staging': ROOT / 'push' / '_staging',
    'local_hub': ROOT / 'push' / '_local_hub',
    'logs': ROOT / 'logs',
}

try:
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE as _ILLEGAL_XLSX
except ImportError:  # pragma: no cover
    _ILLEGAL_XLSX = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')


# ── config ──────────────────────────────────────────────────────────────────

def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_config(path: str | Path | None = None) -> dict:
    """config.yaml + banned_channels.yaml + music_keywords.yaml in one dict."""
    cfg = _read_yaml(Path(path) if path else PATHS['config'])
    banned = _read_yaml(PATHS['banned'])
    cfg['banned'] = {
        'channel_ids': {str(x).strip() for x in banned.get('channel_ids') or [] if str(x).strip()},
        'channel_urls': {str(x).strip().rstrip('/').lower() for x in banned.get('channel_urls') or [] if str(x).strip()},
        'channel_names': {str(x).strip().lower() for x in banned.get('channel_names') or [] if str(x).strip()},
    }
    cfg['music_keywords'] = [str(k) for k in (_read_yaml(PATHS['music']).get('keywords') or [])]
    return cfg


# ── logging ─────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f'pipeline.{name}')


def setup_logging(run_name: str) -> Path:
    """Console + logs/<run_name>_<timestamp>.log. Safe to call more than once."""
    PATHS['logs'].mkdir(parents=True, exist_ok=True)
    log_path = PATHS['logs'] / f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    root = logging.getLogger('pipeline')
    root.setLevel(logging.INFO)
    root.propagate = False
    for handler in list(root.handlers):
        root.removeHandler(handler)
    fmt = logging.Formatter('%(asctime)s %(levelname)-7s %(name)s: %(message)s', '%H:%M:%S')
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    try:
        console.stream.reconfigure(errors='replace')  # Tifinagh/Arabic on Windows consoles
    except Exception:
        pass
    root.addHandler(console)
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)
    return log_path


# ── time / names ────────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_folder_name(name, fallback='Unknown_Channel') -> str:
    """Windows-safe folder name. Only used ONCE per channel (download step);
    after that the folder name is read back from reference.xlsx."""
    text = str(name or '').strip()
    text = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', text)
    text = re.sub(r'\s+', ' ', text).strip('. ')
    return text[:100] or fallback


# ── Excel ───────────────────────────────────────────────────────────────────

def _clean_cell(value):
    if isinstance(value, str):
        return _ILLEGAL_XLSX.sub('', value)
    return value


def write_excel(df: pd.DataFrame, path: str | Path) -> None:
    """Atomic write: temp file, then rename. Illegal XML characters stripped."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(_clean_cell)
    tmp = path.with_name(path.stem + '.tmp' + path.suffix)
    out.to_excel(tmp, index=False, engine='openpyxl')
    os.replace(tmp, path)


def read_excel(path: str | Path) -> pd.DataFrame:
    """Read every cell as text (keeps video IDs like '0123...' intact)."""
    df = pd.read_excel(path, dtype=object, engine='openpyxl')
    for col in df.columns:
        df[col] = df[col].map(lambda v: '' if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip())
    return df


def to_int(value, default=0) -> int:
    try:
        if value is None or value == '' or (isinstance(value, float) and pd.isna(value)):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_float(value, default=0.0) -> float:
    try:
        if value is None or value == '' or (isinstance(value, float) and pd.isna(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def require_ffmpeg() -> None:
    """Fail early and clearly if ffmpeg / ffprobe are not installed."""
    import shutil
    missing = [tool for tool in ('ffmpeg', 'ffprobe') if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(f"{' and '.join(missing)} not found on PATH. Install FFmpeg "
                           "(Windows: winget install Gyan.FFmpeg), close and reopen the terminal, "
                           "then check with: ffprobe -version")


def _first_number(text: str) -> float | None:
    for line in (text or '').splitlines():
        try:
            value = float(line.strip())
            if value > 0:
                return value
        except ValueError:
            continue
    return None


def probe_duration(path) -> float | None:
    """Real duration of an audio file in seconds, or None if it cannot be read.

    1. the container's duration (ffprobe format=duration)
    2. the audio stream's duration (ffprobe stream=duration)
    3. decode the whole file and read the last timestamp (ffmpeg) — needed for
       YouTube .webm files, which often store no duration in their header
       (ffprobe then answers "N/A")
    """
    import re as _re
    import subprocess
    for entries, extra in (('format=duration', []), ('stream=duration', ['-select_streams', 'a:0'])):
        try:
            res = subprocess.run(['ffprobe', '-v', 'error', *extra, '-show_entries', entries,
                                  '-of', 'default=noprint_wrappers=1:nokey=1', str(path)],
                                 capture_output=True, text=True, timeout=120)
            value = _first_number(res.stdout)
            if value:
                return value
        except Exception:
            pass
    try:
        res = subprocess.run(['ffmpeg', '-nostdin', '-v', 'info', '-i', str(path), '-vn', '-f', 'null', '-'],
                             capture_output=True, text=True, timeout=1800, errors='replace')
        times = _re.findall(r'time=(\d+):(\d+):(\d+(?:\.\d+)?)', res.stderr)
        if times:
            h, m, s = times[-1]
            value = int(h) * 3600 + int(m) * 60 + float(s)
            return value if value > 0 else None
    except Exception:
        pass
    return None


def rel_path(path) -> str:
    """Path relative to the repo root, with forward slashes (portable in the reference)."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def abs_path(stored: str) -> Path:
    p = Path(stored)
    return p if p.is_absolute() else ROOT / p


# ── yt-dlp ──────────────────────────────────────────────────────────────────

def ytdlp_base_options(config: dict) -> dict:
    opts = {'quiet': True, 'no_warnings': True, 'noprogress': True}
    yt = config.get('youtube') or {}
    if yt.get('cookies_file'):
        opts['cookiefile'] = str(yt['cookies_file'])
    if yt.get('cookies_from_browser'):
        opts['cookiesfrombrowser'] = (str(yt['cookies_from_browser']),)
    return opts
