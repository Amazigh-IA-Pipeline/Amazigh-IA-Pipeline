"""URL -> 11-character YouTube video ID.

The pipeline matches videos ONLY on this ID, never on URL text or file names
(spec principle 3). Handles every URL form we meet in practice:

    https://www.youtube.com/watch?v=ID&list=...&t=30s
    https://m.youtube.com/watch?v=ID            (mobile)
    https://music.youtube.com/watch?v=ID
    https://youtu.be/ID?si=tracking
    https://www.youtube.com/shorts/ID
    https://www.youtube.com/embed/ID
    https://www.youtube.com/live/ID
    https://www.youtube.com/v/ID
    youtube.com/watch?v=ID                       (no scheme)
    ID                                           (already a bare ID)
"""
from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

_ID_RE = re.compile(r'^[0-9A-Za-z_-]{11}$')
_PATH_RE = re.compile(r'/(?:shorts|embed|live|v|e)/([0-9A-Za-z_-]{11})(?:[/?#&]|$)')


def is_valid_id(value) -> bool:
    return isinstance(value, str) and bool(_ID_RE.match(value.strip()))


def extract_video_id(value) -> str | None:
    """Return the 11-character video ID, or None if it cannot be found."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == 'nan':
        return None
    if _ID_RE.match(text):
        return text
    if not re.match(r'^[a-z]+://', text, re.IGNORECASE):
        text = 'https://' + text
    try:
        parsed = urlparse(text)
    except ValueError:
        return None
    host = (parsed.netloc or '').lower()
    if 'youtu' not in host:
        return None
    query = parse_qs(parsed.query)
    for candidate in query.get('v', []):
        if _ID_RE.match(candidate):
            return candidate
    if host.endswith('youtu.be'):
        candidate = parsed.path.strip('/').split('/')[0]
        if _ID_RE.match(candidate):
            return candidate
    match = _PATH_RE.search(parsed.path + '/')
    if match:
        return match.group(1)
    return None


def watch_url(video_id: str) -> str:
    return f'https://www.youtube.com/watch?v={video_id}'
