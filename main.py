"""Amazigh-IA pipeline — single entry point.

    python main.py                                  # interactive menu
    python main.py --steps scout dedup_url download # chosen steps only
    python main.py --from download --to segment     # a range of steps
    python main.py --all                            # the whole pipeline
    python main.py --all --dry-run                  # show what would run, execute nothing

Steps always run in pipeline order:  scout -> dedup_url -> download -> dedup_audio -> segment -> push
Each step can also be run alone (python download/download.py ...).

Options passed to the steps:
    --push-source downloads|segments|both   (push; default segments)
    --delete-local                          (push)
    --move                                  (dedup_audio)
    --retry-failed                          (download: retry videos marked failed)
    --limit N                               (download, segment: process at most N videos — testing)
"""
from __future__ import annotations

import argparse
import importlib
import multiprocessing
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from common.utils import PATHS, get_logger, load_config, setup_logging  # noqa: E402

STEPS = [
    ('scout', 'scout.scouting'),
    ('dedup_url', 'dedup_url.dedup_url'),
    ('download', 'download.download'),
    ('dedup_audio', 'dedup_audio.dedup_audio'),
    ('segment', 'segment.segment'),
    ('push', 'push.push'),
]
NAMES = [name for name, _ in STEPS]

# (step, its input file, the upstream file it is built from) — for the stale-input warning
STALE_CHECKS = {
    'dedup_url': ('scouted', 'scout_input'),
    'download': ('deduplicated', 'scouted'),
}

log = get_logger('main')


def parse_menu(text: str) -> list[str]:
    text = text.strip().lower()
    if text in ('all', 'a', '*'):
        return list(NAMES)
    chosen = set()
    for token in text.replace(',', ' ').split():
        if '-' in token:
            a, b = token.split('-', 1)
            chosen.update(range(int(a), int(b) + 1))
        elif token.isdigit():
            chosen.add(int(token))
        elif token in NAMES:
            chosen.add(NAMES.index(token) + 1)
        else:
            raise ValueError(f'unknown step: {token}')
    if not chosen or min(chosen) < 1 or max(chosen) > len(NAMES):
        raise ValueError('choose step numbers between 1 and 6')
    return [NAMES[i - 1] for i in sorted(chosen)]


def select_steps(args) -> list[str]:
    if args.all:
        return list(NAMES)
    if args.steps:
        bad = [s for s in args.steps if s not in NAMES]
        if bad:
            raise SystemExit(f'unknown step(s): {bad}. Steps: {NAMES}')
        return [n for n in NAMES if n in args.steps]          # fixed pipeline order
    if args.from_step or args.to_step:
        a = NAMES.index(args.from_step) if args.from_step else 0
        b = NAMES.index(args.to_step) if args.to_step else len(NAMES) - 1
        if a > b:
            raise SystemExit('--from must come before --to in the pipeline order')
        return NAMES[a:b + 1]
    print('\n'.join(f'{i}  {n}' for i, n in enumerate(NAMES, 1)))
    while True:
        try:
            return parse_menu(input('Steps to run (e.g. 1 2 3, 3-5, all): '))
        except ValueError as exc:
            print(exc)


def newest_mtime(key: str) -> float:
    path = PATHS[key]
    if path.is_dir():
        files = [p for p in path.glob('*.xlsx') if not p.name.startswith('~$')]
        return max((p.stat().st_mtime for p in files), default=0.0)
    return path.stat().st_mtime if path.exists() else 0.0


def stale_warning(step: str, selected: list[str]) -> str:
    if step not in STALE_CHECKS:
        return ''
    own_input, upstream = STALE_CHECKS[step]
    previous = NAMES[NAMES.index(step) - 1]
    if previous in selected or not PATHS[own_input].exists():
        return ''
    if newest_mtime(upstream) > newest_mtime(own_input):
        return (f'{PATHS[own_input].name} is older than {upstream.replace("_", " ")} — '
                f'you may want to run {previous} first')
    return ''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--steps', nargs='+', metavar='STEP')
    parser.add_argument('--from', dest='from_step', choices=NAMES)
    parser.add_argument('--to', dest='to_step', choices=NAMES)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--push-source', choices=['downloads', 'segments', 'both'], default='segments')
    parser.add_argument('--delete-local', action='store_true')
    parser.add_argument('--move', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--config', default=None)
    parser.add_argument('--yes', action='store_true', help='do not ask before continuing on a stale-input warning')
    args = parser.parse_args()

    selected = select_steps(args)
    log_path = setup_logging('run')
    config = load_config(args.config)
    options = {'push_source': args.push_source, 'delete_local': args.delete_local, 'move': args.move,
               'retry_failed': args.retry_failed, 'limit': args.limit}
    log.info('steps: %s%s | log: %s', ' -> '.join(selected), ' (DRY RUN)' if args.dry_run else '', log_path)

    summary = []
    exit_code = 0
    for name in selected:
        module = importlib.import_module(dict(STEPS)[name])
        started = time.monotonic()
        try:
            state, message = module.prerequisite(config, **options)
        except Exception as exc:
            state, message = 'fail', f'prerequisite check crashed: {exc}'

        warning = stale_warning(name, selected)
        if warning:
            log.warning('[%s] stale input: %s', name, warning)

        if args.dry_run:
            if state == 'fail' and NAMES.index(name) > 0 and NAMES[NAMES.index(name) - 1] in selected:
                message += ' — will exist after the earlier step(s) of this run'
                state = 'waits'
            log.info('[%s] would run — prerequisite: %s (%s)', name, state, message)
            summary.append((name, f'dry-run: {state}', message, 0, '', ''))
            continue
        if state == 'fail':
            log.error('[%s] FAILED before starting: %s', name, message)
            summary.append((name, 'FAILED', message, 0, '', ''))
            exit_code = 1
            break
        if state == 'skip':
            log.info('[%s] skipped: %s', name, message)
            summary.append((name, 'skipped', message, 0, '', ''))
            continue
        if warning and not args.yes and sys.stdin.isatty():
            if input(f'{warning}. Continue anyway? [y/N] ').strip().lower() != 'y':
                summary.append((name, 'FAILED', 'stopped by user (stale input)', 0, '', ''))
                exit_code = 1
                break

        log.info('=' * 70)
        log.info('[%s] start — %s', name, message)
        try:
            result = module.run(config, **options) or {}
        except KeyboardInterrupt:
            log.error('[%s] interrupted by user — progress saved, re-run to resume', name)
            summary.append((name, 'FAILED', 'interrupted', time.monotonic() - started, '', ''))
            exit_code = 1
            break
        except Exception as exc:
            log.error('[%s] FAILED: %s', name, exc)
            log.debug(traceback.format_exc())
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(traceback.format_exc())
            summary.append((name, 'FAILED', str(exc)[:200], time.monotonic() - started, '', ''))
            exit_code = 1
            break
        elapsed = time.monotonic() - started
        summary.append((name, result.get('status', 'done'), result.get('message', ''), elapsed,
                        result.get('rows_in', ''), result.get('rows_out', '')))
        log.info('[%s] %s in %.0fs — %s', name, result.get('status', 'done'), elapsed, result.get('message', ''))

    log.info('=' * 70)
    log.info('SUMMARY')
    for name, status, message, elapsed, rows_in, rows_out in summary:
        rows = f'rows {rows_in} -> {rows_out}' if rows_in != '' else ''
        log.info('  %-12s %-16s %6.0fs  %-18s %s', name, status, elapsed, rows, message)
    not_run = [n for n in selected if n not in [s[0] for s in summary]]
    if not_run:
        log.info('  not run (stopped at first failure): %s', ', '.join(not_run))
    return exit_code


if __name__ == '__main__':
    multiprocessing.freeze_support()
    sys.exit(main())
