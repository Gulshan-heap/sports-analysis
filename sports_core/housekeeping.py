"""
Keeping the working directories from growing without bound.

A hosted container's disk is as finite as its memory, and nothing in this
project ever deleted anything: every upload, every annotated output video and
every detection cache directory stayed for the life of the deploy. One user
trying a handful of clips is enough to fill the disk, and a full disk fails in
uglier ways than a full heap — the video writer silently produces a truncated
file rather than raising.

These helpers keep the newest N entries in a directory and drop the rest.
Pruning is best-effort by design: a file that is still open, or that another
session is mid-write on, is skipped rather than fought over.
"""

import os
import shutil


def _entries(directory):
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    out = []
    for name in names:
        path = os.path.join(directory, name)
        try:
            out.append((path, os.path.getmtime(path)))
        except OSError:
            continue                      # vanished between listdir and stat
    out.sort(key=lambda pair: pair[1], reverse=True)
    return [path for path, _ in out]


def prune(directory, keep=3, protect=(), suffixes=None):
    """Keep the `keep` most recently modified entries in `directory`.

    Args:
        keep: how many entries to retain.
        protect: paths that must never be removed, whatever their age.
        suffixes: when given, only entries with one of these suffixes are
            considered (both for counting and for removal), so pruning output
            videos cannot touch anything else that shares the directory.

    Returns the number of entries removed.
    """
    protected = {os.path.abspath(p) for p in protect if p}
    candidates = _entries(directory)
    if suffixes:
        candidates = [p for p in candidates
                      if any(p.lower().endswith(s.lower()) for s in suffixes)]

    removed = 0
    for path in candidates[keep:]:
        if os.path.abspath(path) in protected:
            continue
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
            removed += 1
        except OSError:
            continue                      # in use, or already gone
    return removed


def dir_size_mb(directory):
    """Total size of a directory tree in MB (0 when it does not exist)."""
    total = 0
    for root, _dirs, files in os.walk(directory):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total / (1024 * 1024)
