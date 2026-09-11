"""
Fetching model weights
======================
Weight files are gitignored (they are hundreds of MB), so a fresh clone — and
every cloud deploy — starts with no `models/` directory at all. Basketball
already solved this by pulling its three checkpoints from a public Google Drive
folder on first use. Football had no equivalent, which is why its detector
shows as missing on the deployed app while basketball's are ready.

This module holds the generic half of that: work out where the weights should
come from, download them once, and cache them outside the repo.

Sources are tried in order:

    1. a local file already on disk (a clone that has the weights, or a
       previous download)
    2. a URL from Streamlit secrets, then the environment
    3. a URL the user pastes into the sidebar

Supported URL shapes: a Google Drive *folder*, a Google Drive *file*, or any
plain http(s) link to a `.pt`.
"""

import os
import re
import shutil
import tempfile
import urllib.request
from pathlib import Path

CACHE_ROOT = os.path.join(tempfile.gettempdir(), "sports_analysis_weights")


def cache_dir(name):
    """Per-sport download directory, outside the repo so it survives redeploys."""
    path = os.path.join(CACHE_ROOT, name)
    os.makedirs(path, exist_ok=True)
    return path


def configured_url(key, default=None):
    """
    Look up a weights URL from Streamlit secrets, then the environment.

    `key` is used as-is for secrets and upper-cased for the env var, so
    `football_model_url` also reads `FOOTBALL_MODEL_URL`. Missing secrets are
    not an error: `st.secrets` raises when there is no secrets file at all.
    """
    try:
        import streamlit as st
        value = st.secrets.get(key)
        if value:
            return str(value)
    except Exception:
        pass
    return os.environ.get(key.upper()) or default


def _is_drive(url):
    return "drive.google.com" in url


def drive_file_id(url):
    """
    Pull the file id out of any Google Drive share link.

    gdown once offered `fuzzy=True` to do this, but the parameter was removed
    in gdown 6, and requirements.txt allows both 5.x and 6.x. Passing `id=`
    works on every version, so we extract it ourselves.
    """
    for pattern in (r"/file/d/([A-Za-z0-9_-]{10,})",
                    r"[?&]id=([A-Za-z0-9_-]{10,})",
                    r"/d/([A-Za-z0-9_-]{10,})"):
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None


def _download_plain(url, dest_dir, filename=None):
    name = filename or os.path.basename(url.split("?")[0]) or "weights.pt"
    dest = os.path.join(dest_dir, name)
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=120) as response, \
            open(tmp, "wb") as out:
        shutil.copyfileobj(response, out)
    os.replace(tmp, dest)          # never leave a half file looking complete
    return [Path(dest)]


def fetch(url, dest_dir, filename=None, force=False):
    """
    Download weights from `url` into `dest_dir` and return the `.pt` files there.

    Skips the download entirely when `.pt` files are already present and
    `force` is False, so a rerun does not re-fetch hundreds of MB.
    """
    os.makedirs(dest_dir, exist_ok=True)
    existing = sorted(Path(dest_dir).rglob("*.pt"))
    if existing and not force:
        return existing

    if not url:
        return []

    if _is_drive(url):
        import gdown
        if "/folders/" in url:
            gdown.download_folder(url=url, output=dest_dir, quiet=True,
                                  use_cookies=False)
        else:
            file_id = drive_file_id(url)
            if not file_id:
                raise ValueError(f"No Google Drive file id found in {url!r}")
            out = os.path.join(dest_dir, filename or "best.pt")
            gdown.download(id=file_id, output=out, quiet=True)
    else:
        _download_plain(url, dest_dir, filename)

    return sorted(Path(dest_dir).rglob("*.pt"))


def resolve_single(local_path, url, cache_name, filename=None):
    """
    Return (path, source, error) for a one-model sport.

    `source` is "local", "downloaded" or "missing". `error` carries the reason
    a download failed — swallowing it silently just turns a fixable problem
    (an expired link, a changed gdown API) into an unexplained red chip.
    """
    if local_path and os.path.exists(local_path):
        return local_path, "local", None

    dest = cache_dir(cache_name)
    cached = sorted(Path(dest).rglob("*.pt"))
    if cached:
        return str(cached[0]), "downloaded", None

    if not url:
        return None, "missing", "No weights URL configured."

    try:
        found = fetch(url, dest, filename=filename)
    except Exception as exc:
        return None, "missing", f"{type(exc).__name__}: {exc}"

    if not found:
        return None, "missing", "The download produced no .pt file."
    return str(found[0]), "downloaded", None
