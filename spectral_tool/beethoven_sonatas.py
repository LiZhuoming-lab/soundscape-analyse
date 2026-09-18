from __future__ import annotations

import json
import re
import ssl
import time
import urllib.parse
import urllib.request
from typing import Any
from urllib.error import HTTPError, URLError

import certifi
import pandas as pd

BEETHOVEN_SONATAS_REPO = "craigsapp/beethoven-piano-sonatas"
BEETHOVEN_SONATAS_BRANCH = "main"
BEETHOVEN_SONATAS_WEB_URL = f"https://github.com/{BEETHOVEN_SONATAS_REPO}"
BEETHOVEN_SONATAS_TREE_API_URL = (
    f"https://api.github.com/repos/{BEETHOVEN_SONATAS_REPO}/git/trees/{BEETHOVEN_SONATAS_BRANCH}?recursive=1"
)
BEETHOVEN_SONATAS_SUPPORTED_EXTENSIONS = (".krn", ".kern")


def _ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=certifi.where())


def _network_error_message(repository_name: str) -> str:
    return (
        f"暂时无法连接 {repository_name} 语料库。"
        "这通常是网络抖动、GitHub 限速或远端暂时无响应造成的。"
        "请稍后重试，或先切换回本地上传。"
    )


def _fetch_json(url: str, repository_name: str = "Beethoven piano sonatas", retries: int = 3) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "soundscape-analyse-beethoven-loader",
        },
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, context=_ssl_context(), timeout=30) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError, ConnectionResetError, OSError) as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(0.7 * (attempt + 1))
                continue
            raise RuntimeError(_network_error_message(repository_name)) from exc
    raise RuntimeError(_network_error_message(repository_name)) from last_error


def _sonata_display_name(sonata_number: int, movement_number: int) -> str:
    return f"Beethoven Piano Sonata No.{sonata_number} / Movement {movement_number}"


def _entry_from_tree_path(path: str) -> dict[str, str | int]:
    match = re.fullmatch(r"kern/sonata(\d{2})-(\d)\.(krn|kern)", path)
    if match is None:
        filename = path.split("/")[-1]
        return {
            "path": path,
            "sonata_number": 0,
            "movement_number": 0,
            "display_name": filename,
            "score_name": filename,
            "extension": "." + filename.split(".")[-1].lower() if "." in filename else "",
            "raw_url": (
                f"https://raw.githubusercontent.com/{BEETHOVEN_SONATAS_REPO}/{BEETHOVEN_SONATAS_BRANCH}/"
                f"{urllib.parse.quote(path, safe='/')}"
            ),
        }

    sonata_number = int(match.group(1))
    movement_number = int(match.group(2))
    extension = "." + match.group(3).lower()
    filename = path.split("/")[-1]
    return {
        "path": path,
        "sonata_number": sonata_number,
        "movement_number": movement_number,
        "display_name": _sonata_display_name(sonata_number, movement_number),
        "score_name": filename,
        "extension": extension,
        "raw_url": (
            f"https://raw.githubusercontent.com/{BEETHOVEN_SONATAS_REPO}/{BEETHOVEN_SONATAS_BRANCH}/"
            f"{urllib.parse.quote(path, safe='/')}"
        ),
    }


def list_beethoven_sonata_scores() -> pd.DataFrame:
    tree_payload = _fetch_json(BEETHOVEN_SONATAS_TREE_API_URL)
    rows = [
        _entry_from_tree_path(item["path"])
        for item in tree_payload.get("tree", [])
        if item.get("type") == "blob"
        and str(item.get("path", "")).lower().endswith(BEETHOVEN_SONATAS_SUPPORTED_EXTENSIONS)
    ]
    catalog = pd.DataFrame(rows)
    if catalog.empty:
        return catalog
    return catalog.sort_values(["sonata_number", "movement_number", "path"]).reset_index(drop=True)


def download_beethoven_sonata_score(path: str) -> tuple[bytes, str]:
    entry = _entry_from_tree_path(path)
    request = urllib.request.Request(
        str(entry["raw_url"]),
        headers={"User-Agent": "soundscape-analyse-beethoven-loader"},
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, context=_ssl_context(), timeout=60) as response:
                data = response.read()
            return data, str(entry["score_name"])
        except (HTTPError, URLError, TimeoutError, ConnectionResetError, OSError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.7 * (attempt + 1))
                continue
            raise RuntimeError(_network_error_message("Beethoven piano sonatas")) from exc
    raise RuntimeError(_network_error_message("Beethoven piano sonatas")) from last_error
